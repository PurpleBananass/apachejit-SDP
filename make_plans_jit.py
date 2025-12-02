#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_plans_jit_new.py

Semantically equivalent to the old make_plans_jit.py for:
- LIME
- LIME-HPO

but:
- fixes dtype handling and negative thresholds
- adds support for PyExplainer rules
- keeps the same JSON structure: {test_idx: {feature: [candidate_values...]}}
- keeps the same importance_ratio semantics (only for actionable rules)
"""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_utils import get_model, get_true_positives, read_dataset, get_output_dir


# ---------------- utilities ----------------

def _dtype_str(dtype) -> str:
    """Normalize numpy/pandas dtype to 'int', 'float', or raw string."""
    s = str(dtype)
    if s.startswith("int"):
        return "int"
    if s.startswith("float"):
        return "float"
    return s


def perturb(low: float, high: float, current: Any, values: List[Any], dtype) -> List[float]:
    """
    Semantics: same as old version.

    Given [low, high], the current value, and all observed training values for the feature:
    - keep only values within [low, high]
    - for floats: deduplicate by rounding to 2 decimals
    - if >10 values: split into 10 groups and take the median of each
    - remove current if present
    - sort the remaining values by |v - current| ascending
    """
    dtype_kind = _dtype_str(dtype)

    # 1) filter by interval
    in_range = [v for v in values if low <= v <= high]

    if dtype_kind == "int":
        perturbations = [int(v) for v in in_range]
    elif dtype_kind == "float":
        if not in_range:
            return []
        perturbations = []
        candidates = sorted(map(float, in_range))
        last = candidates[0]
        perturbations.append(last)
        for candidate in candidates[1:]:
            # compare to 2 decimal places (same idea as old version)
            if round(last, 2) != round(candidate, 2):
                perturbations.append(candidate)
                last = candidate
    else:
        # fallback: treat as float
        if not in_range:
            return []
        perturbations = sorted(map(float, in_range))

    # 2) downsample to at most 10 candidates by median of bins
    if len(perturbations) > 10:
        groups = np.array_split(np.array(perturbations, dtype=float), 10)
        perturbations = [float(np.median(group)) for group in groups]

    # 3) remove current if present
    cur_f = float(current)
    perturbations = [p for p in perturbations if float(p) != cur_f]

    # 4) sort by closeness to current
    perturbations = sorted(perturbations, key=lambda x: abs(float(x) - cur_f))

    return perturbations


def flip_feature_range(
    feature: str,
    min_val: float,
    max_val: float,
    importance: float,
    rule_str: str,
) -> List[Any] | None:
    """
    Semantics: same as old version.

    From a LIME rule string, derive the "flip interval" [L, feature, R]:

    - 'a < feature <= b':
        if importance > 0  -> move into [min_val, a]
        else               -> move into [b, max_val]
    - 'feature > a'       -> move into [min_val, a]
    - 'feature <= b'      -> move into [b, max_val]
    """
    # Case: a < feature <= b (now supports negative)
    m = re.search(r"([-]?[\d.]+)\s*<\s*" + re.escape(feature) + r"\s*<=\s*([-]?[\d.]+)", rule_str)
    if m:
        a, b = map(float, m.groups())
        if importance > 0:
            return [float(min_val), feature, float(a)]
        else:
            return [float(b), feature, float(max_val)]

    # Case: feature > a
    m = re.search(re.escape(feature) + r"\s*>\s*([-]?[\d.]+)", rule_str)
    if m:
        a = float(m.group(1))
        return [float(min_val), feature, float(a)]

    # Case: feature <= b
    m = re.search(re.escape(feature) + r"\s*<=\s*([-]?[\d.]+)", rule_str)
    if m:
        b = float(m.group(1))
        return [float(b), feature, float(max_val)]

    # Fallback: no actionable interval
    # (old code printed "Not Available" and returned None)
    # print("Not Available", rule_str)
    return None


def px_rule_to_range(
    row: pd.Series,
    train_min: pd.Series,
    train_max: pd.Series,
) -> Tuple[float, str, float] | None:
    """
    PyExplainer-style rule row -> [L, feature, R]

    Expected columns:
      - feature
      - operator: '<' or '>'
      - threshold
      - min, max: optional per-row min/max (fallback only)

    Semantics: choose an interval on the "flip" side of the threshold,
    but bounded by training min/max for feasibility.
    """
    feat = str(row.get("feature"))
    if feat not in train_min.index:
        return None

    op = str(row.get("operator", "")).strip()
    try:
        thr = float(row.get("threshold"))
    except Exception:
        return None

    # prefer dataset-wide bounds
    L = float(train_min[feat])
    R = float(train_max[feat])

    if op == "<":
        # current rule is something like "feature < thr"
        # To flip, move into [thr, R]
        return float(thr), feat, R
    elif op == ">":
        # rule "feature > thr" -> move into [L, thr]
        return L, feat, float(thr)

    # unsupported operator
    return None


# ---------------- core: build plans ----------------

def run_single(
    train: pd.DataFrame,
    test: pd.DataFrame,
    project_name: str,
    model_type: str,
    explainer_type: str,
    search_strategy: str | None,
    verbose: bool = False,
):
    """
    Semantics:
      - For each true positive instance,
        read the explainer CSV and construct a per-feature set of
        candidate values to try (plans_all.json).

    Output path:
      ./plans/<project_name>/<model_type>/<explainer_type>[_<search_strategy>]/plans_all.json
    """

    # Where explanation CSVs are stored
    output_path = get_output_dir(project_name, explainer_type, model_type)

    # Where we store the JSON plans
    proposed_change_path = Path(f"./plans/{project_name}/{model_type}/{explainer_type}")
    if search_strategy is not None:
        proposed_change_path = Path(
            f"./plans/{project_name}/{model_type}/{explainer_type}_{search_strategy}"
        )
        output_path = output_path / search_strategy
        output_path.mkdir(parents=True, exist_ok=True)
    proposed_change_path.mkdir(parents=True, exist_ok=True)

    file_name = "plans_all.json"

    # train_min / train_max semantics: same as old (over all columns)
    train_min = train.min()
    train_max = train.max()

    # feature_values semantics: same as old (all columns, not just numeric),
    # but practically you'll only use numeric columns present in explanations.
    feature_values: Dict[str, List[Any]] = {
        feature: sorted(set(train.loc[:, feature].dropna()))
        for feature in train.columns
        if feature in train.columns
    }

    # TPs defined exactly as in explanation step
    model = get_model(project_name, model_type)
    true_positives = get_true_positives(model, train, test)

    all_plans: Dict[int, Dict[str, List[float]]] = {}

    for test_idx in tqdm(
        true_positives.index, desc=f"{project_name}", leave=True, disable=not verbose
    ):
        test_instance = test.loc[test_idx]
        # safety check: should be defect (1) by construction
        if "target" in test_instance.index:
            assert test_instance["target"] == 1

        explanation_path = output_path / f"{int(test_idx)}.csv"
        if not explanation_path.exists():
            if verbose:
                print(f"[WARN] Missing explanation: {explanation_path}")
            all_plans[int(test_idx)] = {}
            continue

        df = pd.read_csv(explanation_path)
        perturb_features: Dict[str, List[float]] = {}

        # -------------------- LIME / LIME-HPO --------------------
        if explainer_type in ("LIME", "LIME-HPO"):
            if df.empty:
                all_plans[int(test_idx)] = {}
                continue

            # Expect header: feature,value,importance,min,max,rule,importance_ratio
            for _, row in df.iterrows():
                try:
                    feature = row["feature"]
                    importance = float(row["importance"])
                    rule = str(row["rule"])
                except KeyError:
                    # fall back to positional unpack (old behavior)
                    vals = row.values
                    if len(vals) < 7:
                        continue
                    feature, value, importance, min_val, max_val, rule, importance_ratio = vals
                    importance = float(importance)
                    rule = str(rule)

                if feature not in feature_values:
                    continue

                proposed = flip_feature_range(
                    str(feature),
                    train_min[feature],
                    train_max[feature],
                    importance,
                    rule,
                )
                if not proposed:
                    continue

                L, feat, R = proposed
                dtype = train.dtypes[feat]
                cands = perturb(
                    float(L),
                    float(R),
                    test_instance[feat],
                    feature_values[feat],
                    dtype,
                )
                if not cands:
                    continue
                # last one wins (same as old semantics if same feature appears multiple times)
                perturb_features[feat] = cands

        # -------------------- PyExplainer --------------------
        elif explainer_type == "PyExplainer":
            # Expect at least: feature, operator, threshold
            if df.empty or "feature" not in df.columns or "operator" not in df.columns:
                all_plans[int(test_idx)] = {}
                continue

            for _, row in df.iterrows():
                feat = str(row.get("feature"))
                if feat not in feature_values:
                    continue

                res = px_rule_to_range(row, train_min, train_max)
                if res is None:
                    continue

                L, feat, R = res
                dtype = train.dtypes[feat]
                cands = perturb(
                    float(L),
                    float(R),
                    test_instance[feat],
                    feature_values[feat],
                    dtype,
                )
                if not cands:
                    continue
                perturb_features[feat] = cands

        else:
            # unsupported explainer_type here
            all_plans[int(test_idx)] = {}
            continue

        all_plans[int(test_idx)] = perturb_features

    # json dump (handle numpy ints)
    def convert_int64(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        raise TypeError

    with open(proposed_change_path / file_name, "w") as f:
        json.dump(all_plans, f, indent=4, default=convert_int64)


# ---------------- importance aggregator (same semantics as old) ----------------

def get_importance_ratio(
    train: pd.DataFrame,
    test: pd.DataFrame,
    project_name: str,
    model_type: str,
    explainer_type: str,
    verbose: bool = False,
):
    """
    Semantics: exactly like old version.

    For LIME/LIME-HPO:
      - for each TP, scan rules
      - only count importance_ratio if flip_feature_range(...) returned a non-None interval
      - sum those ratios per TP
    """
    output_path = get_output_dir(project_name, explainer_type, model_type)

    train_min = train.min()
    train_max = train.max()

    model = get_model(project_name, model_type)
    true_positives = get_true_positives(model, train, test)

    total: List[float] = []
    for test_idx in tqdm(
        true_positives.index, desc=f"{project_name}", leave=True, disable=not verbose
    ):
        test_instance = test.loc[test_idx]
        if "target" in test_instance.index:
            assert test_instance["target"] == 1

        if explainer_type not in ("LIME", "LIME-HPO"):
            # importance_ratio semantics only defined for LIME-type outputs
            continue

        explanation_path = output_path / f"{int(test_idx)}.csv"
        if not explanation_path.exists():
            if verbose:
                print(f"[WARN] Missing explanation: {explanation_path}")
            continue

        df = pd.read_csv(explanation_path)
        if df.empty:
            continue

        ratios: List[float] = []
        for _, row in df.iterrows():
            try:
                feature = row["feature"]
                importance = float(row["importance"])
                rule = str(row["rule"])
                importance_ratio = float(row["importance_ratio"])
            except KeyError:
                vals = row.values
                if len(vals) < 7:
                    continue
                feature, value, importance, min_val, max_val, rule, importance_ratio = vals
                importance = float(importance)
                rule = str(rule)
                importance_ratio = float(importance_ratio)

            if feature not in train_min.index:
                continue

            proposed = flip_feature_range(
                str(feature),
                train_min[feature],
                train_max[feature],
                importance,
                rule,
            )
            if proposed:
                ratios.append(importance_ratio)

        if ratios:
            total.append(sum(ratios))

    return total


# ---------------- CLI ----------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default="RandomForest")
    parser.add_argument(
        "--explainer_type",
        type=str,
        default="LIME-HPO",
        choices=["LIME", "LIME-HPO", "PyExplainer"],
    )
    parser.add_argument("--project", type=str, default="all")
    parser.add_argument("--search_strategy", type=str, default=None)
    parser.add_argument("--only_minimum", action="store_true")  # kept for compatibility (unused)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compute_importance", action="store_true")

    args = parser.parse_args()
    projects = read_dataset()

    if args.project == "all":
        project_list = list(sorted(projects.keys()))
    else:
        project_list = args.project.split()

    if args.compute_importance:
        if args.explainer_type not in ("LIME", "LIME-HPO"):
            print("importance_ratio is only available for LIME/LIME-HPO outputs.")
        else:
            total: List[float] = []
            for project in tqdm(
                project_list, desc="Projects", leave=True, disable=not args.verbose
            ):
                train, test = projects[project]
                total += get_importance_ratio(
                    train, test, project, args.model_type, args.explainer_type, args.verbose
                )
            if total:
                print(np.mean(np.array(total)))
            else:
                print("No importance ratios found.")
    else:
        for project in tqdm(
            project_list, desc="Projects", leave=True, disable=not args.verbose
        ):
            train, test = projects[project]
            print(project)
            run_single(
                train,
                test,
                project,
                args.model_type,
                args.explainer_type,
                args.search_strategy,
                args.verbose,
            )
