# make_plans_jit.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_utils import get_model, get_true_positives, read_dataset, get_output_dir


# ---------------- utilities ----------------

def _dtype_str(dtype) -> str:
    s = str(dtype)
    return "int" if s.startswith("int") else ("float" if s.startswith("float") else s)

def perturb(low, high, current, values, dtype) -> list:
    """
    Given a target interval [low, high], current value, and the set of feasible values
    seen in training, return a *short* ordered list of candidate perturbations near current.

    Works for both int and float features.
    """
    dtype_kind = _dtype_str(dtype)
    # keep only feasible values in [low, high]
    in_range = [v for v in values if low <= v <= high]

    if dtype_kind == "int":
        cands = sorted(set(int(v) for v in in_range))
    else:  # float (default)
        # keep unique up to 2 decimals to avoid dense grids
        uniq = []
        last = None
        for v in sorted(map(float, in_range)):
            rv = round(v, 2)
            if last is None or rv != last:
                uniq.append(v)
                last = rv
        cands = uniq

    # downsample to at most ~10 by median of bins
    if len(cands) > 10:
        groups = np.array_split(np.array(cands), 10)
        cands = [float(np.median(g)) for g in groups]

    # never include the current value itself
    try:
        cands.remove(float(current))
    except Exception:
        try:
            cands.remove(int(current))
        except Exception:
            pass

    # sort by closeness to current
    cands = sorted(cands, key=lambda x: abs(float(x) - float(current)))
    return cands


def split_inequality(rule, min_val, max_val, pattern):
    """
    Parse rules like:
      "a < feature <= b"  → [a, b]
      "feature > a"       → [a, max_val[feature]]
      "feature <= b"      → [min_val[feature], b]
    Returns: (feature_name, [L, R])
    """
    m = pattern.search(rule)
    if not m:
        return None, [None, None]

    v1, op1, feature_name, op2, v2 = m.groups()
    # Case 1: a < feature <= b
    if v1 is not None and op1 == "<" and op2 in ("<", "<=") and v2 is not None:
        l = float(v1)
        r = float(v2)
        return feature_name, [l, r]

    # Case 2: feature > a  => move into (a, max]
    if op2 == ">" and v2 is not None:
        a = float(v2)
        return feature_name, [a, float(max_val[feature_name])]

    # Case 3: feature <= b => move into [min, b]
    if op2 in ("<", "<=") and v2 is not None:
        b = float(v2)
        return feature_name, [float(min_val[feature_name]), b]

    # Fallback
    return feature_name, [float(min_val.get(feature_name, 0.0)),
                          float(max_val.get(feature_name, 0.0))]

def flip_feature_range(feature, min_val, max_val, importance, rule_str):
    """
    LIME-style rule → target interval [L, feature, R]
    """
    # a < feature <= b
    m = re.search(r"([\d.]+)\s*<\s*" + re.escape(feature) + r"\s*<=\s*([\d.]+)", rule_str)
    if m:
        a, b = map(float, m.groups())
        # if importance > 0 push downwards; else push upwards
        return [min_val, feature, a] if importance > 0 else [b, feature, max_val]

    # feature > a  → move to <= a
    m = re.search(re.escape(feature) + r"\s*>\s*([\d.]+)", rule_str)
    if m:
        a = float(m.group(1))
        return [min_val, feature, a]

    # feature <= b → move to > b
    m = re.search(re.escape(feature) + r"\s*<=\s*([\d.]+)", rule_str)
    if m:
        b = float(m.group(1))
        return [b, feature, max_val]

    # fallback
    return None


def px_rule_to_range(row, train_min, train_max):
    """
    PyExplainer row with columns: feature, value_actual, min, max, operator, threshold
    → target interval [L, feature, R]
    """
    feat = row.get("feature")
    op = str(row.get("operator", "")).strip()
    thr = float(row.get("threshold"))
    # prefer training mins/maxes for feasibility; fall back to per-row min/max if missing
    L = float(train_min.get(feat, row.get("min", np.nan)))
    R = float(train_max.get(feat, row.get("max", np.nan)))
    if op == "<":
        return [L, feat, min(thr, R)]
    else:  # ">"
        return [max(thr, L), feat, R]


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
    Read explainer outputs for JIT (one CSV per TP) and convert into
    per-feature perturbation candidate sets.

    Writes: ./plans/<project>/<model>/<explainer[_strategy]>/plans_all.json
    """
    # where the explanation CSVs are
    output_path = get_output_dir(project_name, explainer_type, model_type)
    proposed_change_path = Path(f"./plans/{project_name}/{model_type}/{explainer_type}")
    if search_strategy:
        proposed_change_path = Path(f"./plans/{project_name}/{model_type}/{explainer_type}_{search_strategy}")
        output_path = output_path / search_strategy
        output_path.mkdir(parents=True, exist_ok=True)
    proposed_change_path.mkdir(parents=True, exist_ok=True)

    file_name = "plans_all.json"

    # numeric training columns only (exclude target)
    num_cols = train.drop(columns=["target"], errors="ignore").select_dtypes(include=[np.number]).columns
    train_min = train[num_cols].min()
    train_max = train[num_cols].max()
    # feasible discrete values per feature (from training)
    feature_values = {f: sorted(set(train.loc[:, f].dropna().astype(float))) for f in num_cols}

    # TPs are identified exactly like during explanation
    model = get_model(project_name, model_type)
    true_positives = get_true_positives(model, train, test)

    all_plans: dict[int, dict[str, list]] = {}
    lime_rule_pattern = re.compile(r"([-]?[\d.]+)?\s*(<|>)?\s*([A-Za-z_]+)\s*(<=|>=|<|>)?\s*([-]?[\d.]+)?")

    for test_idx in tqdm(true_positives.index, desc=f"{project_name}", leave=True, disable=not verbose):
        test_instance = test.loc[test_idx]
        if int(test_instance["target"]) != 1:
            continue  # safety

        explanation_path = output_path / f"{int(test_idx)}.csv"
        if not explanation_path.exists():
            if verbose:
                print(f"[WARN] Missing explanation: {explanation_path}")
            continue

        df = pd.read_csv(explanation_path)

        perturb_features: dict[str, list] = {}

        # ---------- LIME / LIME-HPO ----------
        if explainer_type in ("LIME", "LIME-HPO"):
            if df.empty or "feature" not in df.columns:
                all_plans[int(test_idx)] = {}
                continue

            for _, row in df.iterrows():
                feature = str(row["feature"])
                if feature not in num_cols:
                    continue

                proposed = flip_feature_range(
                    feature,
                    float(train_min[feature]),
                    float(train_max[feature]),
                    float(row.get("importance", 0)),
                    str(row.get("rule", "")),
                )
                if not proposed:
                    continue

                L, feat, R = proposed
                dtype = train.dtypes[feat]
                cands = perturb(L, R, test_instance[feat], feature_values[feat], dtype)
                if cands:
                    perturb_features[feat] = cands

        # ---------- PyExplainer ----------
        elif explainer_type == "PyExplainer":
            # Some files can be meta-only (commit_id,label,pred,score_or_proba,...)
            if df.empty or "feature" not in df.columns or "operator" not in df.columns:
                all_plans[int(test_idx)] = {}
                continue

            for _, row in df.iterrows():
                feat = str(row["feature"])
                if feat not in num_cols:
                    continue

                L, feat, R = px_rule_to_range(row, train_min, train_max)
                # guard against degenerate intervals
                if not np.isfinite(L) or not np.isfinite(R) or L > R:
                    continue
                dtype = train.dtypes[feat]
                cands = perturb(L, R, test_instance[feat], feature_values[feat], dtype)
                if cands:
                    perturb_features[feat] = cands

        else:
            # not requested, but kept for completeness if you add others later
            all_plans[int(test_idx)] = {}
            continue

        all_plans[int(test_idx)] = perturb_features

    # json dump (make numpy ints jsonable)
    def convert_int64(o):
        if isinstance(o, (np.integer, )):
            return int(o)
        raise TypeError

    with open(proposed_change_path / file_name, "w") as f:
        json.dump(all_plans, f, indent=4, default=convert_int64)


# ---------------- optional: importance aggregator (LIME only) ----------------

def get_importance_ratio(train, test, project_name, model_type, explainer_type, verbose=False):
    """
    Sum of importance_ratio per TP (LIME/LIME-HPO only).
    """
    output_path = get_output_dir(project_name, explainer_type, model_type)
    model = get_model(project_name, model_type)
    true_positives = get_true_positives(model, train, test)

    totals = []
    for test_idx in tqdm(true_positives.index, desc=f"{project_name}", leave=True, disable=not verbose):
        path = output_path / f"{int(test_idx)}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or "importance_ratio" not in df.columns:
            continue
        totals.append(float(df["importance_ratio"].fillna(0).sum()))
    return totals


# ---------------- CLI ----------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default="RandomForest")
    parser.add_argument("--explainer_type", type=str, default="PyExplainer",
                        choices=["LIME", "LIME-HPO", "PyExplainer"])
    parser.add_argument("--project", type=str, default="all")
    parser.add_argument("--search_strategy", type=str, default=None)
    parser.add_argument("--only_minimum", action="store_true")   # kept for compatibility (unused)
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
            total = []
            for project in tqdm(project_list, desc="Projects", leave=True, disable=not args.verbose):
                train, test = projects[project]
                total += get_importance_ratio(train, test, project, args.model_type, args.explainer_type, args.verbose)
            if total:
                print(np.mean(np.array(total)))
            else:
                print("No importance ratios found.")
    else:
        for project in tqdm(project_list, desc="Projects", leave=True, disable=not args.verbose):
            train, test = projects[project]
            run_single(train, test, project, args.model_type, args.explainer_type, args.search_strategy, args.verbose)
            if args.verbose:
                print(f"[OK] wrote plans to ./plans/{project}/{args.model_type}/{args.explainer_type}"
                      f"{'_'+args.search_strategy if args.search_strategy else ''}/plans_all.json")
