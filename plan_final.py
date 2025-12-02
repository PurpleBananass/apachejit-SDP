#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert CfExplainer / PyExplainer rules into planner-style plans_all.json for ALL projects.

CfExplainer strategy for each TP test instance:
    1. Try to use TOP "no bug" rule (no_bug_rules_idx<idx>.csv).
       - For each antecedent item (feature bin), propose values INSIDE that bin.
    2. If there are no no-bug rules:
       - Use TOP "bug" rule (bug_rules_idx<idx>.csv).
       - For each antecedent item (feature bin), propose values in the
         "opposite" bin (move out of bug region: low -> high, high/mid -> low).

PyExplainer strategy for each TP test instance:
    - Read <OUTPUT>/<project>/PyExplainer/<model>/<idx>.csv:
        commit_id,label,pred,score_or_proba,feature,value_actual,min,max,operator,threshold
    - For each row, interpret "feature operator threshold" as belonging to the
      *defective rule region* and construct the COMPLEMENT interval:
        * "feature > thr" or ">=" -> [train_min[feature], thr]
        * "feature < thr" or "<=" -> [thr, train_max[feature]]
      Then generate candidate values in that interval.

Plans are written to:
    ./plans/{project}/{model_type}/{explainer_name}/plans_all.json

Format:
    {
      "<test_idx>": {
        "<feature_name>": [candidate_value_1, candidate_value_2, ...],
        ...
      },
      ...
    }
"""

from __future__ import annotations

import ast
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm

from data_utils import read_dataset, get_model, get_true_positives, get_output_dir


# ---------------------------------------------------------------------
# Helper: perturbation generation (adapted from your previous code)
# ---------------------------------------------------------------------


def perturb(
    low: float,
    high: float,
    current: float,
    values: List[float],
    dtype: Any,
) -> List[float]:
    """
    Given:
        - an interval [low, high]
        - the current feature value
        - all unique values for this feature in train
        - the dtype (int / float)
    returns a list of candidate values inside [low, high],
    sorted by distance from current, and downsampled to at most 10 values.
    """
    values = list(values)

    # Robust dtype check
    if np.issubdtype(dtype, np.integer):
        perturbations = [val for val in values if low <= val <= high]

    elif np.issubdtype(dtype, np.floating):
        perturbations = []
        candidates = [val for val in values if low <= val <= high]
        if len(candidates) == 0:
            return []
        last = candidates[0]
        perturbations.append(last)
        for candidate in candidates[1:]:
            # compare up to 2 decimal places
            if round(last, 2) != round(candidate, 2):
                perturbations.append(candidate)
                last = candidate
    else:
        # Non-numeric features are not handled here
        return []

    if len(perturbations) > 10:
        # divide perturbations into 10 groups and select median of each
        groups = np.array_split(perturbations, 10)
        perturbations = [float(np.median(group)) for group in groups]

    # Remove current value if present
    if current in perturbations:
        perturbations.remove(current)

    perturbations = sorted(perturbations, key=lambda x: abs(x - current))
    return perturbations


def convert_int64(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError


# ---------------------------------------------------------------------
# Core helpers for CfExplainer
# ---------------------------------------------------------------------


def _pick_best_rule(rules_df: pd.DataFrame) -> pd.Series:
    """
    Pick the "best" rule by:
        - highest weightSupp (if available),
        - then support,
        - then confidence.
    Fallback: just first row.
    """
    if rules_df.empty:
        raise ValueError("No rules available")

    sort_cols = [c for c in ["weightSupp", "support", "confidence"] if c in rules_df.columns]
    if not sort_cols:
        return rules_df.iloc[0]
    rules_sorted = rules_df.sort_values(sort_cols, ascending=False)
    return rules_sorted.iloc[0]


def _parse_antecedents(cell) -> list[str]:
    """
    Parse the 'antecedents' cell from rules CSV.

    Handles:
      - actual set/frozenset/list/tuple objects
      - strings like "frozenset({'aexp_aexp=0', 'la_la=0'})"
      - strings like "{'aexp_aexp=0', 'la_la=0'}"
    Returns a list of antecedent item strings, or [] if parsing fails.
    """
    # Already a set/frozenset/list/tuple → just list() it
    if isinstance(cell, (set, frozenset, list, tuple)):
        return [str(x) for x in cell]

    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return []

    # Handle "frozenset(...)" wrapper explicitly
    if s.startswith("frozenset(") and s.endswith(")"):
        inner = s[len("frozenset("):-1].strip()
        # empty frozenset()
        if not inner:
            return []
        try:
            parsed = ast.literal_eval(inner)
            if isinstance(parsed, (set, frozenset, list, tuple)):
                return [str(x) for x in parsed]
            # if it's just a single string or something else
            return [str(parsed)]
        except Exception:
            return []

    # Fallback: try literal_eval on the whole string
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (set, frozenset, list, tuple)):
            return [str(x) for x in parsed]
        return [str(parsed)]
    except Exception:
        return []


# ---------------------------------------------------------------------
# Core logic: build plans for ONE project (CfExplainer OR PyExplainer)
# ---------------------------------------------------------------------


def build_plans_for_project(
    project_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_type: str,
    explainer_name: str,
    label_col: str = "target",
    n_bins: int = 3,
    verbose: bool = False,
) -> None:
    """
    For a single project:
        - identify TP instances,
        - depending on explainer_name:
            * CfExplainer: use synthetic neighbourhood + CWCAR rules
            * PyExplainer: use per-feature rule CSVs
        - build per-feature candidate perturbations,
        - write plans_all.json under ./plans/{project}/{model_type}/{explainer_name}.
    """
    if label_col not in train.columns or label_col not in test.columns:
        print(f"[WARN] Skipping {project_name}: label column '{label_col}' missing.")
        return

    feature_cols = [c for c in train.columns if c != label_col]

    # Precompute unique values & dtypes per feature from train
    train_feature_values: Dict[str, List[float]] = {
        feat: sorted(set(train[feat].dropna().values.tolist())) for feat in feature_cols
    }
    train_dtypes = train.dtypes
    train_min = train.min()
    train_max = train.max()

    # Model & true positives
    model = get_model(project_name, model_type)
    tps = get_true_positives(model, train, test, label=label_col)

    if tps.empty:
        print(f"[INFO] No true positives for {project_name}, skipping.")
        return

    if verbose:
        print(f"[INFO] {project_name}: {len(tps)} true positives")

    # Explainer output directory
    expl_out_dir: Path = get_output_dir(project_name, explainer_name, model_type)

    # Plans output directory (same pattern as your previous planner)
    plans_dir = Path(f"./plans/{project_name}/{model_type}/{explainer_name}")
    plans_dir.mkdir(parents=True, exist_ok=True)
    plans_path = plans_dir / "plans_all.json"

    all_plans: Dict[int, Dict[str, List[float]]] = {}

    # ------------------------------ CfExplainer branch ------------------------------
    if explainer_name == "CfExplainer":
        for test_idx in tqdm(
            tps.index,
            desc=f"{project_name} (Cf→plans)",
            leave=False,
            disable=not verbose,
        ):
            test_instance = test.loc[test_idx]
            if test_instance[label_col] != 1:
                # Safety check; get_true_positives should have filtered already
                continue

            syn_path = expl_out_dir / f"synthetic_neighbourhood_idx{test_idx}.csv"
            nobug_rules_path = expl_out_dir / f"no_bug_rules_idx{test_idx}.csv"
            bug_rules_path = expl_out_dir / f"bug_rules_idx{test_idx}.csv"

            if not syn_path.exists():
                continue

            # Load synthetic neighbourhood
            synthetic_df = pd.read_csv(syn_path, index_col=0)
            if synthetic_df.empty:
                continue

            # Decide which rule set to use: prefer no-bug, fallback to bug
            rules_df_nobug = None
            rules_df_bug = None

            if nobug_rules_path.exists():
                tmp = pd.read_csv(nobug_rules_path)
                if not tmp.empty:
                    rules_df_nobug = tmp

            if rules_df_nobug is not None:
                # Use no-bug rule (target region)
                best_rule = _pick_best_rule(rules_df_nobug)
                rule_type = "nobug"
            else:
                # Fallback: bug rules (region to escape)
                if not bug_rules_path.exists():
                    continue
                tmp = pd.read_csv(bug_rules_path)
                if tmp.empty:
                    continue
                rules_df_bug = tmp
                best_rule = _pick_best_rule(rules_df_bug)
                rule_type = "bug"

            antecedent_items = _parse_antecedents(best_rule["antecedents"])
            if not antecedent_items:
                if verbose:
                    print(f"[WARN] No antecedents parsed for test_idx={test_idx} in {project_name}.")
                continue

            # Refit KBinsDiscretizer on this synthetic neighbourhood to recover bin edges
            syn_features = synthetic_df[feature_cols]
            disc = KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy="quantile",
            )
            _ = disc.fit_transform(syn_features.values)
            # disc.bin_edges_: shape (n_features, n_bins+1)

            # Map feature -> bin edges
            feature_to_edges = {
                feat: disc.bin_edges_[i]
                for i, feat in enumerate(feature_cols)
            }

            perturb_features: Dict[str, List[float]] = {}

            for item in antecedent_items:
                item_str = str(item)

                # Expect patterns like 'la_la=0', 'aexp_aexp=2', ...
                if "_" not in item_str or "=" not in item_str:
                    continue

                feature_name, rest = item_str.split("_", 1)  # e.g., "la", "la=0"
                if feature_name not in feature_to_edges:
                    continue

                try:
                    bin_idx_str = rest.split("=", 1)[1]
                    bin_idx = int(bin_idx_str)
                except Exception:
                    continue

                edges = feature_to_edges[feature_name]
                n_bin_edges = len(edges) - 1
                if bin_idx < 0 or bin_idx >= n_bin_edges:
                    continue

                if rule_type == "nobug":
                    # For no-bug rules: move INTO this bin
                    target_bin = bin_idx
                else:
                    # For bug rules: move OUT of this bin
                    # Simple heuristic: if current bin is low (0) -> push to highest bin,
                    # else -> push to lowest bin.
                    if bin_idx == 0:
                        target_bin = n_bin_edges - 1
                    else:
                        target_bin = 0

                low = float(edges[target_bin])
                high = float(edges[target_bin + 1])

                # Pull dtype and training values
                if feature_name not in train_feature_values:
                    continue

                dtype = train_dtypes[feature_name]
                values = train_feature_values.get(feature_name, [])
                if not values:
                    continue

                current_val = float(test_instance[feature_name])

                candidates = perturb(
                    low=low,
                    high=high,
                    current=current_val,
                    values=values,
                    dtype=dtype,
                )
                if candidates:
                    perturb_features[feature_name] = candidates

            if perturb_features:
                all_plans[int(test_idx)] = perturb_features

    # ------------------------------ PyExplainer branch ------------------------------
    elif explainer_name == "PyExplainer":
        for test_idx in tqdm(
            tps.index,
            desc=f"{project_name} (Py→plans)",
            leave=False,
            disable=not verbose,
        ):
            test_instance = test.loc[test_idx]
            if test_instance[label_col] != 1:
                continue

            explanation_path = expl_out_dir / f"{test_idx}.csv"
            if not explanation_path.exists():
                continue

            explanation = pd.read_csv(explanation_path)
            if explanation.empty or "feature" not in explanation.columns:
                continue

            perturb_features: Dict[str, List[float]] = {}

            for _, row in explanation.iterrows():
                feature = row.get("feature")
                op = row.get("operator")
                thr = row.get("threshold")

                if pd.isna(feature) or pd.isna(op) or pd.isna(thr):
                    continue

                feature = str(feature)
                if feature not in train_feature_values:
                    continue

                try:
                    thr_val = float(thr)
                except Exception:
                    continue

                # Build COMPLEMENT intervals of the rule region
                # If rule is "feature > thr", region is (thr, +inf) -> complement [min, thr]
                # If rule is "feature <= thr", region is [min, thr] -> complement [thr, max]
                if op in (">", ">="):
                    low = float(train_min[feature])
                    high = thr_val
                elif op in ("<", "<="):
                    low = thr_val
                    high = float(train_max[feature])
                else:
                    # Unsupported operator
                    continue

                if low > high:
                    low, high = high, low

                dtype = train_dtypes[feature]
                values = train_feature_values[feature]
                current_val = float(test_instance[feature])

                candidates = perturb(
                    low=low,
                    high=high,
                    current=current_val,
                    values=values,
                    dtype=dtype,
                )
                if not candidates:
                    continue

                # Merge with existing candidates for this feature (union)
                existing = perturb_features.get(feature, [])
                merged = sorted(
                    set(existing) | set(candidates),
                    key=lambda x: abs(x - current_val),
                )
                perturb_features[feature] = merged

            if perturb_features:
                all_plans[int(test_idx)] = perturb_features

    else:
        raise ValueError(f"Unsupported explainer_name: {explainer_name}")

    # Save all plans for this project
    if all_plans:
        with open(plans_path, "w") as f:
            json.dump(all_plans, f, indent=4, default=convert_int64)
        print(f"[INFO] Saved plans for {project_name} ({explainer_name}) -> {plans_path}")
    else:
        print(f"[INFO] No plans generated for {project_name} ({explainer_name}).")


# ---------------------------------------------------------------------
# Main: run over ALL (or selected) projects
# ---------------------------------------------------------------------


def main():
    parser = ArgumentParser(
        description="Convert CfExplainer / PyExplainer outputs into planner-style plans_all.json.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="RandomForest",
        help="Model name passed to get_model (default: RandomForest).",
    )
    parser.add_argument(
        "--explainer_name",
        type=str,
        default="CfExplainer",
        help="Folder name used for explainer outputs in OUTPUT "
             "(e.g., CfExplainer, PyExplainer).",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="target",
        help="Name of label column (default: target).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="all",
        help="Project name or 'all' (default: all).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable progress bars and extra logs.",
    )

    args = parser.parse_args()

    projects = read_dataset()
    if args.project == "all":
        project_names = sorted(projects.keys())
    else:
        project_names = args.project.split()

    print(f"[INFO] Projects to process: {project_names}")
    print(f"[INFO] Explainer: {args.explainer_name}")

    for project_name in project_names:
        train, test = projects[project_name]
        build_plans_for_project(
            project_name=project_name,
            train=train,
            test=test,
            model_type=args.model_type,
            explainer_name=args.explainer_name,
            label_col=args.label_col,
            n_bins=3,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()