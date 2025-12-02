#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_utils import (
    read_dataset,
    get_model,
    get_true_positives,
    get_output_dir,
)
from hyparams import PROPOSED_CHANGES

# ----------------- PyExplainer rule parsing helpers ----------------- #

# match things like "la > -0.26", "age <= 10.5", "ns >= 2"
_ATOM_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*(<=|>=|<|>)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)


def parse_pyexp_atom(atom: str):
    """
    Parse a single atomic condition from PyExplainer's rule string.
    Returns (feature, op, threshold) or (None, None, None) if it doesn't match.
    """
    m = _ATOM_RE.match(atom.strip())
    if not m:
        return None, None, None
    feat, op, thr = m.groups()
    return feat, op, float(thr)


def pyexp_flip_range_for_instance(
    feature: str,
    op: str,
    thr: float,
    current_value: float,
    train_min: pd.Series,
    train_max: pd.Series,
):
    """
    Given a defect rule atom (feature op thr) and the current *defect* instance,
    return [low, high] describing a range of values that will BREAK this atom
    for that instance, while staying inside [train_min, train_max].

    If:
      - the atom is already false for this instance, or
      - there is no feasible value in the real domain that can break it,

    return None.
    """
    fmin = float(train_min[feature])
    fmax = float(train_max[feature])

    # --- detect infeasible thresholds wrt real domain ---
    # For "feature > thr": to break it we need x <= thr.
    # If thr < fmin, you can never go below thr while staying >= fmin.
    if op in (">", ">=") and thr <= fmin:
        return None

    # For "feature < thr": to break it we need x >= thr.
    # If thr > fmax, you can never reach thr while staying <= fmax.
    if op in ("<", "<=") and thr >= fmax:
        return None

    # --- now assume thr is inside (fmin, fmax), handle usual cases ---

    # feature > thr  (holds when x > thr)
    if op == ">":
        if current_value > thr:
            # break: move to <= thr but stay in domain
            low = fmin
            high = thr
            if low >= high:
                return None
            return [low, high]
        else:
            # already false, no move needed
            return None

    # feature >= thr
    if op == ">=":
        if current_value >= thr:
            # break: move to < thr
            low = fmin
            high = thr
            if low >= high:
                return None
            return [low, high]
        else:
            return None

    # feature < thr  (holds when x < thr)
    if op == "<":
        if current_value < thr:
            # break: move to >= thr
            low = thr
            high = fmax
            if low >= high:
                return None
            return [low, high]
        else:
            return None

    # feature <= thr
    if op == "<=":
        if current_value <= thr:
            # break: move to > thr
            low = thr
            high = fmax
            if low >= high:
                return None
            return [low, high]
        else:
            return None

    return None


# ----------------- perturbation helper (same logic as before) ----------------- #


def perturb(low, high, current, values, dtype):
    dtype_str = str(dtype)

    if dtype_str.startswith("int"):
        cand = [int(val) for val in values if low <= val <= high]
    elif dtype_str.startswith("float"):
        cand_raw = [float(val) for val in values if low <= val <= high]
        if not cand_raw:
            return []
        # dedupe by rounding to 2 decimals
        cand = []
        last = None
        for v in sorted(cand_raw):
            if last is None or round(last, 2) != round(v, 2):
                cand.append(v)
                last = v
    else:
        # unsupported type for counterfactual numeric tweaking
        return []

    # cap at 10 by grouping and taking medians
    if len(cand) > 10:
        groups = np.array_split(cand, 10)
        cand = [float(np.median(g)) for g in groups]

    # remove current value if present
    cand = [v for v in cand if not np.isclose(v, current)]

    # sort by closeness to current
    cand = sorted(cand, key=lambda x: abs(x - current))
    return cand


# ----------------- main plan generation for PyExplainer ----------------- #


def run_single_project_pyexp(
    train: pd.DataFrame,
    test: pd.DataFrame,
    project_name: str,
    model_type: str,
    verbose: bool = False,
):
    """
    For a single project:
      - load PyExplainer rule CSVs for each true positive
      - build perturbation plans per feature
      - save as PROPOSED_CHANGES/<project>/<model_type>/PyExplainer/plans_all.json
    """
    # Where PyExplainer saved rule CSVs:
    rules_dir = get_output_dir(project_name, "PyExplainer", model_type)
    rules_dir.mkdir(parents=True, exist_ok=True)

    # Where we save plans_all.json:
    plan_dir = (
        Path(PROPOSED_CHANGES) / f"{project_name}/{model_type}/PyExplainer"
    )
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_file = plan_dir / "plans_all.json"

    # numeric training stats (exclude target)
    feat_cols = [c for c in train.columns if c != "target"]
    train_min = train[feat_cols].min()
    train_max = train[feat_cols].max()

    # all discrete value grids for perturbation
    feature_values = {
        feat: sorted(set(train[feat_cols][feat].dropna()))
        for feat in feat_cols
    }

    model = get_model(project_name, model_type)
    true_positives = get_true_positives(model, train, test)

    if len(true_positives) == 0:
        if verbose:
            print(f"{project_name}: no true positives, skipping.")
        # still write an empty file for consistency
        with open(plan_file, "w") as f:
            json.dump({}, f, indent=4)
        return

    if verbose:
        print(f"{project_name}: #true_positives = {len(true_positives)}")

    all_plans = {}

    for test_idx in tqdm(
        true_positives.index,
        desc=f"{project_name}",
        leave=True,
        disable=not verbose,
    ):
        rules_path = rules_dir / f"{test_idx}.csv"
        if not rules_path.exists():
            if verbose:
                print(f"[PyExp] missing rules for {project_name} idx={test_idx}")
            all_plans[int(test_idx)] = {}
            continue

        try:
            rules_df = pd.read_csv(rules_path)
        except pd.errors.EmptyDataError:
            all_plans[int(test_idx)] = {}
            continue

        if rules_df.empty:
            all_plans[int(test_idx)] = {}
            continue

        # full test row (including target)
        test_row = test.loc[test_idx]
        assert test_row["target"] == 1

        perturb_features = {}

        for _, row in rules_df.iterrows():
            rule_str = str(row["rule"])

            # Optional: only use rules that predict 'Defect'
            if "Class" in row and str(row["Class"]) not in ("Defect", "1"):
                continue

            # split conjunction, e.g. "la > -0.26 & asexp > -315.41 & ld > -0.94"
            atoms = [a.strip() for a in rule_str.split("&")]
            for atom in atoms:
                feat, op, thr = parse_pyexp_atom(atom)
                if feat is None:
                    continue
                if feat not in feat_cols:
                    # ignore Unknown / non-feature tokens
                    continue

                current_val = float(test_row[feat])

                rng = pyexp_flip_range_for_instance(
                    feature=feat,
                    op=op,
                    thr=thr,
                    current_value=current_val,
                    train_min=train_min,
                    train_max=train_max,
                )
                if rng is None:
                    continue

                low, high = rng

                cand = perturb(
                    low,
                    high,
                    current_val,
                    feature_values[feat],
                    train.dtypes[feat],
                )
                if not cand:
                    continue

                if feat not in perturb_features:
                    perturb_features[feat] = cand
                else:
                    merged = list(perturb_features[feat]) + list(cand)
                    merged = sorted(set(merged), key=lambda v: abs(v - current_val))
                    perturb_features[feat] = merged[:10]


        all_plans[int(test_idx)] = perturb_features

    def _convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError

    with open(plan_file, "w") as f:
        json.dump(all_plans, f, indent=4, default=_convert)

    if verbose:
        print(f"[PyExp] wrote plans to {plan_file}")


# ----------------- CLI ----------------- #


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default="RandomForest")
    parser.add_argument("--project", type=str, default="all")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    projects = read_dataset()
    if args.project == "all":
        project_list = list(sorted(projects.keys()))
    else:
        project_list = args.project.split()

    for project_name in tqdm(project_list, desc="Projects", leave=True):
        train, test = projects[project_name]
        run_single_project_pyexp(
            train=train,
            test=test,
            project_name=project_name,
            model_type=args.model_type,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
