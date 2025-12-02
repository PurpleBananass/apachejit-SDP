#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import numpy as np

from data_utils import get_model, get_true_positives, read_dataset, get_output_dir
PLANS_ROOT = Path("./plans_closest")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def perturb(low, high, current, values, dtype):
    """
    Given an interval [low, high], a current value, and all possible
    discrete values for a feature, return candidate perturbations inside
    [low, high] sorted by |value - current|.

    The list may be down-sampled to at most 10 values using median pooling.
    """
    if dtype == "int64":
        perturbations = [val for val in values if low <= val <= high]

    elif dtype == "float64":
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
        # Fallback: treat as numeric and just filter in range
        perturbations = [val for val in values if low <= val <= high]

    if len(perturbations) > 10:
        # divide perturbations into 10 groups and select median value per group
        groups = np.array_split(perturbations, 10)
        perturbations = [np.median(group) for group in groups]

    # remove current value if present
    if current in perturbations:
        perturbations.remove(current)

    # sort by closeness to current
    sorted_perturbations = sorted(perturbations, key=lambda x: abs(x - current))

    return sorted_perturbations


def split_inequality(rule, min_val, max_val, pattern):
    """
    Parse a rule string such as:
      - '10 < la <= 20'
      - 'la > 5'
      - 'la <= 5'
    and return (feature_name, [l, r]) with l, r numeric bounds.
    """
    m = pattern.search(rule)
    if not m:
        return None, None

    g1, op1, feature_name, op2, g5 = m.groups()
    # Case: 10 < feature <= 20
    if g1 is not None and op1 == "<" and op2 == "<=" and g5 is not None:
        l, r = float(g1), float(g5)
    # Case: feature > a
    elif g1 is None and op1 is None and op2 == ">" and g5 is not None:
        l, r = float(g5), max_val[feature_name]
    # Case: feature <= b
    elif g1 is None and op1 is None and op2 == "<=" and g5 is not None:
        l, r = min_val[feature_name], float(g5)
    else:
        return None, None

    return feature_name, [l, r]


def flip_feature_range(feature, min_val, max_val, importance, rule_str):
    """
    Given a rule string and feature importance sign, decide which side
    of the interval should be the "good" (flip) side.

    Returns [left_bound, feature, right_bound].
    """
    # Case: a < feature <= b
    matches = re.search(r"([\d.]+) < " + re.escape(feature) + r" <= ([\d.]+)", rule_str)
    if matches:
        a, b = map(float, matches.groups())
        if importance > 0:
            # Positive importance: move towards lower side
            return [min_val, feature, a]
        # Negative importance: move towards upper side
        return [b, feature, max_val]

    # Case: feature > a
    matches = re.search(re.escape(feature) + r" > ([\d.]+)", rule_str)
    if matches:
        a = float(matches.group(1))
        return [min_val, feature, a]

    # Case: feature <= b
    matches = re.search(re.escape(feature) + r" <= ([\d.]+)", rule_str)
    if matches:
        b = float(matches.group(1))
        return [b, feature, max_val]

    print("Not Available", rule_str)
    return None


def np_to_native(o):
    """
    JSON encoder helper to handle numpy types.
    """
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Core: generate closest-only plans
# ---------------------------------------------------------------------------

def run_single(
    train,
    test,
    project_name,
    model_type,
    explainer_type,
    search_strategy,
    verbose=False,
):
    """
    For each true positive instance:
      - read the explanation output,
      - derive a set of [low, feature, high] intervals,
      - for each feature, compute perturbation candidates,
      - **keep only the single closest perturbation per feature**,
      - save as JSON:
            {test_idx: {feature: closest_value, ...}, ...}
    """
    output_path = get_output_dir(project_name, explainer_type, model_type)
    proposed_change_path = PLANS_ROOT / project_name / model_type / explainer_type

    if search_strategy is not None:
        proposed_change_path = (
            PLANS_ROOT / project_name / model_type / f"{explainer_type}_{search_strategy}"
        )
        output_path = output_path / search_strategy
        output_path.mkdir(parents=True, exist_ok=True)

    proposed_change_path.mkdir(parents=True, exist_ok=True)


    file_name = "plans_all.json"

    pattern = re.compile(
        r"([-]?[\d.]+)?\s*(<|>)?\s*([a-zA-Z_]+)\s*(<=|>=|<|>)?\s*([-]?[\d.]+)?"
    )

    train_min = train.min()
    train_max = train.max()

    feature_values = {
        feature: sorted(set(train.loc[:, feature])) for feature in train.columns
    }

    all_plans = {}
    model = get_model(project_name, model_type)
    true_positives = get_true_positives(model, train, test)

    for test_idx in tqdm(
        true_positives.index, desc=f"{project_name}", leave=True, disable=not verbose
    ):
        test_instance = test.loc[test_idx]
        assert test_instance["target"] == 1

        # This will store feature -> closest perturbation value
        closest_plan = {}

        match explainer_type:
            case "LIME" | "LIME-HPO":
                explanation_path = output_path / f"{test_idx}.csv"

                if not explanation_path.exists():
                    if verbose:
                        print(f"[WARN] Missing explanation: {explanation_path}")
                    continue
                explanation = pd.read_csv(explanation_path)

                plan = []
                for row in range(len(explanation)):
                    (
                        feature,
                        value,
                        importance,
                        min_val,
                        max_val,
                        rule,
                        importance_ratio,
                    ) = explanation.iloc[row].values
                    proposed_changes = flip_feature_range(
                        feature,
                        train_min[feature],
                        train_max[feature],
                        importance,
                        rule,
                    )
                    if proposed_changes:
                        plan.append(proposed_changes)

                for low, feature, high in plan:
                    dtype = train.dtypes[feature]
                    candidates = perturb(
                        low,
                        high,
                        test_instance[feature],
                        feature_values[feature],
                        dtype,
                    )
                    if not candidates:
                        continue
                    # only keep the single closest candidate
                    closest_val = candidates[0]
                    closest_plan[feature] = closest_val

            case "TimeLIME":
                explanation_path = Path(f"{output_path}/{test_idx}.csv")

                if not explanation_path.exists():
                    if verbose:
                        print(f"[WARN] Missing explanation: {explanation_path}")
                    continue
                explanation = pd.read_csv(explanation_path)

                plan = []
                for row in range(len(explanation)):
                    (
                        feature,
                        value,
                        importance,
                        left,
                        right,
                        rec,
                        rule,
                        min_val,
                        max_val,
                    ) = explanation.iloc[row].values
                    plan.append([left, feature, right])

                for low, feature, high in plan:
                    dtype = train.dtypes[feature]
                    candidates = perturb(
                        low,
                        high,
                        test_instance[feature],
                        feature_values[feature],
                        dtype,
                    )
                    if not candidates:
                        continue
                    closest_val = candidates[0]
                    closest_plan[feature] = closest_val

            case "SQAPlanner":
                try:
                    plan_df = pd.read_csv(output_path / f"{test_idx}.csv")
                except pd.errors.EmptyDataError:
                    if verbose:
                        print(f"EmptyDataError: {project_name} {test_idx}")
                    continue

                if len(plan_df) == 0:
                    continue

                # We only look at the first non-empty antecedent
                for _, row in plan_df.iterrows():
                    best_rule = row["Antecedent"]
                    if not isinstance(best_rule, str):
                        continue

                    for rule in best_rule.split("&"):
                        feature, ranges = split_inequality(
                            rule, train_min, train_max, pattern
                        )
                        if feature is None or ranges is None:
                            continue
                        low, high = ranges
                        if low > high:
                            continue
                        # clamp to training bounds
                        if low < train_min[feature]:
                            low = max(0, train_min[feature])
                        if high > train_max[feature]:
                            high = train_max[feature]

                        dtype = train.dtypes[feature]
                        candidates = perturb(
                            low,
                            high,
                            test_instance[feature],
                            feature_values[feature],
                            dtype,
                        )
                        if not candidates:
                            continue
                        closest_val = candidates[0]
                        closest_plan[feature] = closest_val

                    # we only use rules from the first row
                    if closest_plan:
                        break

            case _:
                raise ValueError(f"Unsupported explainer_type: {explainer_type}")

        # Store per-instance closest plan
        all_plans[int(test_idx)] = closest_plan

    with open(proposed_change_path / file_name, "w") as f:
        json.dump(all_plans, f, indent=4, default=np_to_native)


def get_importance_ratio(
    train, test, project_name, model_type, explainer_type, verbose=False
):
    """
    Same as your original helper: compute sum of importance_ratio
    over all features that yielded a valid flip interval.
    """
    output_path = get_output_dir(project_name, explainer_type, model_type)

    train_min = train.min()
    train_max = train.max()

    model = get_model(project_name, model_type)
    true_positives = get_true_positives(model, train, test)

    total = []
    for test_idx in tqdm(
        true_positives.index, desc=f"{project_name}", leave=True, disable=not verbose
    ):
        test_instance = test.loc[test_idx]
        assert test_instance["target"] == 1

        match explainer_type:
            case "LIME" | "LIME-HPO":
                explanation_path = output_path / f"{test_idx}.csv"

                if not explanation_path.exists():
                    if verbose:
                        print(f"[WARN] Missing explanation: {explanation_path}")
                    continue
                explanation = pd.read_csv(explanation_path)

                ratios = []
                for row in range(len(explanation)):
                    (
                        feature,
                        value,
                        importance,
                        min_val,
                        max_val,
                        rule,
                        importance_ratio,
                    ) = explanation.iloc[row].values
                    proposed_changes = flip_feature_range(
                        feature,
                        train_min[feature],
                        train_max[feature],
                        importance,
                        rule,
                    )
                    if proposed_changes:
                        ratios.append(importance_ratio)
                total.append(sum(ratios))
            case _:
                # You can extend this if you need importance ratios for other explainers
                continue

    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default="RandomForest")
    # Default to one of the supported values in the match-case
    parser.add_argument("--explainer_type", type=str, default="LIME-HPO")
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
        project_list = args.project.split(" ")

    if args.compute_importance:
        total = []
        for project in tqdm(
            project_list, desc="Projects", leave=True, disable=not args.verbose
        ):
            train, test = projects[project]

            total += get_importance_ratio(
                train, test, project, args.model_type, args.explainer_type, args.verbose
            )
        print(np.mean(np.array(total)))
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
