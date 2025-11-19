import json
import re
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import numpy as np

from data_utils import get_model, get_true_positives, read_dataset, get_output_dir


def perturb(low, high, current, values, dtype):
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
            # compare 소수점 둘 째 자리까지
            if round(last, 2) != round(candidate, 2):
                perturbations.append(candidate)
                last = candidate

    if len(perturbations) > 10:
        # devide perturbations into 10 and select median value
        groups = np.array_split(perturbations, 10)
        perturbations = [np.median(group) for group in groups]

    if current in perturbations:
        perturbations.remove(current)

    sorted_publications = sorted(perturbations, key=lambda x: abs(x - current))

    return sorted_publications


def split_inequality(rule, min_val, max_val, pattern):
    m = pattern.search(rule)
    if not m:
        raise ValueError(f"Could not parse rule: {rule}")

    v1, op1, feature_name, op2, v2 = m.groups()
    # v1, op1 = left number and operator (optional)
    # feature_name = the feature
    # op2, v2 = right operator and number (optional)

    # Case: v1 < feature_name <= v2
    if v1 is not None and op1 == "<" and op2 == "<=" and v2 is not None:
        l, r = float(v1), float(v2)

    # Case: feature_name > v1   (stored as: v1=None, op1=None, feature_name, op2=">", v2)
    elif v1 is None and op1 is None and op2 == ">" and v2 is not None:
        l, r = float(v2), max_val[feature_name]

    # Case: feature_name <= v2  (stored as: v1=None, op1=None, feature_name, op2="<=", v2)
    elif v1 is None and op1 is None and op2 == "<=" and v2 is not None:
        l, r = min_val[feature_name], float(v2)

    else:
        raise ValueError(f"Unhandled rule pattern: {rule} (groups={m.groups()})")

    return feature_name, [l, r]



def flip_feature_range(feature, min_val, max_val, importance, rule_str):
    # Case: a < feature <= b
    matches = re.search(r"([\d.]+) < " + re.escape(feature) + r" <= ([\d.]+)", rule_str)
    if matches:
        a, b = map(float, matches.groups())
        if importance > 0:
            return [min_val, feature, a]
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


# Aussme there are generated explanations
def run_single(
    train,
    test,
    project_name,
    model_type,
    explainer_type,
    search_strategy,
    verbose=False,
):
    output_path = get_output_dir(project_name, explainer_type, model_type)
    proposed_change_path = Path(f"./plans/{project_name}/{model_type}/{explainer_type}")
    if search_strategy is not None:
        proposed_change_path = Path(
            f"./plans/{project_name}/{model_type}/{explainer_type}_{search_strategy}"
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
    perturb_features = {}
    for test_idx in tqdm(
        true_positives.index, desc=f"{project_name}", leave=True, disable=not verbose
    ):
        test_instance = test.loc[test_idx]
        assert test_instance["target"] == 1

        if explainer_type in ("LIME", "LIME-HPO"):
            explanation_path = output_path / f"{test_idx}.csv"

            if not explanation_path.exists():
                print("????")
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

            

            for proposed_changes in plan:
                feature = proposed_changes[1]
                dtype = train.dtypes[feature]
                perturbations = perturb(
                    proposed_changes[0],
                    proposed_changes[2],
                    test_instance[feature],
                    feature_values[feature],
                    dtype,
                )
                if not perturbations:
                    continue
                perturb_features[feature] = perturbations

        elif explainer_type == "TimeLIME":
            explanation_path = Path(f"{output_path}/{test_idx}.csv")

            if not explanation_path.exists():
                print("????")
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

            perturb_features = {}
            for proposed_changes in plan:
                feature = proposed_changes[1]
                dtype = train.dtypes[feature]
                perturbations = perturb(
                    proposed_changes[0],
                    proposed_changes[2],
                    test_instance[feature],
                    feature_values[feature],
                    dtype,
                )
                if not perturbations:
                    continue
                perturb_features[feature] = perturbations

        elif explainer_type == "SQAPlanner":
            try:
                plan = pd.read_csv(output_path / f"{test_idx}.csv")
            except pd.errors.EmptyDataError:
                if verbose:
                    print(f"EmptyDataError: {project_name} {test_idx}")
                continue

            if len(plan) == 0:
                continue

            perturb_features = {}
            for _, row in plan.iterrows():
                if len(perturb_features) > 0:
                    break
                best_rule = row["Antecedent"]
                for rule in best_rule.split("&"):
                    feature, ranges = split_inequality(
                        rule, train_min, train_max, pattern
                    )
                    if ranges[0] > ranges[1]:
                        break
                    if ranges[0] < train_min[feature]:
                        ranges[0] = max(0, train_min[feature])
                    perturbations = perturb(
                        ranges[0],
                        ranges[1],
                        test_instance[feature],
                        feature_values[feature],
                        train.dtypes[feature],
                    )

                    if not perturbations:
                        continue
                    perturb_features[feature] = perturbations


        all_plans[int(test_idx)] = perturb_features
    def convert_int64(o):
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError

    with open(proposed_change_path / file_name, "w") as f:
        json.dump(all_plans, f, indent=4, default=convert_int64)


def get_importance_ratio(
    train, test, project_name, model_type, explainer_type, verbose=False
):
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

        if explainer_type in ("LIME", "LIME-HPO"):
            explanation_path = output_path / f"{test_idx}.csv"

            if not explanation_path.exists():
                print("????")
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
        # If you add other explainer types later, put more elif branches here

    return total


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default="RandomForest")
    parser.add_argument("--explainer_type", type=str, default="LIME-HPO")
    parser.add_argument("--project", type=str, default="all")
    parser.add_argument("--search_strategy", type=str, default=None)
    parser.add_argument("--only_minimum", action="store_true")
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
