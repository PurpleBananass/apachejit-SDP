import math
import json
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from tabulate import tabulate

from hyparams import PROPOSED_CHANGES, EXPERIMENTS
from data_utils import read_dataset, get_model
from flip_jit import get_flip_rates


def generate_all_combinations(data):
    combinations = []
    feature_values = []
    for feature in data:
        feature_values.append(data[feature])
    combinations = list(product(*feature_values))

    df = pd.DataFrame(combinations, columns=data.keys())
    return df


def plan_similarity(project, model_type, explainer):
    results = {}
    plan_path = (
        Path(PROPOSED_CHANGES)
        / f"{project}/{model_type}/{explainer}"
        / "plans_all.json"
    )
    flip_path = Path(EXPERIMENTS) / f"{project}/{model_type}/{explainer}_all.csv"
    with open(plan_path, "r") as f:
        plans = json.load(f)

    if not flip_path.exists():
        return []
    experiment = pd.read_csv(flip_path, index_col=0)
    drops = experiment.dropna().index.to_list()
    model = get_model(project, model_type)
    train, test = read_dataset()[project]
    scaler = StandardScaler()
    scaler.fit(train.drop("target", axis=1).values)
    for test_idx in drops:
        if str(test_idx) in plans:
            original = test.loc[test_idx, test.columns != "target"]
            original_scaled = scaler.transform([original])
            pred_o = model.predict(original_scaled)[0]
            row = experiment.loc[[test_idx], :]
            row_scaled = scaler.transform(row.values)
            pred = model.predict(row_scaled)[0]
            assert pred_o == 1, pred == 0

            plan = {}
            for feature in plans[str(test_idx)]:
                if math.isclose(
                    experiment.loc[test_idx, feature], original[feature], rel_tol=1e-7
                ):
                    continue
                else:
                    plan[feature] = plans[str(test_idx)][feature]

            flipped = experiment.loc[test_idx, [feature for feature in plan]]

            min_changes = [plan[feature][0] for feature in plan]
            min_changes = pd.Series(min_changes, index=flipped.index)
            combi = generate_all_combinations(plan)

            score = normalized_mahalanobis_distance(combi, flipped, min_changes)
            results[test_idx] = {"score": score}

    return results


def normalized_mahalanobis_distance(df, x, y):
    df = df.loc[:, (df.nunique() > 1)]
    if df.shape[1] < 1:
        return 0

    standardized_df = (df - df.mean()) / df.std()

    x_standardized = [
        (x[feature] - df[feature].mean()) / df[feature].std() for feature in df.columns
    ]
    y_standardized = [
        (y[feature] - df[feature].mean()) / df[feature].std() for feature in df.columns
    ]

    cov_matrix = np.cov(standardized_df.T)
    if cov_matrix.ndim == 0:
        inv_cov_matrix = (
            np.array([[1 / cov_matrix]]) if cov_matrix != 0 else np.array([[np.inf]])
        )
    else:
        inv_cov_matrix = np.linalg.pinv(cov_matrix)

    distance = mahalanobis(x_standardized, y_standardized, inv_cov_matrix)

    min_vector = np.array([min(df[feature]) for feature in df.columns])
    max_vector = np.array([max(df[feature]) for feature in df.columns])

    min_vector_standardized = [
        (min_vector[i] - df[feature].mean()) / df[feature].std()
        for i, feature in enumerate(df.columns)
    ]
    max_vector_standardized = [
        (max_vector[i] - df[feature].mean()) / df[feature].std()
        for i, feature in enumerate(df.columns)
    ]

    max_vector_distance = mahalanobis(
        min_vector_standardized, max_vector_standardized, inv_cov_matrix
    )

    normalized_distance = (
        distance / max_vector_distance if max_vector_distance != 0 else 0
    )

    return normalized_distance


def cosine_all(df, x):
    distances = []
    for _, row in df.iterrows():
        distance = cosine_similarity(x, row)
        distances.append(distance)

    return distances


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        print(vec1, vec2)
        return 0

    return dot_product / (norm_vec1 * norm_vec2)


def mahalanobis_all(df, x):
    df = df.loc[:, (df.nunique() > 1)]
    if df.shape[1] < 1:
        return 0

    standardized_df = (df - df.mean()) / df.std()
    x_standardized = [
        (x[feature] - df[feature].mean()) / df[feature].std() for feature in df.columns
    ]

    # 공분산 행렬의 역행렬 계산
    cov_matrix = np.cov(standardized_df.T)
    if cov_matrix.ndim == 0:
        inv_cov_matrix = (
            np.array([[1 / cov_matrix]]) if cov_matrix != 0 else np.array([[np.inf]])
        )
    else:
        inv_cov_matrix = np.linalg.pinv(cov_matrix)

    min_vector = np.array([min(df[feature]) for feature in df.columns])
    max_vector = np.array([max(df[feature]) for feature in df.columns])

    min_vector_standardized = [
        (min_vector[i] - df[feature].mean()) / df[feature].std()
        for i, feature in enumerate(df.columns)
    ]
    max_vector_standardized = [
        (max_vector[i] - df[feature].mean()) / df[feature].std()
        for i, feature in enumerate(df.columns)
    ]

    max_vector_distance = mahalanobis(
        min_vector_standardized, max_vector_standardized, inv_cov_matrix
    )

    # x와 모든 y in df 간의 마할라노비스 거리 계산
    distances = []
    for _, y in df.iterrows():
        y_standardized = [
            (y[feature] - df[feature].mean()) / df[feature].std()
            for feature in df.columns
        ]
        distance = mahalanobis(x_standardized, y_standardized, inv_cov_matrix)
        distances.append(
            distance / max_vector_distance if max_vector_distance != 0 else 0
        )

    return distances


def flip_feasibility(project_list, explainer, model_type, distance="mahalanobis"):
    """Return (results, totals, cannots) PLUS detailed counters so you can see
    why many flips didn't yield a Mahalanobis score.
    """
    # --- aggregate historical deltas across this project group ---
    total_deltas = pd.DataFrame()
    for project in project_list:
        train, test = read_dataset()[project]
        exist_indices = train.index.intersection(test.index)
        deltas = (
            test.loc[exist_indices, test.columns != "target"]
            - train.loc[exist_indices, train.columns != "target"]
        )
        total_deltas = pd.concat([total_deltas, deltas], axis=0)

    # --- counters ---
    totals = 0                      # flips seen (sum across projects)
    cannots = 0                     # your existing cannot
    skipped_no_flipfile = 0         # no *_all.csv for this (project, explainer, model)
    skipped_no_plan = 0             # flip has no matching plan key
    skipped_zero_change = 0         # plan exists but nothing actually changed
    skipped_empty_nonzero = 0       # nonzero_deltas empty after filter
    skipped_rank_too_low = 0        # nonzero_deltas rows < #changed features (mahalanobis)
    written = 0                     # rows actually written to results

    results = []

    for project in project_list:
        plan_path = (
            Path(PROPOSED_CHANGES)
            / f"{project}/{model_type}/{explainer}"
            / "plans_all.json"
        )
        flip_path = Path(EXPERIMENTS) / f"{project}/{model_type}/{explainer}_all.csv"

        # Plan/flip file existence
        if not flip_path.exists():
            skipped_no_flipfile += 1
            continue
        if not plan_path.exists():
            # Without plans you can't compute changes for *any* flip in this project
            # We still count flips to totals for transparency.
            flipped = pd.read_csv(flip_path, index_col=0).dropna()
            totals += len(flipped)
            skipped_no_plan += len(flipped)
            continue

        with open(plan_path, "r") as f:
            plans = json.load(f)

        flipped = pd.read_csv(flip_path, index_col=0).dropna()
        totals += len(flipped)

        train, test = read_dataset()[project]

        for test_idx in flipped.index:
            key = str(test_idx)
            if key not in plans:
                skipped_no_plan += 1
                continue

            original_row = test.loc[test_idx, test.columns != "target"]
            flipped_row = flipped.loc[test_idx, :]

            # Compute per-feature change dictated by the plan
            changed_features = {}
            for feature in plans[key]:
                # treat tiny float diffs as "no change"
                if not math.isclose(flipped_row[feature], original_row[feature], rel_tol=1e-7):
                    changed_features[feature] = flipped_row[feature] - original_row[feature]

            if not changed_features:
                skipped_zero_change += 1
                continue

            changed_feature_names = list(changed_features.keys())
            changed_flipped = pd.Series(changed_features)

            # historical rows where *all* changed features moved (strict)
            nonzero_deltas = total_deltas[changed_feature_names].dropna()
            nonzero_deltas = nonzero_deltas.loc[(nonzero_deltas != 0).all(axis=1)]

            if distance == "cosine":
                if len(nonzero_deltas) == 0:
                    cannots += 1
                    skipped_empty_nonzero += 1
                    continue
                distances = cosine_all(nonzero_deltas, changed_flipped)

            elif distance == "mahalanobis":
                if len(nonzero_deltas) <= len(changed_feature_names):
                    cannots += 1
                    skipped_rank_too_low += 1
                    continue
                distances = mahalanobis_all(nonzero_deltas, changed_flipped)

            # Write one summary row per flip that produced any distances
            if isinstance(distances, (list, np.ndarray)) and len(distances) > 0:
                results.append(
                    {
                        "min": float(np.min(distances)),
                        "max": float(np.max(distances)),
                        "mean": float(np.mean(distances)),
                    }
                )
                written += 1
            else:
                # treat unexpected empty/malformed distances as cannot
                cannots += 1
                skipped_empty_nonzero += 1

    # Optional: print a quick breakdown for this call
    print(
        f"[{model_type} {explainer} {distance}] totals={totals}, written={written}, "
        f"cannot={cannots} | no_flipfile={skipped_no_flipfile}, no_plan={skipped_no_plan}, "
        f"zero_change={skipped_zero_change}, empty_nonzero={skipped_empty_nonzero}, "
        f"rank_too_low={skipped_rank_too_low}"
    )

    return results, totals, cannots



def implications(project, explainer, model_type):
    # Flipped instance's changed steps based on plan
    plan_path = (
        Path(PROPOSED_CHANGES)
        / f"{project}/{model_type}/{explainer}"
        / "plans_all.json"
    )
    flip_path = Path(EXPERIMENTS) / f"{project}/{model_type}/{explainer}_all.csv"
    with open(plan_path, "r") as f:
        plans = json.load(f)

    if not flip_path.exists():
        return []

    flipped = pd.read_csv(flip_path, index_col=0)
    flipped = flipped.dropna()

    train, test = read_dataset()[project]
    scaler = StandardScaler()
    scaler.fit(train.drop("target", axis=1).values)

    totals = []
    for test_idx in flipped.index:
        if str(test_idx) in plans:
            original_row = test.loc[test_idx, test.columns != "target"]
            flipped_row = flipped.loc[test_idx, :]
            changed_features = []
            for feature in plans[str(test_idx)]:
                if math.isclose(
                    flipped_row[feature], original_row[feature], rel_tol=1e-7
                ):
                    continue
                else:
                    changed_features.append(feature)
            scaled_flipped = scaler.transform([flipped_row])[0]
            scaled_original = scaler.transform([original_row])[0]
            scaled_deltas = scaled_flipped - scaled_original
            scaled_deltas = pd.Series(scaled_deltas, index=original_row.index)
            scaled_deltas = scaled_deltas[changed_features].abs()
            total = scaled_deltas.sum()
            totals.append(total)

    return totals


def find_approx_index(lst, value, tol=1e-7):
    for i, v in enumerate(lst):
        if math.isclose(v, value, rel_tol=tol):
            return i
    return -1


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--rq1", action="store_true")
    argparser.add_argument("--rq2", action="store_true")
    argparser.add_argument("--rq3", action="store_true")
    argparser.add_argument("--implications", action="store_true")
    argparser.add_argument("--explainer", type=str, default="all")
    argparser.add_argument("--distance", type=str, default="mahalanobis")

    args = argparser.parse_args()

    model_map = {"SVM": "SVM", "RandomForest": "RF", "LogisticRegression": "LR"}

    explainer_map = {
        "LIME": "LIME",
        "LIME-HPO": "LIME-HPO",
        "PyExplainer": "PyExplainer",
        "CF": "CF",
    }

    if args.explainer == "all":
        explainers = ["LIME", "LIME-HPO", "PyExplainer","CF"]
    else:
        explainers = args.explainer.split(" ")
    projects = read_dataset()
    if args.rq1:
        table = []
        for model_type in ["SVM", "RandomForest", "LogisticRegression"]:
            for explainer in explainers:
                result = get_flip_rates(explainer, None, model_type, verbose=False)
                table.append([model_map[model_type], explainer, result["Rate"]])

            # Add mean per model
            table.append(
                [
                    model_map[model_type],
                    "All",
                    np.mean(
                        [row[2] for row in table if row[0] == model_map[model_type]]
                    ),
                ]
            )
        print(tabulate(table, headers=["Model", "Explainer", "Flip Rate"]))

        # table to csv
        table = pd.DataFrame(table, columns=["Model", "Explainer", "Flip Rate"])
        table.to_csv("./evaluations/flip_rates.csv", index=False)

    if args.rq2:
        table = []
        Path("./evaluations/similarities").mkdir(parents=True, exist_ok=True)
        for model_type in ["SVM", "RandomForest", "LogisticRegression"]:
        # for model_type in ["LightGBM"]:
            similarities = pd.DataFrame()
            for explainer in explainers:
                for project in projects:
                    result = plan_similarity(project, model_type, explainer)
                    df = pd.DataFrame(result).T
                    df["project"] = project
                    df["explainer"] = explainer_map[explainer]
                    df["model"] = model_map[model_type]
                    similarities = pd.concat(
                        [similarities, df], axis=0, ignore_index=False
                    )
                similarities.to_csv(
                    f"./evaluations/similarities/{model_map[model_type]}.csv"
                )

    if args.rq3:
        table = []
        totals = 0
        cannots = 0
        project_lists = [
            ["activemq@0", "activemq@1", "activemq@2", "activemq@3"],
            ["camel@0", "camel@1", "camel@2"],
            ["derby@0", "derby@1"],
            ["groovy@0", "groovy@1"],
            ["hbase@0", "hbase@1"],
            ["hive@0", "hive@1"],
            ["jruby@0", "jruby@1", "jruby@2"],
            ["lucene@0", "lucene@1", "lucene@2"],
            ["wicket@0", "wicket@1"],
        ]

        Path(f"./evaluations/feasibility/{args.distance}").mkdir(
            parents=True, exist_ok=True
        )
        for model_type in ["RandomForest", "SVM", "LogisticRegression"]:
        # for model_type in ["SVM"]:
            for explainer in explainer_map:
                results = []
                for project_list in project_lists:
                    result, total, cannot = flip_feasibility(
                        project_list, explainer, model_type, args.distance
                    )
                    print(f"Processing {project_list} {model_type} {explainer}")
                    
                    if len(result) == 0:
                        totals += total
                        cannots += cannot
                        continue
                    
                    results.extend(result)
                    totals += total
                    cannots += cannot
                df = pd.DataFrame(results)
                if len(df) == 0:
                    continue
                # print(df.head())
                # print(df)
                # save to csv

                df.to_csv(
                    f"./evaluations/feasibility/{args.distance}/{model_map[model_type]}_{explainer_map[explainer]}.csv",
                    index=False,
                )
                table.append(
                    [
                        model_type,
                        explainer,
                        df["min"].mean(),
                        df["max"].mean(),
                        df["mean"].mean(),
                    ]
                )
                print(table)
            # Add mean per model
            table.append(
                [
                    model_type,
                    "Mean",
                    np.mean([row[2] for row in table if row[0] == model_type]),
                    np.mean([row[3] for row in table if row[0] == model_type]),
                    np.mean([row[4] for row in table if row[0] == model_type]),
                ]
            )
        print(tabulate(table, headers=["Model", "Explainer", "Min", "Max", "Mean"]))
        print(f"Total: {totals}, Cannot: {cannots} ({cannots/totals*100:.2f}%)")

        # table to csv
        table = pd.DataFrame(
            table, columns=["Model", "Explainer", "Min", "Max", "Mean"]
        )
        table.to_csv(f"./evaluations/feasibility_{args.distance}.csv", index=False)

    if args.implications:
        table = []
        for model_type in ["RandomForest", "SVM", "LogisticRegression"]:
            for explainer in explainers:
                results = []
                for project in projects:
                    print(f"Processing {project} {model_type} {explainer}")
                    result = implications(project, explainer, model_type)
                    results.extend(result)
                if len(results) == 0:
                    continue
                df = pd.DataFrame(results)
                # save to csv
                df.to_csv(
                    f"./evaluations/abs_changes/{model_map[model_type]}_{explainer_map[explainer]}.csv",
                    index=False,
                )
                table.append([model_type, explainer, df.mean()])
            # Add mean per model
            table.append(
                [
                    model_type,
                    "Mean",
                    np.mean([row[2] for row in table if row[0] == model_type]),
                ]
            )
        print(tabulate(table, headers=["Model", "Explainer", "Mean"]))
        # table to csv
        table = pd.DataFrame(table, columns=["Model", "Explainer", "Mean"])

        per_model_rows = []
        for m in models:
            t = totals_by_model[m]
            c = cannots_by_model[m]
            pct = (c / t * 100.0) if t else 0.0
            per_model_rows.append([model_map[m], c, t, f"{pct:.2f}%"])
        print("\nPer-model 'cannot' summary")
        print(tabulate(per_model_rows, headers=["Model", "Cannot", "Total", "Cannot%"]))
        # <<<