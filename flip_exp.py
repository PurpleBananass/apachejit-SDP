import traceback
import warnings
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from tabulate import tabulate
from tqdm import tqdm

from data_utils import read_dataset, get_true_positives, get_model
from hyparams import PROPOSED_CHANGES, SEED, EXPERIMENTS

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(SEED)


def get_flip_rates(explainer_type, search_strategy, model_type, verbose=True):
    projects = read_dataset()
    project_result = []
    for project_name in projects:
        train, test = projects[project_name]

        scaler = StandardScaler()
        scaler.fit(train.drop("target", axis=1).values)

        if search_strategy is None:
            plan_path = Path(
                f"{PROPOSED_CHANGES}/{project_name}/{model_type}/{explainer_type}/plans_all.json"
            )
            exp_path = Path(
                f"{EXPERIMENTS}/{project_name}/{model_type}/{explainer_type}_all.csv"
            )

        else:
            plan_path = Path(
                f"{PROPOSED_CHANGES}/{project_name}/{model_type}/{explainer_type}_{search_strategy}/plans_all.json"
            )
            exp_path = Path(
                f"{EXPERIMENTS}/{project_name}/{model_type}/{explainer_type}_{search_strategy}_all.csv"
            )

        plan_path.parent.mkdir(parents=True, exist_ok=True)
        exp_path.parent.mkdir(parents=True, exist_ok=True)

        with open(plan_path, "r") as f:
            plans = json.load(f)

        model = get_model(project_name, model_type)
        true_positives = get_true_positives(model, train, test)

        if exp_path.exists():
            all_results_df = pd.read_csv(exp_path, index_col=0)
            test_names = list(plans.keys())
            computed_test_names = list(map(str, all_results_df.index))
            test_names = [
                name for name in test_names if name not in computed_test_names
            ]
            project_result.append(
                [
                    project_name,
                    len(all_results_df.dropna()),
                    len(all_results_df),
                    len(plans.keys()),
                    len(true_positives),
                    len(all_results_df.dropna()) / len(true_positives),
                ]
            )
    project_result.append(
        [
            "Total",
            sum([val[1] for val in project_result]),
            sum([val[2] for val in project_result]),
            sum([val[3] for val in project_result]),
            sum([val[4] for val in project_result]),
            sum([val[1] for val in project_result])
            / sum([val[4] for val in project_result]),
        ]
    )
    if verbose:
        print(
            tabulate(
                project_result,
                headers=["Project", "Flip", "Computed", "#Plan", "#TP", "Flip%"],
            )
        )
    else:
        # return the result
        return {
            "Flip": sum([val[1] for val in project_result]),
            "TP": sum([val[4] for val in project_result]),
            "Rate": sum([val[1] for val in project_result])
            / sum([val[4] for val in project_result]),
        }


def flip_instance(
    original_instance: pd.DataFrame, changeable_features_dict, model, scaler
):
    changeable_features = list(changeable_features_dict.values())
    feature_names = list(changeable_features_dict.keys())
    if isinstance(original_instance, pd.Series):
        original_instance = original_instance.to_frame().T
    for values in product(*changeable_features):
        modified_instance = original_instance.copy()

        modified_instance.loc[:, feature_names] = values
        modified_instance_scaled = scaler.transform(modified_instance)
        prediction = model.predict_proba(modified_instance_scaled)[:, 0]

        if prediction >= 0.5:
            return modified_instance

    nan_row = pd.DataFrame(
        [[np.nan] * len(original_instance.columns)],
        columns=original_instance.columns,
        index=original_instance.index,
    )
    return nan_row


def flip_single_project(
    train,
    test,
    project_name,
    explainer_type,
    search_strategy,
    verbose=True,
    load=True,
    model_type="RandomForest",
):
    scaler = StandardScaler()
    scaler.fit(train.drop("target", axis=1))

    if search_strategy is None:
        plan_path = Path(
            f"{PROPOSED_CHANGES}/{project_name}/{model_type}/{explainer_type}/plans_all.json"
        )
        exp_path = Path(
            f"{EXPERIMENTS}/{project_name}/{model_type}/{explainer_type}_all.csv"
        )

    else:
        plan_path = Path(
            f"{PROPOSED_CHANGES}/{project_name}/{model_type}/{explainer_type}_{search_strategy}/plans_all.json"
        )
        exp_path = Path(
            f"{EXPERIMENTS}/{project_name}/{model_type}/{explainer_type}_{search_strategy}_all.csv"
        )

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    exp_path.parent.mkdir(parents=True, exist_ok=True)

    with open(plan_path, "r") as f:
        plans = json.load(f)

    if exp_path.exists() and load:
        all_results_df = pd.read_csv(exp_path, index_col=0)
        test_names = list(plans.keys())
        computed_test_names = list(map(str, all_results_df.index))
        test_names = [name for name in test_names if name not in computed_test_names]
        print(
            f"{project}:{len(all_results_df.dropna())}/{len(all_results_df)}/{len(plans.keys())}"
        )
    else:
        all_results_df = pd.DataFrame()
        test_names = list(plans.keys())

    model = get_model(project_name, model_type)

    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
        futures = {}
        max_perturbations = []
        for test_name in tqdm(
            test_names,
            desc=f"{project_name} queing...",
            leave=False,
            disable=not verbose,
        ):
            # Calcuate the number of perturbations (len(A) * len(B) * ...)
            features = list(plans[test_name].keys())
            computation = 1
            for feature in features:
                computation *= (
                    len(plans[test_name][feature]) + 1
                )  # +1 for original value
            max_perturbations.append([test_name, computation])
        max_perturbations = sorted(max_perturbations, key=lambda x: x[1])

        # Start from the test with the smallest number of perturbations
        test_indices = [x[0] for x in max_perturbations]

        for test_name in tqdm(
            test_indices, desc=f"{project_name}", leave=False, disable=not verbose
        ):
            original_instance = test.loc[int(test_name), test.columns != "target"]
            features = list(plans[test_name].keys())

            changeable_features_dict = {
                feature: [original_instance[feature]] + plans[test_name][feature]
                for feature in features
            }

            future = executor.submit(
                flip_instance,
                original_instance,
                changeable_features_dict,
                model,
                scaler,
            )

            futures[future] = test_name

        for future in tqdm(
            as_completed(futures),
            desc=f"{project_name}",
            leave=False,
            disable=not verbose,
            total=len(futures),
        ):
            test_name = futures[future]
            try:
                flipped_instance = future.result()
                all_results_df = pd.concat([all_results_df, flipped_instance], axis=0)
                all_results_df.to_csv(exp_path)
            except Exception as e:
                tqdm.write(f"Error occurred: {e} id: {test_name}")
                traceback.print_exc()
                exit()
    print(
        f"{project}:{len(all_results_df.dropna())}/{len(all_results_df)}/{len(plans.keys())}"
    )


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--project", type=str, default="all")
    argparser.add_argument("--explainer_type", type=str, required=True)
    argparser.add_argument("--search_strategy", type=str, default=None)
    argparser.add_argument("--verbose", action="store_true")
    argparser.add_argument("--new", action="store_true")
    argparser.add_argument("--get_flip_rate", action="store_true")
    argparser.add_argument("--model_type", type=str, default="RandomForest")

    args = argparser.parse_args()

    if args.get_flip_rate:
        get_flip_rates(args.explainer_type, args.search_strategy, args.model_type)
    else:
        tqdm.write("Project/Flipped/Computed/Plan")
        projects = read_dataset()
        if args.project == "all":
            project_list = list(sorted(projects.keys()))
        else:
            project_list = args.project.split(" ")

        for project in tqdm(
            project_list, desc="Projects", leave=True, disable=not args.verbose
        ):
            train, test = projects[project]
            flip_single_project(
                train,
                test,
                project,
                args.explainer_type,
                args.search_strategy,
                verbose=args.verbose,
                load=not args.new,
                model_type=args.model_type,
            )