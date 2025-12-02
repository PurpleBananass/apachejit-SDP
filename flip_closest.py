import traceback
import warnings
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from tabulate import tabulate
from tqdm import tqdm

from data_utils import read_dataset, get_true_positives, get_model
from hyparams import PROPOSED_CHANGES, SEED, EXPERIMENTS  # PROPOSED_CHANGES kept for compatibility

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(SEED)

# New root for closest-only plans and experiments
PLANS_ROOT = Path("./plans_closest")
EXPERIMENTS_CLOSEST = f"{EXPERIMENTS}_closest"


def get_flip_rates(explainer_type, search_strategy, model_type, verbose=True):
    projects = read_dataset()
    project_result = []

    for project_name in projects:
        train, test = projects[project_name]

        scaler = StandardScaler()
        scaler.fit(train.drop("target", axis=1).values)

        # read plans from PLANS_ROOT instead of PROPOSED_CHANGES
        if search_strategy is None:
            plan_path = (
                PLANS_ROOT
                / project_name
                / model_type
                / explainer_type
                / "plans_all.json"
            )
            exp_path = Path(
                f"{EXPERIMENTS_CLOSEST}/{project_name}/{model_type}/{explainer_type}_all.csv"
            )
        else:
            plan_path = (
                PLANS_ROOT
                / project_name
                / model_type
                / f"{explainer_type}_{search_strategy}"
                / "plans_all.json"
            )
            exp_path = Path(
                f"{EXPERIMENTS_CLOSEST}/{project_name}/{model_type}/{explainer_type}_{search_strategy}_all.csv"
            )

        plan_path.parent.mkdir(parents=True, exist_ok=True)
        exp_path.parent.mkdir(parents=True, exist_ok=True)

        if not plan_path.exists():
            # No plans → nothing to count
            continue

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
                    len(all_results_df.dropna()) / len(true_positives)
                    if len(true_positives) > 0
                    else 0.0,
                ]
            )

    if not project_result:
        if verbose:
            print("No projects with results/plans found.")
            return
        else:
            return {"Flip": 0, "TP": 0, "Rate": 0.0}

    project_result.append(
        [
            "Total",
            sum(val[1] for val in project_result),
            sum(val[2] for val in project_result),
            sum(val[3] for val in project_result),
            sum(val[4] for val in project_result),
            (
                sum(val[1] for val in project_result)
                / sum(val[4] for val in project_result)
                if sum(val[4] for val in project_result) > 0
                else 0.0
            ),
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
        return {
            "Flip": sum(val[1] for val in project_result),
            "TP": sum(val[4] for val in project_result),
            "Rate": (
                sum(val[1] for val in project_result)
                / sum(val[4] for val in project_result)
                if sum(val[4] for val in project_result) > 0
                else 0.0
            ),
        }


def flip_instance(
    original_instance: pd.DataFrame, changeable_features_dict, model, scaler
):
    """
    NEW BEHAVIOR (closest):
      - No combinatorial search.
      - Build a single modified instance by applying *one* planned value per feature.
      - If the resulting instance predicts class 0 with prob >= 0.5, return it.
      - Otherwise, return a NaN row (same shape), so it counts as "not flipped".
    """
    feature_names = list(changeable_features_dict.keys())
    if isinstance(original_instance, pd.Series):
        original_instance = original_instance.to_frame().T

    modified_instance = original_instance.copy()
    for f in feature_names:
        modified_instance.loc[:, f] = changeable_features_dict[f]

    modified_instance_scaled = scaler.transform(modified_instance)

    try:
        proba = model.predict_proba(modified_instance_scaled)
        proba0 = float(proba[:, 0][0])
    except Exception as e:
        # if something weird happens, surface it instead of silently hanging
        raise RuntimeError(f"predict_proba failed: {e}")

    if proba0 >= 0.5:
        # Flip succeeded (predicted non-defective)
        return modified_instance

    # Flip failed → return NaN row (same index/columns)
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

    # read plans from PLANS_ROOT instead of PROPOSED_CHANGES
    if search_strategy is None:
        plan_path = (
            PLANS_ROOT
            / project_name
            / model_type
            / explainer_type
            / "plans_all.json"
        )
        exp_path = Path(
            f"{EXPERIMENTS_CLOSEST}/{project_name}/{model_type}/{explainer_type}_all.csv"
        )
    else:
        plan_path = (
            PLANS_ROOT
            / project_name
            / model_type
            / f"{explainer_type}_{search_strategy}"
            / "plans_all.json"
        )
        exp_path = Path(
            f"{EXPERIMENTS_CLOSEST}/{project_name}/{model_type}/{explainer_type}_{search_strategy}_all.csv"
        )

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    exp_path.parent.mkdir(parents=True, exist_ok=True)

    if not plan_path.exists():
        tqdm.write(f"[WARN] Plan file not found: {plan_path}")
        return

    with open(plan_path, "r") as f:
        plans = json.load(f)

    if exp_path.exists() and load:
        all_results_df = pd.read_csv(exp_path, index_col=0)
        test_names = list(plans.keys())
        computed_test_names = list(map(str, all_results_df.index))
        test_names = [name for name in test_names if name not in computed_test_names]
        print(
            f"{project_name}:{len(all_results_df.dropna())}/{len(all_results_df)}/{len(plans.keys())}"
        )
    else:
        all_results_df = pd.DataFrame()
        test_names = list(plans.keys())

    model = get_model(project_name, model_type)

    # ---------- use threads instead of processes ----------
    cpu = os.cpu_count() or 1
    max_workers = min(8, cpu)  # be conservative; models are internally multithreaded
    tqdm.write(
        f"{project_name} ({model_type}): using ThreadPoolExecutor with max_workers={max_workers}"
    )

    # small helper so we don't pass big objects around unnecessarily
    def worker(test_name: str):
        original_instance = test.loc[int(test_name), test.columns != "target"]
        features = list(plans[test_name].keys())

        # "closest": pick a single planned value per feature
        changeable_features_dict = {}
        for feature in features:
            raw_vals = plans[test_name][feature]
            # If list → use LAST element (closest); if scalar → use as is
            if isinstance(raw_vals, list) and len(raw_vals) > 0:
                target_val = raw_vals[-1]
            else:
                target_val = raw_vals
            changeable_features_dict[feature] = target_val

        return flip_instance(original_instance, changeable_features_dict, model, scaler)

    # ---- First pass: estimate "size" (kept for ordering)
    max_perturbations = []
    for test_name in tqdm(
        test_names,
        desc=f"{project_name} queing...",
        leave=False,
        disable=not verbose,
    ):
        features = list(plans[test_name].keys())
        computation = len(features)  # simple proxy
        max_perturbations.append([test_name, computation])
    max_perturbations = sorted(max_perturbations, key=lambda x: x[1])
    test_indices = [x[0] for x in max_perturbations]

    # ---- Submit + collect
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_test = {
            executor.submit(worker, test_name): test_name for test_name in test_indices
        }

        for future in tqdm(
            as_completed(future_to_test),
            desc=f"{project_name}",
            leave=False,
            disable=not verbose,
            total=len(future_to_test),
        ):
            test_name = future_to_test[future]
            try:
                flipped_instance = future.result()
                all_results_df = pd.concat([all_results_df, flipped_instance], axis=0)
                all_results_df.to_csv(exp_path)
            except Exception as e:
                tqdm.write(f"Error occurred: {e} id: {test_name}")
                traceback.print_exc()
                # don't exit the whole script on one failure; just continue
                # if you *do* want to hard stop, uncomment:
                # exit(1)

    print(
        f"{project_name}:{len(all_results_df.dropna())}/{len(all_results_df)}/{len(plans.keys())}"
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