# flip_jit.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import traceback
import warnings
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from tabulate import tabulate
from tqdm import tqdm

from data_utils import read_dataset, get_true_positives, get_model
from hyparams import PROPOSED_CHANGES, SEED, EXPERIMENTS

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(SEED)


# ---------- helpers: keep feature processing identical to training ----------
def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.select_dtypes(include=[np.number])
          .replace([np.inf, -np.inf], np.nan)
          .fillna(0.0)
    )

def _train_cols(train: pd.DataFrame) -> pd.Index:
    return _numeric_frame(train.drop(columns=["target"], errors="ignore")).columns

def _align_like_one(x_row: pd.Series, cols: pd.Index) -> pd.DataFrame:
    """
    Create a 1-row numeric DataFrame aligned to training columns.
    """
    X = x_row.to_frame().T
    X = _numeric_frame(X)
    X = X.reindex(columns=list(cols), fill_value=0.0)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


# ---------- summary / flip-rate report ----------
def get_flip_rates(explainer_type: str, search_strategy: Optional[str], model_type: str, verbose: bool = True):
    projects = read_dataset()
    project_result = []
    for project_name in projects:
        train, test = projects[project_name]

        if search_strategy is None:
            plan_path = Path(f"{PROPOSED_CHANGES}/{project_name}/{model_type}/{explainer_type}/plans_all.json")
            exp_path  = Path(f"{EXPERIMENTS}/{project_name}/{model_type}/{explainer_type}_all.csv")
        else:
            plan_path = Path(f"{PROPOSED_CHANGES}/{project_name}/{model_type}/{explainer_type}_{search_strategy}/plans_all.json")
            exp_path  = Path(f"{EXPERIMENTS}/{project_name}/{model_type}/{explainer_type}_{search_strategy}_all.csv")

        plan_path.parent.mkdir(parents=True, exist_ok=True)
        exp_path.parent.mkdir(parents=True, exist_ok=True)

        if not plan_path.exists():
            project_result.append([project_name, 0, 0, 0, 0, 0.0])
            continue

        with open(plan_path, "r") as f:
            plans = json.load(f)

        model = get_model(project_name, model_type)
        true_positives = get_true_positives(model, train, test)

        if exp_path.exists():
            all_results_df = pd.read_csv(exp_path, index_col=0)
            test_names = list(plans.keys())
            computed_test_names = list(map(str, all_results_df.index))
            test_names = [name for name in test_names if name not in computed_test_names]
            project_result.append(
                [
                    project_name,
                    len(all_results_df.dropna()),
                    len(all_results_df),
                    len(plans.keys()),
                    len(true_positives),
                    (len(all_results_df.dropna()) / len(true_positives)) if len(true_positives) else 0.0,
                ]
            )
        else:
            project_result.append([project_name, 0, 0, len(plans.keys()), len(true_positives), 0.0])

    if project_result:
        total = [
            "Total",
            sum(v[1] for v in project_result),
            sum(v[2] for v in project_result),
            sum(v[3] for v in project_result),
            sum(v[4] for v in project_result),
            (sum(v[1] for v in project_result) / max(1, sum(v[4] for v in project_result))),
        ]
        project_result.append(total)

    if verbose:
        print(tabulate(project_result, headers=["Project", "Flip", "Computed", "#Plan", "#TP", "Flip%"]))
    else:
        return {
            "Flip": sum(v[1] for v in project_result) if project_result else 0,
            "TP":   sum(v[4] for v in project_result) if project_result else 0,
            "Rate": (sum(v[1] for v in project_result) / max(1, sum(v[4] for v in project_result))) if project_result else 0.0,
        }


# ---------- flipping ----------
def _same_as_original(
    original: pd.Series, candidate: pd.Series, features: list[str]
) -> bool:
    """True if candidate equals original on all edited features (within tiny tol)."""
    try:
        return np.allclose(
            candidate[features].astype(float).values,
            original[features].astype(float).values,
            rtol=1e-12,
            atol=1e-15,
        )
    except Exception:
        # Fallback exact compare if casting fails
        return bool((candidate[features].values == original[features].values).all())


def flip_instance(
    original_instance: pd.Series,
    changeable_features_dict: Dict[str, list],
    model,
    train_cols: pd.Index,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Try perturbation combinations; return the first modified instance that flips
    the model prediction from Positive (buggy) to Negative (clean).
    - Uses predict_proba[:, pos_idx] for the positive class.
    - No scaling (models were trained on raw features).
    """
    feature_names = list(changeable_features_dict.keys())
    candidate_lists = [list(filter(lambda v: np.isfinite(v), map(float, vals)))
                       for vals in changeable_features_dict.values()]

    # If there are no changeable features, return NaN row
    if len(feature_names) == 0:
        nan_row = pd.DataFrame([[np.nan] * len(original_instance.index)],
                               columns=original_instance.index)
        nan_row.index = pd.Index([original_instance.name])
        return nan_row

    # iterate cartesian product
    for values in product(*candidate_lists):
        mod = original_instance.copy()
        mod.loc[feature_names] = list(values)

        # Never accept the exact original vector as a "flip"
        if _same_as_original(original_instance, mod, feature_names):
            continue

        X = _align_like_one(mod, train_cols)

        # score positive class
        try:
            if hasattr(model, "classes_"):
                pos_idx = int(np.where(model.classes_ == 1)[0][0])
            else:
                pos_idx = 1
            prob_pos = float(model.predict_proba(X)[0, pos_idx])
            is_clean = (prob_pos < threshold)
        except Exception:
            # decision function: default 0 boundary => negative means "clean"
            score = float(model.decision_function(X)[0])
            is_clean = (score < 0.0)

        if is_clean:
            out = mod.to_frame().T
            out.index = pd.Index([original_instance.name])
            return out

    # nothing flipped — return NaN row (signals failure)
    nan_row = pd.DataFrame([[np.nan] * len(original_instance.index)],
                           columns=original_instance.index)
    nan_row.index = pd.Index([original_instance.name])
    return nan_row


def flip_single_project(
    train: pd.DataFrame,
    test: pd.DataFrame,
    project_name: str,
    explainer_type: str,
    search_strategy: Optional[str],
    verbose: bool = True,
    load: bool = True,
    model_type: str = "RandomForest",
    threshold: float = 0.5,
    max_workers: Optional[int] = None,
):
    train_cols = _train_cols(train)

    if search_strategy is None:
        plan_path = Path(f"{PROPOSED_CHANGES}/{project_name}/{model_type}/{explainer_type}/plans_all.json")
        exp_path  = Path(f"{EXPERIMENTS}/{project_name}/{model_type}/{explainer_type}_all.csv")
    else:
        plan_path = Path(f"{PROPOSED_CHANGES}/{project_name}/{model_type}/{explainer_type}_{search_strategy}/plans_all.json")
        exp_path  = Path(f"{EXPERIMENTS}/{project_name}/{model_type}/{explainer_type}_{search_strategy}_all.csv")

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    exp_path.parent.mkdir(parents=True, exist_ok=True)

    if not plan_path.exists():
        tqdm.write(f"[WARN] No plans found for {project_name}: {plan_path}")
        return

    with open(plan_path, "r") as f:
        plans = json.load(f)

    # resume
    if exp_path.exists() and load:
        all_results_df = pd.read_csv(exp_path, index_col=0)
        test_names = list(plans.keys())
        computed_test_names = list(map(str, all_results_df.index))
        test_names = [name for name in test_names if name not in computed_test_names]
        print(f"{project_name}:{len(all_results_df.dropna())}/{len(all_results_df)}/{len(plans.keys())}")
    else:
        all_results_df = pd.DataFrame()
        test_names = list(plans.keys())

    model = get_model(project_name, model_type)

    # schedule small search spaces first
    max_perturbations = []
    for test_name in test_names:
        features = list(plans[test_name].keys())
        comp = 1
        for feat in features:
            comp *= (len(plans[test_name][feat]) + 0)  # original not included
        max_perturbations.append([test_name, comp if comp > 0 else 1])
    max_perturbations.sort(key=lambda x: x[1])
    test_indices = [x[0] for x in max_perturbations]

    # Thread pool (no pickling the model)
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) // 2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for test_name in tqdm(test_indices, desc=f"{project_name}", leave=False, disable=not verbose):
            idx = int(test_name)
            original_instance = test.loc[idx, test.columns != "target"]

            features = list(plans[test_name].keys())
            # Use only planned values; original is implicitly skipped by _same_as_original
            changeable_features_dict = {
                feat: list(plans[test_name][feat])
                for feat in features
            }

            fut = executor.submit(
                flip_instance,
                original_instance,
                changeable_features_dict,
                model,
                train_cols,
                threshold,
            )
            futures[fut] = test_name

        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{project_name}", leave=False, disable=not verbose):
            test_name = futures[fut]
            try:
                flipped_instance = fut.result()
                all_results_df = pd.concat([all_results_df, flipped_instance], axis=0)
                all_results_df.to_csv(exp_path)
            except Exception as e:
                tqdm.write(f"[ERROR] {project_name} id={test_name}: {e}")
                traceback.print_exc()
                # continue others

    print(f"{project_name}:{len(all_results_df.dropna())}/{len(all_results_df)}/{len(plans.keys())}")


# ---------- CLI ----------
if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--project", type=str, default="all")
    argparser.add_argument("--explainer_type", type=str, required=True,
                           choices=["LIME", "LIME-HPO", "PyExplainer"])
    argparser.add_argument("--search_strategy", type=str, default=None)
    argparser.add_argument("--verbose", action="store_true")
    argparser.add_argument("--new", action="store_true", help="Ignore existing experiment CSV; recompute.")
    argparser.add_argument("--get_flip_rate", action="store_true")
    argparser.add_argument("--model_type", type=str, default="RandomForest")
    argparser.add_argument("--threshold", type=float, default=0.5)
    argparser.add_argument("--max_workers", type=int, default=None)

    args = argparser.parse_args()

    if args.get_flip_rate:
        get_flip_rates(args.explainer_type, args.search_strategy, args.model_type)
    else:
        tqdm.write("Project/Flipped/Computed/Plan")
        projects = read_dataset()
        if args.project == "all":
            project_list = list(sorted(projects.keys()))
        else:
            project_list = args.project.split()

        for project_name in tqdm(project_list, desc="Projects", leave=True, disable=not args.verbose):
            train, test = projects[project_name]
            flip_single_project(
                train,
                test,
                project_name,
                args.explainer_type,
                args.search_strategy,
                verbose=args.verbose,
                load=not args.new,
                model_type=args.model_type,
                threshold=args.threshold,
                max_workers=args.max_workers,
            )
