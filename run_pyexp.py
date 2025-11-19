#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
import concurrent.futures
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from sklearn.exceptions import ConvergenceWarning

from data_utils import get_true_positives, read_dataset, get_output_dir, get_model
from pyexplainer_core import PyExplainer

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def save_pyexplainer_rules(rule_obj, output_file: Path) -> None:
    """
    Save PyExplainer's top_k_positive_rules to CSV for later planning.

    Columns kept:
      - rule
      - type
      - coef
      - support
      - importance
      - Class
    """
    pos = rule_obj["top_k_positive_rules"].copy()
    if not isinstance(pos, pd.DataFrame):
        pos = pd.DataFrame(pos)

    wanted_cols = ["rule", "type", "coef", "support", "importance", "Class"]
    cols = [c for c in wanted_cols if c in pos.columns]
    pos = pos[cols]
    pos.to_csv(output_file, index=False)


def process_test_idx(
    test_idx,
    true_positives: pd.DataFrame,
    train_data: pd.DataFrame,
    y_test: pd.Series,
    model,
    output_path: Path,
    top_k: int,
    max_rules: int,
    max_iter: int,
    cv: int,
    search_function: str,
):
    """
    Worker function: generate PyExplainer rules for a single test index
    and save them to <output_path>/<test_idx>.csv
    """
    feature_cols = [c for c in train_data.columns if c != "target"]
    output_file = output_path / f"{test_idx}.csv"

    if output_file.exists():
        return None

    print(f"[START] PyExplainer idx={test_idx} pid={os.getpid()}")

    # PyExplainer is built per project (in each worker) on raw features
    X_train = train_data[feature_cols]
    y_train = train_data["target"]

    pyexp = PyExplainer(
        X_train=X_train,
        y_train=y_train,
        indep=X_train.columns,
        dep="target",
        blackbox_model=model,
        class_label=["Clean", "Defect"],  # adjust if you use different names
        top_k_rules=top_k,
        full_ft_names=[],
    )

    # Single test instance (features) from true_positives subset
    test_row = true_positives.loc[test_idx, feature_cols]
    X_explain = test_row.to_frame().T  # 1-row DataFrame

    # Labels from ORIGINAL test_data (not from true_positives)
    y_explain = y_test.loc[[test_idx]]  # 1-row Series, name="target"

    try:
        rule_obj = pyexp.explain(
            X_explain=X_explain,
            y_explain=y_explain,
            top_k=top_k,
            max_rules=max_rules,
            max_iter=max_iter,
            cv=cv,
            search_function=search_function,  # 'CrossoverInterpolation' or 'RandomPerturbation'
            random_state=None,
            reuse_local_model=False,
        )
    except ValueError as e:
        # Typical case: only one class in synthetic_predictions -> GBM can't fit
        msg = str(e)
        if "y contains 1 class" in msg:
            print(
                f"[SKIP] idx={test_idx} only one class in synthetic neighbourhood; "
                f"PyExplainer cannot build local model."
            )
            return None
        # If something else, re-raise so you see the real bug
        raise

    save_pyexplainer_rules(rule_obj, output_file)

    print(f"[END]   PyExplainer idx={test_idx} pid={os.getpid()}")
    return os.getpid()


def run_single_project(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    project_name: str,
    model_type: str,
    top_k: int,
    max_rules: int,
    max_iter: int,
    cv: int,
    search_function: str,
    verbose: bool = True,
):
    # output dir: <exp_root>/<project>/PyExplainer/<ModelType>
    output_path = get_output_dir(project_name, "PyExplainer", model_type)
    output_path.mkdir(parents=True, exist_ok=True)

    model = get_model(project_name, model_type)

    # true_positives is typically a subset of test_data (often features only)
    true_positives = get_true_positives(model, train_data, test_data)

    if len(true_positives) == 0:
        print(f"{project_name}: No true positives found, skipping...")
        return

    print(f"{project_name}: #true_positives = {len(true_positives)}")

    # labels from full test_data (we'll index into this inside workers)
    y_test = test_data["target"]
    y_test = y_test.rename("target")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_test_idx,
                test_idx,
                true_positives,
                train_data,
                y_test,
                model,
                output_path,
                top_k,
                max_rules,
                max_iter,
                cv,
                search_function,
            )
            for test_idx in true_positives.index
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"{project_name}",
            disable=not verbose,
        ):
            out = future.result()
            if out is not None:
                tqdm.write(f"Process {out} finished")


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, default="RandomForest")
    parser.add_argument("--project", type=str, default="all")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_rules", type=int, default=2000)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument(
        "--search_function",
        type=str,
        default="CrossoverInterpolation",
        choices=["CrossoverInterpolation", "RandomPerturbation"],
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    projects = read_dataset()
    if args.project == "all":
        project_list = list(sorted(projects.keys()))
    else:
        project_list = args.project.split(" ")

    for project in tqdm(project_list, desc="Project", leave=True):
        print(f"\n=== {project} ===")
        train, test = projects[project]
        run_single_project(
            train_data=train,
            test_data=test,
            project_name=project,
            model_type=args.model_type,
            top_k=args.top_k,
            max_rules=args.max_rules,
            max_iter=args.max_iter,
            cv=args.cv,
            search_function=args.search_function,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
