#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# run_explainers_jit.py
from __future__ import annotations

import concurrent.futures
import warnings
import os
from argparse import ArgumentParser
from pathlib import Path

# concurrency
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

# your existing LIME utils
from Explainer.LIME_HPO import LIME_HPO, LIME_Planner
# your data/model helpers
from data_utils import get_true_positives, read_dataset, get_output_dir, get_model
from hyparams import SEED

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

# ============================================================
# ---------------------- Config / constants -------------------
# ============================================================

# Supported combinations
SUPPORTED_MODELS = [
    "RandomForest",
    "SVM",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]

SUPPORTED_EXPLAINERS = [
    "LIME",
    "LIME-HPO",
]

# ============================================================
# ----------------------- LIME helpers ------------------------
# ============================================================

def process_test_idx_lime(
    test_idx,
    true_positives: pd.DataFrame,
    train_df: pd.DataFrame,
    model,
    output_path: Path,
    explainer_type: str,
):
    out_csv = output_path / f"{test_idx}.csv"
    if out_csv.exists():
        return None

    # feature columns used for training
    feat_cols = [c for c in train_df.columns if c != "target"]
    X_train = train_df[feat_cols]

    # take only those columns, in the same order (drop extras like commit_id)
    ti_full = true_positives.loc[test_idx, :]
    ti = ti_full.loc[feat_cols]  # align

    if explainer_type == "LIME":
        LIME_Planner(
            X_train=X_train,
            test_instance=ti,
            training_labels=train_df[["target"]],
            model=model,
            path=out_csv,
        )
    elif explainer_type == "LIME-HPO":
        LIME_HPO(
            X_train=X_train,
            test_instance=ti,
            training_labels=train_df[["target"]],
            model=model,
            path=out_csv,
        )
    else:
        raise ValueError(f"Unsupported explainer_type in process_test_idx_lime: {explainer_type}")

    return os.getpid()


# ============================================================
# ------------------- Unified per-project runner --------------
# ============================================================

def run_single_project(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    project_name: str,
    model_type: str,
    explainer_type: str,
    *,
    seed: int = 42,   # kept for CLI compatibility, even if unused here
    verbose: bool = True,
    max_workers: int = 1,
):
    """
    Unified runner for:
      - LIME / LIME-HPO: per-TP CSV in explainer folder
    """
    output_path = get_output_dir(project_name, explainer_type, model_type)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[RUN  ] {project_name} :: {explainer_type} on {model_type}")
    model = get_model(project_name, model_type)
    true_positives = get_true_positives(model, train_df, test_df)
    if len(true_positives) == 0:
        print(f"[SKIP ] {project_name} :: No true positives found.")
        return

    # Ensure 'target' is present (harmless for LIME)
    if "target" not in true_positives.columns and "target" in test_df.columns:
        true_positives = true_positives.join(test_df["target"])

    print(f"[RUN  ] {project_name} :: {explainer_type} on {len(true_positives)} TPs")

    if explainer_type not in ("LIME", "LIME-HPO"):
        raise ValueError(f"Unsupported explainer_type in run_single_project: {explainer_type}")

    Executor = ThreadPoolExecutor
    ex_kwargs = {"max_workers": max_workers}

    with Executor(**ex_kwargs) as executor:
        futures = [
            executor.submit(
                process_test_idx_lime,
                int(test_idx),
                true_positives,
                train_df,
                model,
                output_path,
                explainer_type,
            )
            for test_idx in true_positives.index
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"{project_name} ({explainer_type})",
            disable=not verbose,
        ):
            out = future.result()
            if out is not None:
                tqdm.write(f"Process {out} finished")


# ============================================================
# ----------------------------- main --------------------------
# ============================================================

def main():
    ap = ArgumentParser()
    ap.add_argument(
        "--model-type",
        type=str,
        default="XGBoost",
        choices=SUPPORTED_MODELS + ["all"],
        help="Model key for get_model, or 'all' to run all supported models.",
    )
    ap.add_argument(
        "--explainer-type",
        type=str,
        default="LIME-HPO",
        choices=SUPPORTED_EXPLAINERS + ["all"],
        help="Explainer to run (LIME, LIME-HPO) or 'all'.",
    )
    ap.add_argument(
        "--project",
        type=str,
        default="all",
        help="'all' or space-separated list of project names",
    )

    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--max-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
    )

    args = ap.parse_args()

    # Resolve projects
    projects = read_dataset()  # dict[project] = (train_df, test_df)
    if args.project == "all":
        project_list = list(sorted(projects.keys()))
    else:
        project_list = args.project.split()

    # Resolve model types
    if args.model_type == "all":
        model_types = SUPPORTED_MODELS
    else:
        model_types = [args.model_type]

    # Resolve explainer types
    if args.explainer_type == "all":
        explainer_types = SUPPORTED_EXPLAINERS
    else:
        explainer_types = [args.explainer_type]

    # Run all combinations: model × explainer × project
    for model_type in model_types:
        for explainer_type in explainer_types:
            print(f"\n====== MODEL: {model_type} | EXPLAINER: {explainer_type} ======")

            for project in tqdm(
                project_list,
                desc=f"{model_type}/{explainer_type}",
                leave=True,
            ):
                train_df, test_df = projects[project]

                run_single_project(
                    train_df=train_df,
                    test_df=test_df,
                    project_name=project,
                    model_type=model_type,
                    explainer_type=explainer_type,
                    seed=args.seed,
                    max_workers=args.max_workers,
                )


if __name__ == "__main__":
    main()
