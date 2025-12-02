#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import warnings
from argparse import ArgumentParser
from pathlib import Path
import concurrent.futures

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.exceptions import ConvergenceWarning

from data_utils import get_true_positives, read_dataset, get_output_dir, get_model
from pyexplainer_core import PyExplainer

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------
# Supported models
# ---------------------------------------------------------------------

SUPPORTED_MODELS = [
    "RandomForest",
    "SVM",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]


def save_pyexplainer_rules(
    rule_obj,
    output_file: Path,
    X_explain: pd.DataFrame,
    train_data: pd.DataFrame,
    test_row: pd.Series,
    y_explain: pd.Series,
    model,
) -> None:
    """
    Convert PyExplainer's best positive rule into a LIME-style per-feature CSV.

    Output columns:
      - commit_id (if available)
      - label
      - pred
      - score_or_proba
      - feature
      - value_actual
      - min
      - max
      - operator
      - threshold
    """
    # Prefer positive rules for now
    pos = rule_obj.get("top_k_positive_rules", None)
    if pos is None:
        pos = pd.DataFrame()
    if not isinstance(pos, pd.DataFrame):
        pos = pd.DataFrame(pos)

    # If there is no usable rule, write meta-only row
    if pos.empty or "rule" not in pos.columns:
        try:
            prob = float(model.predict_proba(X_explain)[0][1])
        except Exception:
            try:
                prob = float(model.decision_function(X_explain)[0])
            except Exception:
                prob = np.nan

        try:
            pred_val = bool(model.predict(X_explain)[0])
        except Exception:
            pred_val = np.nan

        label_val = bool(y_explain.iloc[0]) if len(y_explain) > 0 else np.nan

        meta = {
            "commit_id": test_row.get("commit_id", None),
            "label": label_val,
            "pred": pred_val,
            "score_or_proba": prob,
        }
        pd.DataFrame([meta]).to_csv(output_file, index=False)
        return

    # Take the first rule
    rule_str = str(pos.iloc[0]["rule"]).strip()
    # Split by &, "and", "AND"
    conds = [s.strip() for s in re.split(r"&|and|AND", rule_str) if s.strip()]

    feature_rows = []
    feature_cols = [c for c in train_data.columns if c != "target"]

    for cond in conds:
        # Very simple parser: "<feat> <op> <value>"
        parts = cond.split()
        if len(parts) < 3:
            continue
        feat, op, val_str = parts[0], parts[1], parts[2]

        # Only keep conditions on known features
        if feat not in feature_cols:
            continue

        # threshold
        try:
            thr = float(val_str)
        except ValueError:
            thr = np.nan

        # actual value from this instance
        try:
            value_actual = float(X_explain.iloc[0].get(feat, np.nan))
        except Exception:
            value_actual = np.nan

        # min / max from training data
        if feat in train_data.columns:
            col = train_data[feat].replace([np.inf, -np.inf], np.nan)
            fmin = float(np.nanmin(col.values))
            fmax = float(np.nanmax(col.values))
        else:
            fmin = np.nan
            fmax = np.nan

        feature_rows.append(
            {
                "feature": feat,
                "value_actual": value_actual,
                "min": fmin,
                "max": fmax,
                "operator": op,
                "threshold": thr,
            }
        )

    # meta
    try:
        prob = float(model.predict_proba(X_explain)[0][1])
    except Exception:
        try:
            prob = float(model.decision_function(X_explain)[0])
        except Exception:
            prob = np.nan

    try:
        pred_val = bool(model.predict(X_explain)[0])
    except Exception:
        pred_val = np.nan

    label_val = bool(y_explain.iloc[0]) if len(y_explain) > 0 else np.nan

    meta = {
        "commit_id": test_row.get("commit_id", None),
        "label": label_val,
        "pred": pred_val,
        "score_or_proba": prob,
    }

    if not feature_rows:
        # No parsed conditions → meta-only row
        pd.DataFrame([meta]).to_csv(output_file, index=False)
        return

    plan_df = pd.DataFrame(
        feature_rows,
        columns=["feature", "value_actual", "min", "max", "operator", "threshold"],
    )

    # Repeat meta for each feature row
    meta_df = pd.DataFrame([meta]).loc[[0] * len(plan_df)].reset_index(drop=True)
    out_df = pd.concat([meta_df, plan_df.reset_index(drop=True)], axis=1)

    out_df.to_csv(output_file, index=False)


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
    and save them to <output_path>/<test_idx>.csv in a LIME-like per-feature format.
    """
    feature_cols = [c for c in train_data.columns if c != "target"]
    output_file = output_path / f"{test_idx}.csv"

    if output_file.exists():
        return None

    # print(f"[START] PyExplainer idx={test_idx} pid={os.getpid()}")

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
    test_row = true_positives.loc[test_idx, :]
    test_row_feats = test_row[feature_cols]
    X_explain = test_row_feats.to_frame().T  # 1-row DataFrame

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

    # save in LIME-style per-feature format
    save_pyexplainer_rules(
        rule_obj=rule_obj,
        output_file=output_file,
        X_explain=X_explain,
        train_data=train_data,
        test_row=test_row,
        y_explain=y_explain,
        model=model,
    )

    # print(f"[END]   PyExplainer idx={test_idx} pid={os.getpid()}")
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
    max_workers: int,
    verbose: bool = True,
):
    """
    Run PyExplainer for all TRUE POSITIVES in a single project/model.
    """
    # Light budget shrink for XGBoost (still heavy)
    if model_type == "XGBoost":
        top_k = 1
        max_rules = min(max_rules, 50)
        max_iter = min(max_iter, 1000)
        cv = min(cv, 3)

    # output dir: <exp_root>/<project>/PyExplainer/<ModelType>
    output_path = get_output_dir(project_name, "PyExplainer", model_type)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        model = get_model(project_name, model_type)
    except Exception as e:
        print(f"[WARN] Could not load model '{model_type}' for project '{project_name}': {e}")
        return

    # true_positives is typically a subset of test_data (often features only)
    true_positives = get_true_positives(model, train_data, test_data)

    if len(true_positives) == 0:
        print(f"{project_name} ({model_type}): No true positives found, skipping...")
        return

    print(f"{project_name} ({model_type}): #true_positives = {len(true_positives)}")

    # labels from full test_data (we'll index into this inside workers)
    y_test = test_data["target"].rename("target")

    # Use threads for all models to avoid pickling overhead
    workers = max(1, min(max_workers, os.cpu_count() or 1))
    print(f"{project_name} ({model_type}): using {workers} threads")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
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
            desc=f"{project_name} ({model_type})",
            disable=not verbose,
        ):
            out = future.result()
            # if out is not None:
            #     tqdm.write(f"Thread/process {out} finished")
            _ = out  # keep linter happy


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="all",
        choices=SUPPORTED_MODELS + ["all"],
        help="Model name for get_model, or 'all' to run all supported models.",
    )
    parser.add_argument("--project", type=str, default="all")
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--max_rules", type=int, default=2000)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument(
        "--search_function",
        type=str,
        default="CrossoverInterpolation",
        choices=["CrossoverInterpolation", "RandomPerturbation"],
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Max parallel workers for PyExplainer (default: 4).",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    projects = read_dataset()
    if args.project == "all":
        project_list = list(sorted(projects.keys()))
    else:
        project_list = args.project.split(" ")

    # Resolve models
    if args.model_type == "all":
        model_types = SUPPORTED_MODELS
    else:
        model_types = [args.model_type]

    for project in tqdm(project_list, desc="Project", leave=True):
        print(f"\n=== {project} ===")
        train, test = projects[project]
        for model_type in model_types:
            run_single_project(
                train_data=train,
                test_data=test,
                project_name=project,
                model_type=model_type,
                top_k=args.top_k,
                max_rules=args.max_rules,
                max_iter=args.max_iter,
                cv=args.cv,
                search_function=args.search_function,
                max_workers=args.max_workers,
                verbose=args.verbose,
            )


if __name__ == "__main__":
    main()