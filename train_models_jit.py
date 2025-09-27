# train_models_jit.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tqdm import tqdm

from hyparams import SEED, MODELS, MODEL_EVALUATION
from data_utils import get_model, read_dataset


def _xy(df: pd.DataFrame):
    """Return numeric X, boolean y — no scaling (PyExplainer needs raw feature space)."""
    if "target" not in df.columns:
        raise ValueError("Expected 'target' in dataset.")
    X = (
        df.drop(columns=["target"], errors="ignore")
          .select_dtypes(include=[np.number])
          .replace([np.inf, -np.inf], np.nan)
          .fillna(0.0)
    )
    y = df["target"].astype(bool).values
    return X, y


def evaluate_metrics(model, X, y):
    """Safe metrics for imbalanced data; uses predict_proba when available."""
    y = np.asarray(y).astype(bool)
    y_pred = np.asarray(model.predict(X)).astype(bool)

    # proba or decision score for AUC
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)[:, 1]
        except Exception:
            y_proba = None
    if y_proba is None and hasattr(model, "decision_function"):
        try:
            y_proba = model.decision_function(X)
        except Exception:
            y_proba = None

    metrics = {
        "AUC-ROC": float(roc_auc_score(y, y_proba)) if (y_proba is not None and len(np.unique(y)) == 2) else np.nan,
        "F1-score": float(f1_score(y, y_pred)) if y.size else np.nan,
        "Precision": float(precision_score(y, y_pred)) if y.size else np.nan,
        "Recall": float(recall_score(y, y_pred)) if y.size else np.nan,
    }
    positives = int(y.sum())
    tp = int(((y_pred == True) & (y == True)).sum())
    metrics["TP"] = float(tp / positives) if positives > 0 else 0.0
    metrics["# of TP"] = tp
    return metrics


def train_single_project(project: str, train: pd.DataFrame, test: pd.DataFrame, metrics: dict) -> dict:
    models_path = Path(MODELS) / project
    models_path.mkdir(parents=True, exist_ok=True)

    X_train, y_train = _xy(train)
    X_test, y_test = _xy(test)

    # Train *plain* sklearn classifiers (no Pipelines, no external scalers)
    model_specs = [
        (RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED
        ), "RandomForest"),
        (SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,          # needed for AUC and explainers
            class_weight="balanced",
            random_state=SEED
        ), "SVM"),
        (LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED
        ), "LogisticRegression"),
    ]

    for model, name in model_specs:
        tqdm.write(f"[TRAIN] {project} :: {name} (n={len(X_train)}, d={X_train.shape[1]})")
        model.fit(X_train, y_train)

        # save as .pkl (plain classifier object — PyExplainer-friendly)
        out_path = models_path / f"{name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(model, f)
        tqdm.write(f"[SAVE ] {out_path}")

        # evaluate on test (raw features)
        test_metrics = evaluate_metrics(model, X_test, y_test)
        metrics[name][project] = test_metrics
        tqdm.write(f"[EVAL ] {project} :: {name} -> {test_metrics}")

    return metrics


def train_all_project():
    projects = read_dataset()  # dict[project] = (train_df, test_df)
    Path(MODEL_EVALUATION).mkdir(parents=True, exist_ok=True)

    if not projects:
        tqdm.write("[WARN] read_dataset() returned no projects.")
        return

    metrics = {"RandomForest": {}, "SVM": {}, "LogisticRegression": {}}

    for project in tqdm(projects, desc="projects", leave=True, total=len(projects)):
        try:
            train, test = projects[project]
            metrics = train_single_project(project, train, test, metrics)
        except Exception as e:
            tqdm.write(f"[FAIL ] {project}: {e}")

    # Save metrics per model
    for model_name, model_metrics in metrics.items():
        df = pd.DataFrame(model_metrics).T  # projects as rows
        out_csv = Path(MODEL_EVALUATION) / f"{model_name}.csv"
        df.to_csv(out_csv)
        tqdm.write(f"[METRICS] {out_csv}")


def eval_all_project():
    projects = read_dataset()
    if not projects:
        tqdm.write("[WARN] read_dataset() returned no projects.")
        return

    metrics = {"RandomForest": {}, "SVM": {}, "LogisticRegression": {}}

    for project in tqdm(projects, desc="projects", leave=True, total=len(projects)):
        try:
            train, test = projects[project]
            X_test, y_test = _xy(test)
            for name in metrics.keys():
                model = get_model(project, name)  # should load the .pkl you just saved
                test_metrics = evaluate_metrics(model, X_test, y_test)
                metrics[name][project] = test_metrics
                tqdm.write(f"[EVAL ] {project} :: {name} -> {test_metrics}")
        except Exception as e:
            tqdm.write(f"[FAIL ] {project}: {e}")

    Path(MODEL_EVALUATION).mkdir(parents=True, exist_ok=True)
    for model_name, model_metrics in metrics.items():
        df = pd.DataFrame(model_metrics).T
        out_csv = Path(MODEL_EVALUATION) / f"{model_name}.csv"
        df.to_csv(out_csv)
        tqdm.write(f"[METRICS] {out_csv}")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--test", action="store_true", help="Evaluate saved models instead of training.")
    args = ap.parse_args()

    if args.test:
        eval_all_project()
    else:
        train_all_project()
