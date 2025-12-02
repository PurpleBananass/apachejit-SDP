import joblib
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from tqdm import tqdm

from hyparams import SEED, MODELS, MODEL_EVALUATION
from data_utils import get_model, read_dataset


def evaluate_metrics(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = y_pred.astype(bool)

    return {
        "AUC-ROC": roc_auc_score(y, y_proba),
        "F1-score": f1_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "TP": sum(y_pred & y) / sum(y),
        "# of TP": sum(y_pred & y),
    }


def train_single_project(project, train, test, metrics={}):
    models_path = Path(f"{MODELS}/{project}")
    models_path.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train.iloc[:, train.columns != "target"].values)
    y_train = train["target"].values
    X_test = scaler.transform(test.iloc[:, test.columns != "target"].values)
    y_test = test["target"].values

    model_specs = [
        (
            RandomForestClassifier(
                n_estimators=100,
                random_state=SEED,
            ),
            "RandomForest",
        ),
        (
            LogisticRegression(
                random_state=SEED,
                max_iter=1000,
            ),
            "LogisticRegression",
        ),
        (
            SVC(
                probability=True,
                random_state=SEED,
            ),
            "SVM",
        ),
        (
            LGBMClassifier(
                random_state=SEED,
            ),
            "LightGBM",
        ),
        (
            CatBoostClassifier(
                random_state=SEED,
                verbose=0,
            ),
            "CatBoost",
        ),
        (
            XGBClassifier(
                n_estimators=100,
                random_state=SEED,
                eval_metric="logloss",
                use_label_encoder=False,
            ),
            "XGBoost",
        ),
    ]

    for model, model_name in model_specs:
        tqdm.write(f"Training {project} with {model_name}...")
        model.fit(X_train, y_train)

        model_path = models_path / f"{model_name}.joblib"
        joblib.dump(model, model_path)

        tqdm.write(f"Evaluating {project} with {model_name}...")
        test_metrics = evaluate_metrics(model, X_test, y_test)
        metrics[model_name][project] = test_metrics

    return metrics


def train_all_project():
    projects = read_dataset()
    Path(MODEL_EVALUATION).mkdir(parents=True, exist_ok=True)

    metrics = {
        "RandomForest": {},
        "LogisticRegression": {},
        "SVM": {},
        "LightGBM": {},
        "CatBoost": {},
        "XGBoost": {},
    }

    for project in tqdm(projects, desc="projects", leave=True, total=len(projects)):
        train, test = projects[project]
        metrics = train_single_project(project, train, test, metrics)

    # Save metrics per model as csv
    for model_name, model_metrics in metrics.items():
        df = pd.DataFrame(model_metrics)
        df.to_csv(f"{MODEL_EVALUATION}/{model_name}.csv")


def eval_all_project():
    projects = read_dataset()

    metrics = {
        "RandomForest": {},
        "LogisticRegression": {},
        "SVM": {},
        "LightGBM": {},
        "CatBoost": {},
        "XGBoost": {},
    }

    for project in tqdm(projects, desc="projects", leave=True, total=len(projects)):
        train, test = projects[project]

        # Fit scaler on train to match training-time preprocessing
        scaler = StandardScaler()
        scaler.fit(train.iloc[:, train.columns != "target"].values)

        X_test = scaler.transform(test.iloc[:, test.columns != "target"].values)
        y_test = test["target"].values

        for model_name in metrics.keys():
            tqdm.write(f"[EVAL] {project} with {model_name}...")
            model = get_model(project, model_name)  # should load .joblib models
            test_metrics = evaluate_metrics(model, X_test, y_test)
            metrics[model_name][project] = test_metrics

    # Save metrics per model as csv
    for model_name, model_metrics in metrics.items():
        df = pd.DataFrame(model_metrics)
        df.to_csv(f"{MODEL_EVALUATION}/{model_name}.csv")


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--test", action="store_true")
    args = argparser.parse_args()

    if args.test:
        eval_all_project()
    else:
        train_all_project()
