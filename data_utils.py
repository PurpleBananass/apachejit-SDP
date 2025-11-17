import joblib
from pathlib import Path

# import xgboost
import natsort
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from hyparams import (
    MODELS,
    OUTPUT,
    PROJECT_DATASET,
    RELEASE_DATASET,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_output_dir(project_name: str, explainer_type: str, model_type: str) -> Path:
    path = Path(OUTPUT) / project_name / explainer_type / model_type
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_release_names(project_release, with_num_tp=True):
    project, release_idx = project_release.split("@")
    release_idx = int(release_idx)
    releases = [
        release.stem for release in (Path(PROJECT_DATASET) / project).glob("*.csv")
    ]
    releases = natsort.natsorted(releases)
    if with_num_tp:
        num_tp = get_release_ratio(project_release)

    return f"{project} {releases[release_idx + 1]} ({num_tp:.1f}%)"

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from hyparams import MODELS, OUTPUT, PROJECT_DATASET, RELEASE_DATASET

# --- safer loader (same path layout) ------------------------------------------
def get_model(project_name: str, model_name: str = "RandomForest"):
    model_path = Path(MODELS) / project_name / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)

# --- true positives without external scaling ---------------------------------
def get_true_positives(
    model_or_path,
    train_data: pd.DataFrame,   # unused now, but kept for signature compatibility
    test_data: pd.DataFrame,
    label: str = "target",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Return rows in test_data that are TRUE POSITIVES for the given model.
    Works with plain sklearn classifiers (RF/SVM/LR) and Pipelines.
    No manual scaling here — use the model as trained.
    """
    assert label in test_data.columns, f"'{label}' column missing in test_data"

    # Accept either a loaded model or a path
    if isinstance(model_or_path, (str, Path)):
        model = joblib.load(model_or_path)
    else:
        model = model_or_path

    # Build X_test / y_test (numeric features only)
    X_test = (
        test_data.drop(columns=[label], errors="ignore")
        .select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    y_test = test_data[label].astype(bool).values

    # Predict positives
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold)
    except Exception:
        # fallback for classifiers without predict_proba
        y_pred = np.asarray(model.predict(X_test)).astype(bool)

    mask_tp = (y_test == True) & (y_pred == True)
    # Return empty DF with same columns if no TPs
    if not mask_tp.any():
        return test_data.iloc[0:0].copy()

    return test_data.loc[mask_tp].copy()

# --- fix the ratio helper (was passing only test & a path) --------------------
def get_release_ratio(project_release):
    """
    Computes % of TP in the target release over all releases (using RandomForest by default).
    """
    projects = read_dataset()
    totals = []
    target_release_num = None
    for project in projects:
        train, test = projects[project]
        model = get_model(project, "RandomForest")
        true_positives = get_true_positives(model, train, test)
        num_tp = len(true_positives)
        totals.append(num_tp)
        if project_release == project:
            target_release_num = num_tp

    total_tp = sum(totals) if totals else 0
    if not total_tp or target_release_num is None:
        return 0.0
    return target_release_num / total_tp * 100


def map_indexes_to_file(df, int_to_file):
    df.index = df.index.map(int_to_file)
    return df


def project_dataset(project: Path):
    releases = []
    for release_csv in project.glob("*.csv"):
        release = release_csv.name
        releases.append(release)
    releases = natsort.natsorted(releases)
    k_releases = []
    window = 2
    for i in range(len(releases) - window + 1):
        k_releases.append(releases[i : i + window])
    return k_releases


def all_dataset(dataset: Path = Path("./Dataset/project_dataset")):
    projects = {}
    for project in dataset.iterdir():
        if project.is_dir():
            k_releases = project_dataset(project)
            projects[project.name] = k_releases
    return projects


def read_dataset() -> dict[str, list[pd.DataFrame]]:
    projects = {}
    for project in Path(RELEASE_DATASET).iterdir():
        if not project.is_dir():
            continue
        train = pd.read_csv(project / "train.csv", index_col=0)
        test = pd.read_csv(project / "test.csv", index_col=0)

        projects[project.name] = [train, test]
    return projects