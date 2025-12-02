#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JIT Mahalanobis Feasibility (feature-space, not release deltas)

For each (project, model, explainer):

1. Fit Mahalanobis metric on the TRAIN feature distribution:
   - X_train = numeric(train.drop(columns=["target"]))
   - mu, inv_cov = mean and pseudo-inverse covariance
   - compute train distances d_train and max_d_train for normalization

2. For each flipped JIT instance:
   - original = test features for that test_idx
   - flipped  = flipped candidate features (from *_all.csv / CF_all.csv)
   - dist_orig = Mahalanobis(original)
   - dist_flip = Mahalanobis(flipped)
   - normalize by max_d_train:
       dist_orig_norm = dist_orig / max_d_train
       dist_flip_norm = dist_flip / max_d_train
       delta_norm     = (dist_flip - dist_orig) / max_d_train

3. Write a per-model CSV and one global CSV:
   ./evaluations/jit_maha/{model}.csv
   ./evaluations/jit_maha/all_models.csv
"""

from __future__ import annotations

import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

from hyparams import EXPERIMENTS
from data_utils import read_dataset, get_model

# ----------------------------- config -----------------------------

MODEL_ABBR = {
    "RandomForest": "RF",
    "SVM": "SVM",
    "LogisticRegression": "LR",
}

EXPLAINER_NAME_MAP = {
    "LIME": "LIME",
    "LIME-HPO": "LIME-HPO",
    "CF": "CF",  # CF generator: experiments/{project}/{model}/CF_all.csv
}


# ----------------------------- helpers -----------------------------

def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.select_dtypes(include=[np.number])
          .replace([np.inf, -np.inf], np.nan)
          .fillna(0.0)
    )


def _train_cols(train: pd.DataFrame) -> pd.Index:
    return _numeric_frame(train.drop(columns=["target"], errors="ignore")).columns


def _flip_path(project: str, model_type: str, explainer: str) -> Path:
    """
    Unified flip file paths for JIT experiments.
    - CF:  experiments/{project}/{model}/CF_all.csv
    - LIME/LIME-HPO: experiments/{project}/{model}/{explainer}_all.csv
    """
    if explainer == "CF":
        return Path(EXPERIMENTS) / f"{project}/{model_type}/CF_all.csv"
    return Path(EXPERIMENTS) / f"{project}/{model_type}/{explainer}_all.csv"


def _load_flips_matrix(
    project: str,
    model_type: str,
    explainer: str,
    feat_cols: List[str],
) -> Optional[pd.DataFrame]:
    """
    Load flips in a consistent matrix form:

    Returns a DataFrame with columns:
      - test_idx
      - all feature columns in feat_cols (some may be missing; we'll fill from original)

    Supports:
      - CF_all.csv (has test_idx column)
      - {explainer}_all.csv with index = test_idx and full feature columns.
    """
    flip_path = _flip_path(project, model_type, explainer)
    if not flip_path.exists() or flip_path.stat().st_size == 0:
        return None

    try:
        df = pd.read_csv(flip_path)
        if "test_idx" not in df.columns:
            # old format: index is test_idx
            df = pd.read_csv(flip_path, index_col=0).reset_index().rename(columns={"index": "test_idx"})
    except Exception:
        # fallback: try index_col only
        try:
            df = pd.read_csv(flip_path, index_col=0).reset_index().rename(columns={"index": "test_idx"})
        except Exception:
            return None

    if "test_idx" not in df.columns:
        return None

    df["test_idx"] = pd.to_numeric(df["test_idx"], errors="coerce")
    df = df.dropna(subset=["test_idx"]).copy()
    df["test_idx"] = df["test_idx"].astype(int)

    # Keep only relevant columns: test_idx + features that appear
    keep_cols = ["test_idx"] + [c for c in feat_cols if c in df.columns]
    df = df.loc[:, keep_cols].copy()
    if df.empty:
        return None

    return df


def _fit_mahalanobis(train_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit Mahalanobis parameters on TRAIN features (per project, per model).

    Returns:
      mu            : mean vector
      inv_cov       : pseudo-inverse covariance
      max_d_train   : max Mahalanobis distance among train rows (for normalization)
    """
    X = _numeric_frame(train_features)
    cols = X.columns
    if len(cols) == 0:
        raise ValueError("No numeric features in train.")

    mu = X.mean().values
    cov = np.cov(X.values.T)
    inv_cov = np.linalg.pinv(cov)

    d_train = []
    for row in X.values:
        d = mahalanobis(row, mu, inv_cov)
        if math.isfinite(d):
            d_train.append(d)

    max_d = max(d_train) if d_train else 1.0
    return mu, inv_cov, max_d


def _mahalanobis_vec(x: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> float:
    return float(mahalanobis(x, mu, inv_cov))


def _parse_list(arg: str, default_all: List[str]) -> List[str]:
    if not arg or arg.strip().lower() == "all":
        return list(default_all)
    parts = [a.strip() for a in arg.replace(",", " ").split() if a.strip()]
    return parts if parts else list(default_all)


# ----------------------------- main JIT metric -----------------------------

def jit_maha_for_project(
    project: str,
    model_type: str,
    explainer: str,
) -> List[Dict]:
    """
    Compute JIT Mahalanobis stats for a single (project, model, explainer).

    Returns list of dicts:
      {
        "project": str,
        "model": str,
        "explainer": str,
        "test_idx": int,
        "dist_orig": float,
        "dist_flip": float,
        "dist_orig_norm": float,
        "dist_flip_norm": float,
        "delta": float,
        "delta_norm": float,
        "num_feats_changed": int,
      }
    """
    ds = read_dataset()
    if project not in ds:
        return []

    train, test = ds[project]
    feat_cols = [c for c in test.columns if c != "target"]

    flips = _load_flips_matrix(project, model_type, explainer, feat_cols)
    if flips is None or flips.empty:
        return []

    # Fit Mahalanobis on TRAIN features
    mu, inv_cov, max_d = _fit_mahalanobis(train[feat_cols])

    rows = []
    for _, row in flips.iterrows():
        tidx = int(row["test_idx"])
        if tidx not in test.index:
            continue

        orig = test.loc[tidx, feat_cols].astype(float)
        cand = orig.copy()

        # Overwrite any feature present in the flipped row
        for f in feat_cols:
            if f in row.index and not pd.isna(row[f]):
                cand[f] = float(row[f])

        # Check if anything actually changed
        diff = cand.values - orig.values
        changed_mask = ~np.isclose(diff, 0.0, rtol=1e-7, atol=1e-7)
        num_changed = int(changed_mask.sum())
        if num_changed == 0:
            continue

        v_orig = orig.values.astype(float)
        v_flip = cand.values.astype(float)

        d_orig = _mahalanobis_vec(v_orig, mu, inv_cov)
        d_flip = _mahalanobis_vec(v_flip, mu, inv_cov)

        dist_orig_norm = d_orig / max_d if max_d > 0 else 0.0
        dist_flip_norm = d_flip / max_d if max_d > 0 else 0.0
        delta = d_flip - d_orig
        delta_norm = delta / max_d if max_d > 0 else 0.0

        rows.append(
            {
                "project": project,
                "model": model_type,
                "explainer": explainer,
                "test_idx": tidx,
                "dist_orig": d_orig,
                "dist_flip": d_flip,
                "dist_orig_norm": dist_orig_norm,
                "dist_flip_norm": dist_flip_norm,
                "delta": delta,
                "delta_norm": delta_norm,
                "num_feats_changed": num_changed,
            }
        )

    return rows


# ----------------------------- CLI -----------------------------

def main():
    parser = ArgumentParser()
    parser.add_argument("--models", type=str, default="RandomForest,SVM,LogisticRegression",
                        help='Models spaced/comma, or "all" (default: RF,SVM,LR)')
    parser.add_argument("--explainers", type=str, default="LIME,LIME-HPO,CF",
                        help='Explainers spaced/comma, or "all" (default: LIME,LIME-HPO,CF)')
    parser.add_argument("--projects", type=str, default="all",
                        help='Projects spaced/comma, or "all"')
    args = parser.parse_args()

    ds = read_dataset()
    all_projects = list(sorted(ds.keys()))

    model_types = _parse_list(args.models, list(MODEL_ABBR.keys()))
    explainers = _parse_list(args.explainers, list(EXPLAINER_NAME_MAP.keys()))
    if args.projects.strip().lower() == "all":
        project_list = all_projects
    else:
        project_list = _parse_list(args.projects, all_projects)

    out_root = Path("./evaluations/jit_maha")
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []

    print(f"Projects ({len(project_list)}): {project_list}")
    print(f"Models: {model_types}")
    print(f"Explainers: {explainers}")

    for model_type in model_types:
        model_rows = []
        for explainer in explainers:
            for project in project_list:
                print(f"[JIT-MAHA] {project} / {model_type} / {explainer}")
                rows = jit_maha_for_project(project, model_type, explainer)
                if rows:
                    model_rows.extend(rows)
                    all_rows.extend(rows)

        # per-model CSV
        if model_rows:
            df_m = pd.DataFrame(model_rows)
            m_abbr = MODEL_ABBR.get(model_type, model_type)
            df_m.to_csv(out_root / f"{m_abbr}.csv", index=False)

    # global CSV
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        df_all.to_csv(out_root / "all_models.csv", index=False)
        # quick sanity summary
        print("\n=== Global summary (normalized distances) ===")
        print(
            df_all.groupby(["model", "explainer"])["dist_flip_norm"]
                  .describe(percentiles=[0.5, 0.9, 0.95])
        )
    else:
        print("No rows produced (check that *_all.csv files exist and contain flips).")


if __name__ == "__main__":
    main()
