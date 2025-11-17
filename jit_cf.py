#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICE Counterfactuals for JIT datasets (≤K edits via NICE max_edits)

Assumptions:
- Datasets from read_dataset() are commit-level JIT tables:
    columns: JIT metrics + 'target' (+ maybe some metadata)
- get_model(project, model_type) returns a model that
  accepts X[:, feat_cols] in the SAME order as train_df[feat_cols].
- get_true_positives(model, train_df, test_df) returns true positives
  (we additionally filter to target == 1 so we only CF defective TPs).

We:
- Work in ORIGINAL feature space (no extra scaling).
- Let NICE use predict_fn(X) → 2D proba [p0, p1].
- Ask NICE to flip each defective TP, and we accept any flip 1 → 0.
- Enforce an edit budget only through NICE's internal max_edits=K.
- Output one CF per TP at most.

Output
------
./experiments/{project}/{model}/CF_all.csv

Columns:
    test_idx, candidate_id, <features...>, proba0, proba1,
    num_features_changed, dist_unit_l2
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# make sure we can import your local NICE package
sys.path.append("./NICE")
try:
    from nice import NICE
except Exception as e:
    print(e)
    raise ImportError(
        "Could not import NICE. Ensure the NICEx package is installed "
        "and exposes `from nice import NICE`."
    ) from e

# ---- JIT project helpers (already in your repo) ----
from data_utils import read_dataset, get_model, get_true_positives
from hyparams import EXPERIMENTS, SEED


# ============================ distance utils ============================

def mad_1d(arr: np.ndarray, c: float = 1.4826) -> float:
    med = np.median(arr)
    return float(c * np.median(np.abs(arr - med)) + 1e-12)


class UnitTransformer:
    """Robust units: (x - median) / MAD computed on TRAIN."""
    def __init__(self, X_train: np.ndarray):
        self.median = np.median(X_train, axis=0)
        self.mad = np.array([mad_1d(X_train[:, i]) for i in range(X_train.shape[1])])

    def to_units(self, X: np.ndarray) -> np.ndarray:
        return (X - self.median) / self.mad


def _distance_unit_l2(ut: UnitTransformer, x0: np.ndarray, X: np.ndarray) -> np.ndarray:
    x0u = ut.to_units(x0.reshape(1, -1))
    Xu = ut.to_units(X)
    diff = Xu - x0u
    return np.sqrt(np.sum(diff * diff, axis=1))


def _distance_euclidean_z(z_mean: np.ndarray, z_std: np.ndarray,
                          x0: np.ndarray, X: np.ndarray) -> np.ndarray:
    zstd = np.where(z_std > 0, z_std, 1.0)
    x0z = (x0.reshape(1, -1) - z_mean) / zstd
    Xz = (X - z_mean) / zstd
    diff = Xz - x0z
    return np.sqrt(np.sum(diff * diff, axis=1))


def _distance_raw_l2(x0: np.ndarray, X: np.ndarray) -> np.ndarray:
    diff = X - x0.reshape(1, -1)
    return np.sqrt(np.sum(diff * diff, axis=1))


def _distance_any(kind: str,
                  ut: UnitTransformer,
                  z_mean: Optional[np.ndarray],
                  z_std: Optional[np.ndarray],
                  x0: np.ndarray,
                  X: np.ndarray) -> np.ndarray:
    if kind == "unit_l2":
        return _distance_unit_l2(ut, x0, X)
    elif kind == "euclidean":
        assert z_mean is not None and z_std is not None, "z-stats not provided for euclidean distance"
        return _distance_euclidean_z(z_mean, z_std, x0, X)
    elif kind == "raw_l2":
        return _distance_raw_l2(x0, X)
    else:
        raise ValueError(f"Unknown distance kind: {kind}")


# ============================ prediction helper ============================

def _predict_proba_2d(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Predict P(y=0), P(y=1) in a robust way.

    - If model.predict_proba exists, we use it.
    - Else, we try decision_function.
    - Else, we fall back to hard predictions and fake probabilities.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 1:
            p1 = proba.astype(float)
            return np.stack([1.0 - p1, p1], axis=1)
        return proba.astype(float)

    if hasattr(model, "decision_function"):
        m = model.decision_function(X)
        m = np.asarray(m, dtype=float)
        if m.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-m))
            return np.stack([1.0 - p1, p1], axis=1)
        # multi-class margin: softmax
        expm = np.exp(m - np.max(m, axis=1, keepdims=True))
        return expm / expm.sum(axis=1, keepdims=True)

    # Fallback: use predicted label as "probability"
    y = model.predict(X)
    y = np.asarray(y, dtype=int)
    n = y.shape[0]
    p0 = (y == 0).astype(float)
    if np.unique(y).size == 1 and y[0] == 0:
        return np.stack([p0, 1.0 - p0], axis=1)
    p1 = (y == 1).astype(float)
    return np.stack([p0, p1], axis=1)


# ============================ config ============================

@dataclass
class GenCfg:
    max_features: int
    total_cfs: int
    distance: str = "unit_l2"  # 'unit_l2' | 'euclidean' | 'raw_l2'
    nice_distance_metric: str = "HEOM"
    nice_optimization: str = "proximity"
    nice_justified_cf: bool = True
    seed: int = SEED


# ============================ core: run NICE on JIT ============================

def run_project(project: str,
                model_type: str,
                total_cfs: int,
                max_features: int,
                verbose: bool,
                overwrite: bool,
                cfg_overrides: Dict[str, Any]):

    ds = read_dataset()
    if project not in ds:
        tqdm.write(f"[{project}/{model_type}] dataset not found. Skipping.")
        return

    train, test = ds[project]

    # Use the same feature set convention as your JIT LIME script:
    feat_cols = [c for c in train.columns if c != "target"]
    if not feat_cols:
        tqdm.write(f"[{project}/{model_type}] no feature columns. Skipping.")
        return

    # EXACT same model object as elsewhere in your JIT pipeline
    model = get_model(project, model_type)

    Xtr = train[feat_cols].values.astype(float)
    ytr = train["target"].values.astype(int)

    # True positives according to your own helper
    tp_df = get_true_positives(model, train, test)
    if tp_df.empty:
        tqdm.write(f"[{project}/{model_type}] no true positives. Skipping.")
        return

    # Focus on defective TPs (pred=1 & target=1)
    if "target" in tp_df.columns:
        tp_df = tp_df[tp_df["target"] == 1]
        if tp_df.empty:
            tqdm.write(f"[{project}/{model_type}] no defective TPs (target==1). Skipping.")
            return

    # distances for reporting
    ut = UnitTransformer(Xtr)
    z_mean = Xtr.mean(axis=0)
    z_std = Xtr.std(axis=0)

    d = Xtr.shape[1]

    # NICE configuration
    nice_distance_metric = cfg_overrides.get("nice_distance_metric", "HEOM")
    nice_optimization = cfg_overrides.get("nice_optimization", "proximity")
    nice_justified_cf = cfg_overrides.get("nice_justified_cf", True)

    # Instantiate NICE on ORIGINAL feature space with 2D proba
    nice_kwargs = dict(
        X_train=Xtr,
        y_train=ytr,
        predict_fn=lambda X: _predict_proba_2d(model, X),
        cat_feat=[],                  # all numeric JIT metrics
        num_feat=list(range(d)),
        distance_metric=nice_distance_metric,
        num_normalization="minmax",
        optimization=nice_optimization,
        justified_cf=nice_justified_cf,
        max_edits=max_features,       # ≤ K edits enforced internally
    )

    tqdm.write(f"[NICE] Initialization (reward-free beam), optimization='{nice_optimization}'")
    nice = NICE(**nice_kwargs)

    cfg = GenCfg(
        max_features=max_features,
        total_cfs=total_cfs,
        distance=cfg_overrides.get("distance", "unit_l2"),
        nice_distance_metric=nice_distance_metric,
        nice_optimization=nice_optimization,
        nice_justified_cf=nice_justified_cf,
        seed=cfg_overrides.get("seed", SEED),
    )

    out_path = Path(EXPERIMENTS) / project / model_type / "CF_all.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and out_path.exists():
        out_path.unlink(missing_ok=True)

    results: List[pd.DataFrame] = []
    misses = 0

    for idx in tqdm(
        tp_df.index.astype(int),
        desc=f"{project}/{model_type}/nice (NICE; ≤K={cfg.max_features}, dist={cfg.distance})",
        leave=False,
        disable=not verbose,
    ):
        # Original instance in feature space
        x0_row = test.loc[idx, feat_cols].astype(float)
        x0 = x0_row.values.astype(float)

        # Sanity: original label should be 1 (defective)
        p0 = _predict_proba_2d(model, x0.reshape(1, -1))[0]
        orig_label = int(np.argmax(p0))
        if orig_label != 1:
            # if get_true_positives ever gives something else, just skip
            continue

        # Get CF from NICE
        try:
            cf_raw = nice.explain(x0.reshape(1, -1))
        except Exception:
            misses += 1
            continue

        cf_arr = np.asarray(cf_raw, dtype=float)

        # Robustly coerce: NICE may return (n_cf, d) or (1, d) or (1,1,d)
        if cf_arr.ndim == 3 and cf_arr.shape[0] == 1 and cf_arr.shape[1] == 1:
            cf_arr = cf_arr.reshape(1, cf_arr.shape[-1])

        if cf_arr.ndim == 2:
            cf = cf_arr[0].astype(float)
        elif cf_arr.ndim == 1:
            cf = cf_arr.astype(float)
        else:
            misses += 1
            continue

        # Label after CF
        p_cf = _predict_proba_2d(model, cf.reshape(1, -1))[0]
        label_cf = int(np.argmax(p_cf))

        # We want an actual flip 1 → 0
        if label_cf == orig_label:
            misses += 1
            continue

        # Count changed features (and assert it's ≤ K; NICE should enforce this)
        changed_mask = ~np.isclose(cf, x0, rtol=1e-7, atol=1e-12)
        k_used = int(np.sum(changed_mask))
        if k_used == 0:
            # no actual change → discard
            misses += 1
            continue
        if k_used > cfg.max_features:
            # safety: if NICE ignored max_edits, we drop it to keep protocol clean
            misses += 1
            continue

        # distance for reporting
        dist_val = float(_distance_any(cfg.distance, ut, z_mean, z_std,
                                       x0, cf.reshape(1, -1))[0])

        rec = {
            **{c: float(v) for c, v in zip(feat_cols, cf)},
            "proba0": float(p_cf[0]),
            "proba1": float(p_cf[1]),
            "num_features_changed": k_used,
            "dist_unit_l2": dist_val,
        }
        row = pd.DataFrame([rec])
        row.insert(0, "candidate_id", 0)
        row.insert(0, "test_idx", int(idx))
        results.append(row)

    if results:
        out_df = pd.concat(results, axis=0, ignore_index=True)
        cols = (
            ["test_idx", "candidate_id"]
            + feat_cols
            + ["proba0", "proba1", "num_features_changed", "dist_unit_l2"]
        )
        out_df = out_df[cols]
        out_df.to_csv(out_path, index=False)
        uniq = out_df["test_idx"].nunique()
        tqdm.write(
            f"[OK] {project}/{model_type}/nice: wrote {len(out_df)} rows across "
            f"{uniq} TP(s) → {out_path} | misses={misses}"
        )
    else:
        tqdm.write(f"[{project}/{model_type}/nice] no candidates found. misses={misses}")


# ============================ CLI ============================

def main():
    ap = ArgumentParser(description="NICE CFs for JIT (≤K edits via NICE max_edits)")
    ap.add_argument("--project", type=str, default="all")
    ap.add_argument(
        "--model_types",
        type=str,
        default="RandomForest,SVM,LogisticRegression",
    )
    ap.add_argument("--total_cfs", type=int, default=1)  # 1 CF per TP in this script
    ap.add_argument("--max_features", type=int, default=5)

    # distance for reporting/ranking
    ap.add_argument(
        "--distance",
        type=str,
        default="unit_l2",
        choices=["unit_l2", "euclidean", "raw_l2"],
    )

    # NICE options
    ap.add_argument("--nice_distance_metric", type=str, default="HEOM")
    ap.add_argument("--nice_optimization", type=str, default="plausibility")
    ap.add_argument("--justified_cf", dest="nice_justified_cf", action="store_true")
    ap.add_argument("--no_justified_cf", dest="nice_justified_cf", action="store_false")
    ap.set_defaults(nice_justified_cf=True)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    projects = read_dataset()
    project_list = (
        list(sorted(projects.keys()))
        if args.project == "all"
        else [p.strip() for p in args.project.replace(",", " ").split() if p.strip()]
    )
    model_types = [
        m.strip() for m in args.model_types.replace(",", " ").split() if m.strip()
    ]

    cfg_overrides = dict(
        distance=args.distance,
        nice_distance_metric=args.nice_distance_metric,
        nice_optimization=args.nice_optimization,
        nice_justified_cf=args.nice_justified_cf,
        seed=SEED,
    )

    print(
        f"Running NICE CFs for JIT (≤K={args.max_features}) "
        f"for {len(project_list)} projects × {len(model_types)} models"
    )
    print(
        f"Distance (report): {args.distance} | NICE metric: {args.nice_distance_metric} | "
        f"NICE optimization: {args.nice_optimization}"
    )
    print("Output: ./experiments/{project}/{model}/CF_all.csv\n")

    for p in tqdm(project_list, desc="Projects", disable=not args.verbose):
        for mt in model_types:
            run_project(
                project=p,
                model_type=mt,
                total_cfs=args.total_cfs,
                max_features=args.max_features,
                verbose=args.verbose,
                overwrite=args.overwrite,
                cfg_overrides=cfg_overrides,
            )


if __name__ == "__main__":
    main()
