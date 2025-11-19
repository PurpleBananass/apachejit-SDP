#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NICE Counterfactuals (≤K by default, no refinement/heuristics)

- Uses the SAME StandardScaler behavior as utils.get_true_positives:
  scaler = StandardScaler().fit(TRAIN[features]); predictions on SCALED(X)
- Does NOT re-check positivity (we trust get_true_positives).
- Calls NICE on original feature space; predict_fn internally scales.

Output
------
experiments/{project}/{model}/CF_all.csv
Columns: test_idx, candidate_id, <features...>, proba0, proba1, num_features_changed, dist_unit_l2
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional
import sys
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
sys.path.append("./NICE")
# NICE (pip package must expose `from nice import NICE`)
try:
    from nice import NICE
except Exception as e:
    print(e)
    raise ImportError("Could not import NICE. Ensure the package exposes `from nice import NICE`.") from e

# ---- project helpers (user-provided) ----
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

def _distance_euclidean_z(z_mean: np.ndarray, z_std: np.ndarray, x0: np.ndarray, X: np.ndarray) -> np.ndarray:
    zstd = np.where(z_std > 0, z_std, 1.0)
    x0z = (x0.reshape(1, -1) - z_mean) / zstd
    Xz  = (X - z_mean) / zstd
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


# ============================ prediction (match utils) ============================

def _fit_utils_scaler(train_df: pd.DataFrame, feat_cols: List[str]) -> StandardScaler:
    """Exactly the same scaling as utils.get_true_positives (fit on TRAIN features)."""
    sc = StandardScaler()
    sc.fit(train_df[feat_cols].values.astype(float))
    return sc

def _predict_proba_scaled(model: Any, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Predict proba after applying the SAME scaler as utils.get_true_positives."""
    Xs = scaler.transform(np.asarray(X, dtype=float))
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xs)
        if proba.ndim == 1:
            p1 = proba
            return np.stack([1 - p1, p1], axis=1)
        return proba
    if hasattr(model, "decision_function"):
        m = model.decision_function(Xs)
        if np.ndim(m) == 1:
            p1 = 1.0 / (1.0 + np.exp(-m))
            return np.stack([1 - p1, p1], axis=1)
    y = model.predict(Xs)
    p0 = (y == 0).astype(float)
    return np.stack([p0, 1.0 - p0], axis=1)


# ============================ K enforcement ============================

def greedy_keep_at_most_k(x0: np.ndarray,
                          cf: np.ndarray,
                          k: Optional[int],
                          predict_proba_fn,
                          label_target: int = 0,
                          exact_k: bool = False) -> np.ndarray:
    """
    Enforce ≤k edits (or exactly k if exact_k=True) by reverting least-impact edits first.
    'Impact' = drop in p(target) when reverting that coordinate.
    """
    if k is None:
        return cf

    changed = np.flatnonzero(~np.isclose(cf, x0, rtol=1e-7, atol=1e-12))
    if (not exact_k and len(changed) <= k) or (exact_k and len(changed) == k):
        return cf.copy()

    z = cf.copy()
    if exact_k and len(changed) < k:
        # Can't safely "add edits"; return as-is.
        return z

    while (len(changed) > k) or (exact_k and len(changed) != k):
        base_p = predict_proba_fn(z.reshape(1, -1))[0, label_target]
        best_i, best_drop = None, np.inf
        for i in changed:
            tmp = z.copy()
            tmp[i] = x0[i]
            p = predict_proba_fn(tmp.reshape(1, -1))[0, label_target]
            drop = base_p - p
            if drop < best_drop:
                best_drop = drop
                best_i = i

        # revert the least-impact feature
        z[best_i] = x0[best_i]
        changed = np.flatnonzero(~np.isclose(z, x0, rtol=1e-7, atol=1e-12))

        # preserve flip; if lost, undo and stop relaxing
        if predict_proba_fn(z.reshape(1, -1))[0, label_target] < 0.5:
            z[best_i] = cf[best_i]
            break

        if not exact_k and len(changed) <= k:
            break
        if exact_k and len(changed) == k:
            break

    return z


# ============================ config ============================

@dataclass
class GenCfg:
    # Edit budget (≤K by default; use --exact_k to force exactly K)
    max_features: int
    total_cfs: int
    exact_k: bool = False

    # distance for reporting
    distance: str = "unit_l2"  # 'unit_l2' | 'euclidean' | 'raw_l2'

    # NICE options
    nice_distance_metric: str = "HEOM"      # works for numeric-only as well
    nice_optimization: str = "proximity"    # "proximity" | "sparsity"
    nice_justified_cf: bool = True

    # seed
    seed: int = SEED


# ============================ core: run NICE ============================

def run_project(project: str,
                model_type: str,
                method: str,
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
    feat_cols = [c for c in test.columns if c != "target"]

    # EXACT same model object as used by utils
    base_model = get_model(project, model_type)

    # SAME scaler behavior as utils.get_true_positives (fit on TRAIN features)
    scaler = _fit_utils_scaler(train, feat_cols)

    Xtr = train[feat_cols].values.astype(float)
    ytr = train["target"].values.astype(int)

    # TPs strictly per utils (no re-filtering here)
    tp_df = get_true_positives(base_model, train, test)
    if tp_df.empty:
        tqdm.write(f"[{project}/{model_type}] no true positives. Skipping.")
        return

    # distances for reporting
    ut = UnitTransformer(Xtr)
    z_mean = Xtr.mean(axis=0)
    z_std = Xtr.std(axis=0)

    # NICE on original schema; predict_fn internally scales to match utils
    d = Xtr.shape[1]
    nice = NICE(
        X_train=Xtr,
        y_train=ytr,
        predict_fn=lambda X: _predict_proba_scaled(base_model, scaler, X),
        cat_feat=[],                              # set if you have categoricals
        num_feat=list(range(d)),
        distance_metric=cfg_overrides.get("nice_distance_metric", "HEOM"),
        num_normalization="minmax",
        optimization=cfg_overrides.get("nice_optimization", "proximity"),
        justified_cf=cfg_overrides.get("nice_justified_cf", True),
        max_edits=5  
    )

    cfg = GenCfg(
        max_features=max_features,
        total_cfs=total_cfs,
        exact_k=cfg_overrides.get("exact_k", False),
        distance=cfg_overrides.get("distance", "unit_l2"),
        nice_distance_metric=cfg_overrides.get("nice_distance_metric", "HEOM"),
        nice_optimization=cfg_overrides.get("nice_optimization", "proximity"),
        nice_justified_cf=cfg_overrides.get("nice_justified_cf", True),
        seed=cfg_overrides.get("seed", SEED),
    )

    out_path = Path(EXPERIMENTS) / project / model_type / "CF_all.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and out_path.exists():
        out_path.unlink(missing_ok=True)

    results: List[pd.DataFrame] = []
    misses = 0

    for idx in tqdm(tp_df.index.astype(int),
                    desc=f"{project}/{model_type}/{method} (NICE; ≤K={cfg.max_features}{' | EXACT-K' if cfg.exact_k else ''}, dist={cfg.distance})",
                    leave=False, disable=not verbose):

        # Keep original-space row; predictions will be scaled inside predict_proba
        x0_row = test.loc[idx, feat_cols].astype(float)
        x0 = x0_row.values.astype(float)

        # Get CF from NICE
        try:
            cf = nice.explain(x0.reshape(1, -1))[0].astype(float)
        except Exception:
            misses += 1
            continue

        # Optional ≤K / EXACT-K enforcement with the SAME scaled predictor
        if cfg.max_features is not None:
            cf = greedy_keep_at_most_k(
                x0=x0,
                cf=cf,
                k=int(cfg.max_features),
                predict_proba_fn=lambda X: _predict_proba_scaled(base_model, scaler, X),
                label_target=0,                 # flip to class 0
                exact_k=bool(cfg.exact_k),
            )

        p_cf = _predict_proba_scaled(base_model, scaler, cf.reshape(1, -1))[0]
        if p_cf[0] < 0.5:
            # NICE returned a non-flip (or greedy step broke it)
            misses += 1
            continue

        # distance (stored under legacy name `dist_unit_l2`)
        dist_val = float(_distance_any(cfg.distance, ut, z_mean, z_std, x0, cf.reshape(1, -1))[0])
        k_used = int(np.sum(~np.isclose(cf, x0, rtol=1e-7, atol=1e-12)))

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
        cols = ["test_idx", "candidate_id"] + feat_cols + ["proba0", "proba1", "num_features_changed", "dist_unit_l2"]
        out_df = out_df[cols]
        out_df.to_csv(out_path, index=False)
        uniq = out_df["test_idx"].nunique()
        tqdm.write(
            f"[OK] {project}/{model_type}/{method}: wrote {len(out_df)} rows across "
            f"{uniq} TP(s) → {out_path} | misses={misses}"
        )
    else:
        tqdm.write(f"[{project}/{model_type}/{method}] no candidates found. misses={misses}")


# ============================ CLI ============================

def main():
    ap = ArgumentParser(description="NICE CFs (≤K by default; no refinement)")
    ap.add_argument("--project", type=str, default="all")
    ap.add_argument("--model_types", type=str, default="RandomForest,SVM,LogisticRegression")
    ap.add_argument("--method", type=str, default="nice")
    ap.add_argument("--total_cfs", type=int, default=1)  # 1 CF per instance here
    ap.add_argument("--max_features", type=int, default=5)

    # K policy
    ap.add_argument("--exact_k", action="store_true", help="Enforce exactly K (=max_features) edited features")

    # distance for reporting/ranking
    ap.add_argument("--distance", type=str, default="unit_l2", choices=["unit_l2", "euclidean", "raw_l2"])

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
    project_list = list(sorted(projects.keys())) if args.project == "all" else \
                   [p.strip() for p in args.project.replace(",", " ").split() if p.strip()]
    model_types = [m.strip() for m in args.model_types.replace(",", " ").split() if m.strip()]

    cfg_overrides = dict(
        exact_k=bool(args.exact_k),
        distance=args.distance,
        nice_distance_metric=args.nice_distance_metric,
        # nice_optimization=args.nice_optimization,
        nice_justified_cf=args.nice_justified_cf,
        seed=SEED,
    )

    print(f"Running NICE CFs (≤K={args.max_features}{' | EXACT-K' if args.exact_k else ''}) "
          f"for {len(project_list)} projects × {len(model_types)} models")
    print(f"Distance metric (report): {args.distance} | NICE metric: {args.nice_distance_metric}")
    print(f"Output: experiments/{{project}}/{{model}}/CF_all.csv\n")

    for p in tqdm(project_list, desc="Projects", disable=not args.verbose):
        for mt in model_types:
            run_project(
                project=p,
                model_type=mt,
                method=args.method,
                total_cfs=args.total_cfs,
                max_features=args.max_features,
                verbose=args.verbose,
                overwrite=args.overwrite,
                cfg_overrides=cfg_overrides,
            )

if __name__ == "__main__":
    from argparse import ArgumentParser
    main()
