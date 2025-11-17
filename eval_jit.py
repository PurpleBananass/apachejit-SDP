# eval_jit.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import json
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from tabulate import tabulate

from hyparams import PROPOSED_CHANGES, EXPERIMENTS
from data_utils import read_dataset, get_model

# If you kept get_flip_rates in flip_jit.py:
try:
    from flip_jit import get_flip_rates
except Exception:
    get_flip_rates = None  # RQ1 path will be disabled if not available


# ---------- utilities (match training: raw numeric only, aligned cols) ----------
def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.select_dtypes(include=[np.number])
          .replace([np.inf, -np.inf], np.nan)
          .fillna(0.0)
    )


def _train_cols(train: pd.DataFrame) -> pd.Index:
    return _numeric_frame(train.drop(columns=["target"], errors="ignore")).columns


def _align_like_one(x_row: pd.Series, cols: pd.Index) -> pd.DataFrame:
    X = x_row.to_frame().T
    X = _numeric_frame(X)
    X = X.reindex(columns=list(cols), fill_value=0.0)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _generate_all_combinations(plan_dict: Dict[str, List[float]]) -> pd.DataFrame:
    feats = list(plan_dict.keys())
    values = list(plan_dict.values())
    if len(feats) == 0:
        return pd.DataFrame()
    combos = list(product(*values))
    return pd.DataFrame(combos, columns=feats)


# ---------- distances ----------
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (n1 * n2))


def cosine_all(df: pd.DataFrame, x: pd.Series | np.ndarray) -> List[float]:
    if isinstance(x, pd.Series):
        x = x.values
    dists = []
    for _, row in df.iterrows():
        dists.append(cosine_similarity(x, row.values))
    return dists


def _stdz(col: pd.Series, v: float) -> float:
    mu = float(col.mean())
    sd = float(col.std()) if float(col.std()) != 0 else 1.0
    return (float(v) - mu) / sd


def mahalanobis_all(df: pd.DataFrame, x: pd.Series) -> List[float]:
    """
    Mahalanobis distances of x to each row in df, normalized by the span distance
    (min vs max) along the same covariance. Robust to the single-feature case.
    """
    # keep columns with >1 unique value
    df = df.loc[:, (df.nunique() > 1)]
    if df.shape[1] == 0:
        return []

    # standardize within df (avoid 0 std)
    std_df = (df - df.mean()) / df.std(ddof=0).replace(0, 1.0)
    # x as standardized vector in the same columns/order
    cols = list(std_df.columns)
    x_std = np.array(
        [(float(x[c]) - float(df[c].mean())) / (float(df[c].std(ddof=0)) or 1.0) for c in cols],
        dtype=float,
    )

    # covariance over features (rowvar=False) → always 2D
    cov = np.cov(std_df.values, rowvar=False, ddof=0)
    cov = np.atleast_2d(cov)
    inv_cov = np.linalg.pinv(cov)

    # span normalization (min↔max in standardized space)
    min_vec = std_df.min().values
    max_vec = std_df.max().values
    max_span = float(mahalanobis(min_vec, max_vec, inv_cov))
    if not np.isfinite(max_span) or max_span == 0.0:
        max_span = 1.0

    out: List[float] = []
    for _, row in std_df.iterrows():
        d = float(mahalanobis(x_std, row.values, inv_cov)) / max_span
        out.append(d)
    return out


def normalized_mahalanobis_distance(
    grid: pd.DataFrame, flipped: pd.Series, baseline: pd.Series
) -> float:
    """
    Distance between flipped and baseline normalized by the grid’s covariance span.
    Robust to the single-feature case.
    """
    if grid.shape[1] == 0:
        return 0.0

    cols = [c for c in grid.columns if c in flipped.index and c in baseline.index]
    if not cols:
        return 0.0
    grid = grid[cols]
    x = flipped[cols].astype(float)
    y = baseline[cols].astype(float)

    # standardize within grid
    mu = grid.mean()
    sd = grid.std(ddof=0).replace(0, 1.0)
    std_grid = (grid - mu) / sd
    x_std = ((x - mu) / sd).values.astype(float)
    y_std = ((y - mu) / sd).values.astype(float)

    cov = np.cov(std_grid.values, rowvar=False, ddof=0)
    cov = np.atleast_2d(cov)
    inv_cov = np.linalg.pinv(cov)

    dist = float(mahalanobis(x_std, y_std, inv_cov))

    min_vec = std_grid.min().values
    max_vec = std_grid.max().values
    max_span = float(mahalanobis(min_vec, max_vec, inv_cov))
    if not np.isfinite(max_span) or max_span == 0.0:
        return 0.0
    return dist / max_span


# ---------- plan similarity (RQ2-style per flip) ----------
def plan_similarity(project: str, model_type: str, explainer: str, threshold: float = 0.5):
    """
    For each flipped TP having a plan, compute a normalized Mahalanobis
    between (flipped values on changed features) vs (baseline = first candidate value per feature),
    using the plan’s own candidate grid as the covariance reference.

    NOTE: For NICE there is no plans_all.json; we skip and return {}.
    """
    if explainer.upper() == "NICE":
        # NICE has no per-feature plan grid; similarity is not defined
        return {}

    results = {}
    plan_path = Path(PROPOSED_CHANGES) / f"{project}/{model_type}/{explainer}/plans_all.json"
    flip_path = Path(EXPERIMENTS) / f"{project}/{model_type}/{explainer}_all.csv"
    if (not plan_path.exists()) or (not flip_path.exists()):
        return {}

    with open(plan_path, "r") as f:
        plans = json.load(f)

    flipped_df = pd.read_csv(flip_path, index_col=0).dropna()
    if flipped_df.empty:
        return {}

    train, test = read_dataset()[project]
    cols = _train_cols(train)
    model = get_model(project, model_type)

    for test_idx in flipped_df.index:
        key = str(test_idx)
        if key not in plans:
            continue

        original = test.loc[test_idx, test.columns != "target"]
        flipped = flipped_df.loc[test_idx, :]

        # Verify the flip is indeed to "clean" (robust check)
        X_orig = _align_like_one(original, cols)
        X_flip = _align_like_one(flipped, cols)
        try:
            p_orig = float(model.predict_proba(X_orig)[0, 1])
            p_flip = float(model.predict_proba(X_flip)[0, 1])
            if not (p_orig >= threshold and p_flip < threshold):
                # still compute similarity, but note mismatch
                pass
        except Exception:
            # If no predict_proba, we can’t verify; proceed.
            pass

        # Which features actually changed?
        changed = {
            f
            for f in plans[key].keys()
            if not math.isclose(
                float(flipped[f]),
                float(original[f]),
                rel_tol=1e-7,
                abs_tol=1e-12,
            )
        }
        if not changed:
            continue

        # Build plan grid & baseline
        plan_subset = {f: list(plans[key][f]) for f in changed if len(plans[key][f]) > 0}
        if not plan_subset:
            continue

        grid = _generate_all_combinations(plan_subset)
        flipped_on_changed = flipped[list(plan_subset.keys())]
        baseline = pd.Series(
            [plans[key][f][0] for f in plan_subset.keys()],
            index=list(plan_subset.keys()),
            dtype=float,
        )

        score = normalized_mahalanobis_distance(grid, flipped_on_changed, baseline)
        results[int(test_idx)] = {"score": float(score)}

    return results


# ---------- feasibility (RQ3-like) ----------
def _delta_pool_from_train_test(
    train: pd.DataFrame, test: pd.DataFrame, features: List[str]
) -> pd.DataFrame:
    """Construct a rough 'historical delta' pool from consecutive diffs on concatenated train+test.
    This is a pragmatic fallback for JIT splits where indices don't overlap."""
    df = pd.concat(
        [
            train.drop(columns=["target"], errors="ignore"),
            test.drop(columns=["target"], errors="ignore"),
        ],
        axis=0,
        ignore_index=True,
    )
    df = _numeric_frame(df)
    # only keep features that survived numeric filtering
    features = [f for f in features if f in df.columns]
    if not features:
        return pd.DataFrame()
    df = df[features]
    # consecutive diffs; drop rows with all zeros
    deltas = df.diff().dropna()
    deltas = deltas.loc[(deltas != 0).any(axis=1)]
    return deltas


def flip_feasibility(
    projects: List[str], explainer: str, model_type: str, distance: str = "mahalanobis"
):
    """
    Compute per-flip distance (min/max/mean) of changed vector vs a pool of 'historical' deltas.

    - For LIME / LIME-HPO / PyExplainer: use plans_all.json to know which
      features are allowed to change.
    - For NICE: use all numeric JIT metrics where CF != original as changed features,
      using CF_all.csv instead of <explainer>_all.csv and no plans_all.json.
    """
    totals = 0
    cannots = 0
    written = 0
    skipped_no_flipfile = 0
    skipped_no_plan = 0
    skipped_zero_change = 0
    skipped_empty_pool = 0
    skipped_rank_too_low = 0

    results = []

    ds = read_dataset()
    nice_mode = explainer.upper() == "NICE"

    for project in projects:
        if nice_mode:
            plan_path = None
            flip_path = Path(EXPERIMENTS) / f"{project}/{model_type}/CF_all.csv"
        else:
            plan_path = (
                Path(PROPOSED_CHANGES)
                / f"{project}/{model_type}/{explainer}/plans_all.json"
            )
            flip_path = (
                Path(EXPERIMENTS) / f"{project}/{model_type}/{explainer}_all.csv"
            )

        if not flip_path.exists():
            skipped_no_flipfile += 1
            continue

        flipped = pd.read_csv(flip_path, index_col=0).dropna()
        if flipped.empty:
            continue
        totals += len(flipped)

        if not nice_mode:
            if not plan_path.exists():
                skipped_no_plan += len(flipped)
                continue
            with open(plan_path, "r") as f:
                plans = json.load(f)
        else:
            plans = None  # not used

        train, test = ds[project]
        # numeric metrics in train
        train_num = _numeric_frame(train.drop(columns=["target"], errors="ignore"))
        numeric_feats = list(train_num.columns)

        for test_idx in flipped.index:
            key = str(test_idx)

            original = test.loc[test_idx, test.columns != "target"]
            flip_row = flipped.loc[test_idx, :]

            if nice_mode:
                # NICE: changed = all numeric metrics that differ
                changed = {
                    f: float(flip_row[f]) - float(original[f])
                    for f in numeric_feats
                    if f in flip_row.index
                    and f in original.index
                    and not math.isclose(
                        float(flip_row[f]),
                        float(original[f]),
                        rel_tol=1e-7,
                        abs_tol=1e-12,
                    )
                }
            else:
                if key not in plans:
                    skipped_no_plan += 1
                    continue
                changed = {
                    f: flip_row[f] - original[f]
                    for f in plans[key].keys()
                    if not math.isclose(
                        float(flip_row[f]),
                        float(original[f]),
                        rel_tol=1e-7,
                        abs_tol=1e-12,
                    )
                }

            if not changed:
                skipped_zero_change += 1
                continue

            feat_names = list(changed.keys())
            changed_vec = pd.Series(changed)

            # build delta pool for these features
            pool = _delta_pool_from_train_test(train, test, feat_names)
            if pool.empty:
                cannots += 1
                skipped_empty_pool += 1
                continue

            if distance == "cosine":
                dists = cosine_all(pool, changed_vec)
            else:
                # need at least (features+1) rows for covariance rank
                if len(pool) <= len(feat_names):
                    cannots += 1
                    skipped_rank_too_low += 1
                    continue
                dists = mahalanobis_all(pool, changed_vec)

            if isinstance(dists, list) and len(dists) > 0:
                results.append(
                    {
                        "project": project,
                        "min": float(np.min(dists)),
                        "max": float(np.max(dists)),
                        "mean": float(np.mean(dists)),
                    }
                )
                written += 1
            else:
                cannots += 1
                skipped_empty_pool += 1

    print(
        f"[{model_type} {explainer} {distance}] totals={totals}, written={written}, cannot={cannots} | "
        f"no_flipfile={skipped_no_flipfile}, no_plan={skipped_no_plan}, zero_change={skipped_zero_change}, "
        f"empty_pool={skipped_empty_pool}, rank_too_low={skipped_rank_too_low}"
    )
    return results, totals, cannots


# ---------- absolute-change “implications” ----------
def implications(project: str, explainer: str, model_type: str):
    """
    Sum of |z-scored deltas| over changed features per flipped instance.

    - For LIME / LIME-HPO / PyExplainer: changed features are those in plans[key]
      whose values actually changed.
    - For NICE: changed features are all numeric JIT metrics whose values differ
      between original and CF in CF_all.csv (no plans_all.json).
    """
    nice_mode = explainer.upper() == "NICE"

    if nice_mode:
        plan_path = None
        flip_path = Path(EXPERIMENTS) / f"{project}/{model_type}/CF_all.csv"
    else:
        plan_path = Path(PROPOSED_CHANGES) / f"{project}/{model_type}/{explainer}/plans_all.json"
        flip_path = Path(EXPERIMENTS) / f"{project}/{model_type}/{explainer}_all.csv"

    if not flip_path.exists():
        return []

    if not nice_mode and not plan_path.exists():
        return []

    if nice_mode:
        plans = None
    else:
        with open(plan_path, "r") as f:
            plans = json.load(f)

    flipped = pd.read_csv(flip_path, index_col=0).dropna()
    if flipped.empty:
        return []

    train, test = read_dataset()[project]
    train_num = _numeric_frame(train.drop(columns=["target"], errors="ignore"))
    mu = train_num.mean()
    sd = train_num.std(ddof=0).replace(0, 1.0)

    totals = []
    for test_idx in flipped.index:
        key = str(test_idx)

        orig = test.loc[test_idx, :]
        flip = flipped.loc[test_idx, :]

        if nice_mode:
            # changed = all numeric features that differ
            changed = [
                f
                for f in train_num.columns
                if f in flip.index
                and f in orig.index
                and not math.isclose(
                    float(flip[f]),
                    float(orig[f]),
                    rel_tol=1e-7,
                    abs_tol=1e-12,
                )
            ]
        else:
            if key not in plans:
                continue
            changed = [
                f
                for f in plans[key].keys()
                if not math.isclose(
                    float(flip[f]),
                    float(orig[f]),
                    rel_tol=1e-7,
                    abs_tol=1e-12,
                )
            ]

        if not changed:
            continue

        # z-scored absolute deltas, summed
        # restrict to numeric training metrics
        changed_num = [c for c in changed if c in train_num.columns]
        if not changed_num:
            continue

        z = (flip[changed_num] - orig[changed_num] - mu[changed_num]) / sd[changed_num]
        totals.append(float(z.abs().sum()))
    return totals


# ---------- CLI ----------
if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--rq1",
        action="store_true",
        help="Flip rates per (model, explainer) using flip_jit.get_flip_rates",
    )
    ap.add_argument(
        "--rq2",
        action="store_true",
        help="Plan similarity (normalized Mahalanobis) per flip",
    )
    ap.add_argument(
        "--rq3",
        action="store_true",
        help="Feasibility distances vs historical delta pool",
    )
    ap.add_argument(
        "--implications",
        action="store_true",
        help="Sum of |z-scored delta| over changed features",
    )
    ap.add_argument("--explainer", type=str, default="all")
    ap.add_argument(
        "--distance",
        type=str,
        default="mahalanobis",
        choices=["mahalanobis", "cosine"],
    )

    args = ap.parse_args()

    # What explainers exist in your JIT pipeline?
    expl_map = {
        "LIME": "LIME",
        "LIME-HPO": "LIME-HPO",
        "PyExplainer": "PyExplainer",
        "NICE": "NICE",  # new
    }
    explainers = (
        list(expl_map.keys()) if args.explainer == "all" else args.explainer.split()
    )

    projects = read_dataset()
    proj_names = sorted(projects.keys())

    # RQ1: flip rates (requires flip_jit.get_flip_rates) – we ignore NICE here
    if args.rq1:
        if get_flip_rates is None:
            print("[WARN] flip_jit.get_flip_rates not found; skip RQ1.")
        else:
            table = []
            for model_type in ["RandomForest", "SVM", "LogisticRegression"]:
                for ex in explainers:
                    if ex.upper() == "NICE":
                        # flip_jit is for metamorphic flip pipeline only
                        continue
                    result = get_flip_rates(ex, None, model_type, verbose=False)
                    table.append(
                        [model_type, ex, result["Rate"] if result else 0.0]
                    )
                # per-model "All" row
                rows = [r for r in table if r[0] == model_type]
                if rows:
                    table.append(
                        [
                            model_type,
                            "All",
                            float(np.mean([r[2] for r in rows])),
                        ]
                    )
            print(tabulate(table, headers=["Model", "Explainer", "Flip Rate"]))
            pd.DataFrame(table, columns=["Model", "Explainer", "Flip Rate"]).to_csv(
                "./evaluations/flip_rates_jit.csv", index=False
            )

    # RQ2: plan similarity per flip (not defined for NICE)
    if args.rq2:
        Path("./evaluations/similarities").mkdir(parents=True, exist_ok=True)
        for model_type in ["RandomForest", "SVM", "LogisticRegression"]:
            sim_all = pd.DataFrame()
            for ex in explainers:
                if ex.upper() == "NICE":
                    # No plans_all.json → skip similarity for NICE
                    continue
                for p in proj_names:
                    sim = plan_similarity(p, model_type, ex)
                    if not sim:
                        continue
                    df = pd.DataFrame(sim).T
                    df["project"] = p
                    df["explainer"] = ex
                    df["model"] = model_type
                    sim_all = pd.concat([sim_all, df], axis=0, ignore_index=False)
            if not sim_all.empty:
                out = f"./evaluations/similarities/{model_type}.csv"
                sim_all.to_csv(out)

    # RQ3: feasibility distances (uses per-project delta pool)
    if args.rq3:
        Path(f"./evaluations/feasibility/{args.distance}").mkdir(
            parents=True, exist_ok=True
        )
        table = []
        totals = cannots = 0
        for model_type in ["RandomForest", "SVM", "LogisticRegression"]:
            for ex in explainers:
                results, total, cannot = flip_feasibility(
                    proj_names, ex, model_type, args.distance
                )
                totals += total
                cannots += cannot
                if not results:
                    continue
                df = pd.DataFrame(results)
                df.to_csv(
                    f"./evaluations/feasibility/{args.distance}/{model_type}_{ex}.csv",
                    index=False,
                )
                table.append(
                    [
                        model_type,
                        ex,
                        df["min"].mean(),
                        df["max"].mean(),
                        df["mean"].mean(),
                    ]
                )
            # per-model row
            rows = [r for r in table if r[0] == model_type]
            if rows:
                table.append(
                    [
                        model_type,
                        "Mean",
                        float(np.mean([r[2] for r in rows])),
                        float(np.mean([r[3] for r in rows])),
                        float(np.mean([r[4] for r in rows])),
                    ]
                )
        print(tabulate(table, headers=["Model", "Explainer", "Min", "Max", "Mean"]))
        print(
            f"Total flips: {totals}, Cannot: {cannots} ({(cannots/max(1, totals))*100:.2f}%)"
        )
        pd.DataFrame(
            table, columns=["Model", "Explainer", "Min", "Max", "Mean"]
        ).to_csv(f"./evaluations/feasibility_{args.distance}_jit.csv", index=False)

    # Implications: absolute z-scored change sums
    if args.implications:
        Path("./evaluations/abs_changes").mkdir(parents=True, exist_ok=True)
        table = []
        for model_type in ["RandomForest", "SVM", "LogisticRegression"]:
            for ex in explainers:
                all_vals = []
                for p in proj_names:
                    vals = implications(p, ex, model_type)
                    all_vals.extend(vals)
                if not all_vals:
                    continue
                series = pd.Series(all_vals, name="zsum")
                series.to_csv(
                    f"./evaluations/abs_changes/{model_type}_{ex}.csv", index=False
                )
                table.append([model_type, ex, float(series.mean())])
            rows = [r for r in table if r[0] == model_type]
            if rows:
                table.append(
                    [
                        model_type,
                        "Mean",
                        float(np.mean([r[2] for r in rows])),
                    ]
                )
        print(tabulate(table, headers=["Model", "Explainer", "Mean"]))
        pd.DataFrame(table, columns=["Model", "Explainer", "Mean"]).to_csv(
            "./evaluations/implications_jit.csv", index=False
        )
