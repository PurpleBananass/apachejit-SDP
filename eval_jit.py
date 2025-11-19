# eval_jit.py (JIT version, fully rewritten)
from __future__ import annotations
import math
import json
from pathlib import Path
from argparse import ArgumentParser
from itertools import product

import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis

from hyparams import PROPOSED_CHANGES, EXPERIMENTS
from data_utils import read_dataset, get_model, get_true_positives
from flip_jit import get_flip_rates


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
# ---------- helpers: keep feature processing identical to training ----------
def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.select_dtypes(include=[np.number])
          .replace([np.inf, -np.inf], np.nan)
          .fillna(0.0)
    )

def _train_cols(train: pd.DataFrame) -> pd.Index:
    """Return the numeric feature columns used for training (exclude target)."""
    return _numeric_frame(train.drop(columns=["target"], errors="ignore")).columns

def _align_like_one(x_row: pd.Series, cols: pd.Index) -> pd.DataFrame:
    """
    Turn a single row (Series) into a 1×d DataFrame with the same numeric columns
    and preprocessing as training.
    """
    X = x_row.to_frame().T
    X = _numeric_frame(X)
    X = X.reindex(columns=list(cols), fill_value=0.0)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X

def generate_all_combinations(plan_dict):
    feats = list(plan_dict.keys())
    values = list(plan_dict.values())
    combos = list(product(*values))
    return pd.DataFrame(combos, columns=feats)

def normalized_mahalanobis_distance(df, x, y):
    """
    df : DataFrame of candidate combinations (plan grid)
    x  : pd.Series of flipped values (on *at least* the plan features)
    y  : pd.Series of baseline (first candidate values)

    We:
      1) drop constant columns from df
      2) restrict x, y to the remaining columns
      3) standardize all by df stats
      4) compute Mahalanobis(x, y) normalized by span(min,max)
    """
    # 1) keep only features that actually vary
    df = df.loc[:, df.nunique() > 1]
    if df.shape[1] == 0:
        return 0.0

    cols = df.columns

    # 2) restrict x, y to the same columns (this is what fixes your shape bug)
    x = x[cols].astype(float)
    y = y[cols].astype(float)

    # 3) standardize within df
    mu = df.mean()
    sd = df.std(ddof=0).replace(0, 1.0)
    Z = (df - mu) / sd

    x_std = ((x - mu) / sd).values
    y_std = ((y - mu) / sd).values

    # covariance on standardized grid
    cov = np.cov(Z.values, rowvar=False, ddof=0)
    cov = np.atleast_2d(cov)
    inv_cov = np.linalg.pinv(cov)

    # Mahalanobis distance between x and y
    diff_xy = x_std - y_std
    dist = float(np.sqrt(diff_xy @ inv_cov @ diff_xy))

    # 4) span normalization based on min/max in standardized space
    min_vec = Z.min().values
    max_vec = Z.max().values
    diff_span = max_vec - min_vec
    max_span = float(np.sqrt(diff_span @ inv_cov @ diff_span))

    if not np.isfinite(max_span) or max_span == 0.0:
        # can't normalize; return raw distance
        return dist

    return dist / max_span



# ----------------------------------------------------------------------
# RQ2 – Plan Similarity
# ----------------------------------------------------------------------

def plan_similarity(project: str, model_type: str, explainer: str, threshold: float = 0.5):
    """
    For each flipped TP having a plan, compute normalized Mahalanobis
    between (flipped values on changed features) vs (baseline = original),
    using the plan’s own grid as covariance reference.

    This fixes the 'all scores = 0' issue where baseline == flipped.
    CF/NICE are skipped because they have no plans_all.json in the JIT pipeline.
    """
    # CF/NICE: no plans → no RQ2
    if explainer.upper() in ("CF", "NICE"):
        return {}

    results = {}

    plan_path = Path(PROPOSED_CHANGES) / f"{project}/{model_type}/{explainer}/plans_all.json"
    flip_path = Path(EXPERIMENTS)       / f"{project}/{model_type}/{explainer}_all.csv"

    if (not plan_path.exists()) or (not flip_path.exists()):
        return {}

    # load plans
    with open(plan_path, "r") as f:
        plans = json.load(f)

    # load flipped instances for this (project, model, explainer)
    flipped_df = pd.read_csv(flip_path, index_col=0).dropna()
    if flipped_df.empty:
        return {}

    # dataset + model
    ds = read_dataset()
    train, test = ds[project]
    cols = _train_cols(train)
    model = get_model(project, model_type)

    for test_idx in flipped_df.index:
        key = str(test_idx)
        if key not in plans:
            continue

        original = test.loc[test_idx, test.columns != "target"]
        flipped  = flipped_df.loc[test_idx, :]

        # ---- optional: keep only real TP→non-defective flips ----
        # try:
        #     X_orig = _align_like_one(original, cols)
        #     X_flip = _align_like_one(flipped,   cols)
        #     if hasattr(model, "predict_proba"):
        #         p_orig = float(model.predict_proba(X_orig)[0, 1])
        #         p_flip = float(model.predict_proba(X_flip)[0, 1])
        #         if not (p_orig >= threshold and p_flip < threshold):
        #             print(f"Skipping idx {test_idx}: not TP→non-defective ({p_orig:.3f}→{p_flip:.3f})")
        #             continue
        # except Exception:
        #     # if predict_proba or alignment fails, just skip the TP filter
        #     pass

        # ---- which features actually changed according to the plan? ----
        changed = {
            f
            for f in plans[key].keys()
            if (
                f in flipped.index
                and f in original.index
                and not math.isclose(
                    float(flipped[f]),
                    float(original[f]),
                    rel_tol=1e-7,
                    abs_tol=1e-12,
                )
            )
        }
        if not changed:
            continue

        # only keep plan entries that correspond to changed features and have candidate values
        plan_subset = {f: list(plans[key][f]) for f in changed if len(plans[key][f]) > 0}
        if not plan_subset:
            continue

        # grid of all candidate combinations (same semantics as your old code)
        grid = generate_all_combinations(plan_subset)

        # distance between CF vs original on these features
        flipped_vec  = flipped[list(plan_subset.keys())].astype(float)
        baseline_vec = original[list(plan_subset.keys())].astype(float)

        score = normalized_mahalanobis_distance(grid, flipped_vec, baseline_vec)
        results[int(test_idx)] = {"score": float(score)}

    return results




# ----------------------------------------------------------------------
# RQ3 – Feasibility
# ----------------------------------------------------------------------

def mahalanobis_all(pool, changed_series):
    """
    pool: df of deltas
    changed_series: pd.Series of feature deltas
    """
    # use only >1-unique columns
    pool = pool.loc[:, pool.nunique() > 1]
    if pool.empty:
        return []

    # standardize
    mu = pool.mean()
    sd = pool.std(ddof=0).replace(0, 1.0)
    Z = (pool - mu) / sd

    x = ((changed_series - mu) / sd).values.astype(float)

    cov = np.cov(Z.values, rowvar=False)
    cov = np.atleast_2d(cov)
    inv_cov = np.linalg.pinv(cov)

    # normalization span
    min_vec = Z.min().values
    max_vec = Z.max().values
    span = float(mahalanobis(min_vec, max_vec, inv_cov))
    if not np.isfinite(span) or span == 0:
        span = 1.0

    out = []
    for _, row in Z.iterrows():
        d = float(mahalanobis(x, row.values, inv_cov)) / span
        out.append(d)

    return out


def cosine_similarity(a, b):
    aa = np.linalg.norm(a)
    bb = np.linalg.norm(b)
    if aa == 0 or bb == 0:
        return 0.0
    return float(np.dot(a, b) / (aa * bb))


def cosine_all(pool, changed_series):
    out = []
    for _, row in pool.iterrows():
        out.append(cosine_similarity(changed_series.values, row.values))
    return out



def flip_feasibility(project_list, explainer, model_type, distance="mahalanobis"):

    # Aggregate historical deltas
    total_deltas = pd.DataFrame()
    for project in project_list:
        train, test = read_dataset()[project]
        exist_idx = train.index.intersection(test.index)
        deltas = (
            test.loc[exist_idx, test.columns != "target"]
            - train.loc[exist_idx, train.columns != "target"]
        )
        total_deltas = pd.concat([total_deltas, deltas], axis=0)

    results = []
    totals = 0
    cannots = 0

    for project in project_list:

        plan_path = Path(PROPOSED_CHANGES) / f"{project}/{model_type}/{explainer}/plans_all.json"
        flip_path = Path(EXPERIMENTS) / f"{project}/{model_type}/{explainer}_all.csv"

        if not flip_path.exists():
            continue
        if not plan_path.exists():
            continue

        with open(plan_path, "r") as f:
            plans = json.load(f)

        df_flip = pd.read_csv(flip_path, index_col=0).dropna()
        totals += len(df_flip)

        train, test = read_dataset()[project]

        for idx in df_flip.index:
            if str(idx) not in plans:
                cannots += 1
                continue

            orig = test.loc[idx, test.columns != "target"]
            flip = df_flip.loc[idx]

            changed = {}
            for feat in plans[str(idx)]:
                if not math.isclose(flip[feat], orig[feat], rel_tol=1e-7, abs_tol=1e-12):
                    changed[feat] = flip[feat] - orig[feat]

            if not changed:
                cannots += 1
                continue

            changed_series = pd.Series(changed)

            pool = total_deltas[changed_series.index].dropna()
            pool = pool.loc[(pool != 0).any(axis=1)]  # relaxed (OR)

            if pool.empty:
                cannots += 1
                continue

            if distance == "cosine":
                d = cosine_all(pool, changed_series)
            else:
                if len(pool) <= len(changed_series):
                    cannots += 1
                    continue
                d = mahalanobis_all(pool, changed_series)

            if len(d) == 0:
                cannots += 1
                continue

            results.append({
                "min": float(np.min(d)),
                "max": float(np.max(d)),
                "mean": float(np.mean(d)),
            })

    return results, totals, cannots



# ----------------------------------------------------------------------
# Implications
# ----------------------------------------------------------------------

def implications(project, explainer, model_type):

    plan_path = Path(PROPOSED_CHANGES) / f"{project}/{model_type}/{explainer}/plans_all.json"
    flip_path = Path(EXPERIMENTS) / f"{project}/{model_type}/{explainer}_all.csv"

    if not plan_path.exists() or not flip_path.exists():
        return []

    with open(plan_path, "r") as f:
        plans = json.load(f)

    df_flip = pd.read_csv(flip_path, index_col=0).dropna()
    if df_flip.empty:
        return []

    train, test = read_dataset()[project]
    feat_cols = [c for c in train.columns if c != "target"]

    scaler = StandardScaler().fit(train[feat_cols])

    totals = []
    for idx in df_flip.index:
        if str(idx) not in plans:
            continue

        orig = test.loc[idx, feat_cols]
        flip = df_flip.loc[idx, feat_cols]

        changed = [f for f in plans[str(idx)]
                   if not math.isclose(flip[f], orig[f], rel_tol=1e-7, abs_tol=1e-12)]

        if not changed:
            continue

        s_orig = scaler.transform([orig])[0]
        s_flip = scaler.transform([flip])[0]

        diff = np.abs(s_flip - s_orig)
        total = float(sum(diff[orig.index.get_indexer(changed)]))
        totals.append(total)

    return totals



# ----------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--rq1", action="store_true")
    parser.add_argument("--rq2", action="store_true")
    parser.add_argument("--rq3", action="store_true")
    parser.add_argument("--implications", action="store_true")
    parser.add_argument("--explainer", type=str, default="all")
    parser.add_argument("--distance", type=str, default="mahalanobis")
    args = parser.parse_args()

    model_map = {
        "SVM": "SVM",
        "RandomForest": "RF",
        "XGBoost": "XGB",
        "CatBoost": "CB",
        "LightGBM": "LGBM"
    }

    explainer_map = {
        "LIME": "LIME",
        "LIME-HPO": "LIME-HPO",
        "TimeLIME": "TimeLIME",
        "SQAPlanner_confidence": "SQAPlanner",
    }

    if args.explainer == "all":
        explainers = list(explainer_map.keys())
    else:
        explainers = args.explainer.split()

    projects = read_dataset().keys()

    # --------------------------------------------------------------
    # RQ1 – Flip Rates
    # --------------------------------------------------------------
    if args.rq1:
        table = []
        for model in model_map:
            for ex in explainers:
                if ex.startswith("SQAPlanner"):
                    r = get_flip_rates("SQAPlanner", "confidence", model, verbose=False)
                    rate = r["Rate"]
                    table.append([model_map[model], "SQAPlanner", rate])
                else:
                    r = get_flip_rates(ex, None, model, verbose=False)
                    rate = r["Rate"]
                    table.append([model_map[model], ex, rate])

            # model mean
            rates = [row[2] for row in table if row[0] == model_map[model]]
            table.append([model_map[model], "All", float(np.mean(rates))])

        df = pd.DataFrame(table, columns=["Model", "Explainer", "FlipRate"])
        df.to_csv("./evaluations/flip_rates.csv", index=False)
        print(tabulate(df, headers="keys", tablefmt="psql"))



    # --------------------------------------------------------------
    # RQ2 – Similarities
    # --------------------------------------------------------------
    if args.rq2:
        Path("./evaluations/similarities").mkdir(parents=True, exist_ok=True)
        for model in model_map:
            all_sim = pd.DataFrame()
            for ex in explainers:
                for p in projects:
                    sim = plan_similarity(p, model, ex)
                    if not sim:
                        continue
                    df = pd.DataFrame(sim).T
                    df["project"] = p
                    df["model"] = model_map[model]
                    df["explainer"] = explainer_map[ex]
                    all_sim = pd.concat([all_sim, df])

            out = f"./evaluations/similarities/{model_map[model]}.csv"
            all_sim.to_csv(out)



    # --------------------------------------------------------------
    # RQ3 – Feasibility
    # --------------------------------------------------------------
    if args.rq3:

        Path(f"./evaluations/feasibility/{args.distance}").mkdir(parents=True, exist_ok=True)

        # JIT project groups
        project_groups = [
            ["activemq@0","activemq@1","activemq@2","activemq@3"],
            ["camel@0","camel@1","camel@2"],
            ["derby@0","derby@1"],
            ["groovy@0","groovy@1"],
            ["hbase@0","hbase@1"],
            ["hive@0","hive@1"],
            ["jruby@0","jruby@1","jruby@2"],
            ["lucene@0","lucene@1","lucene@2"],
            ["wicket@0","wicket@1"],
        ]

        totals_all = 0
        cannots_all = 0
        table = []

        for model in model_map:

            for ex in explainers:
                collected = []
                totals = 0
                cannots = 0

                for group in project_groups:
                    r, t, c = flip_feasibility(group, ex, model, args.distance)
                    totals += t
                    cannots += c
                    collected.extend(r)

                totals_all += totals
                cannots_all += cannots

                if collected:
                    df = pd.DataFrame(collected)
                    df.to_csv(
                        f"./evaluations/feasibility/{args.distance}/{model_map[model]}_{explainer_map[ex]}.csv",
                        index=False
                    )
                    table.append([
                        model_map[model],
                        explainer_map[ex],
                        df["min"].mean(),
                        df["max"].mean(),
                        df["mean"].mean(),
                    ])

            # model-level mean
            rows = [r for r in table if r[0] == model_map[model]]
            if rows:
                table.append([
                    model_map[model],
                    "Mean",
                    float(np.mean([r[2] for r in rows])),
                    float(np.mean([r[3] for r in rows])),
                    float(np.mean([r[4] for r in rows])),
                ])

        df = pd.DataFrame(table, columns=["Model","Explainer","Min","Max","Mean"])
        df.to_csv(f"./evaluations/feasibility_{args.distance}.csv", index=False)
        print(tabulate(df, headers="keys", tablefmt="psql"))
        print(f"Total flips: {totals_all}, Cannot: {cannots_all}, ({cannots_all/totals_all*100:.2f}%)")



    # --------------------------------------------------------------
    # Implications
    # --------------------------------------------------------------
    if args.implications:
        Path("./evaluations/abs_changes").mkdir(parents=True, exist_ok=True)
        table = []

        for model in model_map:
            for ex in explainers:
                collected = []

                for p in projects:
                    collected.extend(implications(p, ex, model))

                if not collected:
                    continue

                df = pd.DataFrame(collected)
                df.to_csv(
                    f"./evaluations/abs_changes/{model_map[model]}_{explainer_map[ex]}.csv",
                    index=False,
                )
                table.append([
                    model_map[model],
                    explainer_map[ex],
                    float(df.mean()),
                ])

            # model mean
            rows = [r for r in table if r[0] == model_map[model]]
            table.append([
                model_map[model],
                "Mean",
                float(np.mean([r[2] for r in rows])),
            ])

        df = pd.DataFrame(table, columns=["Model","Explainer","Mean"])
        df.to_csv("./evaluations/implications.csv", index=False)
        print(tabulate(df, headers="keys", tablefmt="psql"))
