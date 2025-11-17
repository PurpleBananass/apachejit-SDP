#!/usr/bin/env python3
# dice_smoketest_v2.py
# -*- coding: utf-8 -*-

import argparse, warnings, time, concurrent.futures
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

def build_dice(tr, te, cols):
    import dice_ml
    from dice_ml import Dice
    all_data = pd.concat([tr[cols + ["target"]], te[cols + ["target"]]], axis=0, ignore_index=True)
    # 1–99% ranges keep search bounded (avoids outliers)
    pr = {}
    for f in cols:
        lo = float(all_data[f].quantile(0.01))
        hi = float(all_data[f].quantile(0.99))
        if lo == hi:
            lo, hi = lo - 1e-6, hi + 1e-6
        pr[f] = [lo, hi]
    data_iface = dice_ml.Data(
        dataframe=all_data,
        continuous_features=cols,
        outcome_name="target",
        permitted_range=pr,
    )
    return data_iface

def pick_nearest_to_boundary(te_df, pred_label):
    """Pick the row with desired predicted label whose proba1 is closest to 0.5."""
    sub = te_df[te_df["_pred"] == pred_label].copy()
    if sub.empty:
        return None
    sub["_dist"] = (sub["_proba1"] - 0.5).abs()
    return sub.sort_values("_dist", ascending=True).iloc[0]

def generate_with_timeout(explainer, x, total_cfs, seed, timeout_s=30):
    """Run DiCE generation with a hard timeout; return CF object or None."""
    def _run():
        try:
            return explainer.generate_counterfactuals(
                x,
                total_CFs=total_cfs,
                desired_class="opposite",
                features_to_vary="all",
                random_seed=seed,
            )
        except TypeError:
            # for older dice-ml without random_seed
            return explainer.generate_counterfactuals(
                x,
                total_CFs=total_cfs,
                desired_class="opposite",
                features_to_vary="all",
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_run)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return None

def try_one(explainer, clf, row, cols, label_str, total_cfs, seed, timeout_s):
    if row is None:
        print(f"[{label_str}] No candidate row available for that predicted class.")
        return
    x = row[cols].to_frame().T
    print(f"\n[{label_str}] proba1={float(row['_proba1']):.3f} pred={int(row['_pred'])} → desired='opposite'")
    t0 = time.time()
    cf = generate_with_timeout(explainer, x, total_cfs, seed, timeout_s=timeout_s)
    dt = time.time() - t0
    if cf is None:
        print(f"[{label_str}] Timed out after {dt:.1f}s with no result.")
        return
    try:
        cfs = cf.cf_examples_list[0].final_cfs_df
    except Exception:
        cfs = None
    if cfs is None or cfs.empty:
        print(f"[{label_str}] No CFs returned (elapsed {dt:.1f}s).")
        return
    # verify flips
    proba = clf.predict_proba(cfs[cols].values)[:, 1]
    pred = (proba >= 0.5).astype(int)
    flips = (pred != int(row["_pred"]))
    n_flips = int(flips.sum())
    print(f"[{label_str}] Returned {len(cfs)} CFs in {dt:.1f}s; verified flips: {n_flips}")
    if n_flips > 0:
        print(cfs.loc[flips, cols].head(3))

def main():
    ap = argparse.ArgumentParser(description="DiCE smoke test with boundary picking + timeout.")
    ap.add_argument("--method", type=str, default="random", choices=["random", "kdtree", "genetic"])
    ap.add_argument("--total_cfs", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=30, help="Per-call timeout (seconds).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Show versions
    import dice_ml, sklearn, numpy, scipy
    print(f"dice-ml: {getattr(dice_ml,'__version__','unknown')} | numpy: {numpy.__version__} | "
          f"scipy: {scipy.__version__} | sklearn: {sklearn.__version__}")

    # Easy synthetic classification
    X, y = make_classification(
        n_samples=600, n_features=6, n_informative=6, n_redundant=0,
        class_sep=1.0, flip_y=0.02, random_state=args.seed
    )
    cols = [f"f{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols); df["target"] = y
    tr, te = train_test_split(df, test_size=0.3, random_state=args.seed, stratify=df["target"])

    # Simple model
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=args.seed)
    clf.fit(tr[cols].values, tr["target"].values)
    te_proba = clf.predict_proba(te[cols].values)[:, 1]
    te_pred  = (te_proba >= 0.5).astype(int)
    print(f"Test accuracy: {((te_pred == te['target'].values).mean()):.3f}")

    # DiCE interfaces
    data_iface = build_dice(tr, te, cols)
    model_iface = dice_ml.Model(model=clf, backend="sklearn")
    try:
        from dice_ml import Dice
        explainer = Dice(data_iface, model_iface, method=args.method)
    except Exception as e:
        print(f"ERROR: constructing DiCE explainer failed: {e}")
        return

    # Pick *closest-to-0.5* rows for each predicted class to avoid “hopeless” points
    te_df = te.copy().reset_index(drop=True)
    te_df["_proba1"] = te_proba
    te_df["_pred"] = te_pred

    q1 = pick_nearest_to_boundary(te_df, pred_label=1)  # ask to flip to 0
    q0 = pick_nearest_to_boundary(te_df, pred_label=0)  # ask to flip to 1

    try_one(explainer, clf, q1, cols, "Query pred=1 → flip to 0", args.total_cfs, args.seed, args.timeout)
    try_one(explainer, clf, q0, cols, "Query pred=0 → flip to 1", args.total_cfs, args.seed, args.timeout)

if __name__ == "__main__":
    main()
