#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate DiCE counterfactuals (k per true-positive instance) using different methods,
restricting changes to the per-instance top-K LIME features (default K=5).
Verifies flips and saves successful candidates in long format.

Output CSV (per project/model/method):
  experiments/{project}/{model_type}/{method}/DiCE_all.csv

Columns:
  - test_idx: original test row index
  - candidate_id: 0..(k-1) per test_idx (after filtering)
  - <all feature columns> (no 'target')
  - proba0, proba1: model probabilities for class 0 and 1
"""

import warnings
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.exceptions import ConvergenceWarning

import dice_ml
from dice_ml import Dice
from lime.lime_tabular import LimeTabularExplainer

# your helpers
from data_utils import read_dataset, get_model, get_true_positives
from hyparams import EXPERIMENTS, SEED

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(SEED)


def _numeric_feature_columns(train: pd.DataFrame, test: pd.DataFrame, label="target"):
    """Keep exactly the numeric columns used by your training code."""
    cols = (
        train.drop(columns=[label], errors="ignore")
             .select_dtypes(include=[np.number])
             .columns
    )
    # ensure they exist in test as well
    cols = [c for c in cols if c in test.columns]
    return cols


def generate_dice_flips_for_project(project: str,
                                    model_type: str,
                                    method: str = "random",
                                    total_cfs: int = 50,
                                    topk: int = 5,
                                    verbose: bool = True,
                                    overwrite: bool = True):
    """
    For the given project/model/method:
      - find true positives on test
      - get LIME top-K features for each TP (raw features, no extra scaling)
      - generate DiCE CFs restricted to those K features (features_to_vary)
      - keep only candidates that actually flip to class 0 and change ≤ topk features
      - save to experiments/{project}/{model_type}/{method}/DiCE_all.csv
    """
    valid_methods = ["random", "kdtree", "genetic"]
    if method not in valid_methods:
        tqdm.write(f"[ERROR] Invalid method '{method}'. Must be one of: {valid_methods}")
        return

    ds = read_dataset()
    if project not in ds:
        tqdm.write(f"[{project}/{model_type}/{method}] dataset not found. Skipping.")
        return

    train, test = ds[project]
    feat_cols = _numeric_feature_columns(train, test, label="target")

    # --- load the raw model (no wrapper, no extra scaler) ---
    model = get_model(project, model_type)

    # quick sanity: are there any predicted positives at 0.5?
    X_test = test[feat_cols].astype(float).values
    if hasattr(model, "predict_proba"):
        pcol = list(model.classes_).index(1) if hasattr(model, "classes_") else 1
        proba1 = model.predict_proba(X_test)[:, pcol]
        y_pred = (proba1 >= 0.5).astype(int)
    else:
        y_pred = np.asarray(model.predict(X_test)).astype(int)
        proba1 = y_pred.astype(float)  # degenerate

    print(f"[DEBUG] test positives: {int(test['target'].sum())}, "
          f"predicted positives: {int((y_pred==1).sum())}")

    if (y_pred == 1).sum() == 0:
        print(f"[CHECK] predicted positives @0.5 = 0 / {len(y_pred)} (frac=0.000)")
        tqdm.write(f"[{project}/{model_type}/{method}] no true positives. Skipping.")
        return

    # true positives using your helper (works with raw model)
    tp_df = get_true_positives(model, train, test)
    if tp_df.empty:
        tqdm.write(f"[{project}/{model_type}/{method}] no true positives. Skipping.")
        return

    # --- LIME on RAW features (same as training space) ---
    lime_explainer = LimeTabularExplainer(
        training_data=train[feat_cols].astype(float).values,
        training_labels=train["target"].values,
        feature_names=feat_cols,
        feature_selection="lasso_path",
        discretizer="entropy",
        random_state=SEED,
    )

    # --- DiCE on RAW features ---
    all_data = pd.concat(
        [train[feat_cols + ["target"]], test[feat_cols + ["target"]]],
        axis=0, ignore_index=True
    )
    dice_data = dice_ml.Data(
        dataframe=all_data,
        continuous_features=feat_cols,
        outcome_name="target",
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    try:
        dice_explainer = Dice(dice_data, dice_model, method=method)
    except Exception as e:
        tqdm.write(f"[ERROR] Failed to create DiCE explainer with method '{method}': {e}")
        return

    out_dir = Path(EXPERIMENTS) / project / model_type / method
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "DiCE_all.csv"
    if overwrite and out_csv.exists():
        out_csv.unlink(missing_ok=True)

    results = []

    for idx in tqdm(tp_df.index.astype(int),
                    desc=f"{project}/{model_type}/{method}",
                    leave=False,
                    disable=not verbose):
        x0_df = test.loc[[idx], feat_cols].astype(float)
        x0 = x0_df.values[0]

        # LIME top-K for this instance (consistent with model's predicted class)
        try:
            label_pred = int(model.predict(x0_df.values)[0])
        except Exception:
            label_pred = 1  # fallback

        lime_exp = lime_explainer.explain_instance(
            x0,
            model.predict_proba,
            num_features=topk,
        )
        amap = lime_exp.as_map()
        label_key = label_pred if label_pred in amap else list(amap.keys())[0]
        top_pairs = amap[label_key][:topk]        # list[(feat_idx, weight)]
        top_idx = [i for (i, _) in top_pairs]
        top_names = [feat_cols[i] for i in top_idx]

        # Generate CFs restricted to these features
        try:
            try:
                cf = dice_explainer.generate_counterfactuals(
                    x0_df,
                    total_CFs=total_cfs,
                    desired_class="opposite",
                    features_to_vary=top_names,
                    random_seed=SEED,          # older Dice may not accept this
                )
            except TypeError:
                cf = dice_explainer.generate_counterfactuals(
                    x0_df,
                    total_CFs=total_cfs,
                    desired_class="opposite",
                    features_to_vary=top_names,
                )
        except Exception as e:
            tqdm.write(f"[{project}/{model_type}/{method}] DiCE error @ {idx}: {e}")
            continue

        # Extract candidate DF
        try:
            cf_df = cf.cf_examples_list[0].final_cfs_df
        except Exception:
            cf_df = None
        if cf_df is None or cf_df.empty:
            continue

        # ensure same columns/order, backfill untouched features with original
        if "target" in cf_df.columns:
            cf_df = cf_df.drop(columns=["target"])
        for c in feat_cols:
            if c not in cf_df.columns:
                cf_df[c] = x0_df.iloc[0][c]
        cf_df = cf_df[feat_cols].astype(float).drop_duplicates()

        # changed-features filter (≤ topk, ≥ 1), ignoring tiny float noise
        orig = x0[None, :]
        cand = cf_df.values
        changed_counts = (~np.isclose(cand, orig, rtol=1e-7, atol=1e-7)).sum(axis=1)
        allowed = (changed_counts > 0) & (changed_counts <= topk)
        if not np.any(allowed):
            continue
        cf_df_allowed = cf_df.loc[allowed].copy()

        # verify flips with the model (class '1' probability)
        pcol = list(model.classes_).index(1) if hasattr(model, "classes_") else 1
        probs = model.predict_proba(cf_df_allowed.values)
        preds = (probs[:, pcol] >= 0.5).astype(int)
        flips = (preds == 0)  # want 1 -> 0
        if not np.any(flips):
            continue

        kept = cf_df_allowed.loc[flips].copy()
        kept["proba0"] = probs[flips, 1 - pcol]
        kept["proba1"] = probs[flips, pcol]
        kept.insert(0, "candidate_id", np.arange(len(kept)))
        kept.insert(0, "test_idx", idx)
        results.append(kept)

    # write
    if results:
        out_df = pd.concat(results, axis=0, ignore_index=False)
        out_df.to_csv(out_csv, index=False)
        flipped = out_df["test_idx"].nunique()
        computed = len(out_df)
        tqdm.write(f"[OK] {project}/{model_type}/{method}: wrote {computed} flipped candidates "
                   f"(restricted to top-{topk} LIME features) for {flipped} TP(s) -> {out_csv}")
    else:
        tqdm.write(f"[{project}/{model_type}/{method}] no flipped candidates found under top-{topk} LIME feature restriction.")


def main():
    ap = ArgumentParser(description="Generate DiCE counterfactuals (restricted to per-instance top-K LIME features)")
    ap.add_argument("--project", type=str, default="all",
                    help="Project name or 'all'")
    ap.add_argument("--model_types", type=str, default="RandomForest,SVM,LogisticRegression",
                    help="Comma-separated: RandomForest,SVM,LogisticRegression")
    ap.add_argument("--methods", type=str, default="random",
                    help="Comma-separated DiCE methods: random,kdtree,genetic")
    ap.add_argument("--total_cfs", type=int, default=50,
                    help="How many CFs to request from DiCE per instance")
    ap.add_argument("--topk", type=int, default=5,
                    help="Number of LIME features to allow DiCE to vary (per instance)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing experiment files")
    ap.add_argument("--verbose", action="store_true",
                    help="Enable verbose output")
    args = ap.parse_args()

    projects = read_dataset()
    if args.project == "all":
        project_list = list(sorted(projects.keys()))
    else:
        project_list = [p.strip() for p in args.project.replace(",", " ").split() if p.strip()]

    model_types = [m.strip() for m in args.model_types.split(",") if m.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    valid_methods = ["random", "kdtree", "genetic"]
    invalid = [m for m in methods if m not in valid_methods]
    if invalid:
        print(f"ERROR: Invalid methods: {invalid}")
        print(f"Valid methods are: {valid_methods}")
        return

    combos = [(p, m, method) for p in project_list for m in model_types for method in methods]

    print(f"Running {len(combos)} combinations:")
    print(f"  Projects: {project_list}")
    print(f"  Models: {model_types}")
    print(f"  Methods: {methods}")
    print(f"  LIME top-K: {args.topk}")
    print()

    for p, m, method in tqdm(combos, desc="Projects/Models/Methods", leave=True, disable=not args.verbose):
        generate_dice_flips_for_project(
            project=p,
            model_type=m,
            method=method,
            total_cfs=args.total_cfs,
            topk=args.topk,
            verbose=args.verbose,
            overwrite=args.overwrite,
        )

    print("Done!")


if __name__ == "__main__":
    main()
