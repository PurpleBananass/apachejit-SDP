#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnostic only (no methodology changes):
- Reproduces LIME top-K features and DiCE calls exactly like your pipeline.
- Logs why CFs may not be found (constant features, ranges, no single-feature flips, etc).
- Saves per-instance diagnostics to experiments_diag/{project}/{model_type}/{method}/dice_diag.csv
"""

import json
import warnings
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

# external libs used in your pipeline
import dice_ml
from dice_ml import Dice
from lime.lime_tabular import LimeTabularExplainer

# your helpers (unchanged methodology)
from data_utils import read_dataset, get_model, get_true_positives
from hyparams import SEED

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(SEED)


# ----------------------------- model wrapper (same idea you used) -----------------------------
class ScaledModel:
    """Wrap sklearn-like model so predict/predict_proba take UNscaled X."""
    def __init__(self, base_model, scaler: StandardScaler):
        self.model = base_model
        self.scaler = scaler
        if hasattr(base_model, "classes_"):
            self.classes_ = base_model.classes_

    def predict(self, X):
        Xs = self.scaler.transform(np.asarray(X))
        return self.model.predict(Xs)

    def predict_proba(self, X):
        Xs = self.scaler.transform(np.asarray(X))
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(Xs)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba
            if proba.ndim == 2 and proba.shape[1] == 1:
                p1 = proba[:, 0]
                return np.stack([1.0 - p1, p1], axis=1)
            if proba.ndim == 1:
                p1 = proba
                return np.stack([1.0 - p1, p1], axis=1)
        if hasattr(self.model, "decision_function"):
            s = self.model.decision_function(Xs)
            s = np.clip(s, -50, 50)
            p1 = 1.0 / (1.0 + np.exp(-s))
            if p1.ndim == 1:
                return np.stack([1.0 - p1, p1], axis=1)
            p1r = p1[:, 0]
            return np.stack([1.0 - p1r, p1r], axis=1)
        y = self.model.predict(Xs)
        p0 = (y == 0).astype(float)
        return np.stack([p0, 1.0 - p0], axis=1)

    def __getattr__(self, name):
        return getattr(self.model, name)


# ----------------------------- diagnostics -----------------------------

def one_dim_probe_flip(model: ScaledModel, x0: pd.Series, feat: str, fmin: float, fmax: float,
                       n=25, rtol=1e-7, atol=1e-7) -> dict:
    """
    Sweep a handful of values for ONE feature (keeping others fixed) to see
    if the model can flip class=1->0 by changing ONLY that feature.
    Purely diagnostic; does not alter methodology.
    """
    v0 = float(x0[feat])
    grid = np.linspace(fmin, fmax, n)
    # avoid duplicates near v0 to reduce pointless evaluations
    grid = [float(g) for g in grid if not np.isclose(g, v0, rtol=rtol, atol=atol)]
    if not grid:
        return {"single_feat_flip": False, "best_proba1": float("nan"), "best_val": float("nan")}

    x_mat = np.tile(x0.values.astype(float), (len(grid), 1))
    j = list(x0.index).index(feat)
    for i, val in enumerate(grid):
        x_mat[i, j] = val

    probs = model.predict_proba(x_mat)  # shape (m, 2)
    p1 = probs[:, 1]
    # flip means predicted class 0
    pred = (p1 >= 0.5).astype(int)
    flips = np.where(pred == 0)[0]
    if len(flips) > 0:
        k = flips[np.argmin(p1[flips])]  # pick the lowest p1 among flipping candidates
        return {"single_feat_flip": True, "best_proba1": float(p1[k]), "best_val": float(grid[k])}
    else:
        k = int(np.argmin(p1))
        return {"single_feat_flip": False, "best_proba1": float(p1[k]), "best_val": float(grid[k])}


def diagnose_project(project: str, model_type: str, method: str, total_cfs: int, topk: int,
                     limit_tp: int = 50, verbose: bool = True):
    ds = read_dataset()
    if project not in ds:
        print(f"[{project}] dataset not found. Skipping.")
        return

    train, test = ds[project]
    feat_cols = [c for c in test.columns if c != "target"]

    base_model = get_model(project, model_type)
    scaler = StandardScaler().fit(train[feat_cols].values)
    model = ScaledModel(base_model, scaler)

    # Basic env info
    env = {
        "dice_ml": getattr(dice_ml, "__version__", "unknown"),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "sklearn": __import__("sklearn").__version__,
        "method": method,
        "model_type": model_type,
    }

    # Class counts
    counts = {
        "train_pos": int(train["target"].sum()),
        "train_neg": int((1 - train["target"]).sum()),
        "test_pos": int(test["target"].sum()),
        "test_neg": int((1 - test["target"]).sum()),
    }
    all_df = pd.concat([train[feat_cols + ["target"]], test[feat_cols + ["target"]]],
                       ignore_index=True)
    counts["all_pos"] = int(all_df["target"].sum())
    counts["all_neg"] = int((1 - all_df["target"]).sum())

    # LIME explainer (same as your pipeline)
    X_train_scaled = scaler.transform(train[feat_cols].values)
    lime_explainer = LimeTabularExplainer(
        training_data=X_train_scaled,
        training_labels=train["target"].values,
        feature_names=feat_cols,
        feature_selection="lasso_path",
        discretizer="entropy",
        random_state=SEED,
    )

    # DiCE interfaces (exactly as your pipeline, no permitted_range tweaks)
    dice_data = dice_ml.Data(
        dataframe=all_df,
        continuous_features=feat_cols,
        outcome_name="target",
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    try:
        dice_explainer = Dice(dice_data, dice_model, method=method)
        dice_ok = True
        dice_init_err = ""
    except Exception as e:
        dice_ok = False
        dice_init_err = str(e)

    # True positives
    tp_df = get_true_positives(base_model, train, test)
    if tp_df.empty:
        print(f"[{project}] no true positives. Nothing to diagnose.")
        return

    if limit_tp and limit_tp > 0:
        tp_indices = list(tp_df.index.astype(int))[:limit_tp]
    else:
        tp_indices = list(tp_df.index.astype(int))

    rows = []
    out_dir = Path("experiments_diag") / project / model_type / method
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "dice_diag.csv"
    meta_path = out_dir / "env_counts.json"
    meta = {"env": env, "counts": counts, "n_tp_examined": len(tp_indices)}
    meta_path.write_text(json.dumps(meta, indent=2))

    # Iterate TPs
    for idx in tqdm(tp_indices, desc=f"{project}/{model_type}/{method}", leave=verbose):
        x0 = test.loc[idx, feat_cols].astype(float)
        x0_df = x0.to_frame().T

        # predicted proba on original
        p = model.predict_proba(x0_df.values)[0]
        p1 = float(p[1])

        # LIME top-k (same logic)
        x0_scaled = scaler.transform(x0_df.values)
        label_pred = int(model.predict(x0_df.values)[0])
        lime_exp = lime_explainer.explain_instance(
            x0_scaled[0],
            model.predict_proba,
            num_features=topk,
        )
        amap = lime_exp.as_map()
        label_key = label_pred if label_pred in amap else list(amap.keys())[0]
        pairs = amap[label_key][:topk]
        top_idx = [i for (i, _) in pairs]
        top_names = [feat_cols[i] for i in top_idx]

        # const flags and min/max
        const_map = {}
        fmin_map = {}
        fmax_map = {}
        for f in top_names:
            col = pd.concat([train[f], test[f]], axis=0).astype(float)
            const_map[f] = bool(col.min() == col.max())
            fmin_map[f] = float(col.min())
            fmax_map[f] = float(col.max())

        # 1-D probe flips (diagnostic only)
        probe_info = {}
        for f in top_names:
            probe_info[f] = one_dim_probe_flip(model, x0, f, fmin_map[f], fmax_map[f], n=25)

        # Try DiCE (exactly as your pipeline — no range tweaks)
        dice_success = False
        dice_num = 0
        dice_err = ""
        if dice_ok:
            try:
                try:
                    cf_obj = dice_explainer.generate_counterfactuals(
                        x0_df,
                        total_CFs=total_cfs,
                        desired_class="opposite",
                        features_to_vary=top_names,
                        random_seed=SEED,
                    )
                except TypeError:
                    cf_obj = dice_explainer.generate_counterfactuals(
                        x0_df,
                        total_CFs=total_cfs,
                        desired_class="opposite",
                        features_to_vary=top_names,
                    )
                try:
                    cf_df = cf_obj.cf_examples_list[0].final_cfs_df
                except Exception:
                    cf_df = None
                if cf_df is not None and not cf_df.empty:
                    dice_success = True
                    dice_num = int(len(cf_df))
            except Exception as e:
                dice_err = str(e)

        # flatten diagnostics into one row
        row = {
            "test_idx": int(idx),
            "p1_origin": p1,
            "lime_topk": ",".join(top_names),
            "any_const_topk": int(any(const_map.values())),
            "dice_init_ok": int(dice_ok),
            "dice_init_err": dice_init_err,
            "dice_success": int(dice_success),
            "dice_num": dice_num,
            "dice_error": dice_err,
        }
        # add per-feature small fields
        for f in top_names:
            row[f"feat_{f}_const"] = int(const_map[f])
            row[f"feat_{f}_min"] = fmin_map[f]
            row[f"feat_{f}_max"] = fmax_map[f]
            row[f"feat_{f}_x0"] = float(x0[f])
            row[f"probe_{f}_single_flip"] = int(probe_info[f]["single_feat_flip"])
            row[f"probe_{f}_best_p1"] = float(probe_info[f]["best_proba1"])
            row[f"probe_{f}_best_val"] = float(probe_info[f]["best_val"])

        rows.append(row)

    diag = pd.DataFrame(rows)
    diag.to_csv(summary_path, index=False)
    print(f"[DIAG] wrote {summary_path}")
    print("Quick summary:")
    print(diag[["dice_success"]].mean().rename({"dice_success": "success_rate"}))


def main():
    ap = ArgumentParser(description="Diagnose DiCE failures without changing methodology.")
    ap.add_argument("--project", type=str, default="all",
                    help="Project name or 'all'")
    ap.add_argument("--model_type", type=str, default="RandomForest",
                    help="Model type key used by get_model()")
    ap.add_argument("--method", type=str, default="random",
                    help="DiCE method (random|kdtree|genetic) — same as your runs")
    ap.add_argument("--total_cfs", type=int, default=100,
                    help="total_CFs (same as your runs)")
    ap.add_argument("--topk", type=int, default=5,
                    help="LIME top-K (same as your runs)")
    ap.add_argument("--limit_tp", type=int, default=50,
                    help="Max number of true positives to inspect per project")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    projects = read_dataset()
    if args.project == "all":
        project_list = list(sorted(projects.keys()))
    else:
        project_list = [p.strip() for p in args.project.replace(",", " ").split() if p.strip()]

    for p in tqdm(project_list, desc="Projects", leave=True, disable=not args.verbose):
        diagnose_project(
            project=p,
            model_type=args.model_type,
            method=args.method,
            total_cfs=args.total_cfs,
            topk=args.topk,
            limit_tp=args.limit_tp,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
