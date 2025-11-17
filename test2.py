# # check_tp_alignment.py
# from __future__ import annotations
# import argparse
# import numpy as np
# import pandas as pd
# from data_utils import read_dataset, get_model, get_true_positives

# def _xy(df: pd.DataFrame):
#     X = (df.drop(columns=["target"])
#            .select_dtypes(include=[np.number])
#            .replace([np.inf, -np.inf], np.nan)
#            .fillna(0.0))
#     y = df["target"].astype(int).values
#     return X, y

# def proba1(model, X):
#     if hasattr(model, "predict_proba"):
#         proba = model.predict_proba(X)
#         classes = list(getattr(model, "classes_", [0,1]))
#         idx = classes.index(1) if 1 in classes else -1
#         return proba[:, idx]
#     if hasattr(model, "decision_function"):
#         s = model.decision_function(X)
#         s = np.clip(np.asarray(s).ravel(), -50, 50)
#         return 1/(1+np.exp(-s))
#     return (model.predict(X) == 1).astype(float)

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--project", required=True)
#     ap.add_argument("--model", required=True, choices=["RandomForest","SVM","LogisticRegression"])
#     ap.add_argument("--thr", type=float, default=0.5)
#     args = ap.parse_args()

#     ds = read_dataset()
#     train, test = ds[args.project]
#     X_test, y_test = _xy(test)
#     model = get_model(args.project, args.model)

#     # what your pipeline uses
#     tp_df = get_true_positives(model, train, test)  # rows with y=1 AND pred=1 (at 0.5)
#     tp_idx_pipeline = set(map(int, tp_df.index))

#     # independent recomputation on raw test
#     p1 = proba1(model, X_test.values)
#     y_pred = (p1 >= args.thr).astype(int)
#     tp_idx_manual = set(map(int, test.index[(y_test == 1) & (y_pred == 1)]))

#     print(f"[{args.project} :: {args.model}] threshold={args.thr}")
#     print(f" Pipeline TPs: {len(tp_idx_pipeline)}")
#     print(f" Manual   TPs: {len(tp_idx_manual)}")
#     only_pipeline = sorted(tp_idx_pipeline - tp_idx_manual)[:10]
#     only_manual   = sorted(tp_idx_manual - tp_idx_pipeline)[:10]
#     if only_pipeline:
#         print(" In pipeline only (first 10):", only_pipeline)
#     if only_manual:
#         print(" In manual only   (first 10):", only_manual)
#     if not only_pipeline and not only_manual:
#         print(" ✔ Indices match.")

# if __name__ == "__main__":
#     main()

# # show projects where a model predicts <10% positives at thr=0.5
# import pandas as pd
# df = pd.read_csv("MODEL_EVALUATION/diagnostics.csv")
# print(df[(df['frac_p1_ge_thr'] < 0.10)][["project","model","AUC_ROC","frac_p1_ge_thr"]])

# show_frac_p1.py
# import argparse
# import pandas as pd

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv", default="MODEL_EVALUATION/diagnostics.csv")
#     ap.add_argument("--sort", choices=["value","project","model"], default="project",
#                     help="How to sort the output")
#     args = ap.parse_args()

#     df = pd.read_csv(args.csv)
#     keep = df[["project","model","frac_p1_ge_thr"]].copy()

#     if args.sort == "value":
#         keep = keep.sort_values("frac_p1_ge_thr", ascending=False)
#     elif args.sort == "model":
#         keep = keep.sort_values(["model","project"])
#     else:
#         keep = keep.sort_values(["project","model"])

#     # print full list
#     print(keep.to_string(index=False))

#     # quick summaries
#     print("\n-- per-model summary (mean/median) --")
#     print(keep.groupby("model")["frac_p1_ge_thr"].agg(["mean","median"]).to_string())

#     print("\n-- lowest 10 by value --")
#     print(keep.nsmallest(10, "frac_p1_ge_thr").to_string(index=False))

# if __name__ == "__main__":
#     main()

# debug_model_probs.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

from data_utils import read_dataset, get_model

def best_f1_threshold(y_true: np.ndarray, p1: np.ndarray):
    thr_grid = np.linspace(0, 1, 101)
    f1s = []
    for t in thr_grid:
        pred = (p1 >= t).astype(int)
        f1s.append(f1_score(y_true, pred))
    i = int(np.argmax(f1s))
    return float(thr_grid[i]), float(f1s[i])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    ds = read_dataset()
    if args.project not in ds:
        raise SystemExit(f"Project not found in read_dataset(): {args.project}")

    train_df, test_df = ds[args.project]
    feat_cols = [c for c in test_df.columns if c != "target"]

    # load model and print file info
    # (get_model should point to your hyparams.MODELS/<project>/<model>.pkl)
    model = get_model(args.project, args.model)
    print(f"[MODEL] type={type(model)}")
    # try to locate the file path by mirroring get_model’s convention
    try_paths = [
        Path("models")/args.project/f"{args.model}.pkl",
        Path("MODELS")/args.project/f"{args.model}.pkl",
        Path("jit_models")/args.project/f"{args.model}.pkl",
        Path("jit_models/apache")/args.project/f"{args.model}.pkl",
    ]
    for p in try_paths:
        if p.exists():
            st = p.stat()
            print(f"[MODEL FILE] {p} (mtime: {time.ctime(st.st_mtime)})")
            break

    # basic sanity on classes_
    if hasattr(model, "classes_"):
        print(f"[CLASSES_] {model.classes_}")

    X = test_df[feat_cols].copy()
    y_true = test_df["target"].astype(int).values

    # shapes and dtypes
    print(f"[SHAPE] X: {X.shape}, y: {y_true.shape}, dtypes=all numeric? {all(np.issubdtype(dt, np.number) for dt in X.dtypes)}")
    print(f"[FEATURES] first 10: {feat_cols[:10]}")

    # probabilities
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X.values)
        # pick the probability for the 'True' class — assume classes_=[False, True]
        if hasattr(model, "classes_"):
            true_col = int(np.where(model.classes_ == True)[0][0])
        else:
            true_col = 1
        p1 = p[:, true_col]
    else:
        # fallback: decision_function or hard preds -> sigmoid
        if hasattr(model, "decision_function"):
            s = model.decision_function(X.values)
            s = np.clip(s, -50, 50)
            p1 = 1/(1+np.exp(-s))
        else:
            yhat = model.predict(X.values)
            p1 = (yhat == 1).astype(float)

    # summary
    q = np.quantile(p1, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    frac_pos_05 = float((p1 >= 0.5).mean())
    auc = roc_auc_score(y_true, p1) if len(np.unique(y_true)) == 2 else float("nan")
    print(f"[AUC] {auc:.6f}")
    print(f"[FRAC p1>=0.5] {frac_pos_05:.6f}")
    print(f"[QUANTILES] q0={q[0]:.6f} q25={q[1]:.6f} q50={q[2]:.6f} q75={q[3]:.6f} q90={q[4]:.6f} q95={q[5]:.6f} q99={q[6]:.6f}")

    # confusion @0.5
    y_pred_05 = (p1 >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_05, labels=[0,1]).ravel()
    print(f"[CONF@0.5] TN={tn} FP={fp} FN={fn} TP={tp}")

    # best-F1 threshold (for reference)
    thr_best, f1_best = best_f1_threshold(y_true, p1)
    print(f"[BEST F1] {f1_best:.6f} @ thr={thr_best:.6f}")
    print(f"[FRAC p1>=thr_best] {(p1 >= thr_best).mean():.6f}")

    # show a few rows with highest p1
    top_idx = np.argsort(-p1)[:10]
    print("\n[TOP-10 by p1] (idx, y_true, p1)")
    for i in top_idx:
        print(f"  {X.index[i]}, {y_true[i]}, {p1[i]:.6f}")

if __name__ == "__main__":
    main()
