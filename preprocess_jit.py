# preprocess_jit.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List
import argparse
import numpy as np
import pandas as pd

# Use AutoSpearman from the installed pyexplainer package (no fallbacks)
from pyexplainer.pyexplainer_pyexplainer import AutoSpearman as PX_AutoSpearman

# Core JIT metrics present in your CSV
JIT_FEATURES: List[str] = [
    "la", "ld", "nf", "nd", "ns", "ent",
    "ndev", "age", "nuc", "aexp", "arexp", "asexp",
]

def _safe_project_dir(name: str) -> str:
    """Filesystem-safe subdir for 'apache/groovy' -> 'apache_groovy'."""
    return name.replace("/", "_").replace("\\", "_")

def _temporal_split(pdf: pd.DataFrame, time_col: str = "author_date", split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by time_col and split (train first part, test last part)."""
    pdf = pdf.sort_values(time_col)
    n = len(pdf)
    k = max(1, int(n * split_ratio))
    train = pdf.iloc[:k].copy()
    test = pdf.iloc[k:].copy()
    return train, test

def map_indexes_to_int(train_df: pd.DataFrame, test_df: pd.DataFrame, *, index_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]:
    """
    Map commit_id -> contiguous int indices across train and test.
    Returns (train_df_with_int_index, test_df_with_int_index, mapping: int->commit_id)
    """
    assert index_col in train_df.columns and index_col in test_df.columns, "missing commit_id"
    tr = train_df.set_index(index_col, drop=True)
    te = test_df.set_index(index_col, drop=True)

    all_keys = tr.index.append(te.index).unique()
    key_to_int = {k: i for i, k in enumerate(all_keys)}
    int_to_key = {i: k for k, i in key_to_int.items()}

    tr.index = tr.index.map(key_to_int)
    te.index = te.index.map(key_to_int)
    return tr, te, int_to_key

def get_df_jit(csv_path: str | Path) -> pd.DataFrame:
    """
    Load a JIT CSV with at least:
      commit_id, project, buggy, author_date, [metrics...]
    """
    df = pd.read_csv(csv_path)
    required = {"commit_id", "project", "buggy", "author_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"JIT CSV missing columns: {missing}")

    # normalize label to 0/1 int
    df["buggy"] = (
        df["buggy"].astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
    ).astype(int)

    return df

def preprocess_one(
    project: str,
    *,
    df: pd.DataFrame,
    split_ratio: float,
    corr_th: float,
    vif_th: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str], List[str]]:
    """
    Build train/test for a single JIT project using AutoSpearman only.
    Returns (train_df, test_df, mapping, selected_features)
    """
    pdf = df[df["project"] == project].copy()
    if pdf.empty:
        raise ValueError(f"project not found in CSV: {project}")

    # dedupe on commit_id keeping earliest by author_date
    pdf = pdf.sort_values("author_date").drop_duplicates(subset=["commit_id"], keep="first")

    # temporal split
    tr_raw, te_raw = _temporal_split(pdf, time_col="author_date", split_ratio=split_ratio)

    # numeric feature matrix
    feat_cols = [c for c in JIT_FEATURES if c in pdf.columns]
    if len(feat_cols) < 3:
        raise ValueError(f"Too few JIT metrics present; found {feat_cols}, expected some of {JIT_FEATURES}")

    trX = tr_raw[feat_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    teX = te_raw[feat_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # feature selection via PyExplainer AutoSpearman ONLY
    selected_df = PX_AutoSpearman(
        trX,
        correlation_threshold=corr_th,
        correlation_method="spearman",
        VIF_threshold=vif_th,
    )
    selected = selected_df.columns.tolist()
    if len(selected) == 0:
        raise RuntimeError("AutoSpearman selected zero features")

    # final frames
    tr = trX.loc[:, selected].copy()
    te = teX.loc[:, selected].copy()
    tr["target"] = tr_raw["buggy"].astype(bool).values
    te["target"] = te_raw["buggy"].astype(bool).values

    # commit mapping
    tr["commit_id"] = tr_raw["commit_id"].values
    te["commit_id"] = te_raw["commit_id"].values
    tr, te, mapping = map_indexes_to_int(tr, te, index_col="commit_id")

    # ensure column order: selected + target
    tr = tr.loc[:, selected + ["target"]]
    te = te.loc[:, selected + ["target"]]
    return tr, te, mapping, selected

def main():
    ap = argparse.ArgumentParser(description="Preprocess JIT dataset with AutoSpearman (PyExplainer).")
    ap.add_argument("--csv", type=str, default="./Dataset/apachejit_total.csv", help="Path to total.csv (JIT dataset).")
    ap.add_argument("--project", type=str, default="all",
                    help="Project name (e.g., 'apache/groovy') or 'all' to process every project in the CSV.")
    ap.add_argument("--split", type=float, default=0.8, help="Temporal split ratio for train (default: 0.8).")
    ap.add_argument("--corr-th", type=float, default=0.7, help="Spearman correlation threshold (default: 0.7).")
    ap.add_argument("--vif-th", type=float, default=10.0, help="VIF threshold (default: 10.0).")
    ap.add_argument("--outdir", type=str, default="./Dataset/jit_preprocessed", help="Output root directory.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = get_df_jit(csv_path)

    if args.project.lower() == "all":
        projects = sorted(df["project"].dropna().unique().tolist())
        print(f"[INFO] Found {len(projects)} projects.")
        ok, fail = 0, 0
        for pj in projects:
            subdir = out_root / _safe_project_dir(pj)
            subdir.mkdir(parents=True, exist_ok=True)
            try:
                tr, te, mapping, selected = preprocess_one(
                    pj, df=df, split_ratio=args.split, corr_th=args.corr_th, vif_th=args.vif_th
                )
                tr.to_csv(subdir / "train.csv", index=True)
                te.to_csv(subdir / "test.csv", index=True)
                mapping_df = pd.DataFrame.from_dict(mapping, orient="index", columns=["commit_id"])
                mapping_df.to_csv(subdir / "mapping.csv", index=True, header=True)
                print(f"[OK] {pj}: {len(selected)} features | train={len(tr)} (bugs={int(tr['target'].sum())}) "
                      f"| test={len(te)} (bugs={int(te['target'].sum())}) -> {subdir}")
                ok += 1
            except Exception as e:
                print(f"[FAIL] {pj}: {e}")
                fail += 1
        print(f"[DONE] success={ok}, failed={fail}, outdir={out_root}")
    else:
        pj = args.project
        subdir = out_root / _safe_project_dir(pj)
        subdir.mkdir(parents=True, exist_ok=True)

        tr, te, mapping, selected = preprocess_one(
            pj, df=df, split_ratio=args.split, corr_th=args.corr_th, vif_th=args.vif_th
        )
        tr.to_csv(subdir / "train.csv", index=True)
        te.to_csv(subdir / "test.csv", index=True)
        mapping_df = pd.DataFrame.from_dict(mapping, orient="index", columns=["commit_id"])
        mapping_df.to_csv(subdir / "mapping.csv", index=True, header=True)

        print(f"[OK] {pj}: {len(selected)} features | train={len(tr)} (bugs={int(tr['target'].sum())}) "
              f"| test={len(te)} (bugs={int(te['target'].sum())}) -> {subdir}")

if __name__ == "__main__":
    main()
