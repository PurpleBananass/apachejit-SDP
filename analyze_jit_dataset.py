#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze JIT-style CSV dataset to debug projects with abnormal results.

Expected columns (at least):
    commit_id, project, buggy, fix, year, author_date,
    la, ld, nf, nd, ns, ent, ndev, age, nuc, aexp, arexp, asexp
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# ----------------------------- config -----------------------------

META_COLS = [
    "commit_id",
    "project",
    "fix",
    "year",
    "author_date",
]
LABEL_COL = "buggy"


# ----------------------------- helpers -----------------------------

def _infer_feature_cols(df: pd.DataFrame) -> List[str]:
    """All numeric columns that are not metadata or label."""
    excluded = set(META_COLS + [LABEL_COL])
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in excluded]


def _safe_buggy_to_int(series: pd.Series) -> pd.Series:
    """Convert buggy to {0,1} robustly (bool, string, etc.)."""
    s = series.copy()
    if s.dtype == bool:
        return s.astype(int)
    # handle common string forms
    if s.dtype == object:
        lowered = s.astype(str).str.lower()
        mapping = {
            "true": 1, "t": 1, "1": 1, "yes": 1, "y": 1,
            "false": 0, "f": 0, "0": 0, "no": 0, "n": 0,
        }
        return lowered.map(mapping).astype("Int64").fillna(0).astype(int)
    # numeric already
    return s.astype(int)


def describe_global(df: pd.DataFrame, feat_cols: List[str]) -> None:
    print("\n=== Global dataset info ===")
    print(f"# rows: {len(df)}")
    print(f"# projects: {df['project'].nunique()}")
    print("Dtypes:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isna().sum().sort_values(ascending=False))
    print("\nQuick feature summary (global):")
    print(df[feat_cols].describe(percentiles=[0.25, 0.5, 0.75]).T)


def summarize_projects(df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    """
    Per-project summary:
        n_rows, n_buggy, bug_rate,
        min_year, max_year,
        #constant_features, #low_var_features
    """
    rows = []
    for proj, g in df.groupby("project"):
        n = len(g)
        bug = int(g[LABEL_COL].sum())
        bug_rate = bug / n if n > 0 else np.nan

        years = g["year"].dropna().astype(int) if "year" in g.columns else pd.Series([], dtype=int)
        min_year = int(years.min()) if not years.empty else np.nan
        max_year = int(years.max()) if not years.empty else np.nan

        # constant / low-variance features (within project)
        const_count = 0
        low_var_count = 0
        for c in feat_cols:
            vals = pd.to_numeric(g[c], errors="coerce").dropna()
            if vals.empty:
                continue
            if vals.nunique() == 1:
                const_count += 1
            elif vals.std() == 0 or vals.std() < 1e-6:
                low_var_count += 1

        rows.append({
            "project": proj,
            "n_rows": n,
            "n_buggy": bug,
            "bug_rate": bug_rate,
            "min_year": min_year,
            "max_year": max_year,
            "n_constant_feats": const_count,
            "n_lowvar_feats": low_var_count,
        })

    proj_df = pd.DataFrame(rows).sort_values("bug_rate")
    print("\n=== Per-project label / feature summary ===")
    print(proj_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    return proj_df


def correlation_per_project(df: pd.DataFrame, feat_cols: List[str], min_rows: int = 50) -> pd.DataFrame:
    """
    Compute Pearson correlation between each feature and buggy per project.

    Returns long DataFrame: (project, feature, corr, p_value, n)
    """
    records = []
    for proj, g in df.groupby("project"):
        if len(g) < min_rows:
            continue
        y = g[LABEL_COL].astype(float)
        if y.nunique() < 2:
            # cannot compute correlation if label is constant
            continue

        for c in feat_cols:
            x = pd.to_numeric(g[c], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 10:
                continue
            try:
                r, p = pearsonr(x[mask], y[mask])
            except Exception:
                continue
            records.append({
                "project": proj,
                "feature": c,
                "corr_buggy": r,
                "p_value": p,
                "n": int(mask.sum()),
            })

    corr_df = pd.DataFrame(records)
    if corr_df.empty:
        print("\n[WARN] No valid feature–label correlations could be computed.")
        return corr_df

    print("\n=== Strongest correlations per project (top 5 by |r|) ===")
    for proj, g in corr_df.groupby("project"):
        top = g.reindex(g["corr_buggy"].abs().sort_values(ascending=False).index).head(5)
        print(f"\nProject: {proj}")
        print(top[["feature", "corr_buggy", "p_value", "n"]].to_string(index=False,
              float_format=lambda x: f"{x:.4f}"))

    return corr_df


def detect_outliers_per_project(df: pd.DataFrame, feat_cols: List[str], iqr_factor: float = 3.0) -> pd.DataFrame:
    """
    For each project & feature, compute IQR-based outlier counts.
    Marks features where a large fraction of points are outliers (which
    can indicate mis-scaling or data issues).
    """
    rows = []
    for proj, g in df.groupby("project"):
        for c in feat_cols:
            vals = pd.to_numeric(g[c], errors="coerce").dropna()
            if len(vals) < 10:
                continue
            q1 = vals.quantile(0.25)
            q3 = vals.quantile(0.75)
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr <= 0:
                continue
            lower = q1 - iqr_factor * iqr
            upper = q3 + iqr_factor * iqr
            is_out = (vals < lower) | (vals > upper)
            out_frac = is_out.mean()
            if out_frac > 0.05:  # only record if >5% are outliers
                rows.append({
                    "project": proj,
                    "feature": c,
                    "outlier_fraction": out_frac,
                    "q1": q1,
                    "q3": q3,
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                })

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        print("\nNo substantial outlier patterns (based on IQR) found.")
        return out_df

    out_df = out_df.sort_values(["project", "outlier_fraction"], ascending=[True, False])
    print("\n=== Features with many IQR-outliers per project (outlier_fraction > 0.05) ===")
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return out_df


def inspect_project_detail(df: pd.DataFrame, project: str, feat_cols: List[str], head_n: int = 10) -> None:
    """
    Print detailed stats + a few rows for a particular project.
    Useful once you know which project looks bad.
    """
    g = df[df["project"] == project].copy()
    if g.empty:
        print(f"\n[inspect_project_detail] No rows for project={project!r}")
        return

    print(f"\n=== Detailed inspection: {project} ===")
    print(f"# rows: {len(g)}, #buggy: {int(g[LABEL_COL].sum())}, bug_rate: {g[LABEL_COL].mean():.4f}")
    print("\nNumeric feature summary:")
    print(g[feat_cols].describe(percentiles=[0.25, 0.5, 0.75]).T)

    # show most extreme rows w.r.t. la, ld, ent etc (just an example)
    key_feats = [c for c in ["la", "ld", "nf", "nd", "ns", "ent", "age", "aexp", "arexp", "asexp"] if c in feat_cols]
    for c in key_feats:
        print(f"\nTop {head_n} rows by {c} (descending):")
        print(
            g.sort_values(c, ascending=False)
             .head(head_n)[["commit_id", "buggy", c]]
             .to_string(index=False)
        )


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=str, help="Path to the JIT-style CSV dataset")
    ap.add_argument(
        "--inspect_project",
        type=str,
        default=None,
        help="Optional: specific project name to inspect in detail (e.g., 'apache/groovy')",
    )
    ap.add_argument(
        "--min_rows_corr",
        type=int,
        default=50,
        help="Minimum rows per project to compute feature–label correlations (default=50)",
    )
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    # Basic coercions
    if LABEL_COL not in df.columns:
        raise SystemExit(f"Label column '{LABEL_COL}' not found in CSV.")

    if "project" not in df.columns:
        raise SystemExit("Column 'project' not found in CSV.")

    df[LABEL_COL] = _safe_buggy_to_int(df[LABEL_COL])

    # Infer features
    feat_cols = _infer_feature_cols(df)
    if not feat_cols:
        raise SystemExit("No numeric feature columns inferred; check META_COLS / LABEL_COL configuration.")

    print(f"Using {len(feat_cols)} feature columns:")
    print(", ".join(feat_cols))

    # Global
    describe_global(df, feat_cols)

    # Per-project basics
    proj_summary = summarize_projects(df, feat_cols)

    # Feature-label correlations
    corr_df = correlation_per_project(df, feat_cols, min_rows=args.min_rows_corr)

    # Outlier patterns
    out_df = detect_outliers_per_project(df, feat_cols, iqr_factor=3.0)

    # Optional detailed view of a problematic project
    if args.inspect_project is not None:
        inspect_project_detail(df, args.inspect_project, feat_cols, head_n=10)

    # Optionally save summaries for offline inspection
    out_dir = path.parent / "analysis_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    proj_summary.to_csv(out_dir / "project_summary.csv", index=False)
    if not corr_df.empty:
        corr_df.to_csv(out_dir / "feature_label_correlations.csv", index=False)
    if not out_df.empty:
        out_df.to_csv(out_dir / "outlier_features.csv", index=False)

    print(f"\nSaved summaries to: {out_dir}")


if __name__ == "__main__":
    main()
