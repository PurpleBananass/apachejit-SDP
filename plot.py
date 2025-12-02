#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from glob import glob
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy.stats import ranksums, mannwhitneyu, wilcoxon
from sklearn.preprocessing import StandardScaler
from cliffs_delta import cliffs_delta
from tabulate import tabulate

from data_utils import get_model, get_true_positives, read_dataset

# ----------------------------- config & fallbacks -----------------------------

try:
    from hyparams import EXPERIMENTS, PROPOSED_CHANGES
except Exception:
    # sensible defaults if hyparams isn't available
    EXPERIMENTS = "flipped_instances"
    PROPOSED_CHANGES = "proposed_changes"

# Try to import helper from your CF evaluator; otherwise provide a compatible fallback.
try:
    from evaluate_cf import _flip_path as _cf_flip_path
except Exception:
    # Fallback: {EXPERIMENTS}/{project}/{ModelFull}/{ExplainerToken}_all.csv
    def _cf_flip_path(project: str, model_abbr: str, explainer_key: str) -> Path:
        ABBR2FULL = {
            "RF": "RandomForest",
            "SVM": "SVM",
            "LR": "LogisticRegression",
        }
        model_full = ABBR2FULL.get(model_abbr, model_abbr)
        token = explainer_key  # e.g., "CF"
        return Path(EXPERIMENTS) / f"{project}/{model_full}/{token}_all.csv"


ABBR2FULL = {
    "RF": "RandomForest",
    "SVM": "SVM",
    "LR": "LogisticRegression",
}
FULL2ABBR = {v: k for k, v in ABBR2FULL.items()}

SEL_DIR = Path("./evaluations/feasibility/mahalanobis/selected")

EXPL_ABBR_FILE = "CF"   # token used in filenames in /selected/
EXPL_LABEL = "CF"       # how it appears in plots

# consistent model / explainer orders
MODEL_ABBRS = ["RF", "SVM", "LR"]
MODEL_FULLS = [ABBR2FULL[a] for a in MODEL_ABBRS]

# grayscale palette per model
MODEL_GRAY = {
    "RF": "0.30",
    "SVM": "0.45",
    "LR": "0.60",
}
MODEL_FULL2GRAY = {full: MODEL_GRAY[abbr] for abbr, full in ABBR2FULL.items()}

# explainer order + hatches
EXPLAINERS_ORDER = ["LIME", "LIME-HPO", "PyExplainer", "CF"]
EXPLAINER_HATCHES = {
    "LIME": "////",
    "LIME-HPO": "\\\\\\\\",
    "PyExplainer": "xxxx",
    "CF": "++++",
}

# ----------------------------- EMSE-style plotting config -----------------------------

def apply_emse_bw_style():
    """
    EMSE/Springer-friendly, grayscale style:
      - Times New Roman
      - White background, light grid
      - Grayscale only (no color)
      - Consistent font sizes/line widths
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.edgecolor": "0.2",
        "axes.linewidth": 0.8,
        "grid.color": "0.85",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })
    sns.set_theme(style="whitegrid", context="paper")


# ----------------------------- small helpers -----------------------------

def _save_csv(rows, columns, path):
    df = pd.DataFrame(rows, columns=columns)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def _save_pretty_csv(pretty_rows, headers, path):
    df = pd.DataFrame(pretty_rows, columns=headers)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _load_all_selected_for_model_abbr(
    model_abbr: str,
    expl_abbr: str = EXPL_ABBR_FILE,
) -> pd.DataFrame | None:
    """
    Load all cached 'selected' rows for a given model abbr (RF/XGB/...) and explainer token (CF).
    """
    if not SEL_DIR.exists():
        return None
    paths = glob(str(SEL_DIR / f"{model_abbr}_{expl_abbr}_*.csv"))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception:
            pass
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _load_flips_df(path: Path, feat_cols: list[str]) -> pd.DataFrame | None:
    """
    Load a flips CSV and try to standardize 'test_idx' as index if available.
    Ensures feature columns exist (drops missing ones if necessary).
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df is None or df.empty:
        return None

    # standardize 'test_idx'
    if "test_idx" in df.columns:
        df["test_idx"] = df["test_idx"].astype(int)
        df.set_index("test_idx", drop=True, inplace=True)
    else:
        first = df.columns[0].lower()
        if first in {"idx", "id", "index"}:
            df.set_index(df.columns[0], drop=True, inplace=True)
            try:
                df.index = df.index.astype(int)
            except Exception:
                pass

    present = [c for c in feat_cols if c in df.columns]
    if len(present) == 0:
        return None
    return df


def _cliffs_magnitude(delta: float) -> str:
    """
    Cliff's delta magnitude thresholds:
      |δ| < 0.147   -> negligible
      0.147–0.33    -> small
      0.33–0.474    -> medium
      >= 0.474      -> large
    """
    if delta is None or not np.isfinite(delta):
        return "NA"
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def _format_p(p: float) -> str:
    """
    p-value displayed as 0.xxx (3 decimals).
    Very small values (e.g., 1.35E-142) become '0.000'.
    """
    if p is None or not np.isfinite(p):
        return "NA"
    p = max(min(float(p), 1.0), 0.0)
    return f"{p:.3f}"


def _wilcoxon_paired(other, cf):
    """
    Shared helper:
      - Takes two paired samples (other, cf)
      - Runs Wilcoxon signed-rank (paired rank-sum)
      - Returns (N, p_value, cliff_delta, magnitude, median_other_minus_cf)

    NOTE:
      * other and cf must already be aligned instance-wise.
    """
    s_other = pd.to_numeric(pd.Series(other), errors="coerce").dropna()
    s_cf = pd.to_numeric(pd.Series(cf), errors="coerce").dropna()

    n = int(min(len(s_other), len(s_cf)))
    if n == 0:
        return 0, np.nan, np.nan, "NA", np.nan

    s_other = s_other.iloc[:n]
    s_cf = s_cf.iloc[:n]

    try:
        _, p = wilcoxon(s_other, s_cf, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        # e.g., all differences zero; treat as p=1.0 (no evidence of difference)
        p = 1.0

    d, _ = cliffs_delta(s_other, s_cf)
    d = float(d) if d is not None else np.nan
    mag = _cliffs_magnitude(d)
    med_diff = float(np.median(s_other - s_cf))

    return n, float(p), d, mag, med_diff


# ----------------------------- CF-based computations -----------------------------

def cf_selected_flip_rates_df() -> pd.DataFrame:
    """
    Flip Rate for CF (selected) per *full model name*.
    Uses selected rows to reconstruct candidate values and re-predict.
    """
    ds = read_dataset()
    out_rows = []

    for abbr, model_full in ABBR2FULL.items():
        sel = _load_all_selected_for_model_abbr(abbr, EXPL_ABBR_FILE)
        if sel is None or sel.empty or "project" not in sel.columns or "test_idx" not in sel.columns:
            continue

        total_tp = 0
        flipped_tp = 0

        for project, g in sel.groupby("project", sort=False):
            project = str(project)
            if project not in ds:
                continue

            train, test = ds[project]
            feat_cols = [c for c in test.columns if c != "target"]
            present_cols = [c for c in feat_cols if c in g.columns]

            model = get_model(project, model_full)
            scaler = StandardScaler().fit(train[feat_cols].values)

            tp_df = get_true_positives(model, train, test)
            tp_idx = set(tp_df.index.astype(int).tolist())
            if not tp_idx:
                continue
            total_tp += len(tp_idx)

            flipped_here = set()
            for ti, gi in g.groupby("test_idx", sort=False):
                ti = int(ti)
                if ti not in tp_idx:
                    continue
                orig = test.loc[ti, feat_cols].astype(float)
                r = gi.iloc[0]
                cand = orig.copy()
                if present_cols:
                    cand[present_cols] = r[present_cols].astype(float).values
                X = scaler.transform([cand.values])
                if hasattr(model, "predict_proba"):
                    pred = int((model.predict_proba(X)[:, 1] >= 0.5)[0])
                else:
                    pred = int(model.predict(X)[0])
                if pred == 0:
                    flipped_here.add(ti)

            flipped_tp += len(flipped_here)

        if total_tp > 0:
            out_rows.append({
                "Explainer": EXPL_LABEL,
                "Model": model_full,
                "Flip Rate": flipped_tp / total_tp,
            })

    return pd.DataFrame(out_rows)


def _read_abs_scores(path: str) -> pd.Series:
    """
    Read a CSV with a 'score' column, or a 1-col CSV to be named 'score'.
    Returns a float Series (may be empty).
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)

    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "score" not in df.columns:
        if df.shape[1] == 1:
            df.columns = ["score"]
        else:
            return pd.Series(dtype=float)
    return pd.to_numeric(df["score"], errors="coerce").dropna()


# ----------------------------- Implications: stats (paired Wilcoxon) -----------------------------

def run_implications_stats(
    baseline: str = "CF",
    save_csv: str = "./evaluations/implications_vs_CF_stats.csv",
    save_pretty_csv: str = "./evaluations/implications_vs_CF_table.csv",
):
    """
    For each model and each explainer in [LIME, LIME-HPO, TimeLIME, SQAPlanner],
    compare the distributions of 'total amount of changes required' vs CF
    using Wilcoxon signed-rank (paired rank-sum) + Cliff's delta.

    Paired files used:
        ./evaluations/abs_changes/{Model}_{Explainer}_pairedCF.csv

    Columns expected:
        - score_explainer
        - score_cf
    """
    others = ["LIME", "LIME-HPO", "PyExplainer", "CF"]
    models = MODEL_ABBRS

    raw_rows = []
    pretty_rows = []

    for model in models:
        for other in others:
            pth = f"./evaluations/abs_changes/{model}_{other}_pairedCF.csv"
            try:
                df = pd.read_csv(pth)
            except FileNotFoundError:
                raw_rows.append({
                    "Model": model,
                    "Other": other,
                    "Baseline": baseline,
                    "Test": "Wilcoxon (paired rank-sum)",
                    "N": 0,
                    "p_value": np.nan,
                    "cliffs_delta": np.nan,
                    "Magnitude": "NA",
                    "Median(other−CF)": np.nan,
                })
                pretty_rows.append([
                    model,
                    other,
                    baseline,
                    "Wilcoxon (paired rank-sum)",
                    0,
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                ])
                continue
            except Exception:
                raw_rows.append({
                    "Model": model,
                    "Other": other,
                    "Baseline": baseline,
                    "Test": "Wilcoxon (paired rank-sum)",
                    "N": 0,
                    "p_value": np.nan,
                    "cliffs_delta": np.nan,
                    "Magnitude": "NA",
                    "Median(other−CF)": np.nan,
                })
                pretty_rows.append([
                    model,
                    other,
                    baseline,
                    "Wilcoxon (paired rank-sum)",
                    0,
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                ])
                continue

            needed = {"score_explainer", "score_cf"}
            if df is None or df.empty or not needed.issubset(df.columns):
                raw_rows.append({
                    "Model": model,
                    "Other": other,
                    "Baseline": baseline,
                    "Test": "Wilcoxon (paired rank-sum)",
                    "N": 0,
                    "p_value": np.nan,
                    "cliffs_delta": np.nan,
                    "Magnitude": "NA",
                    "Median(other−CF)": np.nan,
                })
                pretty_rows.append([
                    model,
                    other,
                    baseline,
                    "Wilcoxon (paired rank-sum)",
                    0,
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                ])
                continue

            n, p, d, mag, med_diff = _wilcoxon_paired(
                df["score_explainer"], df["score_cf"]
            )

            raw_rows.append({
                "Model": model,
                "Other": other,
                "Baseline": baseline,
                "Test": "Wilcoxon (paired rank-sum)",
                "N": n,
                "p_value": p,
                "cliffs_delta": d,
                "Magnitude": mag,
                "Median(other−CF)": med_diff,
            })

            pretty_rows.append([
                model,
                other,
                baseline,
                "Wilcoxon (paired rank-sum)",
                n,
                _format_p(p),
                ("NA" if not np.isfinite(d) else f"{d:.3f}"),
                mag,
                (f"{med_diff:.3f}" if np.isfinite(med_diff) else "NA"),
            ])

    Path("./evaluations").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_rows).to_csv(save_csv, index=False)

    headers = [
        "Model",
        "Other",
        "Baseline",
        "Test",
        "N",
        "p-value",
        "Cliff’s δ",
        "Magnitude",
        "Median(other−CF)",
    ]
    _save_pretty_csv(pretty_rows, headers, save_pretty_csv)

    print(tabulate(pretty_rows, headers=headers, tablefmt="grid"))
    print(f"Saved {save_csv}")
    print(f"Saved {save_pretty_csv}")


# ----------------------------- RQ3 stats helpers (paired Wilcoxon) -----------------------------

def _read_rq3_df(model_abbr: str, explainer: str) -> pd.DataFrame:
    """
    Returns a DataFrame with normalized 'min' values for a given (model_abbr, explainer).

    - Reads ./evaluations/feasibility/mahalanobis/{abbr}_{expl}.csv (+ shards)
    - Applies the same non-NaN mask to all propagated columns (min_norm, test_idx, project, idx)
    """
    base = "./evaluations/feasibility/mahalanobis"
    frames: list[pd.DataFrame] = []

    # main file
    path = f"{base}/{model_abbr}_{explainer}.csv"
    try:
        df = pd.read_csv(path)
        if df is not None and not df.empty:
            frames.append(df)
    except FileNotFoundError:
        pass

    # sharded files
    for p in glob(f"{base}/{model_abbr}_{explainer}_*.csv"):
        try:
            d = pd.read_csv(p)
            if d is not None and not d.empty:
                frames.append(d)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(columns=["min_norm"])

    all_df = pd.concat(frames, ignore_index=True)

    if "min" not in all_df.columns:
        return pd.DataFrame(columns=["min_norm"])

    # convert and build a mask for valid min values
    mins = pd.to_numeric(all_df["min"], errors="coerce")
    mask = mins.notna()
    if not mask.any():
        return pd.DataFrame(columns=["min_norm"])

    # normalized [0,1]
    min_norm = mins[mask].clip(0, 1)

    out = pd.DataFrame({"min_norm": min_norm.values})

    # propagate identifiers aligned with the mask
    for k in ("project", "test_idx", "idx"):
        if k in all_df.columns:
            out[k] = all_df.loc[mask, k].values

    return out


def _pair_rq3(model_abbr: str, other: str, baseline: str = "CF"):
    """
    Build paired samples (other, CF) for RQ3 based on shared test_idx / idx.
    Returns:
        other_vals, cf_vals (aligned)
    """
    cf_df = _read_rq3_df(model_abbr, baseline)
    oth_df = _read_rq3_df(model_abbr, other)

    if cf_df.empty or oth_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    key = None
    for k in ("test_idx", "idx"):
        if k in cf_df.columns and k in oth_df.columns:
            key = k
            break

    if key is None:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    cf_sub = cf_df[[key, "min_norm"]].dropna()
    oth_sub = oth_df[[key, "min_norm"]].dropna()

    merged = pd.merge(
        oth_sub.rename(columns={"min_norm": "min_other"}),
        cf_sub.rename(columns={"min_norm": "min_cf"}),
        on=key,
        how="inner",
    )

    if merged.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    other_vals = pd.to_numeric(merged["min_other"], errors="coerce").dropna()
    cf_vals = pd.to_numeric(merged["min_cf"], errors="coerce").dropna()

    n = int(min(len(other_vals), len(cf_vals)))
    if n == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    return other_vals.iloc[:n], cf_vals.iloc[:n]


def run_rq3_stat_tests(
    baseline: str = "CF",
    save_csv: str = "./evaluations/rq3_stats.csv",
    save_pretty_csv: str = "./evaluations/rq3_stats_table.csv",
):
    """
    Paired Wilcoxon signed-rank (paired rank-sum) + Cliff's delta for RQ3
    (feasibility distances), comparing each explainer vs CF per model.

    Pairing is done on shared instance IDs (test_idx/idx).
    """
    models = MODEL_ABBRS
    others = ["LIME", "LIME-HPO", "PyExplainer", "CF"]

    raw_rows = []
    pretty_rows = []

    for m in models:
        for other in others:
            oth_vals, cf_vals = _pair_rq3(m, other, baseline=baseline)

            if len(oth_vals) == 0 or len(cf_vals) == 0:
                raw_rows.append({
                    "Model": m,
                    "Other": other,
                    "Baseline": baseline,
                    "Test": "Wilcoxon (paired rank-sum)",
                    "N": 0,
                    "p_value": np.nan,
                    "cliffs_delta": np.nan,
                    "Magnitude": "NA",
                    "Median(other−CF)": np.nan,
                })
                pretty_rows.append([
                    m,
                    other,
                    baseline,
                    "Wilcoxon (paired rank-sum)",
                    0,
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                ])
                continue

            n, p, d, mag, med_diff = _wilcoxon_paired(oth_vals, cf_vals)

            raw_rows.append({
                "Model": m,
                "Other": other,
                "Baseline": baseline,
                "Test": "Wilcoxon (paired rank-sum)",
                "N": n,
                "p_value": p,
                "cliffs_delta": d,
                "Magnitude": mag,
                "Median(other−CF)": med_diff,
            })

            pretty_rows.append([
                m,
                other,
                baseline,
                "Wilcoxon (paired rank-sum)",
                n,
                _format_p(p),
                ("NA" if not np.isfinite(d) else f"{d:.3f}"),
                mag,
                (f"{med_diff:.3f}" if np.isfinite(med_diff) else "NA"),
            ])

    Path("./evaluations").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_rows).to_csv(save_csv, index=False)

    headers = [
        "Model",
        "Other",
        "Baseline",
        "Test",
        "N",
        "p-value",
        "Cliff’s δ",
        "Magnitude",
        "Median(other−CF)",
    ]
    _save_pretty_csv(pretty_rows, headers, save_pretty_csv)

    print(tabulate(pretty_rows, headers=headers, tablefmt="grid"))
    print(f"Saved {save_csv}")
    print(f"Saved {save_pretty_csv}")


# ----------------------------- RQ1: Flip rates -----------------------------

def visualize_rq1():
    apply_emse_bw_style()

    base_df = pd.read_csv("./evaluations/flip_rates.csv")
    try:
        cf_sel = cf_selected_flip_rates_df()
        df = pd.concat([base_df, cf_sel], ignore_index=True) if not cf_sel.empty else base_df
    except Exception as e:
        print(f"[rq1] Could not add {EXPL_LABEL}: {e}")
        df = base_df

    df = df[df["Explainer"] != "All"].copy()

    expl_order = EXPLAINERS_ORDER
    present_expl = [e for e in expl_order if e in set(df["Explainer"])]
    if not present_expl:
        present_expl = sorted(df["Explainer"].unique().tolist())

    model_order = (
        df.groupby("Model", as_index=False)["Flip Rate"]
          .mean()
          .sort_values("Flip Rate", ascending=False)["Model"]
          .tolist()
    )
    if not model_order:
        print("[rq1] No data to plot.")
        return

    n_models = len(model_order)
    fig, axes = plt.subplots(
        1,
        n_models,
        sharey=True,
        figsize=(1.7 * n_models + 1.5, 4.0),
    )
    if n_models == 1:
        axes = [axes]

    for ax, model_full in zip(axes, model_order):
        sub = df[df["Model"] == model_full].copy()
        if sub.empty:
            ax.axis("off")
            continue

        expls_here = [e for e in present_expl if e in set(sub["Explainer"])]
        sub = (
            sub.set_index("Explainer")
               .reindex(expls_here)
               .dropna(subset=["Flip Rate"])
        )
        values = sub["Flip Rate"].values
        labels = sub.index.tolist()

        x = np.arange(len(labels))
        bar_width = 0.6

        for xi, label, v in zip(x, labels, values):
            if np.isnan(v):
                continue
            hatch = EXPLAINER_HATCHES.get(label, "")
            ax.bar(
                xi,
                v,
                width=bar_width,
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
                hatch=hatch,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax, left=False, bottom=False, right=True, top=True)

        for xi, v in zip(x, values):
            if np.isnan(v):
                continue
            ax.text(
                xi,
                v + 0.02,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontfamily="monospace",
            )

        model_abbr = FULL2ABBR.get(model_full, model_full)
        ax.set_title(model_abbr, fontsize=11)

    axes[0].set_ylabel("Flip Rate", fontsize=11)
    for ax in axes[1:]:
        ax.set_ylabel("")

    legend_handles = []
    for expl in present_expl:
        legend_handles.append(
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=EXPLAINER_HATCHES.get(expl, ""),
                label=expl,
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        frameon=False,
        ncol=len(present_expl),
        bbox_to_anchor=(0.5, 1.08),
    )

    plt.tight_layout()
    fig.savefig("./evaluations/rq1.png", dpi=300, bbox_inches="tight")


# ----------------------------- RQ2 (similarity distributions) -----------------------------

def visualize_rq2():
    apply_emse_bw_style()

    explainers = {
        "LIME": "LIME",
        "LIME-HPO": "LIME-HPO",
        "PyExplainer": "PyExplainer",
        "CF": "CF",
    }
    models = {
        "RF": "RandomForest",
        "SVM": "SVM",
        "LR": "LogisticRegression",
    }

    total_df = pd.DataFrame()
    for model in models:
        try:
            df = pd.read_csv(f"./evaluations/similarities/{model}.csv", index_col=0)
            total_df = pd.concat([total_df, df], ignore_index=False)
        except FileNotFoundError:
            print(f"Warning: similarities file for {model} not found")
            continue

    if total_df.empty:
        print("No data to plot for RQ2.")
        return

    total_df.index.set_names("idx", inplace=True)
    total_df = total_df.set_index([total_df.index, total_df["project"]])
    total_df = total_df.drop(columns=["project"])

    dset = read_dataset()
    for project in dset:
        train, test = dset[project]
        for model_type, model_full in models.items():
            try:
                true_positives = get_true_positives(
                    get_model(project, model_full), train, test
                )
            except Exception as e:
                print(f"Warning: could not get TPs for {project} {model_type}: {e}")
                continue

            for expl_label, expl_token in explainers.items():
                flip_path = (
                    Path(EXPERIMENTS)
                    / f"{project}/{model_full}/{expl_token}_all.csv"
                )
                if not flip_path.exists():
                    continue

                try:
                    df = pd.read_csv(flip_path, index_col=0)
                except Exception:
                    continue

                df["model"] = model_type
                df["explainer"] = expl_label
                df["project"] = project

                flipped = df.dropna()

                unflipped_index = true_positives.index.difference(flipped.index)
                unflipped = pd.DataFrame(index=unflipped_index)
                unflipped["model"] = model_type
                unflipped["explainer"] = expl_label
                unflipped["project"] = project
                unflipped["score"] = None
                unflipped.set_index(
                    [unflipped.index, unflipped["project"]], inplace=True
                )
                unflipped = unflipped.drop(columns=["project"])

                total_df = pd.concat(
                    [total_df, unflipped[["model", "explainer", "score"]]],
                    ignore_index=False,
                )

    if total_df.empty:
        print("No data to plot for RQ2 after adding flipped/unflipped.")
        return

    max_count = {}
    for expl in explainers.keys():
        max_count[expl] = 0
        for model in models.keys():
            df = total_df[
                (total_df["explainer"] == expl) & (total_df["model"] == model)
            ]
            max_count[expl] = max(max_count[expl], len(df))

    expl_list = list(explainers.keys())
    model_list = list(models.keys())
    n_rows, n_cols = len(expl_list), len(model_list)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(13, 5.5),
        sharex=True,
        sharey=False,
    )
    axes = np.array(axes).reshape(n_rows, n_cols)

    for r, expl in enumerate(expl_list):
        for c, model in enumerate(model_list):
            ax = axes[r, c]
            df = total_df[
                (total_df["explainer"] == expl) & (total_df["model"] == model)
            ]

            if len(df) > 0:
                sns.histplot(
                    data=df,
                    x="score",
                    ax=ax,
                    color="0.3",
                    stat="count",
                    common_norm=False,
                    common_bins=True,
                    cumulative=True,
                    bins=10,
                )

            ax.set_ylim(0, max_count[expl] + 250)
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.set_xlabel("")

            if c == 0:
                sns.despine(ax=ax, left=False, right=True, top=False, bottom=True)
            elif c == n_cols - 1:
                sns.despine(ax=ax, left=True, right=False, top=False, bottom=True)
            else:
                sns.despine(ax=ax, left=True, right=True, top=False, bottom=True)

            if r == 0:
                ax.set_title(model, fontsize=12)

            if c == 0:
                ax.set_ylabel(
                    expl,
                    fontsize=12,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=25,
                )

            if len(df) > 0:
                for container in ax.containers:
                    for bar_idx, bar in enumerate(container):
                        if bar_idx == 0 or bar_idx == len(container) - 1:
                            ax.text(
                                bar.get_x() + bar.get_width() * (0.35 if bar_idx == 0 else 0.5),
                                bar.get_height() + 20,
                                f".{bar.get_height()/len(df)*100:.0f}",
                                ha="center",
                                va="bottom",
                                fontsize=9,
                                fontfamily="monospace",
                            )

            if r < n_rows - 1:
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    top=False,
                    labelbottom=False,
                    labeltop=False,
                )
            else:
                ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                ax.set_xticks(ticks)
                ax.set_xticklabels(ticks, fontsize=10)
                ax.tick_params(
                    axis="x",
                    which="both",
                    bottom=True,
                    top=False,
                    labelbottom=True,
                    labeltop=False,
                    pad=2,
                )

    fig.text(0.5, 0.04, "Similarity Score", ha="center", fontsize=12)
    plt.tight_layout(rect=[0.04, 0.08, 0.99, 0.98])
    plt.savefig("./evaluations/rq2_combined.png", dpi=300)


# ----------------------------- Implications (CF baseline; no DiCE) -----------------------------

def visualize_implications():
    """
    Boxplot of distributions of total amount of changes required.
    Grayscale, EMSE-style, models in gray levels.
    """
    apply_emse_bw_style()

    explainers = ["LIME", "LIME-HPO", "PyExplainer", "CF"]
    models = MODEL_ABBRS
    total_df = pd.DataFrame()

    def _read_scores_df(path: str) -> pd.DataFrame | None:
        s = _read_abs_scores(path)
        if len(s) == 0:
            return None
        return pd.DataFrame({"score": s})

    for model in models:
        for explainer in explainers:
            if explainer == "CF":
                parts = []
                main = _read_scores_df(f"./evaluations/abs_changes/{model}_CF.csv")
                if main is not None:
                    parts.append(main)
                else:
                    for p in glob(f"./evaluations/abs_changes/{model}_CF_*.csv"):
                        d = _read_scores_df(p)
                        if d is not None and not empty:
                            parts.append(d)
                if not parts:
                    print(f"Warning: no CF abs_changes files found for {model}")
                    continue
                df = pd.concat(parts, ignore_index=True)
            else:
                df = _read_scores_df(f"./evaluations/abs_changes/{model}_{explainer}.csv")
                if df is None:
                    print(f"Warning: abs_changes file not found for {model}_{explainer}")
                    continue

            df["Model"] = model
            df["Explainer"] = explainer
            total_df = pd.concat([total_df, df], ignore_index=True)

    if total_df.empty:
        print("No data to plot for implications.")
        return

    Path("./evaluations").mkdir(parents=True, exist_ok=True)
    out_long = "./evaluations/implications_data_long.csv"
    total_df[["Model", "Explainer", "score"]].to_csv(out_long, index=False)
    print(f"Saved {out_long}")

    summary = (
        total_df.groupby(["Model", "Explainer"])["score"]
                .agg(N="count", mean="mean", median="median", std="std")
                .reset_index()
    )
    out_summary = "./evaluations/implications_data_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"Saved {out_summary}")

    present = [e for e in explainers if e in set(total_df["Explainer"])]
    plt.figure(figsize=(6.2, 3.2))

    palette = {m: MODEL_GRAY[m] for m in models}
    ax = sns.boxplot(
        data=total_df,
        x="Explainer",
        y="score",
        hue="Model",
        order=present,
        hue_order=models,
        palette=palette,
        showfliers=False,
    )
    ax.set_ylabel("Total Amount of Changes Required", rotation=90, labelpad=3, fontsize=12)
    ax.set_xlabel("")
    plt.yticks(fontsize=12, ticks=[])
    ax.set_yticklabels(labels=[])
    ax.set_xticklabels(fontsize=12, labels=present)
    ax.get_legend().set_title("")
    ax.legend(loc="upper right", title="", fontsize=10, frameon=False)

    plt.ylim(-0.5, 15)
    plt.tight_layout()
    plt.savefig("./evaluations/implications.png", dpi=300)


# ----------------------------- RQ3 – bar version (boxplot) -----------------------------

def visualize_rq3_bar():
    """
    RQ3 boxplot:

    - Normalized Mahalanobis distance (min_norm) distributions
      per Explainer × Model combination.
    - x-axis: Explainer (LIME, LIME-HPO, TimeLIME, SQAPlanner, CF)
    - y-axis: Normalized Mahalanobis Distance
    - hue: Model (RF, XGB, SVM, LGBM, CatB) – grayscale palette.

    Also saves:
      ./evaluations/rq3_data_long.csv
      ./evaluations/rq3_data_summary.csv
    """
    apply_emse_bw_style()

    models = MODEL_ABBRS
    explainers = ["LIME", "LIME-HPO", "PyExplainer", "CF"]

    rows = []
    for m in models:
        
        for expl in explainers:
            if m != "RF":
                continue
            df = _read_rq3_df(m, expl)
            if df is None or df.empty or "min_norm" not in df:
                print(f"[RQ3-bar] No data for {m}_{expl}")
                continue

            vals = pd.to_numeric(df["min_norm"], errors="coerce").dropna()
            if vals.empty:
                continue

            for v in vals:
                rows.append({
                    "Model": m,
                    "Explainer": expl,
                    "min_norm": float(v),
                })

    total_df = pd.DataFrame(rows)
    if total_df.empty:
        print("No feasibility data found for RQ3 boxplot.")
        return

    Path("./evaluations").mkdir(parents=True, exist_ok=True)

    out_long = "./evaluations/rq3_data_long.csv"
    total_df.to_csv(out_long, index=False)
    print(f"Saved {out_long}")

    summary = (
        total_df.groupby(["Model", "Explainer"])["min_norm"]
                .agg(N="count", mean="mean", median="median", std="std")
                .reset_index()
    )
    out_summary = "./evaluations/rq3_data_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"Saved {out_summary}")

    total_df["Explainer"] = pd.Categorical(
        total_df["Explainer"], categories=explainers, ordered=True
    )
    total_df["Model"] = pd.Categorical(
        total_df["Model"], categories=models, ordered=True
    )

    plt.figure(figsize=(7.0, 4.0))

    gray_palette = {m: MODEL_GRAY[m] for m in models}
    ax = sns.boxplot(
        data=total_df,
        x="Explainer",
        y="min_norm",
        hue="Model",
        hue_order=models,
        order=explainers,
        palette=gray_palette,
        showfliers=False,
    )

    ax.set_xlabel("Explainer")
    ax.set_ylabel("Normalized Mahalanobis Distance")
    ax.set_ylim(0, 1.0)

    ax.legend(
        title="Model",
        loc="upper right",
        frameon=False,
        fontsize=10,
    )
    sns.despine(ax=ax, left=True, right=False, top=True, bottom=False)

    plt.tight_layout()
    plt.savefig("./evaluations/rq3_bar.png", dpi=300, bbox_inches="tight")
    print("Saved ./evaluations/rq3_bar.png")


# ----------------------------- RQ3 – main strip + mean markers -----------------------------

def visualize_rq3():
    apply_emse_bw_style()

    explainers = ["LIME", "LIME-HPO", "PyExplainer", "CF"]
    models = MODEL_ABBRS
    distance_dir = "./evaluations/feasibility/mahalanobis"

    total_df = pd.DataFrame()
    for model_abbr in models:
        for explainer in explainers:
            path = f"{distance_dir}/{model_abbr}_{explainer}.csv"
            frames = []
            try:
                df = pd.read_csv(path)
                if df is not None and not df.empty:
                    frames.append(df)
            except FileNotFoundError:
                for p in glob(f"{distance_dir}/{model_abbr}_{explainer}_*.csv"):
                    try:
                        d = pd.read_csv(p)
                        if d is not None and not d.empty:
                            frames.append(d)
                    except Exception:
                        pass

            if not frames:
                print(f"Warning: feasibility file(s) not found for {model_abbr}_{explainer}")
                continue

            df_all = pd.concat(frames, ignore_index=True)
            if "min" not in df_all.columns:
                print(f"[RQ3] Missing 'min' column for {model_abbr}_{explainer}; skipping.")
                continue

            df_all["Model"] = model_abbr
            df_all["Explainer"] = explainer
            total_df = pd.concat([total_df, df_all], ignore_index=True)

    if total_df.empty:
        print("No feasibility data found.")
        return

    plot_df = total_df.loc[:, ["Model", "Explainer", "min"]].copy()
    plot_df["min"] = pd.to_numeric(plot_df["min"], errors="coerce")
    plot_df.dropna(subset=["min"], inplace=True)
    plot_df["min_norm"] = plot_df["min"].clip(0, 1)

    fig = plt.figure(figsize=(6.8, 5.6))

    gray_palette = {m: MODEL_GRAY[m] for m in models}

    sns.stripplot(
        data=plot_df,
        x="Explainer",
        y="min_norm",
        hue="Model",
        palette=gray_palette,
        dodge=True,
        jitter=0.2,
        size=3,
        alpha=0.25,
        legend=False,
    )

    mean_palette = {m: "0.1" for m in models}
    ax = sns.pointplot(
        data=plot_df,
        x="Explainer",
        y="min_norm",
        hue="Model",
        palette=mean_palette,
        dodge=0.8 - 0.8 / len(models),
        errorbar=None,
        markers="x",
        markersize=4,
        linestyles="none",
        legend=False,
        zorder=10,
    )

    mean_df = plot_df.groupby(["Model", "Explainer"], as_index=False)["min_norm"].mean()
    offsets = (-0.4, -0.2, 0, 0.2, 0.4)
    for _, row in mean_df.iterrows():
        model_name = row["Model"]
        expl = row["Explainer"]
        mi = models.index(model_name)
        x = explainers.index(expl) + offsets[mi]
        y = float(row["min_norm"])
        label = f".{y:.2f}".replace("0.", "")
        ax.text(
            x,
            min(max(y, 0.0), 1.0) + 0.01,
            label,
            va="bottom",
            ha="center",
            fontsize=11,
            fontfamily="monospace",
            color="black",
        )

    plt.ylabel("Normalized Mahalanobis Distance")
    plt.xlabel("Explainer")

    plt.ylim(0, 1.0)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    legend_elements = [
        Patch(facecolor=gray_palette[m], edgecolor="black", label=m)
        for m in models
    ]
    fig.legend(
        handles=legend_elements,
        title="Model",
        loc="upper center",
        fontsize=10,
        frameon=False,
        ncols=5,
        bbox_to_anchor=(0.525, 0.94),
    )

    plt.tight_layout()
    plt.savefig("./evaluations/rq3.png", dpi=300, bbox_inches="tight")
    print("Saved ./evaluations/rq3.png")


# ----------------------------- Misc utilities -----------------------------

def group_diff(d1, d2):
    d1 = pd.to_numeric(pd.Series(d1), errors="coerce").dropna()
    d2 = pd.to_numeric(pd.Series(d2), errors="coerce").dropna()
    if len(d1) == 0 or len(d2) == 0:
        return np.nan, np.nan
    _, p = ranksums(d1, d2)
    d, _ = cliffs_delta(d1, d2)
    return p, d


def list_status(
    model_type="XGBoost",
    explainers=("LIME-HPO", "LIME", "PyExplainer", "DiCE"),
):
    dset = read_dataset()
    table = []
    headers = ["Project"] + [exp[:8] for exp in explainers] + ["common", "left"]
    total = 0
    total_left = 0
    for project in sorted(dset.keys()):
        row = {}
        table_row = [project]
        for explainer in explainers:
            flipped_path = Path(
                f"flipped_instances/{project}/{model_type}/{explainer}_all.csv"
            )
            if not flipped_path.exists():
                print(f"{flipped_path} not exists")
                row[explainer] = set()
            else:
                flipped = pd.read_csv(flipped_path, index_col=0)
                computed_names = set(flipped.index)
                row[explainer] = computed_names

        plan_path = Path(
            f"proposed_changes/{project}/{model_type}/{explainers[0]}/plans_all.json"
        )
        if plan_path.exists():
            with open(plan_path, "r") as f:
                plans = json.load(f)
                total_names = set(plans.keys())
        else:
            total_names = set()

        common_names = row.get(explainers[0], set())
        for explainer in explainers[1:]:
            common_names = common_names.intersection(row.get(explainer, set()))
        row["common"] = common_names
        row["total"] = total_names
        for explainer in explainers:
            table_row.append(len(row.get(explainer, set())))
        table_row.append(f"{len(common_names)}/{len(total_names)}")
        table_row.append(len(total_names) - len(common_names))
        table.append(table_row)
        total += len(common_names)
        total_left += len(total_names) - len(common_names)
    table.append(["Total"] + [""] * len(explainers) + [total, total_left])
    print(f"Model: {model_type}")
    print(tabulate(table, headers=headers))


# ----------------------------- CLI -----------------------------

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--rq1", action="store_true")
    argparser.add_argument("--rq2", action="store_true")
    argparser.add_argument("--rq3", action="store_true")
    argparser.add_argument("--implications", action="store_true")
    argparser.add_argument("--rq3_stats", action="store_true")
    args = argparser.parse_args()

    if args.rq1:
        visualize_rq1()
    if args.rq2:
        visualize_rq2()
    if args.rq3:
        visualize_rq3()
        visualize_rq3_bar()
    if args.implications:
        visualize_implications()
        run_implications_stats(
            baseline="CF",
            save_csv="./evaluations/implications_vs_CF_stats.csv",
            save_pretty_csv="./evaluations/implications_vs_CF_table.csv",
        )
    if args.rq3_stats:
        run_rq3_stat_tests(
            baseline="CF",
            save_csv="./evaluations/rq3_stats.csv",
            save_pretty_csv="./evaluations/rq3_stats_table.csv",
        )
        run_implications_stats(
            baseline="CF",
            save_csv="./evaluations/abs_changes_vs_CF_stats.csv",
            save_pretty_csv="./evaluations/abs_changes_vs_CF_table.csv",
        )
