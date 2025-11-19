#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
jit_plots.py

Plot RQ1 / RQ2 / RQ3 / Implications for the JIT experiments
using outputs from eval_jit.py and jit_cf.py.

Expected files:

- RQ1:
    ./evaluations/flip_rates_jit.csv
    columns: Model, Explainer, Flip Rate

- RQ2 (plan similarity):
    ./evaluations/similarities/{Model}.csv
    columns: score, project, explainer, model

- RQ3 (feasibility distances, 'min'):
    ./evaluations/feasibility/mahalanobis/{Model}_{Explainer}.csv
    columns include: min, max, mean

- Implications (absolute z-scored change sums):
    ./evaluations/abs_changes/{Model}_{Explainer}.csv
    single column (e.g. 'zsum') → treated as 'score'
"""

from __future__ import annotations

import argparse
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_utils import read_dataset, get_model, get_true_positives

# ----------------------------- config -----------------------------

try:
    from hyparams import EXPERIMENTS
except Exception:
    EXPERIMENTS = "experiments"

JIT_MODELS = ["RandomForest", "SVM", "LogisticRegression"]
JIT_MODEL_LABELS = {
    "RandomForest": "RF",
    "SVM": "SVM",
    "LogisticRegression": "LR",
}
JIT_EXPLAINERS = ["LIME", "LIME-HPO", "PyExplainer", "CF"]


def _ensure_eval_dir() -> None:
    Path("./evaluations").mkdir(parents=True, exist_ok=True)


def _read_abs_scores(path: str) -> pd.Series:
    """
    Read a CSV with a single score-like column.
    - If 'score' exists, use it.
    - If there's exactly one column, rename to 'score'.
    Returns float Series (may be empty).
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


# ----------------------------- RQ1: Flip rates -----------------------------

def visualize_rq1_jit(
    csv_path: str = "./evaluations/flip_rates_jit.csv",
    out_path: str = "./evaluations/jit_rq1.png",
) -> None:
    """
    JIT RQ1: Flip rates per (Explainer, Model).

    Expects flip_rates_jit.csv with at least:
        Model, Explainer, FlipRate
    (optionally Flips, TPs, and 'Mean' rows per model).
    """
    _ensure_eval_dir()
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[RQ1-JIT] File not found: {csv_path}")
        return
    except Exception as e:
        print(f"[RQ1-JIT] Could not read {csv_path}: {e}")
        return

    if df.empty:
        print("[RQ1-JIT] flip_rates_jit.csv is empty.")
        return

    # --- normalize column names to what the plotting code expects ---
    rename_map = {}
    if "Model" in df.columns:
        rename_map["Model"] = "Model"
    if "Explainer" in df.columns:
        rename_map["Explainer"] = "Explainer"
    if "FlipRate" in df.columns:
        rename_map["FlipRate"] = "Flip Rate"   # <- this is the key fix

    df = df.rename(columns=rename_map)

    # sanity-check required columns
    for col in ["Model", "Explainer", "Flip Rate"]:
        if col not in df.columns:
            print(f"[RQ1-JIT] Missing required column: {col}. Columns present: {list(df.columns)}")
            return

    # make sure rates are numeric
    df["Flip Rate"] = pd.to_numeric(df["Flip Rate"], errors="coerce")
    df = df.dropna(subset=["Flip Rate"])
    if df.empty:
        print("[RQ1-JIT] No valid numeric flip rates after cleaning.")
        return

    # Optional: keep or drop the 'Mean' lines per model.
    # If you DON'T want a separate bar for the aggregated per-model mean, uncomment:
    # df = df[df["Explainer"] != "Mean"]

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")
    sns.set_palette("crest")
    plt.rcParams["font.family"] = "Times New Roman"

    # Order explainers (if present)
    expl_order_pref = ["LIME", "LIME-HPO", "PyExplainer", "CF", "Mean"]
    present_expl = [e for e in expl_order_pref if e in df["Explainer"].unique()]
    if not present_expl:
        present_expl = sorted(df["Explainer"].unique().tolist())

    # Order models by mean flip rate (descending)
    model_order = (
        df.groupby("Model", as_index=False)["Flip Rate"]
          .mean()
          .sort_values("Flip Rate", ascending=False)["Model"]
          .tolist()
    )

    plt.figure(figsize=(6, 4.5))
    ax = sns.barplot(
        data=df,
        y="Explainer",
        x="Flip Rate",
        hue="Model",
        order=present_expl,
        hue_order=model_order,
        edgecolor="0.2",
    )

    # annotate each bar with the numeric flip rate
    for container in ax.containers:
        for bar in container:
            bar.set_edgecolor("black")
            bar.set_linewidth(0.5)
            w = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            ax.text(
                w + 0.01,
                y,
                f"{w:.2f}",
                va="center",
                ha="left",
                fontsize=9,
                fontfamily="monospace",
            )

    ax.set_xlim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax.legend(title="", frameon=False, loc="lower right", fontsize=10)
    plt.xticks(fontsize=11, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    plt.yticks(fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[RQ1-JIT] Saved {out_path}")



# ----------------------------- RQ2: Plan similarity -----------------------------

def visualize_rq2_jit(
    out_path: str = "./evaluations/jit_rq2.png",
) -> None:
    """
    Histograms of plan similarity scores by (Explainer, Model) for JIT.
    Uses ./evaluations/similarities/{Model}.csv from eval_jit.py --rq2.

    Simpler than the original: only uses the 'score' values for flipped instances
    (no dummy rows for unflipped).
    """
    _ensure_eval_dir()
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "Times New Roman"

    explainers = ["LIME", "LIME-HPO", "PyExplainer"]
    models = JIT_MODELS

    total_df = pd.DataFrame()
    for model in models:
        path = Path("./evaluations/similarities") / f"{model}.csv"
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"[RQ2] Similarities file not found for model {model}: {path}")
            continue
        except Exception as e:
            print(f"[RQ2] Error reading {path}: {e}")
            continue

        if df is None or df.empty or "score" not in df.columns:
            print(f"[RQ2] No usable data in {path}")
            continue

        df["model"] = df.get("model", model)
        total_df = pd.concat([total_df, df], ignore_index=True)

    if total_df.empty:
        print("[RQ2] No similarity data found.")
        return

    # Clean up
    total_df["score"] = pd.to_numeric(total_df["score"], errors="coerce")
    total_df.dropna(subset=["score"], inplace=True)

    nrows = len(explainers)
    ncols = len(models)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 2.6 * nrows),
        sharex=True,
        sharey=True,
    )
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    colors = sns.color_palette("crest", ncols)

    for i, expl in enumerate(explainers):
        for j, model in enumerate(models):
            ax = axes[i, j]
            sub = total_df[
                (total_df["explainer"] == expl) & (total_df["model"] == model)
            ]
            if not sub.empty:
                sns.histplot(
                    data=sub,
                    x="score",
                    ax=ax,
                    bins=10,
                    stat="count",
                    kde=False,
                    color=colors[j],
                )

            if i == 0:
                ax.set_title(JIT_MODEL_LABELS.get(model, model), fontsize=11)
            if j == 0:
                ax.set_ylabel(expl, fontsize=11)
            else:
                ax.set_ylabel("")

            if i < nrows - 1:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Similarity score", fontsize=11)
                ax.tick_params(axis="x", labelsize=10)

            ax.tick_params(axis="y", labelleft=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[RQ2] Saved {out_path}")


# ----------------------------- RQ3: Feasibility distances -----------------------------

def _load_rq3_jit_raw(distance: str = "mahalanobis") -> pd.DataFrame:
    """
    Load all per-project feasibility results into one DataFrame:
    Model, Explainer, min, max, mean.

    Expects: ./evaluations/feasibility/{distance}/{Model}_{Explainer}.csv
    where Model ∈ JIT_MODELS and Explainer ∈ JIT_EXPLAINERS.
    """
    base_dir = Path("./evaluations/feasibility") / distance
    frames = []
    for model in JIT_MODELS:
        for expl in JIT_EXPLAINERS:
            path = base_dir / f"{model}_{expl}.csv"
            if not path.exists():
                # optional shards: {model}_{expl}_*.csv
                shard_paths = glob(str(base_dir / f"{model}_{expl}_*.csv"))
                for sp in shard_paths:
                    try:
                        d = pd.read_csv(sp)
                        if d is not None and not d.empty:
                            d["Model"] = model
                            d["Explainer"] = expl
                            frames.append(d)
                    except Exception:
                        pass
                continue

            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            df["Model"] = model
            df["Explainer"] = expl
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def visualize_rq3_jit(
    distance: str = "mahalanobis",
    out_path: str = "./evaluations/jit_rq3.png",
) -> None:
    """
    Strip + point plot of normalized feasibility minima (0–1) vs explainer, per model.
    Uses ./evaluations/feasibility/{distance}/{Model}_{Explainer}.csv.
    """
    _ensure_eval_dir()
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "Times New Roman"

    raw_df = _load_rq3_jit_raw(distance=distance)
    if raw_df.empty or "min" not in raw_df.columns:
        print(f"[RQ3] No feasibility data found for distance='{distance}'.")
        return

    plot_df = raw_df[["Model", "Explainer", "min"]].copy()
    plot_df["min"] = pd.to_numeric(plot_df["min"], errors="coerce")
    plot_df.dropna(subset=["min"], inplace=True)
    plot_df["min_norm"] = plot_df["min"].clip(0, 1)

    # Only keep explainers we have data for
    explainers_present = [e for e in JIT_EXPLAINERS if e in plot_df["Explainer"].unique()]
    if not explainers_present:
        print("[RQ3] No explainers found in feasibility data.")
        return

    fig = plt.figure(figsize=(6.6, 4.8))
    sns.stripplot(
        data=plot_df,
        x="Explainer",
        y="min_norm",
        hue="Model",
        palette="crest",
        dodge=True,
        jitter=0.2,
        size=4,
        alpha=0.25,
        legend=False,
    )
    ax = sns.pointplot(
        data=plot_df,
        x="Explainer",
        y="min_norm",
        hue="Model",
        palette=["black"] * len(JIT_MODELS),
        dodge=0.6,
        errorbar=None,
        markers="x",
        markersize=4,
        linestyles="none",
        legend=False,
        zorder=10,
    )

    # Mean labels per (Model, Explainer)
    mean_df = (
        plot_df.groupby(["Model", "Explainer"], as_index=False)["min_norm"].mean()
    )
    offsets = np.linspace(-0.3, 0.3, len(JIT_MODELS))
    for _, row in mean_df.iterrows():
        model = row["Model"]
        expl = row["Explainer"]
        if expl not in explainers_present:
            continue
        m_idx = JIT_MODELS.index(model)
        e_idx = explainers_present.index(expl)
        x = e_idx + offsets[m_idx]
        y = float(row["min_norm"])
        label = f".{y:.2f}".replace("0.", ".")
        ax.text(
            x,
            min(max(y, 0.0), 1.0) + 0.01,
            label,
            va="bottom",
            ha="center",
            fontsize=10,
            fontfamily="monospace",
        )

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.0)
    ax.set_xticklabels(explainers_present, fontsize=11)
    ax.tick_params(axis="y", labelsize=11)

    # Legend for models
    colors = sns.color_palette("crest", len(JIT_MODELS))
    handles = []
    for i, m in enumerate(JIT_MODELS):
        h = plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=colors[i],
            markerfacecolor=colors[i],
            markeredgecolor="black",
            label=JIT_MODEL_LABELS.get(m, m),
        )
        handles.append(h)

    fig.legend(
        handles=handles,
        title="",
        loc="upper center",
        fontsize=10,
        frameon=False,
        ncols=len(JIT_MODELS),
        bbox_to_anchor=(0.5, 0.96),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[RQ3] Saved {out_path}")


# ----------------------------- Implications (amount of change) -----------------------------

def visualize_implications_jit(
    out_path: str = "./evaluations/jit_implications.png",
) -> None:
    """
    Boxplot of 'total amount of changes required' for each (Explainer, Model)
    in the JIT setting. Reads ./evaluations/abs_changes/{Model}_{Explainer}.csv
    created by eval_jit.py --implications.
    """
    _ensure_eval_dir()
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "Times New Roman"

    frames = []
    for model in JIT_MODELS:
        for expl in JIT_EXPLAINERS:
            # accept shards {model}_{expl}.csv or {model}_{expl}_*.csv
            main_path = f"./evaluations/abs_changes/{model}_{expl}.csv"
            s = _read_abs_scores(main_path)
            if s.empty:
                # check shards
                shard_paths = glob(f"./evaluations/abs_changes/{model}_{expl}_*.csv")
                parts = []
                for p in shard_paths:
                    sp = _read_abs_scores(p)
                    if not sp.empty:
                        parts.append(sp)
                if not parts:
                    print(f"[Implications] No data for {model}_{expl}")
                    continue
                s = pd.concat(parts, ignore_index=True)

            df = pd.DataFrame({"score": s})
            df["Model"] = model
            df["Explainer"] = expl
            frames.append(df)

    if not frames:
        print("[Implications] No abs_changes data found.")
        return

    total_df = pd.concat(frames, ignore_index=True)
    total_df["score"] = pd.to_numeric(total_df["score"], errors="coerce")
    total_df.dropna(subset=["score"], inplace=True)

    # Save the long-form data & a small summary (optional)
    out_long = "./evaluations/jit_implications_long.csv"
    out_summary = "./evaluations/jit_implications_summary.csv"
    total_df[["Model", "Explainer", "score"]].to_csv(out_long, index=False)
    (
        total_df.groupby(["Model", "Explainer"])["score"]
        .agg(N="count", mean="mean", median="median", std="std")
        .reset_index()
        .to_csv(out_summary, index=False)
    )
    print(f"[Implications] Saved {out_long}")
    print(f"[Implications] Saved {out_summary}")

    explainers_present = [e for e in JIT_EXPLAINERS if e in total_df["Explainer"].unique()]
    plt.figure(figsize=(6.4, 3.4))
    ax = sns.boxplot(
        data=total_df,
        x="Explainer",
        y="score",
        hue="Model",
        order=explainers_present,
        hue_order=JIT_MODELS,
        palette="crest",
        showfliers=False,
    )

    ax.set_ylabel("Total Amount of Changes Required", fontsize=12)
    ax.set_xlabel("")
    ax.set_xticklabels(explainers_present, fontsize=11)
    ax.tick_params(axis="y", labelsize=11)

    if ax.legend_ is not None:
        ax.legend_.set_title("")
        ax.legend_.set_loc = "upper right"

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Implications] Saved {out_path}")


# ----------------------------- CLI -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Plots for JIT experiments (RQ1/RQ2/RQ3/Implications)")
    parser.add_argument("--rq1", action="store_true", help="Generate RQ1 JIT flip-rate bar plot")
    parser.add_argument("--rq2", action="store_true", help="Generate RQ2 JIT plan-similarity histograms")
    parser.add_argument("--rq3", action="store_true", help="Generate RQ3 JIT feasibility plot")
    parser.add_argument("--implications", action="store_true", help="Generate JIT implications boxplot")
    args = parser.parse_args()

    if args.rq1:
        visualize_rq1_jit()
    if args.rq2:
        visualize_rq2_jit()
    if args.rq3:
        visualize_rq3_jit()
    if args.implications:
        visualize_implications_jit()


if __name__ == "__main__":
    main()
