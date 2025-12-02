#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute L1 norm of changes (in z-scored feature space) for all explainers,
including CF (DeFlip), plot boxplots, and run **unpaired** Mann–Whitney U
tests vs DeFlip.

Outputs (per model, where MODEL is RF, XGB, etc.):

  - ./evaluations/norms/{MODEL}_L1.csv
      columns: Model, Explainer, Project, TestIdx, Score

  - ./evaluations/norms/{MODEL}_norms.png
      boxplots for L1 per explainer

  - ./evaluations/norms/{MODEL}_stats.csv
      unpaired Mann–Whitney vs DeFlip (L1 only):
      Model, Metric, Baseline, Explainer, N_baseline, N_other,
      p_value, cliffs_delta, cd_magnitude

Additionally:

  - ./evaluations/norms/AllModels_norms.png
      one figure with subplots for all models.
"""

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
from cliffs_delta import cliffs_delta

from hyparams import EXPERIMENTS
from data_utils import read_dataset


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

MODEL_ABBR = {
    "SVM": "SVM",
    "RandomForest": "RF",
    "XGBoost": "XGB",
    "LightGBM": "LGBM",
    "CatBoost": "CatB",
}

# Display names – CF => DeFlip; SQAPlanner_confidence => SQAPlanner
EXPLAINER_NAME_MAP = {
    "LIME": "LIME",
    "LIME-HPO": "LIME-HPO",
    "PyExplainer": "PyExplainer",
    "CfExplainer": "CfExplainer",
    "CF": "DeFlip",
}


def parse_models(arg: str) -> List[str]:
    if not arg or arg.strip().lower() == "all":
        return ["RandomForest", "SVM", "XGBoost", "CatBoost", "LightGBM"]
    return [m.strip() for m in arg.replace(",", " ").split() if m.strip()]


def parse_projects(arg: str, ds_keys: List[str]) -> List[str]:
    if not arg or arg.strip().lower() == "all":
        return list(sorted(ds_keys))
    return [p.strip() for p in arg.replace(",", " ").split() if p.strip()]


def _flip_path(project: str, model_type: str, explainer: str) -> Path:
    """
    Same convention as in your main evaluation script.
      CF → experiments/{project}/{model}/CF_all.csv
      others → experiments/{project}/{model}/{EXPLAINER}_all.csv
    """
    if explainer == "CF":
        return Path(EXPERIMENTS) / f"{project}/{model_type}/CF_all.csv"
    return Path(EXPERIMENTS) / f"{project}/{model_type}/{explainer}_all.csv"


def _load_flips_df(flip_path: Path, feat_cols: List[str]) -> Optional[pd.DataFrame]:
    """
    CF / long-format loader from your evaluation script.
    Keeps: test_idx, optional candidate_id, and any feature columns present.
    """
    if not flip_path.exists() or flip_path.stat().st_size == 0:
        return None

    try:
        df = pd.read_csv(flip_path)
        if "test_idx" not in df.columns:
            df = (
                pd.read_csv(flip_path, index_col=0)
                .reset_index()
                .rename(columns={"index": "test_idx"})
            )
    except Exception:
        return None

    if df is None or df.empty or "test_idx" not in df.columns:
        return None

    keep = (
        ["test_idx"]
        + ([c for c in ("candidate_id",) if c in df.columns])
        + [c for c in feat_cols if c in df.columns]
    )
    df = df.loc[:, [c for c in keep if c in df.columns]].copy()
    df["test_idx"] = pd.to_numeric(df["test_idx"], errors="coerce")
    df = df.dropna(subset=["test_idx"]).copy()
    df["test_idx"] = df["test_idx"].astype(int)

    if "candidate_id" in df.columns:
        df = df.sort_values(["test_idx", "candidate_id"], kind="stable")
    else:
        df = df.sort_values(["test_idx"], kind="stable")
    return df


def _first_candidates(cf_df: pd.DataFrame) -> pd.DataFrame:
    """Pick first candidate per test_idx."""
    if "candidate_id" in cf_df.columns:
        return (
            cf_df.sort_values(["test_idx", "candidate_id"], kind="stable")
            .groupby("test_idx", as_index=False)
            .head(1)
        )
    return cf_df.drop_duplicates(subset=["test_idx"], keep="first")


# ------------------------------------------------------------------ #
# Core: compute L1 norms (and keep per-(project,test_idx) for CSV)
# ------------------------------------------------------------------ #

def collect_norms_for_model(
    model_type: str,
    explainers: List[str],
    projects: List[str],
) -> Tuple[
    Dict[str, List[float]],
    Dict[str, int],
    Dict[str, Dict[Tuple[str, int], float]],
]:
    """
    Returns:
      - l1_scores[disp] -> list of L1 norms (all projects / test_idx)
      - succ_counts[disp] -> # of successful flips counted
      - l1_by_id[disp][(project, test_idx)] -> L1 norm  (for per-instance CSV)
    """
    ds = read_dataset()

    l1_scores = defaultdict(list)
    succ_counts = defaultdict(int)
    l1_by_id: Dict[str, Dict[Tuple[str, int], float]] = defaultdict(dict)

    for project in projects:
        if project not in ds:
            continue

        train, test = ds[project]
        feat_cols = [c for c in test.columns if c != "target"]

        # scaler on train
        scaler = StandardScaler().fit(train[feat_cols].values)

        for explainer in explainers:
            disp_name = EXPLAINER_NAME_MAP.get(explainer, explainer)
            flip_path = _flip_path(project, model_type, explainer)

            if not flip_path.exists():
                continue

            # ---------------------- CF / DeFlip ---------------------- #
            if explainer == "CF":
                cf_df = _load_flips_df(flip_path, feat_cols)
                if cf_df is None or cf_df.empty:
                    continue
                cf_df = _first_candidates(cf_df)

                for _, row in cf_df.iterrows():
                    t = int(row["test_idx"])
                    if t not in test.index:
                        continue

                    orig = test.loc[t, feat_cols].astype(float)
                    cand = orig.copy()
                    present = [c for c in feat_cols if c in row.index]
                    cand[present] = row[present].astype(float).values

                    diff = cand.values - orig.values
                    if np.allclose(diff, 0.0, rtol=1e-7, atol=1e-7):
                        continue

                    z_orig = scaler.transform([orig.values])[0]
                    z_cand = scaler.transform([cand.values])[0]
                    dz = z_cand - z_orig

                    l1 = float(np.sum(np.abs(dz)))

                    l1_scores[disp_name].append(l1)
                    succ_counts[disp_name] += 1

                    key = (project, t)
                    l1_by_id[disp_name][key] = l1

                continue  # done with CF for this project

            # ---------------------- Non-CF explainers ---------------------- #
            # flip files: wide format, index=test_idx, columns=features
            try:
                flipped_full = pd.read_csv(flip_path, index_col=0).dropna()
            except Exception:
                continue
            if flipped_full.empty:
                continue

            for t_raw in flipped_full.index:
                t = int(t_raw)
                if t not in test.index:
                    continue

                orig = test.loc[t, feat_cols].astype(float)
                cand = flipped_full.loc[t_raw, feat_cols].astype(float)

                diff = cand.values - orig.values
                if np.allclose(diff, 0.0, rtol=1e-7, atol=1e-7):
                    continue

                z_orig = scaler.transform([orig.values])[0]
                z_cand = scaler.transform([cand.values])[0]
                dz = z_cand - z_orig

                l1 = float(np.sum(np.abs(dz)))

                l1_scores[disp_name].append(l1)
                succ_counts[disp_name] += 1

                key = (project, t)
                l1_by_id[disp_name][key] = l1

    return l1_scores, succ_counts, l1_by_id


# ------------------------------------------------------------------ #
# Unpaired Mann–Whitney vs DeFlip + Cliff's delta
# ------------------------------------------------------------------ #

def compute_unpaired_stats(
    model_abbr: str,
    l1_scores: Dict[str, List[float]],
    min_n: int = 5,
) -> pd.DataFrame:
    """
    For each explainer vs DeFlip, run unpaired Mann–Whitney U test (L1 only).

    Returns a DataFrame with:
      Model, Metric, Baseline, Explainer,
      N_baseline, N_other,
      p_value,
      cliffs_delta, cd_magnitude
    """
    baseline = "DeFlip"
    if baseline not in l1_scores or len(l1_scores[baseline]) < min_n:
        print(f"[stats] No sufficient DeFlip data for {model_abbr}; skipping stats.")
        return pd.DataFrame(
            columns=[
                "Model",
                "Metric",
                "Baseline",
                "Explainer",
                "N_baseline",
                "N_other",
                "p_value",
                "cliffs_delta",
                "cd_magnitude",
            ]
        )

    base_vals = np.asarray(l1_scores[baseline], dtype=float)
    base_vals = base_vals[np.isfinite(base_vals)]

    rows = []
    for expl, vals in l1_scores.items():
        if expl == baseline:
            continue

        other_vals = np.asarray(vals, dtype=float)
        other_vals = other_vals[np.isfinite(other_vals)]

        n_b = base_vals.size
        n_o = other_vals.size
        if n_b < min_n or n_o < min_n:
            continue

        # Mann–Whitney U (independent)
        _, p_val = mannwhitneyu(
            other_vals, base_vals, alternative="two-sided"
        )

        cd, cd_mag = cliffs_delta(other_vals.tolist(), base_vals.tolist())

        rows.append(
            [
                model_abbr,
                "L1",
                baseline,
                expl,
                int(n_b),
                int(n_o),
                float(p_val),
                float(cd),
                cd_mag,
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Metric",
            "Baseline",
            "Explainer",
            "N_baseline",
            "N_other",
            "p_value",
            "cliffs_delta",
            "cd_magnitude",
        ],
    )

    # round numeric stats to 2 decimals
    if not df.empty:
        df["p_value"] = df["p_value"].round(2)
        df["cliffs_delta"] = df["cliffs_delta"].round(2)

    return df


# ------------------------------------------------------------------ #
# Plotting (L1 only)
# ------------------------------------------------------------------ #

def plot_norms_for_model(
    model_abbr: str,
    l1_scores: dict,
    succ_counts: dict,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Generate boxplots for L1 norm.
    DeFlip is highlighted in dark gray; others in light gray.
    Annotates each box with its median value placed on the median line.
    If `ax` is provided, draws into that axis (for combined figure).
    """

    # Put DeFlip first
    desired_order = ["DeFlip", "CfExplainer", "PyExplainer", "LIME-HPO", "LIME"]
    labels = [lab for lab in desired_order if lab in l1_scores and l1_scores[lab]]

    if not labels:
        print(f"[plot] No data to plot for {model_abbr}. Skipping.")
        return

    data_l1 = [l1_scores[lab] for lab in labels]
    n_counts = [succ_counts[lab] for lab in labels]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        created_fig = True
    else:
        fig = ax.figure

    bplot = ax.boxplot(
        data_l1,
        patch_artist=True,
        notch=True,
        vert=True,
        widths=0.9,
        showfliers=False,
    )

    # fill colors (DeFlip dark)
    fill_colors = []
    for lab in labels:
        if lab == "DeFlip":
            fill_colors.append("#555555")
        else:
            fill_colors.append("#E0E0E0")

    for patch, color in zip(bplot["boxes"], fill_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)
        patch.set_alpha(1.0)

    # median: white for DeFlip, black otherwise
    for i, median in enumerate(bplot["medians"]):
        color = "white" if labels[i] == "DeFlip" else "black"
        median.set(color=color, linewidth=1.5)

    # annotate median values ON the median line with white box
    for i, vals in enumerate(data_l1):
        x = i + 1
        med_y = float(np.median(vals))

        ax.text(
            x,
            med_y,
            f"{med_y:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=1.0,
                pad=1.0,
            ),
            zorder=5,
        )

        # single-line labels with counts in parentheses
    new_labels = [f"{lab} (n={n:,})" for lab, n in zip(labels, n_counts)]

    ax.set_xticklabels(
        new_labels,
        fontsize=8,
        fontweight="normal",
        color="black",
        rotation=30,   # slanted labels
    )

    # make sure alignment matches the rotation
    for tick in ax.get_xticklabels():
        tick.set_ha("right")

    ax.set_ylabel(
        "Magnitude of Change",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(f"{model_abbr}", fontsize=14, pad=15)

    ax.yaxis.grid(True, linestyle="-", which="major", color="#cccccc", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if created_fig and save_path is not None:
        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        print(f"[plot] Saved figure -> {save_path}")
        plt.close(fig)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--models",
        type=str,
        default="RandomForest,SVM,XGBoost,CatBoost,LightGBM",
        help='Models spaced/comma (e.g., "RandomForest XGBoost") or "all"',
    )
    ap.add_argument(
        "--explainer",
        type=str,
        default="LIME LIME-HPO PyExplainer CfExplainer CF",
        help='Explainers spaced/comma (e.g., "LIME LIME-HPO SQAPlanner_confidence CF")',
    )
    ap.add_argument(
        "--projects",
        type=str,
        default="all",
        help='Projects spaced/comma or "all"',
    )

    args = ap.parse_args()

    ds = read_dataset()
    all_projects = list(sorted(ds.keys()))

    model_types = parse_models(args.models)
    explainers = [e for e in args.explainer.replace(",", " ").split() if e.strip()]
    project_list = parse_projects(args.projects, all_projects)

    out_dir = Path("./evaluations/norms")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Projects ({len(project_list)}): {project_list}")
    print(f"Models: {model_types}")
    print(f"Explainers(raw): {explainers}\n")

    # store results for combined plot
    combined_results: Dict[str, Tuple[dict, dict]] = {}

    for model_type in model_types:
        model_abbr = MODEL_ABBR.get(model_type, model_type)
        print(f"\n=== Processing model: {model_abbr} ===")

        l1_scores, succ_counts, l1_by_id = collect_norms_for_model(
            model_type, explainers, project_list
        )

        combined_results[model_abbr] = (l1_scores, succ_counts)

        # ---------------- save per-instance L1 CSV ----------------
        rows_L1 = []
        for expl_disp, id_dict in l1_by_id.items():
            for (proj, tid), v in id_dict.items():
                rows_L1.append([model_abbr, expl_disp, proj, tid, v])

        if rows_L1:
            df_L1 = pd.DataFrame(
                rows_L1, columns=["Model", "Explainer", "Project", "TestIdx", "Score"]
            )
            L1_path = out_dir / f"{model_abbr}_L1.csv"
            df_L1.to_csv(L1_path, index=False)
            print(f"[save] L1 scores -> {L1_path}")

        # ---------------- unpaired stats vs DeFlip (L1 only) ----------------
        stats_df = compute_unpaired_stats(model_abbr, l1_scores)
        if not stats_df.empty:
            stats_path = out_dir / f"{model_abbr}_stats.csv"
            stats_df.to_csv(stats_path, index=False)
            print(f"[stats] Saved unpaired Mann–Whitney stats -> {stats_path}")
        else:
            print(f"[stats] No usable unpaired data for DeFlip comparisons in {model_abbr}.")

        # ---------------- plot L1 only (per model) ----------------
        fig_path = out_dir / f"{model_abbr}_norms.png"
        plot_norms_for_model(model_abbr, l1_scores, succ_counts, save_path=str(fig_path))

    # ------------------------------------------------------------------ #
    # Combined figure with one subplot per model
    # ------------------------------------------------------------------ #
    if combined_results:
        model_order = [MODEL_ABBR.get(m, m) for m in model_types]
        # filter only models that actually have data
        model_order = [m for m in model_order if m in combined_results]

        n_models = len(model_order)
        if n_models > 0:
            fig, axes = plt.subplots(
                1,
                n_models,
                figsize=(4.5 * n_models, 6),
                sharey=True,
            )
            if n_models == 1:
                axes = [axes]

            for i, (ax, model_abbr) in enumerate(zip(axes, model_order)):
                l1_scores, succ_counts = combined_results[model_abbr]
                plot_norms_for_model(
                    model_abbr,
                    l1_scores,
                    succ_counts,
                    save_path=None,
                    ax=ax,
                )
                if i > 0:
                    ax.set_ylabel("")  # keep y-label only on first subplot

            fig.tight_layout()
            combined_path = out_dir / "AllModels_norms.png"
            fig.savefig(combined_path, dpi=300)
            print(f"[plot] Saved combined figure -> {combined_path}")
            plt.close(fig)
