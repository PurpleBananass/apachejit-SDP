#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQ3: Mahalanobis distance realism analysis with direct-labeled KDE plots.

- Build global "Actual History" deltas across selected projects
- For each explainer (incl. CF=DeFlip), compute Mahalanobis distance
  of each flip's change-vector to the history mean
- For each model, plot grayscale KDEs with:
    * Actual History: filled background
    * DeFlip: bold black line
    * Others: distinct gray linestyles
    * All labels / overlap values moved to legend (except Actual History name)
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.stats import gaussian_kde

from hyparams import EXPERIMENTS
from data_utils import read_dataset

# ----------------------------- basic maps -----------------------------

MODEL_ABBR = {
    "SVM": "SVM",
    "RandomForest": "RF",
    "XGBoost": "XGB",
    "LightGBM": "LGBM",
    "CatBoost": "CatB",
}

EXPLAINER_NAME_MAP = {
    "LIME": "LIME",
    "LIME-HPO": "LIME-HPO",
    "PyExplainer": "PyExplainer",
    "CfExplainer": "CfExplainer",
    "CF": "DeFlip",  # CF == DeFlip
}

# ----------------------------- CLI helpers -----------------------------

def parse_models(arg: str) -> List[str]:
    if not arg or arg.strip().lower() == "all":
        return ["RandomForest", "SVM", "XGBoost", "CatBoost", "LightGBM"]
    return [m.strip() for m in arg.replace(",", " ").split() if m.strip()]


def parse_projects(arg: str, ds_keys: List[str]) -> List[str]:
    if not arg or arg.strip().lower() == "all":
        return list(sorted(ds_keys))
    return [p.strip() for p in arg.replace(",", " ").split() if p.strip()]

# ----------------------------- flip file helpers -----------------------------

def _flip_path(project: str, model_type: str, explainer: str) -> Path:
    """
    Unified flip file paths.
    - CF → experiments/{project}/{model}/CF_all.csv
    - Others → experiments/{project}/{model}/{EXPLAINER}_all.csv
    """
    if explainer == "CF":
        return Path(EXPERIMENTS) / f"{project}/{model_type}/CF_all.csv"
    return Path(EXPERIMENTS) / f"{project}/{model_type}/{explainer}_all.csv"


def _load_flips_df(flip_path: Path, feat_cols: List[str]) -> Optional[pd.DataFrame]:
    """
    Load flips in *long* format:
      - expects a 'test_idx' column; if missing, try reading index_col=0 then reset.
      - keeps 'test_idx', optional 'candidate_id', and any present feature columns.
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
        + [c for c in ("candidate_id",) if c in df.columns]
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


def _first_candidate_per_test(flips: pd.DataFrame) -> pd.DataFrame:
    """Take one row per test_idx (first candidate)."""
    if "candidate_id" in flips.columns:
        flips = (
            flips.sort_values(["test_idx", "candidate_id"], kind="stable")
            .groupby("test_idx", as_index=False)
            .head(1)
        )
    else:
        flips = flips.drop_duplicates(subset=["test_idx"], keep="first")
    return flips

# ----------------------------- history distribution -----------------------------

def build_history_distribution(ds, project_list):
    """
    Build global history delta distribution across project_list.

    Returns:
      - total_deltas: (N x F) dataframe of deltas
      - feat_cols_md: list of feature names used
      - mean_vec: mean over total_deltas (length F)
      - inv_cov: pseudo-inverse covariance matrix (F x F)
    """
    # union of all feature names
    feat_union = set()
    for project in project_list:
        if project not in ds:
            continue
        train, test = ds[project]
        feat_union |= {c for c in train.columns if c != "target"}
        feat_union |= {c for c in test.columns if c != "target"}

    if not feat_union:
        raise RuntimeError("No features found across selected projects.")

    feat_union = sorted(feat_union)

    total_deltas_list = []
    for project in project_list:
        if project not in ds:
            continue
        train, test = ds[project]
        common_idx = train.index.intersection(test.index)
        if len(common_idx) == 0:
            continue

        t_cols = [c for c in test.columns if c != "target"]
        tr_cols = [c for c in train.columns if c != "target"]
        common_feats = sorted(set(t_cols) & set(tr_cols))
        if not common_feats:
            continue

        deltas = test.loc[common_idx, common_feats] - train.loc[common_idx, common_feats]
        deltas = deltas.reindex(columns=feat_union, fill_value=0.0)
        total_deltas_list.append(deltas)

    if not total_deltas_list:
        raise RuntimeError("No historical deltas could be built.")

    total_deltas = pd.concat(total_deltas_list, axis=0)

    # drop constant columns
    total_deltas = total_deltas.loc[:, total_deltas.nunique() > 1]
    feat_cols_md = list(total_deltas.columns)

    if len(feat_cols_md) == 0:
        raise RuntimeError("All features were constant; cannot compute Mahalanobis.")

    X = total_deltas.values.astype(float)
    mean_vec = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.pinv(cov)

    return total_deltas, feat_cols_md, mean_vec, inv_cov

# ----------------------------- Mahalanobis for flips -----------------------------

def compute_mahalanobis_scores(
    ds,
    project_list,
    model_type: str,
    explainer: str,
    feat_cols_md: List[str],
    mean_vec: np.ndarray,
    inv_cov: np.ndarray,
) -> pd.DataFrame:
    """
    For each flip (per project & explainer), compute Mahalanobis distance of
    the change-vector to the global history mean.
    """
    scores = []
    feat_index = {f: i for i, f in enumerate(feat_cols_md)}

    for project in project_list:
        if project not in ds:
            continue

        train, test = ds[project]
        feat_cols_proj = [c for c in test.columns if c != "target"]

        flip_path = _flip_path(project, model_type, explainer)
        flips = _load_flips_df(flip_path, feat_cols_proj)
        if flips is None or flips.empty:
            continue

        flips = _first_candidate_per_test(flips)

        for _, row in flips.iterrows():
            t = int(row["test_idx"])
            if t not in test.index:
                continue

            orig = test.loc[t, feat_cols_proj].astype(float)
            cand = orig.copy()
            present = [c for c in feat_cols_proj if c in row.index]
            if present:
                cand[present] = row[present].astype(float).values

            diff_series = cand - orig

            # skip zero-change
            if np.allclose(diff_series.values, 0.0, rtol=1e-7, atol=1e-7):
                continue

            delta_full = np.zeros(len(feat_cols_md), dtype=float)
            for f, val in diff_series.items():
                if f in feat_index:
                    delta_full[feat_index[f]] = float(val)

            if np.allclose(delta_full, 0.0, rtol=1e-7, atol=1e-7):
                continue

            d = float(mahalanobis(delta_full, mean_vec, inv_cov))
            scores.append(
                {"project": project, "test_idx": t, "distance": d}
            )

    return pd.DataFrame(scores)

# ----------------------------- overlap helper -----------------------------

def get_overlap_score(kde1, kde2, x_grid: np.ndarray) -> float:
    """Compute overlap between two KDEs on a common x_grid."""
    y1 = kde1(x_grid)
    y2 = kde2(x_grid)
    return float(np.trapz(np.minimum(y1, y2), x_grid))

# ----------------------------- plotting (legend-based, log scale) -----------------------------

def plot_rq3_direct_labels(dist_data_dict, model_abbr, save_path):
    # squished vertically
    fig, ax = plt.subplots(figsize=(12, 4))

    # keep full KDE range 0~30 (no stretching)
    x_grid = np.linspace(0.0, 30.0, 1000)

    # KDE for history (baseline)
    hist_vals = np.asarray(dist_data_dict["Actual History"], float)
    hist_vals = hist_vals[np.isfinite(hist_vals) & (hist_vals >= 0)]
    if hist_vals.size < 2:
        print(f"[RQ3] Not enough Actual History points to plot for {model_abbr}.")
        plt.close()
        return

    kde_history = gaussian_kde(hist_vals)
    y_hist = kde_history(x_grid)

    styles = {
        "Actual History": {"c": "#cccccc", "ls": "-",  "lw": 1.0, "fill": True,  "z": 1},
        "CfExplainer":     {"c": "#999999", "ls": "--", "lw": 1.5, "fill": False, "z": 3},
        "DeFlip":         {"c": "black",   "ls": "-",  "lw": 2.5, "fill": False, "z": 5},
        "PyExplainer":       {"c": "#666666", "ls": "-.", "lw": 1.5, "fill": False, "z": 2},
        "LIME":           {"c": "#555555", "ls": ":",  "lw": 1.5, "fill": False, "z": 2},
        "LIME-HPO":       {"c": "#999999", "ls": "-.", "lw": 1.5, "fill": False, "z": 2},
    }

    plot_order = ["Actual History", "CfExplainer",
                  "DeFlip", "PyExplainer", "LIME", "LIME-HPO"]
    plot_order = [n for n in plot_order if n in dist_data_dict]

    handles, labels = [], []

    # ----- Actual History -----
    st_hist = styles["Actual History"]
    ax.fill_between(x_grid, y_hist, color=st_hist["c"], alpha=0.4, zorder=st_hist["z"])
    ax.plot(x_grid, y_hist, color="#aaaaaa", lw=1.0, zorder=st_hist["z"])

    # label very close to the curve
    peak_idx = int(np.argmax(y_hist))
    x_peak = x_grid[peak_idx]
    y_peak = y_hist[peak_idx]

    ax.text(
        x_peak,
        y_peak * 1.02,
        "Actual History\n(ref)",
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1),
    )

    # ----- other explainers -----
    for name in plot_order:
        if name == "Actual History":
            continue

        data = np.asarray(dist_data_dict[name], dtype=float)
        data = data[np.isfinite(data) & (data >= 0)]
        if len(data) < 2:
            continue

        kde = gaussian_kde(data)
        y_vals = kde(x_grid)
        st = styles[name]

        line_handle, = ax.plot(
            x_grid, y_vals,
            color=st["c"], linestyle=st["ls"],
            linewidth=st["lw"], zorder=st["z"]
        )

        ov_score = get_overlap_score(kde, kde_history, x_grid)
        if name == "CfExplainer":
            legend_label = f"{name} (Overlap: {ov_score:.2f})"
        else:
            legend_label = f"{name}"

        handles.append(line_handle)
        labels.append(legend_label)

    # ---- axes styling: cut at 20 ----
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Mahalanobis Distance", fontsize=12, fontweight="bold")
    ax.set_ylabel("Probability Density", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- shrink plot width to make right margin ----
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

    # ---- legend in the empty right margin (not on top of curves) ----
    if handles:
        ax.legend(
            handles,
            labels,
            frameon=True,
            framealpha=1.0,
            edgecolor="black",
            loc="center left",
            bbox_to_anchor=(0.5, 0.55),  # outside axes, in right margin
            fontsize=10,
        )

    # ---- model label in right margin above legend ----
    fig.text(
        0.86, 0.9,           # x,y in figure coords (right area)
        model_abbr,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[Graph] Saved to {save_path}")
    plt.close()

def plot_rq3_on_axis(ax, dist_data_dict, model_abbr):
    """
    Draw one Mahalanobis KDE plot for a single model onto the given axis.
    Legend is placed in the middle-right white space (inside the axes).
    """
    x_grid = np.linspace(0.0, 30.0, 1000)

    # --- history KDE ---
    hist_vals = np.asarray(dist_data_dict["Actual History"], float)
    hist_vals = hist_vals[np.isfinite(hist_vals) & (hist_vals >= 0)]
    if hist_vals.size < 2:
        print(f"[RQ3] Not enough Actual History points to plot for {model_abbr}.")
        return

    kde_history = gaussian_kde(hist_vals)
    y_hist = kde_history(x_grid)

    styles = {
        "Actual History": {"c": "#cccccc", "ls": "-",  "lw": 1.0, "fill": True,  "z": 1},
        "CfExplainer":     {"c": "#999999", "ls": "--", "lw": 1.5, "fill": False, "z": 3},
        "DeFlip":         {"c": "black",   "ls": "-",  "lw": 2.5, "fill": False, "z": 5},
        "PyExplainer":       {"c": "#666666", "ls": "-.", "lw": 1.5, "fill": False, "z": 2},
        "LIME":           {"c": "#555555", "ls": ":",  "lw": 1.5, "fill": False, "z": 2},
        "LIME-HPO":       {"c": "#999999", "ls": "-.", "lw": 1.5, "fill": False, "z": 2},
    }

    plot_order = ["Actual History", "CfExplainer",
                  "DeFlip", "PyExplainer", "LIME", "LIME-HPO"]
    plot_order = [n for n in plot_order if n in dist_data_dict]

    handles, labels = [], []

    # --- Actual History (filled) ---
    st_hist = styles["Actual History"]
    ax.fill_between(x_grid, y_hist, color=st_hist["c"], alpha=0.4, zorder=st_hist["z"])
    ax.plot(x_grid, y_hist, color="#aaaaaa", lw=1.0, zorder=st_hist["z"])

    # label close to the peak
    peak_idx = int(np.argmax(y_hist))
    x_peak = x_grid[peak_idx]
    y_peak = y_hist[peak_idx]
    ax.text(
        x_peak,
        y_peak * 1.02,
        "Actual History\n(ref)",
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1),
    )

    # --- other explainers ---
    for name in plot_order:
        if name == "Actual History":
            continue

        data = np.asarray(dist_data_dict[name], dtype=float)
        data = data[np.isfinite(data) & (data >= 0)]
        if len(data) < 2:
            continue

        kde = gaussian_kde(data)
        y_vals = kde(x_grid)
        st = styles[name]

        line_handle, = ax.plot(
            x_grid, y_vals,
            color=st["c"], linestyle=st["ls"],
            linewidth=st["lw"], zorder=st["z"],
        )

        ov_score = get_overlap_score(kde, kde_history, x_grid)
        if name == "CfExplainer":
            legend_label = f"{name} (Overlap: {ov_score:.2f})"
        else:
            legend_label = f"{name} ({ov_score:.2f})"

        handles.append(line_handle)
        labels.append(legend_label)

    # --- axes styling ---
    ax.set_xlim(0.0, 20.0)          # cut at 20
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # model label at top-right of the subplot
    ax.text(
        0.98, 0.92,
        model_abbr,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
    )

    # LEGEND: inside the axes, middle-right where curves are almost flat
    # (axes coords: x≈0.7~0.75 corresponds to ~14–15 on 0–20 range)
    if handles:
        ax.legend(
            handles,
            labels,
            frameon=True,
            framealpha=1.0,
            edgecolor="black",
            loc="center",              # "center" of the anchor box below
            bbox_to_anchor=(0.6, 0.75),  # middle-right inside the axes
            fontsize=9,
        )

def plot_rq3_vertical(all_dist_dicts, save_path):
    model_abbrs = list(all_dist_dicts.keys())
    n_models = len(model_abbrs)

    # fig, axes = plt.subplots(
    #     nrows=n_models,
    #     ncols=1,
    #     figsize=(12, 2.4 * n_models),   # tall but squished rows
    #     sharex=True,
    # )
    if n_models == 1:
        axes = [axes]

    fig, axes = plt.subplots(
        nrows=n_models,
        ncols=1,
        figsize=(6, 2.4 * n_models),   # width from 12 -> 8
        sharex=True,
    )

    fig.subplots_adjust(
        left=0.12, right=0.98, top=0.95, bottom=0.10, hspace=0.35
    )


    for ax, model_abbr in zip(axes, model_abbrs):
        dist_data_dict = all_dist_dicts[model_abbr]
        plot_rq3_on_axis(ax, dist_data_dict, model_abbr)

    axes[-1].set_xlabel("Mahalanobis Distance", fontsize=12, fontweight="bold")
    for ax in axes[:-1]:
        ax.set_xlabel("")

    fig.text(
        0.02, 0.5,
        "Probability Density",
        va="center",
        ha="left",
        rotation="vertical",
        fontsize=12,
        fontweight="bold",
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[Graph] Saved combined figure to {save_path}")
    plt.close()



# ----------------------------- main -----------------------------

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--models",
        type=str,
        default="RandomForest",
        help='Models spaced/comma (e.g., "RandomForest,SVM") or "all"',
    )
    ap.add_argument(
        "--explainer",
        type=str,
        default="all",
        help=('Explainers spaced/comma (e.g., "CF LIME") or "all" -> '
              'LIME LIME-HPO PyExplainer CfExplainer CF'),
    )
    ap.add_argument(
        "--projects",
        type=str,
        default="all",
        help='Projects spaced/comma or "all"',
    )

    args = ap.parse_args()

    ds = read_dataset()
    all_projects = sorted(ds.keys())

    model_types = parse_models(args.models)

    if args.explainer.strip().lower() == "all":
        explainer_raw = ["LIME", "LIME-HPO", "PyExplainer",
                         "CfExplainer", "CF"]
    else:
        explainer_raw = [
            e for e in args.explainer.replace(",", " ").split() if e.strip()
        ]
    explainers = [e for e in explainer_raw]

    project_list = parse_projects(args.projects, all_projects)

    print(f"Projects ({len(project_list)}): {project_list}")
    print(f"Models: {model_types}")
    print(f"Explainers(raw): {explainers}\n")

    out_root = Path("./evaluations/rq3_mahalanobis")
    out_root.mkdir(parents=True, exist_ok=True)

    # Build global history distribution once
    total_deltas, feat_cols_md, mean_vec, inv_cov = build_history_distribution(
        ds, project_list
    )

    # Mahalanobis distances for "Actual History"
    # Mahalanobis distances for "Actual History"
    hist_vals = []
    X_hist = total_deltas[feat_cols_md].values.astype(float)
    for i in range(X_hist.shape[0]):
        hist_vals.append(float(mahalanobis(X_hist[i], mean_vec, inv_cov)))
    hist_vals = np.asarray(hist_vals, dtype=float)

    # collect all models' distributions
    all_dist_dicts: Dict[str, Dict[str, np.ndarray]] = {}

    for model_type in model_types:
        model_abbr = MODEL_ABBR.get(model_type, model_type)
        print(f"\n=== Model: {model_abbr} ===")

        dist_dict: Dict[str, np.ndarray] = {"Actual History": hist_vals}

        for explainer in explainers:
            disp_name = EXPLAINER_NAME_MAP.get(explainer, explainer)
            csv_path = out_root / f"{model_abbr}_{disp_name}.csv"

        # for explainer in explainers:
        #     disp_name = EXPLAINER_NAME_MAP.get(explainer, explainer)
        #     csv_path = out_root / f"{model_abbr}_{disp_name}.csv"

            # ----------- reuse existing CSV if available -----------
            if csv_path.exists() and csv_path.stat().st_size > 0:
                try:
                    df_scores = pd.read_csv(csv_path)
                    if "distance" not in df_scores.columns:
                        print(f"[RQ3] {csv_path} missing 'distance' column; recomputing.")
                        df_scores = compute_mahalanobis_scores(
                            ds,
                            project_list,
                            model_type,
                            explainer,
                            feat_cols_md,
                            mean_vec,
                            inv_cov,
                        )
                        if df_scores.empty:
                            print(f"[RQ3] {model_abbr}/{disp_name}: no flips with non-zero changes.")
                            continue
                        df_scores.to_csv(csv_path, index=False)
                    else:
                        print(
                            f"[RQ3] Using cached distances for {model_abbr}/{disp_name} "
                            f"({len(df_scores)} flips) from {csv_path}"
                        )
                except Exception:
                    # If anything goes wrong, recompute and overwrite
                    print(f"[RQ3] Failed to read {csv_path}; recomputing.")
                    df_scores = compute_mahalanobis_scores(
                        ds,
                        project_list,
                        model_type,
                        explainer,
                        feat_cols_md,
                        mean_vec,
                        inv_cov,
                    )
                    if df_scores.empty:
                        print(f"[RQ3] {model_abbr}/{disp_name}: no flips with non-zero changes.")
                        continue
                    df_scores.to_csv(csv_path, index=False)
            else:
                # ----------- compute and save if not cached -----------
                df_scores = compute_mahalanobis_scores(
                    ds,
                    project_list,
                    model_type,
                    explainer,
                    feat_cols_md,
                    mean_vec,
                    inv_cov,
                )
                if df_scores.empty:
                    print(f"[RQ3] {model_abbr}/{disp_name}: no flips with non-zero changes.")
                    continue

                df_scores.to_csv(csv_path, index=False)
                print(
                    f"[RQ3] Saved distances for {model_abbr}/{disp_name} "
                    f"({len(df_scores)} flips) -> {csv_path}"
                )

            dist_dict[disp_name] = df_scores["distance"].values.astype(float)

        if len(dist_dict) > 1:
            all_dist_dicts[model_abbr] = dist_dict
        else:
            print(f"[RQ3] Not enough data to plot for model {model_abbr}.")

    # one combined vertical figure
    if all_dist_dicts:
        png_path = out_root / "rq3_mahalanobis_all_models_vertical.png"
        plot_rq3_vertical(all_dist_dicts, save_path=str(png_path))
    else:
        print("[RQ3] No models had enough data to plot.")