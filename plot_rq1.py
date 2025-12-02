#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================================================
# [설정] – explainer 토큰 → (표시 이름, Cost, Label offset)
# =========================================================
COST_AND_OFFSETS = {
    "CF":         ("DeFlip",     0.00, (0.04, 0)),   # our method
    "LIME-HPO":   ("LIME-HPO",   0.95, (0.03, 0)),
    "LIME":       ("LIME",       0.90, (0.03, -4)),
    "PyExplainer":   ("PyExplainer",   0.40, (0.03, 0)),
    "CfExplainer": ("CfExplainer", 0.85, (0.03, 0)),
}

MARKERS_BY_TOOL = {
    "DeFlip": "o",
    "LIME": "^",
    "LIME-HPO": "s",
    "PyExplainer": "D",
    "CfExplainer": "P",
}

# (minimal_size, full_size) per explainer
MARKER_SIZES = {
    "DeFlip":     (20, 20),
    "LIME":       (25, 25),
    "LIME-HPO":   (25, 25),
    "PyExplainer":   (18, 18),
    "CfExplainer": (40, 40),
}

# =========================================================
# [단일 축에 그리기]
# =========================================================
def _plot_single_axis(ax, model_name, data):
    # background + vertical guides (unchanged)
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.grid(True, which='major', axis='x', linestyle=':', alpha=0.3, zorder=0)
    ax.axvline(0, color='black', linewidth=1, alpha=0.2, zorder=0)
    for v in [0.25, 0.5, 0.75]:
        ax.axvline(v, color='black', linestyle=':', linewidth=0.8, alpha=0.3, zorder=0)

    EDGE_COLOR = 'black'
    START_FILL = 'white'
    FULL_FILL  = "0.5"   # gray

    for tool_name, val in data.items():
        start_rate, end_rate, cost, offset = val
        marker = MARKERS_BY_TOOL.get(tool_name, "o")
        size_start, size_end = MARKER_SIZES.get(tool_name, (40, 24))

        # ---------------- DeFlip: one gray circle with border ----------------
        if tool_name == "DeFlip":
            ax.scatter(
                cost, end_rate,
                s=size_end,
                marker=marker,
                facecolors=FULL_FILL,
                edgecolors=EDGE_COLOR,   # border ON for DeFlip
                linewidth=1.0,
                zorder=7,
            )
            ax.text(
                cost,
                end_rate + 4.0,
                f"{end_rate:.0f}",
                fontsize=7.0,
                ha="center",
                va="bottom",
                color="black",
            )
            continue  # skip generic minimal/full logic

        # ---------------- minimal impl. (x = 0) ----------------
        ax.scatter(
            0, start_rate,
            s=size_start,
            marker=marker,
            facecolors=START_FILL,
            edgecolors=EDGE_COLOR,
            linewidth=1.0,
            zorder=6,
        )

        start_label = f"{start_rate:.0f}"

        # horizontal position for minimal labels
        if tool_name in ("LIME", "CfExplainer"):
            min_x = -0.06
            min_ha = "right"
        else:
            min_x = 0.07
            min_ha = "left"

        # *** NEW: vertical offsets to avoid overlaps at x = 0 ***
        if tool_name == "LIME-HPO" and model_name in ("LGBM", "RF", "XGB"):
            min_y = start_rate - 1.5       # nudge LIME up a bit
        elif tool_name == "PyExplainer":
            min_y = start_rate + 1.5       # nudge CfExplainer down
        else:
            min_y = start_rate

        ax.text(
            min_x,
            min_y,
            start_label,
            fontsize=7.0,
            ha=min_ha,
            va="center",
            color="black",
        )

        # ---------------- full impl. (x = cost) ----------------
        ax.scatter(
            cost, end_rate,
            s=size_end,
            marker=marker,
            facecolors=FULL_FILL,
            edgecolors="none",   # NO border for others
            linewidth=0.0,
            zorder=7,
        )

        end_label = f"{end_rate:.0f}"

        if tool_name == "LIME-HPO":
            ax.text(
                cost, end_rate + 4.0,
                end_label,
                fontsize=7.0,
                ha="center",
                va="bottom",
                color="black",
            )
        else:
            ax.text(
                cost, end_rate - 5.0,
                end_label,
                fontsize=7.0,
                ha="center",
                va="top",
                color="black",
            )

    # axes cosmetics – keep your existing code
    LEFT_PAD = 0.3
    RIGHT_PAD = 0.07
    ax.set_xlim(-LEFT_PAD, 1.0 + RIGHT_PAD)
    ax.set_ylim(0, 115)

    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(
        ["0.0\n", "0.5", "1.0\n"],
        fontsize=7.5
    )
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_title(f"{model_name}", fontsize=9.5, pad=6)


# =========================================================
# [전체 모델을 하나의 figure에 가로로 배치]
# =========================================================
def plot_all_models(experiments, save_path):
    # model_names = sorted(experiments.keys())
    desired_order = ["RF", "SVM", "XGB", "CatB", "LGBM"]
    model_names = [m for m in desired_order if m in experiments]

    n_models = len(model_names)
    if n_models == 0:
        print("[WARN] No models to plot.")
        return

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(10, 3.0),
        sharey=True,
    )
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        _plot_single_axis(ax, model_name, experiments[model_name])

    # y-label only on first axis
    axes[0].set_ylabel(
        "Predicted Defect Flip Rate (%)",
        fontsize=9.5,
        fontweight="bold",
        labelpad=10,
    )
    for ax in axes[1:]:
        ax.set_ylabel("")

    # common x-label
    fig.text(
        0.5, 0.12,   # was -0.02 → closer to axes
        "Normalized Exploration Depth (0=Start, 1=Limit)",
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
    )

    # fig.subplots_adjust(top=0.76, bottom=0.12, left=0.08, right=0.99, wspace=0.18)


    # ---------- build handles: shapes per explainer ----------
    # Legend 1: MINIMAL (hollow)
        # ---------- build handles: shapes per explainer ----------
    # Legend 1: MINIMAL (hollow)
    minimal_handles = []
    for expl_name, marker in MARKERS_BY_TOOL.items():
        # get per-explainer sizes (fallback to 40,24)
        size_start, size_end = MARKER_SIZES.get(expl_name, (40, 24))
        ms = (size_start ** 0.5)  # convert scatter area -> legend markersize

        minimal_handles.append(
            Line2D(
                [0], [0],
                marker=marker,
                linestyle="None",
                color="w",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.0,
                markersize=ms,
                label=expl_name,
            )
        )

    # Legend 2: FULL (gray)
        # Legend 2: FULL (gray)
    full_handles = []
    for expl_name, marker in MARKERS_BY_TOOL.items():
        size_start, size_end = MARKER_SIZES.get(expl_name, (40, 24))
        ms = (size_end ** 0.5)

        full_handles.append(
            Line2D(
                [0], [0],
                marker=marker,
                linestyle="None",
                color="w",
                markerfacecolor="0.5",   # same gray as FULL_FILL
                markeredgecolor="none",   # <- NO border
                markeredgewidth=0.0,
                markersize=ms,
                label=expl_name,
            )
        )



    # ---------- TWO VERTICAL LEGENDS ----------
    # Top legend: Minimal Plan (hollow)
    fig.legend(
        handles=minimal_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),  # higher row
        frameon=True,
        fontsize=8,
        borderpad=0.4,
        edgecolor="black",
        ncol=len(minimal_handles),
        title="Immediate Suggestion",
        title_fontsize=8.5,
    )

    # Second legend: Full Plan (gray), just below the first
    fig.legend(
        handles=full_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),  # lower row
        frameon=True,
        fontsize=8,
        borderpad=0.4,
        edgecolor="black",
        ncol=len(full_handles),
        title="Validated Solution (at Flip Depth)",
        title_fontsize=8.5,
    )

    # leave enough top margin for stacked legends
    fig.subplots_adjust(top=0.76, bottom=0.22, left=0.08, right=0.99, wspace=0.18)

    # make sure legends are not cut off
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Graph saved: {save_path}")
    plt.close(fig)


# =========================================================
# [Flip rate 로딩 & experiments dict 구성]
# =========================================================
def load_flip_rates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Explainer" not in df.columns or "Model" not in df.columns:
        raise ValueError(f"flip_rates file missing required columns: {path}")

    if "All" in df["Explainer"].values:
        df = df[df["Explainer"] != "All"].copy()
    return df


def build_experiments(start_df: pd.DataFrame, end_df: pd.DataFrame):
    start_pivot = start_df.pivot(index="Model", columns="Explainer", values="Flip Rate")
    end_pivot = end_df.pivot(index="Model", columns="Explainer", values="Flip Rate")

    experiments = {}
    common_models = sorted(set(start_pivot.index) & set(end_pivot.index))

    for model_name in common_models:
        model_dict = {}
        for expl_token, (disp_name, cost, offset) in COST_AND_OFFSETS.items():
            if expl_token not in start_pivot.columns or expl_token not in end_pivot.columns:
                print(f"[WARN] Missing explainer={expl_token} for model={model_name}")
                continue

            start_val = start_pivot.loc[model_name, expl_token]
            end_val = end_pivot.loc[model_name, expl_token]

            if pd.isna(start_val) or pd.isna(end_val):
                print(f"[WARN] NaN flip rate for {model_name}, {expl_token}")
                continue

            start_pct = float(start_val) * 100.0
            end_pct = float(end_val) * 100.0
            model_dict[disp_name] = (start_pct, end_pct, cost, offset)

        if model_dict:
            experiments[model_name] = model_dict

    return experiments


# =========================================================
# [실행부]
# =========================================================
if __name__ == "__main__":
    closest_path = Path("./evaluations_closest/flip_rates.csv")
    full_path = Path("./evaluations/flip_rates.csv")

    if not closest_path.exists():
        raise FileNotFoundError(closest_path)
    if not full_path.exists():
        raise FileNotFoundError(full_path)

    start_df = load_flip_rates(closest_path)
    end_df = load_flip_rates(full_path)
    experiments = build_experiments(start_df, end_df)

    out_dir = Path("./figures_rq1_plans")
    out_dir.mkdir(parents=True, exist_ok=True)

    # single horizontal panel for all models
    save_path = out_dir / "figure_rq1_all_models.png"
    plot_all_models(experiments, save_path)
