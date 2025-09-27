# run_explainers_jit.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import concurrent.futures
import warnings
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

# your existing LIME utils
from Explainer.LIME_HPO import LIME_HPO, LIME_Planner
# your data/model helpers
from data_utils import get_true_positives, read_dataset, get_output_dir, get_model
from hyparams import SEED
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

# ---------- PyExplainer helpers ----------
# --- add near the top of your run_explanations.py (PyExplainer branch) ---
import re
import numpy as np
import pandas as pd

def _pyexp_to_lime_like_df(px, rule_obj, X_train: pd.DataFrame, X_row: pd.Series) -> pd.DataFrame:
    """
    Convert PyExplainer's rule_obj into LIME-like rows:
    feature,value,importance,min,max,rule,importance_ratio
    - Uses 'to follow' items (toward Clean = negative rules).
    - One row per suggested feature (thresholded).
    """
    parsed = px.parse_top_rules(rule_obj["top_k_positive_rules"],
                                rule_obj["top_k_negative_rules"])
    tofollow = parsed.get("top_tofollow_rules", [])  # [{'variable','lessthan','value'}, ...]

    neg_df = rule_obj.get("top_k_negative_rules")
    rows = []
    for itm in tofollow:
        feat = itm["variable"]
        thr  = float(itm["value"])
        rule_str = f"{feat} <= {thr}" if itm["lessthan"] else f"{feat} > {thr}"

        # importance: take the max importance among negative rules mentioning this feature
        imp = np.nan
        if isinstance(neg_df, pd.DataFrame) and not neg_df.empty and "rule" in neg_df.columns:
            m = neg_df[neg_df["rule"].astype(str).str.contains(rf"\b{re.escape(feat)}\b", regex=True, na=False)]
            if not m.empty and "importance" in m.columns:
                imp = float(m["importance"].max())

        rows.append({
            "feature": feat,
            "value": float(X_row.get(feat, np.nan)),
            "importance": imp,
            "min": float(X_train[feat].min()) if feat in X_train.columns else np.nan,
            "max": float(X_train[feat].max()) if feat in X_train.columns else np.nan,
            "rule": rule_str,
        })

    df = pd.DataFrame(rows, columns=["feature","value","importance","min","max","rule"])
    if df.empty:
        # return an empty table with the right columns
        df["importance_ratio"] = pd.Series(dtype=float)
        return df[["feature","value","importance","min","max","rule","importance_ratio"]]

    # normalize to importance_ratio like your LIME code
    abs_imp = df["importance"].abs().replace({np.inf: np.nan})
    denom = abs_imp.sum()
    df["importance_ratio"] = abs_imp / denom if pd.notnull(denom) and denom > 0 else np.nan

    # optional: tidy numbers
    for c in ["value","min","max","importance","importance_ratio"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["feature","value","importance","min","max","rule","importance_ratio"]]

def _px_build(train_df: pd.DataFrame, model, dep: str = "target"):
    """Build a PyExplainer object from training data + model (created inside worker)."""
    # Lazy import to keep main process light
    from pyexplainer.pyexplainer_pyexplainer import PyExplainer
    X_train = train_df.loc[:, train_df.columns != dep]
    X_train = X_train.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train_df[dep].astype(bool)
    return PyExplainer(
        X_train=X_train,
        y_train=y_train,
        indep=pd.Index(X_train.columns),
        dep=dep,
        blackbox_model=model,
        class_label=["Clean", "Defect"],
        top_k_rules=5,
        full_ft_names=[],
    )


def _px_explain_one(
    px,
    test_instance: pd.Series,
    model,
    *,
    top_k: int,
    max_rules: int,
    max_iter: int,
    cv: int,
    random_state: int,
    reuse_local_model: bool,
) -> tuple[pd.DataFrame, dict]:
    """Run PyExplainer on one instance -> (plan_df, meta_dict)."""
    # features-only 1-row DF
    X_explain = test_instance.drop(labels=[c for c in ["commit_id", "target"] if c in test_instance.index],
                                   errors="ignore").to_frame().T
    y_explain = pd.Series([bool(test_instance["target"])], name="target")
    # X_explain / y_explain already built
    # rule_obj = px.explain(
    #     X_explain=X_explain,
    #     y_explain=y_explain,
    #     top_k=top_k, max_rules=max_rules, max_iter=max_iter, cv=cv,
    #     search_function="CrossoverInterpolation",
    #     random_state=random_state,
    #     reuse_local_model=reuse_local_model,
    # )

    # X_train_only = train_df.drop(columns=["target"], errors="ignore")
    # out_df = _pyexp_to_lime_like_df(px, rule_obj, X_train_only, X_explain.iloc[0])

    # out_df.to_csv(out_csv, index=False)   # <-- writes exactly: feature,value,importance,min,max,rule,importance_ratio

    rule_obj = px.explain(
        X_explain=X_explain,
        y_explain=y_explain,
        top_k=top_k,
        max_rules=max_rules,
        max_iter=max_iter,
        cv=cv,
        search_function="CrossoverInterpolation",
        random_state=random_state,
        reuse_local_model=reuse_local_model,
    )

    parsed = px.parse_top_rules(rule_obj["top_k_positive_rules"], rule_obj["top_k_negative_rules"])
    plan = pd.DataFrame(parsed.get("top_tofollow_rules", []))  # [{'variable','lessthan','value'}, ...]

    if not plan.empty:
        plan["operator"] = np.where(plan["lessthan"], "<", ">")
        plan["threshold"] = pd.to_numeric(plan["value"], errors="coerce")
        plan["feature"] = plan["variable"]
        plan["value_actual"] = [float(X_explain.iloc[0][f]) for f in plan["feature"]]

        # 🔧 NEW: add min/max per feature (computed from PyExplainer's training frame)
        mins, maxs = [], []
        for f in plan["feature"]:
            if f in px.X_train.columns:
                col = px.X_train[f]
                mins.append(float(np.nanmin(col.values)))
                maxs.append(float(np.nanmax(col.values)))
            else:
                mins.append(np.nan)
                maxs.append(np.nan)
        plan["min"] = mins
        plan["max"] = maxs

        # Column order to match your LIME-style expectation
        plan = plan[["feature", "value_actual", "min", "max", "operator", "threshold"]].copy()

    # meta
    try:
        prob = float(model.predict_proba(X_explain)[0][1])
    except Exception:
        try:
            prob = float(model.decision_function(X_explain)[0])
        except Exception:
            prob = np.nan

    meta = {
        "commit_id": test_instance.get("commit_id", None),
        "label": bool(test_instance.get("target", False)),
        "pred": bool(model.predict(X_explain)[0]),
        "score_or_proba": prob,
    }
    return plan, meta


def _process_tp_idx_pyexp(
    test_idx: int,
    true_positives: pd.DataFrame,
    train_df: pd.DataFrame,
    model,
    out_dir: Path,
    *,
    top_k: int = 5,
    max_rules: int = 2000,
    max_iter: int = 10000,
    cv: int = 5,
    random_state: int = 42,
    reuse_local_model: bool = False,
) -> Optional[int]:
    """Worker: explain one true positive with PyExplainer and write <out_dir>/<test_idx>.csv."""
    out_csv = out_dir / f"{test_idx}.csv"
    if out_csv.exists():
        return None

    ti = true_positives.loc[test_idx, :]
    px = _px_build(train_df, model)
    plan, meta = _px_explain_one(
        px, ti, model,
        top_k=top_k, max_rules=max_rules, max_iter=max_iter, cv=cv,
        random_state=random_state, reuse_local_model=reuse_local_model,
    )

    if plan is None or plan.empty:
        pd.DataFrame([meta]).to_csv(out_csv, index=False)
    else:
        meta_df = pd.DataFrame([meta]).loc[[0] * len(plan)].reset_index(drop=True)
        pd.concat([meta_df, plan.reset_index(drop=True)], axis=1).to_csv(out_csv, index=False)
    return os.getpid()


# ---------- Main runner (LIME / LIME-HPO / PyExplainer) ----------

def process_test_idx_lime(
    test_idx, true_positives, train_df, model, output_path, explainer_type
):
    """Your original LIME/LIME-HPO worker (unchanged behavior)."""
    ti = true_positives.loc[test_idx, :]
    out_csv = output_path / f"{test_idx}.csv"
    if out_csv.exists():
        return None

    if explainer_type == "LIME":
        LIME_Planner(
            X_train=train_df.drop(columns=["target"]),
            test_instance=ti,
            training_labels=train_df[["target"]],
            model=model,
            path=out_csv,
        )
    elif explainer_type == "LIME-HPO":
        LIME_HPO(
            X_train=train_df.drop(columns=["target"]),
            test_instance=ti,
            training_labels=train_df[["target"]],
            model=model,
            path=out_csv,
        )
    return os.getpid()


def run_single_project(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    project_name: str,
    model_type: str,
    explainer_type: str,
    *,
    top_k: int = 5,
    max_rules: int = 2000,
    max_iter: int = 10000,
    cv: int = 5,
    seed: int = 42,
    reuse_local_model: bool = False,
    verbose: bool = True,
):
    output_path = get_output_dir(project_name, explainer_type, model_type)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[RUN  ] {project_name} :: {explainer_type} on {model_type}")    
    model = get_model(project_name, model_type)
    true_positives = get_true_positives(model, train_df, test_df)
    if len(true_positives) == 0:
        print(f"[SKIP ] {project_name} :: No true positives found.")
        return

    worker = None
    worker_kwargs = {}
    if explainer_type in ("LIME", "LIME-HPO"):
        worker = process_test_idx_lime
    elif explainer_type == "PyExplainer":
        worker = _process_tp_idx_pyexp
        worker_kwargs = dict(
            top_k=top_k, max_rules=max_rules, max_iter=max_iter, cv=cv,
            random_state=seed, reuse_local_model=reuse_local_model,
        )
    else:
        raise ValueError(f"Unsupported explainer_type: {explainer_type}")
    print(f"[RUN  ] {project_name} :: {explainer_type} on {len(true_positives)} TPs")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                worker,
                int(test_idx),
                true_positives,
                train_df,
                model,
                output_path,
                explainer_type,
                **worker_kwargs,
            ) if explainer_type in ("LIME", "LIME-HPO")
            else executor.submit(
                worker,
                int(test_idx),
                true_positives,
                train_df,
                model,
                output_path,
                **worker_kwargs,
            )
            for test_idx in true_positives.index
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"{project_name} ({explainer_type})",
            disable=not verbose,
        ):
            out = future.result()
            if out is not None:
                tqdm.write(f"Process {out} finished")


def main():
    ap = ArgumentParser()
    ap.add_argument("--model_type", type=str, default="RandomForest", help="Your get_model key (e.g., RandomForest|SVM|LogisticRegression).")
    ap.add_argument("--explainer_type", type=str, default="LIME-HPO", choices=["LIME", "LIME-HPO", "PyExplainer"])
    ap.add_argument("--project", type=str, default="all", help="'all' or space-separated list")
    # PyExplainer-specific (safe defaults)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_rules", type=int, default=2000)
    ap.add_argument("--max_iter", type=int, default=10000)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--reuse_local_model", action="store_true")
    args = ap.parse_args()

    projects = read_dataset()  # dict[project] = (train_df, test_df)
    if args.project == "all":
        project_list = list(sorted(projects.keys()))
    else:
        project_list = args.project.split()

    for project in tqdm(project_list, desc="Project", leave=True):
        train_df, test_df = projects[project]
        run_single_project(
            train_df=train_df,
            test_df=test_df,
            project_name=project,
            model_type=args.model_type,
            explainer_type=args.explainer_type,
            top_k=args.top_k,
            max_rules=args.max_rules,
            max_iter=args.max_iter,
            cv=args.cv,
            seed=args.seed,
            reuse_local_model=args.reuse_local_model,
        )


if __name__ == "__main__":
    main()
