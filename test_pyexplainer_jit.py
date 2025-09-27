# test_pyexplainer_jit.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import argparse
import joblib
import pandas as pd

# PyExplainer (your vendored code)
from pyexplainer.pyexplainer_pyexplainer import PyExplainer

SUPPORTED = {"RandomForest", "SVM", "LogReg"}  # sklearn-only for PyExplainer


def load_project_frames(root: str | Path, project: str):
    project = "apache_" + project
    proj_dir = Path(root) / f"{project.replace('/', '_')}"
    tr_path = proj_dir / "train.csv"
    te_path = proj_dir / "test.csv"
    if not tr_path.exists() or not te_path.exists():
        raise FileNotFoundError(f"Missing train/test under {proj_dir}")

    train = pd.read_csv(tr_path)
    test = pd.read_csv(te_path)

    # split features/target
    feat_cols = [c for c in train.columns if c != "target"]
    Xtr = train[feat_cols].copy()
    ytr = train["target"].astype(bool).copy()
    Xte = test[feat_cols].copy()
    yte = test["target"].astype(bool).copy()
    return proj_dir, feat_cols, Xtr, ytr, Xte, yte


def load_model_and_scaler(models_root: str | Path, project: str, model_name: str):
    proj_dir = Path(models_root) / project
    model_path = proj_dir / f"{model_name}.pkl"
    scaler_path = proj_dir / "scaler.pkl"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not scaler_path.exists():
        raise FileNotFoundError(scaler_path)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# test_pyexplainer_jit.py (drop-in replacement for the middle section)

def _load_feature_list(models_root: str | Path, data_root: str | Path, project: str, feat_cols_from_data):
    # Prefer the features used at training time
    f1 = Path(models_root) / project / "features.txt"
    if f1.exists():
        return [ln.strip() for ln in f1.read_text().splitlines() if ln.strip()]
    # Fallback: features produced by preprocessing bundle
    f2 = Path(data_root) / f"{project.replace('/','_')}@0" / "features.txt"
    if f2.exists():
        return [ln.strip() for ln in f2.read_text().splitlines() if ln.strip()]
    # Last resort: whatever the data has now
    return [c for c in feat_cols_from_data]

def _align_features(df: pd.DataFrame, needed: list[str]) -> pd.DataFrame:
    # add any missing columns with zeros, drop extras, and reorder
    for col in needed:
        if col not in df.columns:
            df[col] = 0.0
    return df.reindex(columns=needed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="e.g., apache/activemq")
    ap.add_argument("--model", required=True, choices=sorted(SUPPORTED))
    ap.add_argument("--data-root", default="./Dataset/release_dataset_jit")
    ap.add_argument("--models-root", default="jit_models/apache")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    if args.model not in SUPPORTED:
        raise SystemExit(f"Model {args.model} not supported by PyExplainer. Use one of {sorted(SUPPORTED)}")

    # 1) load data
    proj_dir, feat_cols_data, Xtr_raw, ytr, Xte_raw, yte = load_project_frames(args.data_root, args.project)

    # 2) load model + scaler
    model, scaler = load_model_and_scaler(args.models_root, args.project, args.model)

    # 3) get the exact feature order used when training; align frames
    needed = _load_feature_list(args.models_root, args.data_root, args.project, feat_cols_data)
    Xtr = _align_features(Xtr_raw.copy(), needed)
    Xte = _align_features(Xte_raw.copy(), needed)

    # Debug aid if shapes still mismatch
    if hasattr(scaler, "n_features_in_") and Xtr.shape[1] != scaler.n_features_in_:
        have = list(Xtr.columns)
        msg = [
            f"Scaler expects {scaler.n_features_in_} features but got {Xtr.shape[1]}.",
            f"Missing: {[c for c in needed if c not in have]}",
            f"Extra:   {[c for c in have if c not in needed]}",
        ]
        raise ValueError("\n".join(msg))

    # 4) scale using the training scaler
    Xtr_s = pd.DataFrame(scaler.transform(Xtr.values), columns=Xtr.columns, index=Xtr.index)
    Xte_s = pd.DataFrame(scaler.transform(Xte.values), columns=Xte.columns, index=Xte.index)

    # choose one test instance
    if args.idx < 0 or args.idx >= len(Xte_s):
        raise SystemExit(f"--idx out of range (0..{len(Xte_s)-1})")
    X_explain = Xte_s.iloc[[args.idx]].copy()
    y_explain = pd.Series([yte.iloc[args.idx]], index=X_explain.index, name="target")

    # 5) PyExplainer on scaled space
    px = PyExplainer(
        X_train=Xtr_s,
        y_train=ytr.rename("target"),
        indep=Xtr_s.columns,
        dep="target",
        blackbox_model=model,
        class_label=["Clean", "Defect"],
        top_k_rules=args.top_k,
        full_ft_names=[],
    )
    rule_obj = px.explain(
        X_explain=X_explain,
        y_explain=y_explain,
        top_k=args.top_k,
        max_rules=30,
        max_iter=5,
        cv=5,
        search_function="CrossoverInterpolation",
        reuse_local_model=False,
    )

    # ... (keep the rest of your printing/summary code as-is)

    # 6) print a quick sanity summary
    pos_rules = rule_obj["top_k_positive_rules"]
    neg_rules = rule_obj["top_k_negative_rules"]
    print("\n=== PyExplainer quick check ===")
    print(f"Project: {args.project}")
    print(f"Model:   {args.model}")
    print(f"Test idx: {X_explain.index[0]}")
    print(f"#positive_rules: {len(pos_rules)}")
    print(f"#negative_rules: {len(neg_rules)}")
    if len(pos_rules):
        print("\nTop positive rules (→ 'Defect'):")
        print(pos_rules[["rule", "importance", "coef"]].head(args.top_k).to_string(index=False))
    if len(neg_rules):
        print("\nTop negative rules (→ 'Clean'):")
        print(neg_rules[["rule", "importance", "coef"]].head(args.top_k).to_string(index=False))

    # Optional: parse into actionable thresholds toward CLEAN
    parsed = px.parse_top_rules(pos_rules, neg_rules)
    to_follow = parsed["top_tofollow_rules"]  # list of {"variable","lessthan", "value"}
    print("\nRules to FOLLOW (toward CLEAN):")
    if to_follow:
        for r in to_follow:
            op = "<" if r["lessthan"] else "≥"
            print(f" - {r['variable']} {op} {r['value']}")
    else:
        print(" (none)")

    # Optional: risk prediction on this row
    risk = px.generate_risk_data(X_explain)[0]
    print(f"\nModel prediction: {risk['riskPred'][0]}, risk score: {risk['riskScore'][0]}")


if __name__ == "__main__":
    main()
