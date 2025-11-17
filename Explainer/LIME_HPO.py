# Explainer/LIME_HPO.py
# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from scipy.optimize import differential_evolution
from sklearn.preprocessing import StandardScaler
from hyparams import SEED


def _restore_rule_text(lime_scaler, dummy_scaled, feature_idx, a=None, b=None, feature=None, op=None):
    """Convert thresholds from LIME's scaled space back to original feature scale."""
    a_orig = b_orig = None
    if a is not None:
        dummy_scaled[feature_idx] = a
        a_orig = lime_scaler.inverse_transform([dummy_scaled])[0][feature_idx]
    if b is not None:
        dummy_scaled[feature_idx] = b
        b_orig = lime_scaler.inverse_transform([dummy_scaled])[0][feature_idx]

    if op == ">":
        return f"{feature} > {a_orig}"
    elif op == "<=":
        return f"{feature} <= {b_orig}"
    else:
        return f"{a_orig} < {feature} <= {b_orig}"


def _rules_df_from_explanation(explanation, X_train, test_instance_raw, lime_scaler, top_k=5):
    """
    Build a LIME-style DataFrame WITHOUT using explanation.data:
    columns: feature, value, importance, min, max, rule, importance_ratio
    """
    asmap = explanation.as_map()
    label = 1 if 1 in asmap else next(iter(asmap.keys()))
    top_pairs = asmap[label][:top_k]  # [(feat_idx, weight), ...]
    if not top_pairs:
        return pd.DataFrame(columns=["feature","value","importance","min","max","rule","importance_ratio"])

    feat_idx = [i for i, _ in top_pairs]
    top_feature_names = X_train.columns[feat_idx]

    # importance values from text form
    try:
        pairs = explanation.as_list(label=label)[:top_k]
        all_pairs = explanation.as_list(label=label)
    except TypeError:
        pairs = explanation.as_list()[:top_k]
        all_pairs = explanation.as_list()

    if not pairs:
        return pd.DataFrame(columns=["feature","value","importance","min","max","rule","importance_ratio"])

    rules_txt, importances = zip(*pairs)
    denom = np.sum(np.abs([w for _, w in all_pairs])) or 1.0
    importance_ratio = np.abs(np.array(importances)) / denom

    # mins/maxs in original space
    mins = X_train.min().iloc[feat_idx].values
    maxs = X_train.max().iloc[feat_idx].values

    # actual (original-space) values for this instance
    values = test_instance_raw.iloc[feat_idx].values

    # prepare scaled dummy vector for inverse-threshold restoration
    test_scaled = lime_scaler.transform(test_instance_raw.values.reshape(1, -1))[0]
    dummy = test_scaled.copy().tolist()

    restored_rules = []
    for i, rule in enumerate(rules_txt):
        fname = top_feature_names[i]
        fidx = feat_idx[i]
        # (a < feat <= b)
        m = re.search(rf"([-.\d]+)\s*<\s*{re.escape(fname)}\s*<=\s*([-.\d]+)", rule)
        if m:
            a, b = map(float, m.groups())
            restored_rules.append(_restore_rule_text(lime_scaler, dummy, fidx, a=a, b=b, feature=fname))
            continue
        # (feat > a)
        m = re.search(rf"{re.escape(fname)}\s*>\s*([-.\d]+)", rule)
        if m:
            a = float(m.group(1))
            restored_rules.append(_restore_rule_text(lime_scaler, dummy, fidx, a=a, feature=fname, op=">"))
            continue
        # (feat <= b)
        m = re.search(rf"{re.escape(fname)}\s*<=\s*([-.\d]+)", rule)
        if m:
            b = float(m.group(1))
            restored_rules.append(_restore_rule_text(lime_scaler, dummy, fidx, b=b, feature=fname, op="<="))
            continue
        # fallback: keep the original LIME text
        restored_rules.append(rule)

    df = pd.DataFrame({
        "feature": top_feature_names,
        "value": values,
        "importance": importances,
        "min": mins,
        "max": maxs,
        "rule": restored_rules,
        "importance_ratio": importance_ratio,
    })
    return df


def LIME_HPO(X_train, test_instance, training_labels, model, path, model_scaler=None):
    """
    Hyper-parameter optimized LIME with rule restoration.
    Heavy settings preserved (bounds 100–10000, maxiter=10, popsize=10).
    """
    lime_scaler = StandardScaler()
    X_train_scaled = lime_scaler.fit_transform(X_train.values)
    test_scaled = lime_scaler.transform(test_instance.values.reshape(1, -1))

    # Ensure the model sees data in its own feature space
    def _proba_in_model_space(X_scaled):
        X_orig = lime_scaler.inverse_transform(np.asarray(X_scaled))
        X_for_model = model_scaler.transform(X_orig) if model_scaler is not None else X_orig
        return model.predict_proba(X_for_model)

    explainer = LimeTabularExplainer(
        training_data=X_train_scaled,
        training_labels=training_labels,
        feature_names=X_train.columns,
        feature_selection="lasso_path",
        discretizer="entropy",
        random_state=SEED,
    )

    # Original heavy HPO settings
    def objective(params):
        num_samples = int(params[0])
        exp = explainer.explain_instance(test_scaled[0], _proba_in_model_space, num_samples=num_samples)
        local = exp.local_pred
        model_p = _proba_in_model_space(test_scaled)[0]
        res = model_p - local
        ss_res = np.sum(res**2)
        ss_tot = np.sum((model_p - np.mean(model_p))**2)
        if ss_tot == 0:
            return 100.0
        return -(1 - ss_res/ss_tot)

    bounds = [(100, 10000)]
    result = differential_evolution(
        objective,
        bounds,
        strategy="currenttobest1bin",
        maxiter=10,
        popsize=10,
        mutation=0.8,
        recombination=0.5,
        seed=SEED,
    )
    num_samples = int(result.x[0])

    exp = explainer.explain_instance(
        test_scaled[0],
        _proba_in_model_space,
        num_samples=num_samples,
        num_features=len(X_train.columns),  # keep as before
    )
    df = _rules_df_from_explanation(exp, X_train, test_instance, lime_scaler, top_k=5)
    df.to_csv(path, index=False)


def LIME_Planner(X_train, test_instance, training_labels, model, path, model_scaler=None):
    """
    Plain LIME with rule restoration.
    Heavy default kept: we DO NOT set num_samples here (LIME default is ~5000).
    """
    lime_scaler = StandardScaler()
    X_train_scaled = lime_scaler.fit_transform(X_train.values)
    test_scaled = lime_scaler.transform(test_instance.values.reshape(1, -1))

    def _proba_in_model_space(X_scaled):
        X_orig = lime_scaler.inverse_transform(np.asarray(X_scaled))
        X_for_model = model_scaler.transform(X_orig) if model_scaler is not None else X_orig
        return model.predict_proba(X_for_model)

    explainer = LimeTabularExplainer(
        training_data=X_train_scaled,
        training_labels=training_labels,
        feature_names=X_train.columns,
        feature_selection="lasso_path",
        discretizer="entropy",
        random_state=SEED,
    )

    exp = explainer.explain_instance(
        test_scaled[0],
        _proba_in_model_space,
        num_features=len(X_train.columns),  # as you had it
        # no num_samples -> heavy default
    )
    df = _rules_df_from_explanation(exp, X_train, test_instance, lime_scaler, top_k=5)
    df.to_csv(path, index=False)
