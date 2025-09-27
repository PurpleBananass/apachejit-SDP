# LIME_HPO.py
import re
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.preprocessing import StandardScaler
from hyparams import SEED


def LIME_HPO(X_train, test_instance, training_labels, model, path, model_scaler=None):
    """Hyper-parameter optimized LIME explainer with feature restoration.

    Minimal change:
      - Accept optional `model_scaler` (the training-time scaler).
      - Use a local wrapper so model.predict_proba sees data transformed with `model_scaler`.
    """

    # Apply StandardScaler to preserve column names (LIME's internal scale)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    test_instance_scaled = scaler.transform(test_instance.values.reshape(1, -1))

    # Build a predict_proba wrapper:
    #   LIME generates samples in *this* (internal) scaled space -> inverse to original
    #   -> apply model_scaler (if provided) -> model.predict_proba
    def _proba_in_model_space(X_scaled_from_lime):
        X_orig = scaler.inverse_transform(np.asarray(X_scaled_from_lime))
        if model_scaler is not None:
            X_for_model = model_scaler.transform(X_orig)
        else:
            X_for_model = X_orig
        return model.predict_proba(X_for_model)

    # Initialize LIME explainer with training data and feature names
    explainer = LimeTabularExplainer(
        training_data=X_train_scaled,
        training_labels=training_labels,
        feature_names=X_train.columns,
        feature_selection="lasso_path",
        discretizer="entropy",
        random_state=SEED,
    )

    # Objective function to minimize residuals between model and LIME predictions
    def objective(params):
        num_samples = int(params[0])
        explanation = explainer.explain_instance(
            test_instance_scaled[0], _proba_in_model_space, num_samples=num_samples
        )
        local_model_predictions = explanation.local_pred
        model_predictions = _proba_in_model_space(test_instance_scaled)[0]

        residuals = model_predictions - local_model_predictions
        SS_res = np.sum(residuals**2)
        SS_tot = np.sum((model_predictions - np.mean(model_predictions)) ** 2)

        if SS_tot == 0:
            return 100  # large penalty for non-variance cases

        R2 = 1 - (SS_res / SS_tot)
        return -R2  # Negative R² for minimization

    # Hyperparameter optimization using differential evolution
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

    # Generate explanation with optimized number of samples
    explanation = explainer.explain_instance(
        test_instance_scaled[0],
        _proba_in_model_space,
        num_samples=num_samples,
        num_features=len(X_train.columns),
    )

    # Extract top 5 features and their importance
    top_features_rule = explanation.as_list()[:5]
    top_features = explanation.as_map()[1]
    top_features_index = [feature[0] for feature in top_features][:5]
    top_feature_names = X_train.columns[top_features_index]

    min_val = X_train.min()
    max_val = X_train.max()

    rules, importances = zip(*top_features_rule)
    rules = list(rules)
    denominator = np.sum(np.abs(np.array([imp for _, imp in explanation.as_list()]))) or 1.0
    importance_ratio = np.abs(np.array(importances)) / denominator

    dummy = test_instance_scaled[0].copy().tolist()
    # Restore original feature values in the rules
    for i, rule in enumerate(rules):
        feature = top_feature_names[i]
        # Restore range (a < feature <= b)
        matches = re.search(
            r"([-.\d]+) < " + re.escape(feature) + r" <= ([-.\d]+)", rule
        )
        if matches:
            a, b = map(float, matches.groups())
            rules[i] = restore_rule(scaler, dummy, top_features_index[i], a, b, feature)
            continue

        # Restore case (feature > a)
        matches = re.search(re.escape(feature) + r" > ([-.\d]+)", rule)
        if matches:
            a = float(matches.group(1))
            rules[i] = restore_rule(
                scaler, dummy, top_features_index[i], a, None, feature, ">"
            )
            continue

        # Restore case (feature <= b)
        matches = re.search(re.escape(feature) + r" <= ([-.\d]+)", rule)
        if matches:
            b = float(matches.group(1))
            rules[i] = restore_rule(
                scaler, dummy, top_features_index[i], None, b, feature, "<="
            )
            continue

    # Create DataFrame to save the rules and export as CSV
    rules_df = pd.DataFrame(
        {
            "feature": top_feature_names,
            "value": test_instance[top_features_index],
            "importance": importances,
            "min": min_val[top_features_index],
            "max": max_val[top_features_index],
            "rule": rules,
            "importance_ratio": importance_ratio,
        }
    )

    rules_df.to_csv(path, index=False)


def restore_rule(
    scaler, dummy, feature_idx, a=None, b=None, feature=None, operator=None
):
    """Helper function to restore original feature values."""
    if a is not None:
        dummy[feature_idx] = a
        a = scaler.inverse_transform([dummy])[0][feature_idx]
    if b is not None:
        dummy[feature_idx] = b
        b = scaler.inverse_transform([dummy])[0][feature_idx]

    if operator == ">":
        return f"{feature} > {a}"
    elif operator == "<=":
        return f"{feature} <= {b}"
    else:
        return f"{a} < {feature} <= {b}"


def LIME_Planner(X_train, test_instance, training_labels, model, path, model_scaler=None):
    """LIME explainer with feature restoration.

    Minimal change:
      - Accept optional `model_scaler`; use same wrapper idea as LIME_HPO.
    """

    # Apply StandardScaler to preserve column names (LIME's internal scale)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    test_instance_scaled = scaler.transform(test_instance.values.reshape(1, -1))

    def _proba_in_model_space(X_scaled_from_lime):
        X_orig = scaler.inverse_transform(np.asarray(X_scaled_from_lime))
        if model_scaler is not None:
            X_for_model = model_scaler.transform(X_orig)
        else:
            X_for_model = X_orig
        return model.predict_proba(X_for_model)

    # Initialize LIME explainer with training data and feature names
    explainer = LimeTabularExplainer(
        training_data=X_train_scaled,
        training_labels=training_labels,
        feature_names=X_train.columns,
        feature_selection="lasso_path",
        discretizer="entropy",
        random_state=SEED,
    )

    # Generate explanation
    explanation = explainer.explain_instance(
        test_instance_scaled[0],
        _proba_in_model_space,
        num_features=len(X_train.columns)
    )

    # Extract top 5 features and their importance
    top_features_rule = explanation.as_list()[:5]
    top_features = explanation.as_map()[1]
    top_features_index = [feature[0] for feature in top_features][:5]
    top_feature_names = X_train.columns[top_features_index]

    min_val = X_train.min()
    max_val = X_train.max()

    rules, importances = zip(*top_features_rule)
    rules = list(rules)
    denominator = np.sum(np.abs(np.array([imp for _, imp in explanation.as_list()]))) or 1.0
    importance_ratio = np.abs(np.array(importances)) / denominator

    dummy = test_instance_scaled[0].copy().tolist()
    # Restore original feature values in the rules
    feature_name_to_index = {name: i for i, name in enumerate(X_train.columns)}
    for i, scaled_rule in enumerate(rules):
        matches = re.search(
            r"([-]?[\d.]+)?\s*(<|>)?\s*([a-zA-Z_]+)\s*(<=|>=|<|>)?\s*([-]?[\d.]+)?",
            scaled_rule,
        )
        if not matches:
            continue
        v1, _, feature_name, op, v2 = matches.groups()
        if v1 and v2 and op in ("<=","<"):
            feature_idx = feature_name_to_index[feature_name]
            dummy[feature_idx] = float(v1)
            l = scaler.inverse_transform([dummy])[0][feature_idx]
            dummy[feature_idx] = float(v2)
            r = scaler.inverse_transform([dummy])[0][feature_idx]
            rule = f"{l} < {feature_name} <= {r}"
        elif op == ">" and v2:
            feature_idx = feature_name_to_index[feature_name]
            dummy[feature_idx] = float(v2)
            r = scaler.inverse_transform([dummy])[0][feature_idx]
            rule = f"{feature_name} > {r}"
        elif op == "<=" and v2:
            feature_idx = feature_name_to_index[feature_name]
            dummy[feature_idx] = float(v2)
            l = scaler.inverse_transform([dummy])[0][feature_idx]
            rule = f"{feature_name} <= {l}"
        else:
            rule = scaled_rule
        rules[i] = rule

    # Create DataFrame to save the rules and export as CSV
    rules_df = pd.DataFrame(
        {
            "feature": top_feature_names,
            "value": test_instance[top_features_index],
            "importance": importances,
            "min": min_val[top_features_index],
            "max": max_val[top_features_index],
            "rule": rules,
            "importance_ratio": importance_ratio,
        }
    )

    rules_df.to_csv(path, index=False)
