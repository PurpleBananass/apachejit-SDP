# pyexplainer_core.py
# Minimal, non-visual PyExplainer for modern pandas/sklearn

import copy
import math
import os
import re
import sys
import string
import warnings
import pickle

import numpy as np
import pandas as pd
import scipy as sp

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state, all_estimators

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Try relative import first (if packaged), then local.
try:
    from .rulefit import RuleFit
except ImportError:  # pragma: no cover
    from rulefit import RuleFit


# ---------------------------------------------------------------------------
# AutoSpearman (unchanged in logic, modern pandas-compatible)
# ---------------------------------------------------------------------------

def AutoSpearman(
    X_train: pd.DataFrame,
    correlation_threshold: float = 0.7,
    correlation_method: str = "spearman",
    VIF_threshold: float = 5,
) -> pd.DataFrame:
    """Automated feature selection to address collinearity and multicollinearity.

    Parameters
    ----------
    X_train : DataFrame
        Training features.
    correlation_threshold : float
        Absolute correlation threshold.
    correlation_method : str
        Method passed to DataFrame.corr (e.g. 'spearman', 'pearson').
    VIF_threshold : float
        Variance Inflation Factor threshold.

    Returns
    -------
    DataFrame
        Reduced feature set (columns) after AutoSpearman.
    """
    X_AS_train = X_train.copy()
    AS_metrics = list(X_AS_train.columns)
    count = 1

    # Part 1: correlation-based pruning
    print(
        "(Part 1) Automatically select non-correlated metrics "
        "based on a Spearman rank correlation test"
    )
    while True:
        corrmat = X_AS_train.corr(method=correlation_method)
        top_corr_features = corrmat.index
        abs_corrmat = abs(corrmat)

        highly_correlated_metrics = (
            ((corrmat > correlation_threshold) | (corrmat < -correlation_threshold))
            & (corrmat != 1)
        )
        n_correlated_metrics = int(np.sum(highly_correlated_metrics.values))

        if n_correlated_metrics > 0:
            # strongest pair-wise correlation
            find_top_corr = pd.melt(abs_corrmat, ignore_index=False)
            find_top_corr.reset_index(inplace=True)
            find_top_corr = find_top_corr[find_top_corr["value"] != 1]
            top_corr_index = find_top_corr["value"].idxmax()
            top_corr_i = find_top_corr.loc[top_corr_index, :]

            correlated_metric_1 = top_corr_i["index"]
            correlated_metric_2 = top_corr_i["variable"]
            print(
                "> Step",
                count,
                "comparing between",
                correlated_metric_1,
                "and",
                correlated_metric_2,
            )

            # correlation with other metrics
            others = [
                i
                for i in top_corr_features
                if i not in [correlated_metric_1, correlated_metric_2]
            ]
            correlation_with_other_metrics_1 = float(
                np.mean(abs_corrmat[correlated_metric_1][others])
            )
            correlation_with_other_metrics_2 = float(
                np.mean(abs_corrmat[correlated_metric_2][others])
            )
            print(
                ">>",
                correlated_metric_1,
                "has the average correlation of",
                np.round(correlation_with_other_metrics_1, 3),
                "with other metrics",
            )
            print(
                ">>",
                correlated_metric_2,
                "has the average correlation of",
                np.round(correlation_with_other_metrics_2, 3),
                "with other metrics",
            )

            if correlation_with_other_metrics_1 < correlation_with_other_metrics_2:
                exclude_metric = correlated_metric_2
            else:
                exclude_metric = correlated_metric_1
            print(">>", "Exclude", exclude_metric)
            count += 1

            AS_metrics = list(set(AS_metrics) - {exclude_metric})
            X_AS_train = X_AS_train[AS_metrics]
        else:
            break

    print("According to Part 1 of AutoSpearman,", AS_metrics, "are selected.")

    # Part 2: VIF-based pruning
    print(
        "(Part 2) Automatically select non-correlated metrics "
        "based on a Variance Inflation Factor analysis"
    )

    X_AS_train = add_constant(X_AS_train)
    selected_features = list(X_AS_train.columns)
    count = 1

    while True:
        # Calculate VIF scores
        vif_values = [
            variance_inflation_factor(np.array(X_AS_train.values, dtype=float), i)
            for i in range(X_AS_train.shape[1])
        ]
        vif_scores = pd.DataFrame(
            {"Feature": X_AS_train.columns, "VIFscore": vif_values}
        )

        vif_scores = vif_scores.loc[vif_scores["Feature"] != "const", :]
        vif_scores.sort_values(
            by=["VIFscore"], ascending=False, inplace=True, kind="mergesort"
        )

        filtered_vif_scores = vif_scores[vif_scores["VIFscore"] >= VIF_threshold]

        if len(filtered_vif_scores) == 0:
            break

        metric_to_exclude = str(filtered_vif_scores["Feature"].iloc[0])
        print("> Step", count, "- exclude", metric_to_exclude)
        count += 1

        selected_features = [f for f in selected_features if f != metric_to_exclude]
        X_AS_train = X_AS_train.loc[:, selected_features]

    print(
        "Finally, according to Part 2 of AutoSpearman,",
        X_AS_train.columns,
        "are selected.",
    )
    if "const" in X_AS_train.columns:
        X_AS_train = X_AS_train.drop("const", axis=1)
    return X_AS_train


# ---------------------------------------------------------------------------
# Utility: environment / sample data (unchanged; optional)
# ---------------------------------------------------------------------------

def get_base_prefix_compat():
    """Get base/real prefix, or sys.prefix if there is none."""
    return (
        getattr(sys, "base_prefix", None)
        or getattr(sys, "real_prefix", None)
        or sys.prefix
    )


def in_virtualenv():
    return get_base_prefix_compat() != sys.prefix


INSIDE_VIRTUAL_ENV = in_virtualenv()


def load_sample_data() -> pd.DataFrame:
    """Load bundled sample data (activemq-5.0.0.csv) if present."""
    this_dir, _ = os.path.split(__file__)
    path = os.path.join(this_dir, "default_data", "activemq-5.0.0.csv")
    if INSIDE_VIRTUAL_ENV:
        cwd = os.getcwd()
        path = os.path.join(cwd, "pyexplainer", "default_data", "activemq-5.0.0.csv")
    return pd.read_csv(path)


def get_dflt():
    """Obtain the default data and model (if default_data exists)."""
    this_dir, _ = os.path.split(__file__)
    path_rf_model = os.path.join(this_dir, "default_data", "sample_model.pkl")
    path_X_train = os.path.join(this_dir, "default_data", "X_train.csv")
    path_y_train = os.path.join(this_dir, "default_data", "y_train.csv")
    path_X_explain = os.path.join(this_dir, "default_data", "X_explain.csv")
    path_y_explain = os.path.join(this_dir, "default_data", "y_explain.csv")
    if INSIDE_VIRTUAL_ENV:
        cwd = os.getcwd()
        base = os.path.join(cwd, "tests", "default_data")
        path_rf_model = os.path.join(base, "sample_model.pkl")
        path_X_train = os.path.join(base, "X_train.csv")
        path_y_train = os.path.join(base, "y_train.csv")
        path_X_explain = os.path.join(base, "X_explain.csv")
        path_y_explain = os.path.join(base, "y_explain.csv")

    with open(path_rf_model, "rb") as f:
        rf_model = pickle.load(f)
    X_train = pd.read_csv(path_X_train)
    if "File" in X_train.columns:
        X_train = X_train.drop(["File"], axis=1)
    y_train = pd.read_csv(path_y_train)["RealBug"]
    X_explain = pd.read_csv(path_X_explain)
    y_explain = pd.read_csv(path_y_explain)["RealBug"]

    full_ft_names = ["nCommit", "AddedLOC", "nCoupledClass", "LOC", "CommentToCodeRatio"]
    return {
        "X_train": X_train,
        "y_train": y_train,
        "indep": X_train.columns,
        "dep": "RealBug",
        "blackbox_model": rf_model,
        "X_explain": X_explain,
        "y_explain": y_explain,
        "full_ft_names": full_ft_names,
    }


# ---------------------------------------------------------------------------
# Rule filtering helper (used by PyExplainer.explain)
# ---------------------------------------------------------------------------

def filter_rules(rules: pd.DataFrame, X_explain: pd.DataFrame) -> pd.DataFrame:
    """Return rules that are actually satisfied by the given instance X_explain.

    - keeps only 'rule'-type rules with positive coef and importance
    - evaluates each rule on the row in X_explain
    """

    def eval_rule(rule: str, x_df: pd.DataFrame):
        var_in_rule = list(set(re.findall(r"[a-z_*A-Z]+", rule)))
        rule_ = re.sub(r"\b=\b", "==", rule)
        if "or" in var_in_rule:
            var_in_rule.remove("or")
        if "e" in var_in_rule and "e" not in x_df.columns:
            var_in_rule.remove("e")
        rule_ = rule_.replace("&", "and")

        eval_result_list = []
        for i in range(len(x_df)):
            x = x_df.iloc[[i]]
            var_dict = {var: float(x[var]) for var in var_in_rule}
            eval_result = eval(rule_, var_dict)  # nosec - trusted rules from RuleFit
            eval_result_list.append(eval_result)
        return eval_result_list

    # (1) rule-type, positive coef and importance
    rules_f = rules[
        (rules["type"] == "rule") & (rules["coef"] > 0) & (rules["importance"] > 0)
    ]
    rules_list = list(rules_f["rule"])
    rule_eval_result = []

    X_explain = X_explain.copy()
    for r in rules_list:
        py_exp_pred = eval_rule(r, X_explain)[0]
        rule_eval_result.append(py_exp_pred)

    df_flags = pd.DataFrame({"is_satisfy_instance": rule_eval_result})
    rules_f = pd.concat([rules_f.reset_index(drop=True), df_flags], axis=1)
    rules_f = rules_f.loc[rules_f["is_satisfy_instance"] == True]  # noqa: E712
    sorted_rules = rules_f.sort_values(
        by="importance", ascending=False, kind="mergesort"
    )
    return sorted_rules


# ---------------------------------------------------------------------------
# Core PyExplainer (no widgets / HTML)
# ---------------------------------------------------------------------------

class PyExplainer:
    """Core PyExplainer (no visualisation).

    Parameters
    ----------
    X_train : DataFrame
        Training features.
    y_train : Series
        Training labels.
    indep : Index
        Feature column names.
    dep : str
        Label column name.
    blackbox_model : sklearn classifier
        Global model with predict/predict_proba.
    class_label : list
        Binary class labels, e.g. ['Clean', 'Defect'].
    top_k_rules : int
        Number of top rules per sign to retrieve.
    full_ft_names : list
        Optional full feature names (same length as X_train.columns).
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        indep,
        dep: str,
        blackbox_model,
        class_label=None,
        top_k_rules: int = 3,
        full_ft_names=None,
    ):
        if class_label is None:
            class_label = ["Clean", "Defect"]
        if full_ft_names is None:
            full_ft_names = []

        if not isinstance(X_train, pd.core.frame.DataFrame):
            raise TypeError("X_train should be type 'pandas.core.frame.DataFrame'")
        if not isinstance(y_train, pd.core.series.Series):
            raise TypeError("y_train should be type 'pandas.core.series.Series'")
        if not isinstance(indep, pd.Index):
            raise TypeError(
                "indep (feature column names) should be type 'pandas.Index'"
            )
        if not isinstance(dep, str):
            raise TypeError("dep (label column name) should be type 'str'")

        self.X_train = X_train
        self.y_train = y_train
        self.indep = indep
        self.dep = dep

        # Relaxed check: accept any sklearn-style classifier
        all_clf = all_estimators(type_filter="classifier")
        supported_algo = [clf[1] for clf in all_clf]
        if type(blackbox_model) in supported_algo:
            self.blackbox_model = blackbox_model
        else:
            # If it's not in all_estimators, still accept if it quacks like a classifier
            if not (hasattr(blackbox_model, "predict") and hasattr(blackbox_model, "predict_proba")):
                raise TypeError(
                    "blackbox_model should be an sklearn classifier with predict/predict_proba"
                )
            self.blackbox_model = blackbox_model

        if not isinstance(class_label, list):
            raise TypeError("class_label should be type 'list'")
        if len(class_label) != 2:
            raise ValueError("class_label should be a list with length of 2")
        self.class_label = class_label

        if not isinstance(top_k_rules, int):
            raise TypeError("top_k_rules should be type 'int'")
        if top_k_rules <= 0 or top_k_rules > 15:
            raise ValueError("top_k_rules should be in range 1 - 15")
        self.top_k_rules = top_k_rules

        if full_ft_names:
            short_ft_names = X_train.columns.to_list()
            if len(short_ft_names) != len(full_ft_names):
                raise ValueError(
                    "short feature names and full feature names must have same length"
                )
            self.full_ft_names = dict(zip(short_ft_names, full_ft_names))
        else:
            self.full_ft_names = {}

        self.X_explain = None
        self.y_explain = None
        self.local_model = None

    # -------------------- basic getters / setters --------------------

    def get_full_ft_names(self):
        return self.full_ft_names

    def set_full_ft_names(self, full_ft_names: dict):
        self.full_ft_names = full_ft_names

    def get_top_k_rules(self) -> int:
        return self.top_k_rules

    def set_top_k_rules(self, top_k_rules: int):
        if not isinstance(top_k_rules, int) or not (1 <= top_k_rules <= 15):
            raise ValueError("top_k_rules should be int in range 1 - 15")
        self.top_k_rules = top_k_rules

    def set_X_train(self, X_train: pd.DataFrame):
        if not isinstance(X_train, pd.core.frame.DataFrame):
            raise TypeError(
                "set_X_train failed, X_train should be DataFrame"
            )
        self.X_train = X_train

    # -------------------- AutoSpearman wrapper --------------------

    def auto_spearman(
        self,
        apply_to_X_train: bool = True,
        correlation_threshold: float = 0.7,
        correlation_method: str = "spearman",
        VIF_threshold: float = 5,
    ):
        X_AS_train = AutoSpearman(
            self.X_train,
            correlation_threshold=correlation_threshold,
            correlation_method=correlation_method,
            VIF_threshold=VIF_threshold,
        )
        if apply_to_X_train:
            self.set_X_train(X_AS_train)
            if self.get_full_ft_names():
                full_ft_names = self.get_full_ft_names()
                new_full = {k: full_ft_names[k] for k in X_AS_train.columns}
                self.set_full_ft_names(new_full)
            print(
                "X_train inside PyExplainer was updated based on the selected features above"
            )
        else:
            return X_AS_train

    # -------------------- core explain --------------------

    def explain(
        self,
        X_explain: pd.DataFrame,
        y_explain: pd.Series,
        top_k: int = 3,
        max_rules: int = 2000,
        max_iter: int = 10000,
        cv: int = 5,
        search_function: str = "CrossoverInterpolation",
        random_state=None,
        reuse_local_model: bool = False,
    ):
        """Generate local RuleFit rules around X_explain.

        Returns
        -------
        dict with keys:
            'synthetic_data', 'synthetic_predictions', 'X_explain', 'y_explain',
            'indep', 'dep', 'top_k_positive_rules', 'top_k_negative_rules',
            'local_rulefit_model'
        """
        if not isinstance(X_explain, pd.core.frame.DataFrame):
            raise TypeError("X_explain should be DataFrame")
        if len(X_explain.columns) != len(self.X_train.columns):
            raise ValueError(
                "X_explain should have the same number of columns as X_train"
            )
        if not isinstance(y_explain, pd.core.series.Series):
            raise TypeError("y_explain should be Series")

        self.set_top_k_rules(top_k)

        # Step 1: synthetic instances
        if search_function.lower() == "crossoverinterpolation":
            synthetic_object = self.generate_instance_crossover_interpolation(
                X_explain, y_explain, random_state=random_state
            )
        elif search_function.lower() == "randomperturbation":
            synthetic_object = self.generate_instance_random_perturbation(
                X_explain=X_explain
            )
        else:
            warnings.warn(
                f"Unknown search_function={search_function}, "
                "defaulting to crossoverinterpolation."
            )
            synthetic_object = self.generate_instance_crossover_interpolation(
                X_explain, y_explain, random_state=random_state
            )

        synthetic_instances = synthetic_object["synthetic_data"].loc[:, self.indep]
        synthetic_predictions = self.blackbox_model.predict(synthetic_instances)

        # one-class fall-back
        if 1 in synthetic_predictions and 0 in synthetic_predictions:
            one_class_problem = False
        else:
            one_class_problem = True

        if one_class_problem:
            print(
                "Random Perturbation only generated one class; "
                "falling back to crossoverinterpolation."
            )
            synthetic_object = self.generate_instance_crossover_interpolation(
                X_explain, y_explain
            )
            synthetic_instances = synthetic_object["synthetic_data"].loc[:, self.indep]
            synthetic_predictions = self.blackbox_model.predict(synthetic_instances)

        # Step 3: local RuleFit
        if reuse_local_model and self.local_model is not None:
            local_rulefit_model = self.local_model
        else:
            local_rulefit_model = RuleFit(
                rfmode="classify",
                exp_rand_tree_size=False,
                random_state=random_state,
                max_rules=max_rules,
                cv=cv,
                max_iter=max_iter,
                n_jobs=-1,
            )
            local_rulefit_model.fit(
                synthetic_instances.values,
                synthetic_predictions,
                feature_names=self.indep,
            )
            self.local_model = local_rulefit_model

        # Step 4: get rules
        rules = local_rulefit_model.get_rules()
        rules = rules[rules.coef != 0].sort_values(
            "importance", ascending=False, kind="mergesort"
        )
        rules = rules[rules.type == "rule"].sort_values(
            "importance", ascending=False, kind="mergesort"
        )

        positive_filtered_rules = filter_rules(rules, X_explain)

        # top positive rules (satisfied by instance)
        top_k_positive_rules = (
            positive_filtered_rules.loc[positive_filtered_rules["coef"] > 0]
            .sort_values("importance", ascending=False, kind="mergesort")
            .head(top_k)
            .copy()
        )
        top_k_positive_rules["Class"] = self.class_label[1]
        top_k_positive_rules = top_k_positive_rules.dropna()

        # top negative rules (global)
        top_k_negative_rules = (
            rules.loc[rules["coef"] < 0]
            .sort_values("importance", ascending=False, kind="mergesort")
            .head(top_k)
            .copy()
        )
        top_k_negative_rules["Class"] = self.class_label[0]
        top_k_negative_rules = top_k_negative_rules.dropna()

        rule_obj = {
            "synthetic_data": synthetic_instances,
            "synthetic_predictions": synthetic_predictions,
            "X_explain": X_explain,
            "y_explain": y_explain,
            "indep": self.indep,
            "dep": self.dep,
            "top_k_positive_rules": top_k_positive_rules,
            "top_k_negative_rules": top_k_negative_rules,
            "local_rulefit_model": local_rulefit_model,
        }
        return rule_obj

    # -------------------- sampling: crossover + interpolation --------------------

    def generate_instance_crossover_interpolation(
        self, X_explain: pd.DataFrame, y_explain: pd.Series, random_state=None, debug=False
    ):
        """Generate instances via crossover + interpolation (PyExplainer's main sampler)."""

        X_train_i = self.X_train.copy()
        X_explain_i = X_explain.copy()
        y_explain_i = y_explain.copy()

        X_train_i.reset_index(inplace=True)
        X_explain_i.reset_index(inplace=True)
        X_train_i = X_train_i.loc[:, self.indep]
        X_explain_i = X_explain_i.loc[:, self.indep]
        y_explain_i = y_explain_i.reset_index()[[self.dep]]

        target_train = self.blackbox_model.predict(X_train_i)

        scaler = StandardScaler()
        trainset_normalize = X_train_i.copy()
        if debug:
            print(list(X_train_i), "columns")
        cases_normalize = X_explain_i.copy()

        train_objs_num = len(trainset_normalize)
        dataset = pd.concat([trainset_normalize, cases_normalize], axis=0)
        if debug:
            print(self.indep, "continuous")
            print(type(self.indep))
        dataset[self.indep] = scaler.fit_transform(dataset[self.indep])

        trainset_normalize = dataset.iloc[:train_objs_num].copy()
        cases_normalize = dataset.iloc[train_objs_num:].copy()

        dist_df = pd.DataFrame(index=trainset_normalize.index.copy())
        width = math.sqrt(len(X_train_i.columns)) * 0.75

        for _, case in cases_normalize.iterrows():
            dist = np.linalg.norm(trainset_normalize.sub(np.array(case)), axis=1)
            similarity = np.exp(-(dist ** 2) / (2 * (width ** 2)))
            dist_df["dist"] = similarity
            dist_df["t_target"] = target_train

            unique_classes = dist_df.t_target.unique()
            dist_df = dist_df.sort_values(
                by=["dist"], ascending=False, kind="mergesort"
            )

            # top 40 per class
            parts = []
            for clz in unique_classes:
                parts.append(dist_df[dist_df["t_target"] == clz].head(40))
            if parts:
                top_fourty_df = pd.concat(parts, axis=0)
            else:
                top_fourty_df = pd.DataFrame(columns=dist_df.columns)

            cutoff_similarity = (
                top_fourty_df.nsmallest(1, "dist", keep="last").index.values.astype(int)[0]
            )
            min_loc = dist_df.index.get_loc(cutoff_similarity)
            train_neigh_sampling_b = dist_df.iloc[0 : min_loc + 1]

            target_details = train_neigh_sampling_b.groupby(["t_target"]).size()
            target_details_df = pd.DataFrame(
                {"target": target_details.index, "target_count": target_details.values}
            )

            final_parts = []
            for _, row in target_details_df.iterrows():
                cls = row["target"]
                cls_mask = train_neigh_sampling_b["t_target"] == cls
                cls_set = train_neigh_sampling_b.loc[cls_mask]
                if row["target_count"] > 200:
                    cls_set = cls_set.sample(n=200, random_state=random_state)
                final_parts.append(cls_set)

            if final_parts:
                final_neighbours_similarity_df = pd.concat(final_parts, axis=0)
            else:
                final_neighbours_similarity_df = pd.DataFrame(columns=dist_df.columns)

            train_set_neigh = X_train_i[X_train_i.index.isin(
                final_neighbours_similarity_df.index
            )]
            if debug:
                print(train_set_neigh, "train set neigh")

            train_class_neigh = y_explain_i[
                y_explain_i.index.isin(final_neighbours_similarity_df.index)
            ]

            new_con_df = pd.DataFrame([])
            sample_classes_arr = []

            rng = check_random_state(random_state)

            # Crossover
            for num in range(0, 1000):
                if len(train_set_neigh) < 2:
                    break
                rand_rows = train_set_neigh.sample(2, random_state=rng)
                sample_classes = train_class_neigh[
                    train_class_neigh.index.isin(rand_rows.index)
                ]
                sample_classes = np.array(
                    sample_classes.to_records().view(type=np.matrix)
                )
                if len(sample_classes) > 0:
                    sample_classes_arr.append(sample_classes[0].tolist())

                alpha_n = rng.uniform(low=0.0, high=1.0)
                x = rand_rows.iloc[0]
                y = rand_rows.iloc[1]
                new_ins = x + (y - x) * alpha_n
                new_ins = new_ins.to_frame().T
                new_ins.index = [num]
                new_con_df = pd.concat([new_con_df, new_ins], axis=0)

            # Mutation
            for num in range(1000, 2000):
                if len(train_set_neigh) < 3:
                    break
                rand_rows = train_set_neigh.sample(3, random_state=rng)
                sample_classes = train_class_neigh[
                    train_class_neigh.index.isin(rand_rows.index)
                ]
                sample_classes = np.array(
                    sample_classes.to_records().view(type=np.matrix)
                )
                if len(sample_classes) > 0:
                    sample_classes_arr.append(sample_classes[0].tolist())

                mu_f = rng.uniform(low=0.5, high=1.0)
                x = rand_rows.iloc[0]
                y = rand_rows.iloc[1]
                z = rand_rows.iloc[2]
                new_ins = x + (y - z) * mu_f
                new_ins = new_ins.to_frame().T
                new_ins.index = [num]
                new_con_df = pd.concat([new_con_df, new_ins], axis=0)

            # combine original neighbours + generated instances
            predict_dataset = pd.concat(
                [train_set_neigh, new_con_df], axis=0, ignore_index=True
            )
            target = self.blackbox_model.predict(predict_dataset)
            target_df = pd.DataFrame(target)

            new_df_case = pd.concat([predict_dataset, target_df], axis=1)
            new_df_case = np.round(new_df_case, 2)
            new_df_case.rename(columns={0: y_explain_i.columns[0]}, inplace=True)
            sampled_class_frequency = new_df_case.groupby([self.dep]).size()

            return {
                "synthetic_data": new_df_case,
                "sampled_class_frequency": sampled_class_frequency,
            }

        # Fallback: should not reach here
        return {
            "synthetic_data": X_train_i.copy(),
            "sampled_class_frequency": pd.Series(dtype=int),
        }

    # -------------------- sampling: random perturbation (LIME-style) --------------------

    def generate_instance_random_perturbation(
        self, X_explain: pd.DataFrame, debug: bool = False
    ):
        """Random perturbation approach (LIME-style)."""

        random_seed = 0
        data_row = X_explain.loc[:, self.indep].values
        num_samples = 1000
        sampling_method = "gaussian"
        discretizer = None
        sample_around_instance = True
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        scaler.fit(self.X_train.loc[:, self.indep])

        random_state = check_random_state(random_seed)
        is_sparse = sp.sparse.issparse(data_row)

        if is_sparse:
            num_cols = data_row.shape[1]
            data = sp.sparse.csr_matrix((num_samples, num_cols), dtype=data_row.dtype)
        else:
            num_cols = data_row.shape[0]
            data = np.zeros((num_samples, num_cols))

        if discretizer is None:
            instance_sample = data_row
            scale = scaler.scale_

            if is_sparse:
                non_zero_indexes = data_row.nonzero()[1]
                num_cols = len(non_zero_indexes)
                instance_sample = data_row[:, non_zero_indexes]
                scale = scale[non_zero_indexes]

            if sampling_method == "gaussian":
                data = random_state.normal(0, 1, num_samples * num_cols).reshape(
                    num_samples, num_cols
                )
            else:
                warnings.warn(
                    "Invalid sampling_method; defaulting to Gaussian.",
                    UserWarning,
                )
                data = random_state.normal(0, 1, num_samples * num_cols).reshape(
                    num_samples, num_cols
                )

            if sample_around_instance:
                data = data * scale + instance_sample

            if is_sparse:
                if num_cols == 0:
                    data = sp.sparse.csr_matrix(
                        (num_samples, data_row.shape[1]), dtype=data_row.dtype
                    )
                else:
                    indexes = np.tile(non_zero_indexes, num_samples)
                    indptr = np.array(
                        range(
                            0,
                            len(non_zero_indexes) * (num_samples + 1),
                            len(non_zero_indexes),
                        )
                    )
                    data_1d = data.reshape(data.shape[0] * data.shape[1])
                    data = sp.sparse.csr_matrix(
                        (data_1d, indexes, indptr),
                        shape=(num_samples, data_row.shape[1]),
                    )

        data[0] = data_row.copy()
        inverse = data.copy()
        inverse[0] = data_row

        if sp.sparse.issparse(data):
            scaled_data = data.multiply(scaler.scale_)
            if not sp.sparse.isspmatrix_csr(scaled_data):
                scaled_data = scaled_data.tocsr()
        else:
            scaled_data = (data - scaler.mean_) / scaler.scale_

        new_df_case = pd.DataFrame(data=scaled_data, columns=self.indep)
        sampled_class_frequency = 0

        n_defect_class = np.sum(self.blackbox_model.predict(new_df_case.loc[:, self.indep]))
        if debug:
            print("Random seed", random_seed, "nDefective", n_defect_class)

        return {
            "synthetic_data": new_df_case,
            "sampled_class_frequency": sampled_class_frequency,
        }

    # -------------------- parsing top rules into feature-level advice --------------------

    def parse_top_rules(
        self,
        top_k_positive_rules: pd.DataFrame,
        top_k_negative_rules: pd.DataFrame,
    ):
        """Parse top k positive & negative rules into variable-level thresholds.

        Returns
        -------
        dict with keys 'top_tofollow_rules' and 'top_toavoid_rules'
        """
        smaller_top_rule = min(len(top_k_positive_rules), len(top_k_negative_rules))
        if self.get_top_k_rules() > smaller_top_rule:
            self.set_top_k_rules(smaller_top_rule)

        top_variables = []
        top_k_toavoid_rules = []
        top_k_tofollow_rules = []

        # Positive rules -> to avoid (defect side)
        for i in range(len(top_k_positive_rules)):
            tmp_rule = str(top_k_positive_rules["rule"].iloc[i]).strip()
            subrules = [s.strip() for s in tmp_rule.split("&") if s.strip()]
            for j in subrules:
                tmp_sub_rule = j.split()
                if len(tmp_sub_rule) < 3:
                    continue
                tmp_variable = tmp_sub_rule[0]
                tmp_condition_variable = tmp_sub_rule[1]
                tmp_value = tmp_sub_rule[2]
                if tmp_variable not in top_variables:
                    top_variables.append(tmp_variable)
                    top_k_toavoid_rules.append(
                        {
                            "variable": tmp_variable,
                            "lessthan": tmp_condition_variable[0] == "<",
                            "value": tmp_value,
                        }
                    )
                if len(top_k_toavoid_rules) == self.get_top_k_rules():
                    break
            if len(top_k_toavoid_rules) == self.get_top_k_rules():
                break

        # Negative rules -> to follow (clean side)
        for i in range(len(top_k_negative_rules)):
            tmp_rule = str(top_k_negative_rules["rule"].iloc[i]).strip()
            subrules = [s.strip() for s in tmp_rule.split("&") if s.strip()]
            for j in subrules:
                tmp_sub_rule = j.split()
                if len(tmp_sub_rule) < 3:
                    continue
                tmp_variable = tmp_sub_rule[0]
                tmp_condition_variable = tmp_sub_rule[1]
                tmp_value = tmp_sub_rule[2]
                if tmp_variable not in top_variables:
                    top_variables.append(tmp_variable)
                    top_k_tofollow_rules.append(
                        {
                            "variable": tmp_variable,
                            "lessthan": tmp_condition_variable[0] == "<",
                            "value": tmp_value,
                        }
                    )
                if len(top_k_tofollow_rules) == self.get_top_k_rules():
                    break
            if len(top_k_tofollow_rules) == self.get_top_k_rules():
                break

        if not top_k_tofollow_rules:
            print("PyExplainer could not find rules to follow (clean side).")
        if not top_k_toavoid_rules:
            print("PyExplainer could not find rules to avoid (defect side).")

        return {
            "top_tofollow_rules": top_k_tofollow_rules,
            "top_toavoid_rules": top_k_toavoid_rules,
        }
