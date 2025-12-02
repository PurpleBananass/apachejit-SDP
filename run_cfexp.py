#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CfExplainer-style pipeline over ALL projects using existing data_utils helpers.

What it does
------------
For every project returned by data_utils.read_dataset(), and for each model type:

    - loads train/test and the corresponding model (get_model)
    - finds TRUE POSITIVES in the test set (get_true_positives)
    - for EACH TP instance:
        * builds a synthetic neighbourhood via
          generate_instance_crossover_interpolation(...)
        * mines CWCAR-style class association rules on that neighbourhood
          (separate bug / no-bug rule sets, conflict & redundancy pruning)
        * saves neighbourhood & rules to:
              OUTPUT/{project}/{explainer_name}/{model_type}/
                  synthetic_neighbourhood_idx<test_idx>.csv
                  bug_rules_idx<test_idx>.csv
                  no_bug_rules_idx<test_idx>.csv

If the three files already exist for a given test_idx, that instance is skipped.

Requirements
------------
- data_utils.py with:
    read_dataset, get_model, get_true_positives, get_output_dir
- crossover_interpolation.py (from CfExplainer repo, with DataFrame.append fixed)
- pip install: mlxtend scikit-learn pandas numpy tqdm
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm

from data_utils import (
    read_dataset,
    get_model,
    get_true_positives,
    get_output_dir,
)
from crossover_interpolation import generate_instance_crossover_interpolation


# ---------------------------------------------------------------------
# Supported models
# ---------------------------------------------------------------------

SUPPORTED_MODELS = [
    "RandomForest",
    "SVM",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]


# ---------------------------------------------------------------------
# CWCAR-style Rule Miner (approximation of rule_minning)
# ---------------------------------------------------------------------


class CWCARRuleMiner:
    """
    Approximate CfExplainer's CWCAR rule learner.

    - Discretizes features into bins.
    - Builds transactions: feature items + 'bug'/'no bug' label token.
    - Runs apriori once.
    - Extracts bug vs no-bug rules.
    - Removes conflicting and redundant rules (using support as proxy for weightSupp).
    """

    def __init__(self, T_min_support: float = 0.05, F_min_support: float = 0.05, max_rules: int = 100):
        self.T_min_support = T_min_support
        self.F_min_support = F_min_support
        self.max_rules = max_rules

    def _discretize_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        n_bins: int = 3,
    ) -> pd.DataFrame:
        """
        Quantile-bin numeric features into n_bins and return items like 'la=0', 'ns=2', ...
        """
        if df.empty:
            return pd.DataFrame(index=df.index)

        disc = KBinsDiscretizer(
            n_bins=n_bins,
            encode="ordinal",
            strategy="quantile",
        )
        X_cont = df[feature_cols].values
        X_binned = disc.fit_transform(X_cont)

        df_binned = pd.DataFrame(X_binned, columns=feature_cols, index=df.index)
        df_cat = df_binned.copy()
        for col in feature_cols:
            df_cat[col] = df_cat[col].astype(int).astype(str)
            df_cat[col] = col + "=" + df_cat[col]
        return df_cat

    def mine_rules(
        self,
        synthetic_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        defective_class: int = 1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Mine CWCAR-style rules from the synthetic neighbourhood.

        Returns:
            Trules: rules whose consequent is 'bug'
            Frules: rules whose consequent is 'no bug'
        """
        if synthetic_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        syn_features = synthetic_df[feature_cols]
        syn_labels = synthetic_df[label_col]

        # 1) Discretize features into categorical items
        syn_cat = self._discretize_features(syn_features, feature_cols, n_bins=3)
        if syn_cat.empty:
            return pd.DataFrame(), pd.DataFrame()

        # 2) Build transactional one-hot matrix
        transactions = pd.get_dummies(syn_cat)
        if transactions.empty:
            return pd.DataFrame(), pd.DataFrame()

        # 3) Add label tokens
        bug_token = "bug"
        nobug_token = "no bug"

        transactions[bug_token] = (syn_labels == defective_class).astype(int)
        transactions[nobug_token] = (syn_labels != defective_class).astype(int)

        # 4) Frequent itemsets with a global minimum support
        global_min_support = min(self.T_min_support, self.F_min_support)
        frequent_itemsets = apriori(
            transactions,
            min_support=global_min_support,
            use_colnames=True,
        )
        if frequent_itemsets.empty:
            return pd.DataFrame(), pd.DataFrame()

        # 5) Association rules (use support as metric, like original rule_minning)
        rules = association_rules(
            frequent_itemsets,
            metric="support",
            min_threshold=global_min_support,
        )
        if rules.empty:
            return pd.DataFrame(), pd.DataFrame()

        # 6) Split into bug / no-bug rule sets
        def has_token(consequents, token):
            return token in consequents

        Trules = rules[rules["consequents"].apply(has_token, token=bug_token)].copy()
        Frules = rules[rules["consequents"].apply(has_token, token=nobug_token)].copy()

        # Apply class-specific support thresholds
        Trules = Trules[Trules["support"] >= self.T_min_support].copy()
        Frules = Frules[Frules["support"] >= self.F_min_support].copy()

        # Add length, generate_time, and weightSupp (approximate = support)
        for df in (Trules, Frules):
            if df.empty:
                continue
            df["length"] = df["antecedents"].apply(len)
            df["generate_time"] = range(1, len(df) + 1)
            df["weightSupp"] = df["support"]

        # Rank & trim
        Trules = self._rank_and_trim(Trules)
        Frules = self._rank_and_trim(Frules)

        # Remove conflicting and redundant rules
        Trules, Frules = self._remove_conflicts(Trules, Frules)
        Trules = self._remove_redundant(Trules)
        Frules = self._remove_redundant(Frules)

        return Trules, Frules

    def _rank_and_trim(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.sort_values(
            by=["support", "length", "generate_time"],
            ascending=False,
        ).reset_index(drop=True)
        if len(df) > self.max_rules:
            df = df.iloc[: self.max_rules].copy()
        return df

    def _remove_conflicts(self, Trules: pd.DataFrame, Frules: pd.DataFrame):
        """
        Remove rules whose antecedent maps to >1 distinct consequent (bug vs no bug).
        """
        if Trules.empty and Frules.empty:
            return Trules, Frules

        columns = Trules.columns if not Trules.empty else Frules.columns
        rules = pd.concat([Trules, Frules], axis=0, ignore_index=True)

        # antecedents that have more than one distinct consequent
        conflict_mask = rules.groupby("antecedents")["consequents"].nunique().gt(1)
        conflict_ants = conflict_mask[conflict_mask].index

        rules_clean = (
            rules[~rules["antecedents"].isin(conflict_ants)]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        rulesT = []
        rulesF = []
        for _, row in rules_clean.iterrows():
            cons = row["consequents"]
            if "bug" in cons:
                rulesT.append(row)
            if "no bug" in cons:
                rulesF.append(row)

        Trules_out = (
            pd.DataFrame(rulesT, columns=columns).reset_index(drop=True)
            if rulesT
            else pd.DataFrame(columns=columns)
        )
        Frules_out = (
            pd.DataFrame(rulesF, columns=columns).reset_index(drop=True)
            if rulesF
            else pd.DataFrame(columns=columns)
        )
        return Trules_out, Frules_out

    def _remove_redundant(self, rules: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rules whose antecedent is a superset with lower weightSupp
        (approximate of original remove_redundant_rule).
        """
        if rules.empty:
            return rules

        drop_idx = set()
        n = len(rules)
        for i in range(n):
            for j in range(i + 1, n):
                ants_i = frozenset(rules.loc[i, "antecedents"])
                ants_j = frozenset(rules.loc[j, "antecedents"])
                wi = rules.loc[i, "weightSupp"]
                wj = rules.loc[j, "weightSupp"]

                if ants_i.issubset(ants_j) and wi > wj:
                    drop_idx.add(j)
                elif ants_j.issubset(ants_i) and wj > wi:
                    drop_idx.add(i)

        if drop_idx:
            rules = rules.drop(list(drop_idx), axis=0)
        return rules.reset_index(drop=True)


# ---------------------------------------------------------------------
# Per-project + per-model processing
# ---------------------------------------------------------------------


def run_project(
    project_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_type: str,
    label_col: str,
    explainer_name: str,
    miner: CWCARRuleMiner,
):
    """
    Run CfExplainer (neighbourhood + CWCAR rules) for all TRUE POSITIVES
    in the given project, for a specific model_type.
    """
    if label_col not in train_df.columns or label_col not in test_df.columns:
        print(f"[WARN] Skipping {project_name} ({model_type}): label column '{label_col}' missing.")
        return

    feature_cols = [c for c in train_df.columns if c != label_col]
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[label_col].copy()

    print(f"\n[PROJECT] {project_name} | MODEL {model_type}")
    print(f"[INFO] Feature columns ({len(feature_cols)}): {feature_cols}")
    print("[INFO] Train label distribution:\n", train_df[label_col].value_counts())
    print("[INFO] Test label distribution:\n", y_test.value_counts())

    # Load model
    try:
        model = get_model(project_name, model_type)
    except Exception as e:
        print(f"[WARN] Could not load model '{model_type}' for project '{project_name}': {e}")
        return

    print(f"[INFO] Loaded model for {project_name} ({model_type}).")

    # True positives to explain
    tps = get_true_positives(model, train_df, test_df, label=label_col)
    if tps.empty:
        print(f"[INFO] No true positives for {project_name} ({model_type}), skipping.")
        return

    print(f"[INFO] Number of true positives in test set: {len(tps)}")

    out_dir = get_output_dir(project_name, explainer_name, model_type)

    for test_idx in tqdm(tps.index, desc=f"{project_name} ({model_type}) TPs"):
        # Paths for this instance
        syn_path = out_dir / f"synthetic_neighbourhood_idx{test_idx}.csv"
        bug_rules_path = out_dir / f"bug_rules_idx{test_idx}.csv"
        nobug_rules_path = out_dir / f"no_bug_rules_idx{test_idx}.csv"

        # Skip if all outputs already exist
        if syn_path.exists() and bug_rules_path.exists() and nobug_rules_path.exists():
            continue

        # X / y to explain
        X_explain = X_test.loc[[test_idx]]
        y_explain = y_test.loc[[test_idx]]

        if y_explain.iloc[0] != 1:
            # Safety check: get_true_positives should guarantee this, but just in case
            continue

        # Generate synthetic neighbourhood
        syn = generate_instance_crossover_interpolation(
            X_train=X_train,
            X_explain=X_explain,
            y_explain=y_explain.to_frame(),
            indep=feature_cols,
            dep=label_col,
            blackbox_model=model,
            random_state=0,
            debug=False,
        )
        synthetic_df = syn["synthetic_data"]

        # Save neighbourhood regardless of rule success
        synthetic_df.to_csv(syn_path, index=True)

        # Mine rules
        Trules, Frules = miner.mine_rules(
            synthetic_df=synthetic_df,
            feature_cols=feature_cols,
            label_col=label_col,
            defective_class=1,
        )

        if Trules.empty and Frules.empty:
            # Still have the neighbourhood saved; nothing more to do here
            continue

        Trules.to_csv(bug_rules_path, index=False)
        Frules.to_csv(nobug_rules_path, index=False)


# ---------------------------------------------------------------------
# Main pipeline over ALL projects × ALL models
# ---------------------------------------------------------------------


def main():
    parser = ArgumentParser(
        description="CfExplainer-style local rule explanations over ALL projects and models.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="all",
        choices=SUPPORTED_MODELS + ["all"],
        help="Model name passed to get_model, or 'all' to run all supported models.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="target",
        help="Name of label column (default: target).",
    )
    parser.add_argument(
        "--T-min-support",
        type=float,
        default=0.05,
        help="Min support for defective (bug) rules (default: 0.05).",
    )
    parser.add_argument(
        "--F-min-support",
        type=float,
        default=0.05,
        help="Min support for non-defective (no bug) rules (default: 0.05).",
    )
    parser.add_argument(
        "--max-rules",
        type=int,
        default=1,
        help="Maximum number of rules per class after ranking (default: 100).",
    )
    parser.add_argument(
        "--explainer-name",
        type=str,
        default="CfExplainer",
        help="Name for explainer folder in OUTPUT (default: CfExplainer).",
    )

    args = parser.parse_args()

    # Build a single miner instance with the desired thresholds
    miner = CWCARRuleMiner(
        T_min_support=args.T_min_support,
        F_min_support=args.F_min_support,
        max_rules=args.max_rules,
    )

    # Determine which models to run
    if args.model_type == "all":
        model_types = SUPPORTED_MODELS
    else:
        model_types = [args.model_type]

    # Load all projects
    projects = read_dataset()
    project_names = sorted(projects.keys())

    print(f"[INFO] Found {len(project_names)} projects: {project_names}")
    print(f"[INFO] Running models: {model_types}")

    for project_name in project_names:
        train_df, test_df = projects[project_name]
        for model_type in model_types:
            run_project(
                project_name=project_name,
                train_df=train_df,
                test_df=test_df,
                model_type=model_type,
                label_col=args.label_col,
                explainer_name=args.explainer_name,
                miner=miner,
            )


if __name__ == "__main__":
    main()
