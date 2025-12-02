import os
import dice_ml
import pandas as pd
from pyexplainer_core import PyExplainer as pyexplainer_pyexplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
import warnings
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import math
import numpy as np

import math
import copy
import numpy as np
import pandas as pd
import dice_ml
from sklearn.preprocessing import StandardScaler


def generate_instance_counterfactual(
    X_train,
    X_explain,
    y_explain,
    indep,
    dep,
    blackbox_model,
    pretain_model,
    random_state=None,
    debug=False,
):
    """CfExplainer-style generation using neighbourhood similarity + DiCE."""

    # --- defensive copies & basic shaping --- #
    X_train_i = X_train.copy()
    X_explain = X_explain.copy()
    y_explain = y_explain.copy()

    X_train_i.reset_index(drop=True, inplace=True)
    X_explain.reset_index(drop=True, inplace=True)

    X_train_i = X_train_i.loc[:, indep]
    X_explain = X_explain.loc[:, indep]

    # y_explain -> DataFrame with column dep
    if isinstance(y_explain, pd.Series):
        if y_explain.name != dep:
            y_explain = y_explain.rename(dep)
        y_explain = y_explain.reset_index(drop=True).to_frame()
    else:
        y_explain = y_explain.reset_index(drop=True)[[dep]]

    # global model predictions on train
    target_train = blackbox_model.predict(X_train_i)

    # --- scaling --- #
    scaler = StandardScaler()
    trainset_normalize = X_train_i.copy()
    cases_normalize = X_explain.copy()

    train_objs_num = len(trainset_normalize)
    dataset = pd.concat([trainset_normalize, cases_normalize], axis=0)

    if debug:
        print("[DEBUG] continuous cols:", indep)
        print("[DEBUG] type(indep):", type(indep))

    dataset[indep] = scaler.fit_transform(dataset[indep])

    trainset_normalize = dataset.iloc[:train_objs_num].copy()
    cases_normalize = dataset.iloc[train_objs_num:].copy()

    # --- similarity container --- #
    dist_df = pd.DataFrame(index=trainset_normalize.index.copy())
    width = math.sqrt(len(X_train_i.columns)) * 0.75

    # we only return for the *first* explained instance (same as original code)
    for count, case in cases_normalize.iterrows():
        # euclidean distance in normalized space
        dist = np.linalg.norm(trainset_normalize.sub(np.array(case)), axis=1)
        similarity = np.exp(-(dist ** 2) / (2 * (width ** 2)))

        dist_df["dist"] = similarity
        dist_df["t_target"] = target_train

        unique_classes = dist_df["t_target"].unique()

        # sort by similarity (descending)
        dist_sorted = dist_df.sort_values(
            by=["dist"], ascending=False, inplace=False, kind="mergesort"
        )

        # top 40 per class (no .append)
        top_fourty_parts = []
        for clz in unique_classes:
            part = dist_sorted[dist_sorted["t_target"] == clz].head(40)
            if not part.empty:
                top_fourty_parts.append(part)

        if not top_fourty_parts:
            if debug:
                print("[DEBUG] no top-40 neighbours; skipping case", count)
            continue

        top_fourty_df = pd.concat(top_fourty_parts, axis=0)

        cutoff_similarity = (
            top_fourty_df.nsmallest(1, "dist", keep="last")
            .index.values.astype(int)[0]
        )

        # location of cutoff in sorted df
        min_loc = dist_sorted.index.get_loc(cutoff_similarity)

        train_neigh_sampling_b = dist_sorted.iloc[0 : min_loc + 1]

        target_details = train_neigh_sampling_b.groupby(["t_target"]).size()
        if debug:
            print("[DEBUG] target_details\n", target_details)

        target_details_df = pd.DataFrame(
            {"target": target_details.index, "target_count": target_details.values}
        )

        # undersample majority classes (cap at 200 per class)
        final_neigh_parts = []
        for _, row in target_details_df.iterrows():
            cls = row["target"]
            cnt = row["target_count"]
            cls_df = train_neigh_sampling_b[
                train_neigh_sampling_b["t_target"] == cls
            ]
            if cnt > 200:
                cls_df = cls_df.sample(n=200, random_state=random_state)
            final_neigh_parts.append(cls_df)

        final_neighbours_similarity_df = (
            pd.concat(final_neigh_parts, axis=0)
            if final_neigh_parts
            else pd.DataFrame(columns=dist_sorted.columns)
        )

        if debug:
            print(
                "[DEBUG] final_neighbours_similarity_df shape:",
                final_neighbours_similarity_df.shape,
            )

        # neighbourhood in original feature space
        train_set_neigh = X_train_i[
            X_train_i.index.isin(final_neighbours_similarity_df.index)
        ]
        if debug:
            print("[DEBUG] train_set_neigh shape:", train_set_neigh.shape)

        train_class_neigh = y_explain[
            y_explain.index.isin(final_neighbours_similarity_df.index)
        ]

        # --- DiCE setup --- #
        # Use model predictions on train as outcome (so both classes visible)
        train_for_dice = X_train_i.copy()
        train_for_dice[dep] = target_train

        d = dice_ml.Data(
            dataframe=train_for_dice,
            continuous_features=list(indep),
            outcome_name=dep,
        )

        m = dice_ml.Model(model=pretain_model, backend="sklearn")
        exp = dice_ml.Dice(d, m, method="random")

        synthetic_instance = pd.DataFrame()

        # permitted range: min/max per feature from training data
        permitted_range = {}
        for key in indep:
            col = X_train_i[key]
            permitted_range[key] = [float(col.min()), float(col.max())]

        # generate CFs for each training instance until ~2000 synthetic rows
        for i in X_train_i.index:
            try:
                cf_obj = exp.generate_counterfactuals(
                    X_train_i.iloc[i : i + 1, :],
                    total_CFs=50,
                    permitted_range=permitted_range,
                    desired_class="opposite",
                )
            except Exception as e:
                if debug:
                    print(f"[DEBUG] DiCE failed at i={i}: {e}")
                continue

            if not cf_obj.cf_examples_list:
                continue

            cf = cf_obj.cf_examples_list[0].final_cfs_df
            if cf is None or cf.empty:
                continue

            synthetic_instance = pd.concat(
                [synthetic_instance, cf], axis=0, ignore_index=True
            )

            if len(synthetic_instance) > 2000:
                break

        # --- merge neighbourhood + synthetic and predict --- #
        predict_dataset = pd.concat(
            [train_set_neigh, synthetic_instance], axis=0, ignore_index=True
        )
        target = blackbox_model.predict(predict_dataset)
        target_df = pd.DataFrame(target, columns=[dep])

        new_df_case = pd.concat([predict_dataset, target_df], axis=1)
        new_df_case = np.round(new_df_case, 2)

        sampled_class_frequency = new_df_case.groupby([dep]).size()

        return {
            "synthetic_data": new_df_case,
            "sampled_class_frequency": sampled_class_frequency,
        }

    # if X_explain is empty
    return {
        "synthetic_data": pd.DataFrame(),
        "sampled_class_frequency": pd.Series(dtype=int),
    }
