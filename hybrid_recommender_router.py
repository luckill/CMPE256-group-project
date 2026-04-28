import numpy as np
import pandas as pd

import cold_case1_eval_cpu_optimized as cold_case1
import cold_case2_eval_cpu_optimized as cold_case2
import cold_case3_eval_cpu_optimized as cold_case3
from bandit.bandit_data import load_bandit_table, load_ground_truth
from bandit.bandit_features import get_feature_cols, zscore_apply, zscore_fit_transform
from bandit.bandit_model import LinUCB, select_hybrid_top_k
from evaluation import (
    attach_rating_rmse,
    build_hybrid_summary,
    merge_rmse_user_metrics,
    precision_recall_ndcg_at_k,
)
from utils import rating_to_reward, set_seed


USER_COL = "user_id"
ITEM_COL = "item_id"
RATING_COL = "rating"
NAME_COL = "beer/name"
BREWER_COL = "beer/brewerId"
ABV_COL = "beer/ABV"
STYLE_COL = "beer/style"
METADATA_COLS = [NAME_COL, BREWER_COL, ABV_COL, STYLE_COL]

TRAIN_FILE = "advanced_train_temporal.parquet"
TEST_FILE = "advanced_test_temporal.parquet"
CALIBRATION_FILE = "advanced_calibration_temporal_warm.parquet"

VAL_CANDIDATE_PATH = "top100_candidates_val_warm.parquet"
VAL_USER_EMB_PATH = "user_embeddings_val_warm.parquet"
VAL_ITEM_EMB_PATH = "item_embeddings_val_warm.parquet"
VAL_USER_STATS_PATH = "user_stats_val_warm.parquet"
VAL_ITEM_STATS_PATH = "item_stats_val_warm.parquet"

TEST_CANDIDATE_PATH = "top100_candidates_with_scores_warm.parquet"
TEST_USER_EMB_PATH = "user_embeddings_warm.parquet"
TEST_ITEM_EMB_PATH = "item_embeddings_warm.parquet"
TEST_USER_STATS_PATH = "user_stats_warm.parquet"
TEST_ITEM_STATS_PATH = "item_stats_warm.parquet"

TOP_K = 10
RELEVANCE_THRESHOLD = 4.0
ALPHA = 0.15
LAMBDA_NCF = 0.30
TEST_UPDATE_MODEL = True
WARM_FEATURE_SET = "no_emb_prod"
WARM_FEATURE_EXCLUDE_PREFIXES = ("emb_prod_",)

RECOMMENDATIONS_OUT = "hybrid_router_recommendations.csv"
USER_METRICS_OUT = "hybrid_router_user_metrics.csv"

ROUTE_WARM = "warm_ncf_bandit"
ROUTE_CASE2 = "cold_case2"
ROUTE_CASE1 = "cold_case1"
ROUTE_CASE3 = "cold_case3"
ROUTES = [ROUTE_WARM, ROUTE_CASE2, ROUTE_CASE1, ROUTE_CASE3]


def normalize_raw_df(df):
    df = df.copy()
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")
    df = df.dropna(subset=[USER_COL, ITEM_COL, RATING_COL]).copy()
    df[USER_COL] = df[USER_COL].astype(str)
    df[ITEM_COL] = df[ITEM_COL].astype(str)

    for col in [NAME_COL, BREWER_COL, STYLE_COL]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    if ABV_COL in df.columns:
        df[ABV_COL] = pd.to_numeric(df[ABV_COL], errors="coerce")

    return df


def build_metadata_lookup(train_df, test_df):
    keep_cols = [ITEM_COL] + [col for col in METADATA_COLS if col in train_df.columns or col in test_df.columns]
    metadata_df = pd.concat(
        [
            train_df[[col for col in keep_cols if col in train_df.columns]],
            test_df[[col for col in keep_cols if col in test_df.columns]],
        ],
        ignore_index=True,
    )

    def first_non_null(series):
        values = series.dropna()
        return values.iloc[0] if not values.empty else pd.NA

    metadata_df = metadata_df.dropna(subset=[ITEM_COL]).copy()
    metadata_df[ITEM_COL] = metadata_df[ITEM_COL].astype(str)
    aggregations = {
        col: first_non_null
        for col in METADATA_COLS
        if col in metadata_df.columns
    }
    return metadata_df.groupby(ITEM_COL, sort=False, as_index=False).agg(aggregations)


def build_route_test_context(train_df, test_df):
    routed_test_df = test_df[[USER_COL, ITEM_COL, RATING_COL]].copy()
    routed_test_df[USER_COL] = routed_test_df[USER_COL].astype(str)
    routed_test_df[ITEM_COL] = routed_test_df[ITEM_COL].astype(str)

    seen_users = set(train_df[USER_COL].astype(str).unique())
    seen_items = set(train_df[ITEM_COL].astype(str).unique())
    is_seen_user = routed_test_df[USER_COL].isin(seen_users)
    is_seen_item = routed_test_df[ITEM_COL].isin(seen_items)

    routed_test_df["route"] = np.select(
        [
            is_seen_user & is_seen_item,
            (~is_seen_user) & is_seen_item,
            is_seen_user & (~is_seen_item),
            (~is_seen_user) & (~is_seen_item),
        ],
        [
            ROUTE_WARM,
            ROUTE_CASE2,
            ROUTE_CASE1,
            ROUTE_CASE3,
        ],
    )

    route_users = {
        route: set(route_df[USER_COL].astype(str))
        for route, route_df in routed_test_df.groupby("route", sort=False)
    }

    relevant_items_by_route_user = {}
    relevant_df = routed_test_df[routed_test_df[RATING_COL] >= RELEVANCE_THRESHOLD].copy()
    for (route, user_id), group in relevant_df.groupby(["route", USER_COL], sort=False):
        relevant_items_by_route_user[(route, user_id)] = set(group[ITEM_COL].astype(str))

    return routed_test_df, route_users, relevant_items_by_route_user


def add_metadata(recommendations_df, metadata_lookup):
    if recommendations_df.empty:
        return recommendations_df

    df = recommendations_df.merge(
        metadata_lookup,
        on=ITEM_COL,
        how="left",
        suffixes=("", "_metadata"),
    )

    for col in METADATA_COLS:
        metadata_col = f"{col}_metadata"
        if metadata_col not in df.columns:
            continue

        if col in df.columns:
            df[col] = df[col].combine_first(df[metadata_col])
        else:
            df[col] = df[metadata_col]

        df = df.drop(columns=[metadata_col])

    return df


def load_bandit_artifacts():
    val_bandit_df = load_bandit_table(
        candidate_path=VAL_CANDIDATE_PATH,
        user_emb_path=VAL_USER_EMB_PATH,
        item_emb_path=VAL_ITEM_EMB_PATH,
        user_stats_path=VAL_USER_STATS_PATH,
        item_stats_path=VAL_ITEM_STATS_PATH,
    )
    test_bandit_df = load_bandit_table(
        candidate_path=TEST_CANDIDATE_PATH,
        user_emb_path=TEST_USER_EMB_PATH,
        item_emb_path=TEST_ITEM_EMB_PATH,
        user_stats_path=TEST_USER_STATS_PATH,
        item_stats_path=TEST_ITEM_STATS_PATH,
    )

    _, val_relevant_items, val_reward_map, val_user_order = load_ground_truth(
        CALIBRATION_FILE,
        relevance_threshold=RELEVANCE_THRESHOLD,
    )
    _, test_relevant_items, test_reward_map, test_user_order = load_ground_truth(
        TEST_FILE,
        relevance_threshold=RELEVANCE_THRESHOLD,
    )

    feature_cols = get_feature_cols(
        val_bandit_df,
        exclude_prefixes=WARM_FEATURE_EXCLUDE_PREFIXES,
    )
    missing_feature_cols = [col for col in feature_cols if col not in test_bandit_df.columns]
    if missing_feature_cols:
        missing_features_df = pd.DataFrame(
            0.0,
            index=test_bandit_df.index,
            columns=missing_feature_cols,
        )
        test_bandit_df = pd.concat([test_bandit_df, missing_features_df], axis=1).copy()

    _, mean, std = zscore_fit_transform(val_bandit_df, feature_cols)

    return {
        "val_bandit_df": val_bandit_df,
        "test_bandit_df": test_bandit_df,
        "val_relevant_items": val_relevant_items,
        "val_reward_map": val_reward_map,
        "val_user_order": val_user_order,
        "test_relevant_items": test_relevant_items,
        "test_reward_map": test_reward_map,
        "test_user_order": test_user_order,
        "feature_cols": feature_cols,
        "mean": mean,
        "std": std,
    }


def replay_bandit_users(
    bandit_df,
    reward_by_user_item,
    user_order,
    feature_cols,
    mean,
    std,
    bandit,
    update_model,
    collect_recommendations,
):
    bandit_df = bandit_df.copy()
    bandit_df[USER_COL] = bandit_df[USER_COL].astype(str)
    bandit_df[ITEM_COL] = bandit_df[ITEM_COL].astype(str)

    available_users = set(bandit_df[USER_COL].unique())
    ordered_users = [str(user_id) for user_id in user_order if str(user_id) in available_users]
    extra_users = sorted(available_users - set(ordered_users))
    ordered_users.extend(extra_users)

    recommendation_rows = []

    for user_id in ordered_users:
        event_df = bandit_df[bandit_df[USER_COL] == user_id].copy()
        if event_df.empty:
            continue

        event_df = event_df.sort_values(["rank", ITEM_COL]).reset_index(drop=True)
        x_event = zscore_apply(event_df, feature_cols, mean, std)
        top_idx, bandit_scores, final_scores = select_hybrid_top_k(
            event_df=event_df,
            X_event=x_event,
            bandit=bandit,
            top_k=TOP_K,
            lambda_ncf=LAMBDA_NCF,
        )

        chosen_df = event_df.iloc[top_idx].copy()
        chosen_df["ncf_rank"] = chosen_df["rank"].astype(np.int64)
        chosen_df["rank"] = np.arange(1, len(chosen_df) + 1, dtype=np.int32)
        chosen_df["bandit_score"] = bandit_scores[top_idx]
        chosen_df["final_score"] = final_scores[top_idx]
        chosen_df["route"] = ROUTE_WARM

        if update_model:
            recommended_items = chosen_df[ITEM_COL].astype(str).tolist()
            rewards_selected = np.array(
                [
                    rating_to_reward(reward_by_user_item.get((user_id, item_id)))
                    for item_id in recommended_items
                ],
                dtype=np.float64,
            )
            bandit.update(x_event[top_idx], rewards_selected)

        if collect_recommendations:
            recommendation_rows.append(chosen_df)

    if not recommendation_rows:
        return pd.DataFrame()

    return pd.concat(recommendation_rows, ignore_index=True)


def build_warm_recommendations():
    artifacts = load_bandit_artifacts()
    bandit = LinUCB(d=len(artifacts["feature_cols"]), alpha=ALPHA)

    replay_bandit_users(
        bandit_df=artifacts["val_bandit_df"],
        reward_by_user_item=artifacts["val_reward_map"],
        user_order=artifacts["val_user_order"],
        feature_cols=artifacts["feature_cols"],
        mean=artifacts["mean"],
        std=artifacts["std"],
        bandit=bandit,
        update_model=True,
        collect_recommendations=False,
    )

    warm_recommendations_df = replay_bandit_users(
        bandit_df=artifacts["test_bandit_df"],
        reward_by_user_item=artifacts["test_reward_map"],
        user_order=artifacts["test_user_order"],
        feature_cols=artifacts["feature_cols"],
        mean=artifacts["mean"],
        std=artifacts["std"],
        bandit=bandit,
        update_model=TEST_UPDATE_MODEL,
        collect_recommendations=True,
    )

    return warm_recommendations_df


def build_cold_case_recommendations(train_df, test_df):
    case1_config = cold_case1.HYPERPARAMETER_CONFIGS[0]
    _, case1_eval_df, case1_recs_df, _ = cold_case1.evaluate_cold_case1_batch(
        train_df=train_df,
        test_df=test_df,
        top_k=TOP_K,
        relevance_threshold=RELEVANCE_THRESHOLD,
        style_weight=case1_config["style_weight"],
        abv_weight=case1_config["abv_weight"],
        brewer_weight=case1_config["brewer_weight"],
        abv_scale=case1_config["abv_scale"],
        config_name=case1_config["config_name"],
    )

    _, case2_eval_df, case2_recs_df, _ = cold_case2.evaluate_cold_case2_batch(
        train_df=train_df,
        test_df=test_df,
        top_k=TOP_K,
        relevance_threshold=RELEVANCE_THRESHOLD,
        bayesian_m=cold_case2.BAYESIAN_M_CONFIGS[0],
        config_name=f"bayesian_m_{cold_case2.BAYESIAN_M_CONFIGS[0]}",
    )

    case3_config = cold_case3.HYPERPARAMETER_CONFIGS[0]
    _, case3_eval_df, case3_recs_df, _ = cold_case3.evaluate_cold_case3_batch(
        train_df=train_df,
        test_df=test_df,
        top_k=TOP_K,
        relevance_threshold=RELEVANCE_THRESHOLD,
        style_weight=case3_config["style_weight"],
        brewer_weight=case3_config["brewer_weight"],
        abv_weight=case3_config["abv_weight"],
        config_name=case3_config["config_name"],
    )

    return case1_recs_df, case2_recs_df, case3_recs_df, case1_eval_df, case2_eval_df, case3_eval_df


def build_route_rmse_source(routed_test_df, case1_eval_df, case2_eval_df, case3_eval_df):
    frames = []

    cold_route_frames = [
        (ROUTE_CASE1, case1_eval_df),
        (ROUTE_CASE2, case2_eval_df),
        (ROUTE_CASE3, case3_eval_df),
    ]
    for route, route_df in cold_route_frames:
        if route_df.empty or "predicted_rating" not in route_df.columns:
            continue

        df = route_df[[USER_COL, ITEM_COL, RATING_COL, "predicted_rating"]].copy()
        df["route"] = route
        df["actual_rating"] = pd.to_numeric(df[RATING_COL], errors="coerce")
        df["predicted_rating"] = pd.to_numeric(df["predicted_rating"], errors="coerce").clip(1.0, 5.0)
        valid = df["predicted_rating"].notna() & df["actual_rating"].notna()
        df["squared_error"] = np.nan
        df.loc[valid, "squared_error"] = (
            df.loc[valid, "predicted_rating"] - df.loc[valid, "actual_rating"]
        ) ** 2
        frames.append(df[[USER_COL, ITEM_COL, "route", "predicted_rating", "actual_rating", "squared_error"]])

    warm_df = routed_test_df[routed_test_df["route"] == ROUTE_WARM].copy()
    if not warm_df.empty:
        warm_df = warm_df[[USER_COL, ITEM_COL, "route", RATING_COL]].copy()
        warm_df["predicted_rating"] = np.nan
        warm_df["actual_rating"] = pd.to_numeric(warm_df[RATING_COL], errors="coerce")
        warm_df["squared_error"] = np.nan
        frames.append(warm_df[[USER_COL, ITEM_COL, "route", "predicted_rating", "actual_rating", "squared_error"]])

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def prepare_cold_recommendations(recommendations_df, route, score_col, prediction_col=None):
    if recommendations_df.empty:
        return recommendations_df

    df = recommendations_df.copy()
    df[USER_COL] = df[USER_COL].astype(str)
    df[ITEM_COL] = df[ITEM_COL].astype(str)
    df["route"] = route
    df["final_score"] = pd.to_numeric(df[score_col], errors="coerce")
    if prediction_col is not None and prediction_col in df.columns:
        df["predicted_rating"] = pd.to_numeric(df[prediction_col], errors="coerce")
    elif prediction_col is not None:
        df["predicted_rating"] = np.nan
    elif "predicted_rating" not in df.columns:
        df["predicted_rating"] = df["final_score"]
    df["predicted_rating"] = pd.to_numeric(df["predicted_rating"], errors="coerce").clip(1.0, 5.0)
    return df


def select_final_recommendations(warm_df, case1_df, case2_df, case3_df, route_users):
    selected_frames = []

    route_frames = [
        (ROUTE_WARM, warm_df),
        (ROUTE_CASE2, case2_df),
        (ROUTE_CASE1, case1_df),
        (ROUTE_CASE3, case3_df),
    ]

    for route, route_df in route_frames:
        if route_df.empty:
            continue

        users_to_keep = route_users.get(route, set())
        if not users_to_keep:
            continue

        route_df = route_df.copy()
        route_df[USER_COL] = route_df[USER_COL].astype(str)
        selected_frames.append(route_df[route_df[USER_COL].isin(users_to_keep)].copy())

    if not selected_frames:
        return pd.DataFrame()

    final_df = pd.concat(selected_frames, ignore_index=True)
    return final_df.sort_values(["route", USER_COL, "rank"], ascending=[True, True, True]).reset_index(drop=True)


def evaluate_route_recommendations_at_k(
    recommendations_df,
    relevant_items_by_route_user,
    user_col=USER_COL,
    item_col=ITEM_COL,
    route_col="route",
    k=TOP_K,
):
    if recommendations_df.empty:
        return recommendations_df, pd.DataFrame()

    df = recommendations_df.copy()
    df[user_col] = df[user_col].astype(str)
    df[item_col] = df[item_col].astype(str)
    df = df.sort_values([route_col, user_col, "rank"], ascending=[True, True, True]).reset_index(drop=True)

    metric_rows = []
    is_relevant_values = []

    for (route, user_id), user_recs in df.groupby([route_col, user_col], sort=False):
        relevant_items = set(relevant_items_by_route_user.get((route, user_id), set()))
        recommended_items = user_recs[item_col].astype(str).tolist()
        user_is_relevant = [1 if item_id in relevant_items else 0 for item_id in recommended_items]
        is_relevant_values.extend(user_is_relevant)

        if not relevant_items:
            continue

        precision, recall, ndcg = precision_recall_ndcg_at_k(
            recommended_items,
            relevant_items,
            k=k,
        )
        metric_rows.append(
            {
                user_col: user_id,
                route_col: route,
                "num_recommended_items": int(len(recommended_items)),
                "num_relevant_items": int(len(relevant_items)),
                f"Precision@{k}": float(precision),
                f"Recall@{k}": float(recall),
                f"NDCG@{k}": float(ndcg),
            }
        )

    df["is_relevant"] = is_relevant_values
    return df, pd.DataFrame(metric_rows)


def order_output_columns(recommendations_df):
    if recommendations_df.empty:
        return recommendations_df

    preferred_cols = [
        USER_COL,
        "rank",
        ITEM_COL,
        "route",
        "final_score",
        "predicted_rating",
        "actual_rating",
        "squared_error",
        "is_relevant",
        "ncf_score",
        "ncf_predicted_rating",
        "bandit_score",
        "ncf_rank",
        "cold_case1_score",
        "style_score",
        "abv_score",
        "brewer_score",
        "cold_case2_score",
        "bayesian_score",
        "item_avg_rating",
        "item_rating_count",
        "cold_case3_score",
        "style_prior_score",
        "brewer_prior_score",
        "abv_prior_score",
        "style_prior_count",
        "brewer_prior_count",
        "abv_prior_count",
        NAME_COL,
        BREWER_COL,
        ABV_COL,
        STYLE_COL,
    ]

    cols = [col for col in preferred_cols if col in recommendations_df.columns]
    return recommendations_df[cols]


def main():
    set_seed(42)

    train_df = normalize_raw_df(pd.read_parquet(TRAIN_FILE))
    test_df = normalize_raw_df(pd.read_parquet(TEST_FILE))
    metadata_lookup = build_metadata_lookup(train_df, test_df)
    routed_test_df, route_users, relevant_items_by_route_user = build_route_test_context(train_df, test_df)

    print("Building warm NCF + LinUCB recommendations...")
    warm_recs_df = build_warm_recommendations()

    print("Building cold-case fallback recommendations...")
    (
        case1_recs_df,
        case2_recs_df,
        case3_recs_df,
        case1_eval_df,
        case2_eval_df,
        case3_eval_df,
    ) = build_cold_case_recommendations(
        train_df=train_df,
        test_df=test_df,
    )
    route_rmse_source_df = build_route_rmse_source(
        routed_test_df=routed_test_df,
        case1_eval_df=case1_eval_df,
        case2_eval_df=case2_eval_df,
        case3_eval_df=case3_eval_df,
    )

    warm_recs_df = prepare_cold_recommendations(
        warm_recs_df,
        route=ROUTE_WARM,
        score_col="final_score",
        prediction_col="ncf_predicted_rating",
    )
    case1_recs_df = prepare_cold_recommendations(
        case1_recs_df,
        route=ROUTE_CASE1,
        score_col="cold_case1_score",
    )
    case2_recs_df = prepare_cold_recommendations(
        case2_recs_df,
        route=ROUTE_CASE2,
        score_col="cold_case2_score",
    )
    case3_recs_df = prepare_cold_recommendations(
        case3_recs_df,
        route=ROUTE_CASE3,
        score_col="cold_case3_score",
    )

    final_recs_df = select_final_recommendations(
        warm_df=warm_recs_df,
        case1_df=case1_recs_df,
        case2_df=case2_recs_df,
        case3_df=case3_recs_df,
        route_users=route_users,
    )
    final_recs_df = add_metadata(final_recs_df, metadata_lookup)
    final_recs_df, user_metrics_df = evaluate_route_recommendations_at_k(
        recommendations_df=final_recs_df,
        relevant_items_by_route_user=relevant_items_by_route_user,
        user_col=USER_COL,
        item_col=ITEM_COL,
        route_col="route",
        k=TOP_K,
    )
    final_recs_df, rmse_user_metrics_df = attach_rating_rmse(
        recommendations_df=final_recs_df,
        truth_df=routed_test_df,
        prediction_col="predicted_rating",
        user_col=USER_COL,
        item_col=ITEM_COL,
        rating_col=RATING_COL,
        route_col="route",
    )
    if not route_rmse_source_df.empty and "ncf_predicted_rating" in final_recs_df.columns:
        warm_predictions = final_recs_df[
            (final_recs_df["route"] == ROUTE_WARM)
            & final_recs_df["ncf_predicted_rating"].notna()
            & final_recs_df["actual_rating"].notna()
        ][[USER_COL, ITEM_COL, "route", "ncf_predicted_rating", "actual_rating"]].copy()
        if not warm_predictions.empty:
            warm_predictions = warm_predictions.rename(
                columns={"ncf_predicted_rating": "predicted_rating"}
            )
            warm_predictions["predicted_rating"] = pd.to_numeric(
                warm_predictions["predicted_rating"],
                errors="coerce",
            ).clip(1.0, 5.0)
            warm_predictions["actual_rating"] = pd.to_numeric(
                warm_predictions["actual_rating"],
                errors="coerce",
            )
            valid = warm_predictions["predicted_rating"].notna() & warm_predictions["actual_rating"].notna()
            warm_predictions["squared_error"] = np.nan
            warm_predictions.loc[valid, "squared_error"] = (
                warm_predictions.loc[valid, "predicted_rating"]
                - warm_predictions.loc[valid, "actual_rating"]
            ) ** 2
            route_rmse_source_df = pd.concat(
                [
                    route_rmse_source_df[route_rmse_source_df["route"] != ROUTE_WARM],
                    warm_predictions[
                        [USER_COL, ITEM_COL, "route", "predicted_rating", "actual_rating", "squared_error"]
                    ],
                ],
                ignore_index=True,
            )
    user_metrics_df = merge_rmse_user_metrics(
        user_metrics_df=user_metrics_df,
        rmse_user_metrics_df=rmse_user_metrics_df,
        user_col=USER_COL,
        route_col="route",
    )
    final_recs_df = order_output_columns(final_recs_df)

    summary = build_hybrid_summary(
        user_metrics_df=user_metrics_df,
        recommendations_df=final_recs_df,
        train_df=train_df,
        test_df=test_df,
        routes=ROUTES,
        top_k=TOP_K,
        relevance_threshold=RELEVANCE_THRESHOLD,
        alpha=ALPHA,
        lambda_ncf=LAMBDA_NCF,
        rmse_source_df=route_rmse_source_df,
        user_col=USER_COL,
        item_col=ITEM_COL,
        route_col="route",
    )
    summary["bandit_test_mode"] = (
        "online_simulation" if TEST_UPDATE_MODEL else "frozen_offline"
    )
    summary["test_update_model"] = bool(TEST_UPDATE_MODEL)
    summary["warm_feature_set"] = WARM_FEATURE_SET
    summary_df = pd.DataFrame([summary])

    final_recs_df.to_csv(RECOMMENDATIONS_OUT, index=False)
    user_metrics_df.to_csv(USER_METRICS_OUT, index=False)

    print("\nHybrid router summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved recommendations to: {RECOMMENDATIONS_OUT}")
    print(f"Saved user metrics to: {USER_METRICS_OUT}")


if __name__ == "__main__":
    main()
