import numpy as np
import pandas as pd

from evaluation import (
    attach_rating_rmse,
    evaluate_ranked_recommendations_by_user,
    merge_rmse_user_metrics,
    rmse_from_prediction_columns,
)


USER_COL = "user_id"
ITEM_COL = "item_id"
RATING_COL = "rating"
NAME_COL = "beer/name"
BREWER_COL = "beer/brewerId"
ABV_COL = "beer/ABV"
STYLE_COL = "beer/style"

REQUIRED_COLS = [USER_COL, ITEM_COL, RATING_COL]
METADATA_COLS = [NAME_COL, BREWER_COL, ABV_COL, STYLE_COL]

TRAIN_FILE = "advanced_train_temporal.parquet"
TEST_FILE = "advanced_test_temporal.parquet"
TOP_K = 10
RELEVANCE_THRESHOLD = 4.0
VAL_RATIO = 0.125
BAYESIAN_M_CONFIGS = [100, 150, 200, 250, 300, 350, 400, 450, 500]

# Tuning grid is active in BAYESIAN_M_CONFIGS above.
# Keep this historical list for reference when editing the sweep.
# BAYESIAN_M_TUNING_GRID = [100, 150, 200, 250, 300, 350, 400, 450, 500]
BEST_METRIC = "NDCG"

CASE2_RECORDS_OUT = "cold_case2_records.csv"
RECOMMENDATIONS_OUT = "cold_case2_ranked_recommendations.csv"
USER_METRICS_OUT = "cold_case2_user_metrics.csv"
SUMMARY_OUT = "cold_case2_summary.csv"


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def load_df(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def temporal_split_raw(df, val_ratio):
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    elif "review/time" in df.columns:
        df = df.sort_values("review/time").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    split_idx = int(len(df) * (1.0 - val_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def validate_required_columns(df, df_name):
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def normalize_df(df, df_name):
    validate_required_columns(df, df_name)

    df = df.copy()
    df[RATING_COL] = to_numeric(df[RATING_COL])
    df = df.dropna(subset=[USER_COL, ITEM_COL, RATING_COL]).copy()

    df[USER_COL] = df[USER_COL].astype(str)
    df[ITEM_COL] = df[ITEM_COL].astype(str)

    for col in [NAME_COL, BREWER_COL, STYLE_COL]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    if ABV_COL in df.columns:
        df[ABV_COL] = to_numeric(df[ABV_COL])

    return df


def build_cold_case2_df(train_df, test_df):
    seen_user_ids = set(train_df[USER_COL].unique())
    seen_item_ids = set(train_df[ITEM_COL].unique())

    return test_df[
        (~test_df[USER_COL].isin(seen_user_ids))
        & test_df[ITEM_COL].isin(seen_item_ids)
    ].copy()


def build_candidate_items_df(case2_df, train_df):
    keep_cols = [ITEM_COL]
    for col in METADATA_COLS:
        if col in case2_df.columns:
            keep_cols.append(col)

    candidate_items_df = (
        case2_df[keep_cols]
        .copy()
        .drop_duplicates(subset=[ITEM_COL])
        .reset_index(drop=True)
    )

    missing_metadata_cols = [
        col
        for col in METADATA_COLS
        if col in train_df.columns and col not in candidate_items_df.columns
    ]
    if missing_metadata_cols:
        train_metadata_cols = [ITEM_COL] + missing_metadata_cols
        train_metadata_df = (
            train_df[train_metadata_cols]
            .drop_duplicates(subset=[ITEM_COL])
            .reset_index(drop=True)
        )
        candidate_items_df = candidate_items_df.merge(
            train_metadata_df,
            on=ITEM_COL,
            how="left",
        )

    return candidate_items_df


def build_item_popularity_stats(train_df, bayesian_m):
    global_avg_rating = float(train_df[RATING_COL].mean())

    item_stats_df = (
        train_df.groupby(ITEM_COL, sort=False)[RATING_COL]
        .agg(item_avg_rating="mean", item_rating_count="size")
        .reset_index()
    )

    item_stats_df["item_rating_count"] = item_stats_df["item_rating_count"].astype(np.int64)
    item_stats_df["bayesian_score"] = (
        (
            item_stats_df["item_rating_count"] * item_stats_df["item_avg_rating"]
            + float(bayesian_m) * global_avg_rating
        )
        / (item_stats_df["item_rating_count"] + float(bayesian_m))
    )

    return item_stats_df, global_avg_rating


def build_ranked_candidates(candidate_items_df, item_stats_df):
    ranked_df = candidate_items_df.merge(item_stats_df, on=ITEM_COL, how="inner")
    ranked_df["cold_case2_score"] = ranked_df["bayesian_score"]

    return ranked_df.sort_values(
        ["cold_case2_score", "item_rating_count", ITEM_COL],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def topk_items_from_ranked_candidates(ranked_candidates_df, top_k):
    cols = [
        ITEM_COL,
        "cold_case2_score",
        "bayesian_score",
        "item_avg_rating",
        "item_rating_count",
    ]
    for col in METADATA_COLS:
        if col in ranked_candidates_df.columns:
            cols.append(col)

    topk_df = ranked_candidates_df[cols].head(top_k).copy()
    topk_df["rank"] = np.arange(1, len(topk_df) + 1, dtype=np.int32)
    return topk_df


def add_case2_predictions(case2_df, item_stats_df, global_avg_rating):
    if case2_df.empty:
        df = case2_df.copy()
        df["predicted_rating"] = pd.Series(dtype="float64")
        return df

    prediction_df = item_stats_df[[ITEM_COL, "bayesian_score"]].copy()
    prediction_df[ITEM_COL] = prediction_df[ITEM_COL].astype(str)
    df = case2_df.copy()
    df[ITEM_COL] = df[ITEM_COL].astype(str)
    df = df.merge(prediction_df, on=ITEM_COL, how="left")
    df["predicted_rating"] = (
        pd.to_numeric(df["bayesian_score"], errors="coerce")
        .fillna(float(global_avg_rating))
        .clip(lower=1.0, upper=5.0)
    )
    return df.drop(columns=["bayesian_score"])


def build_relevant_items_by_user(case2_df, relevance_threshold):
    positives = case2_df[case2_df[RATING_COL] >= relevance_threshold].copy()
    if positives.empty:
        return {}

    return (
        positives.groupby(USER_COL, sort=False)[ITEM_COL]
        .apply(lambda s: set(s.dropna().astype(str)))
        .to_dict()
    )


def build_recommendations_and_metrics(
    topk_df,
    user_ids,
    relevant_items_by_user,
    top_k,
):
    recommendation_rows = []

    for user_id in user_ids:
        user_recs = topk_df.copy()
        user_recs.insert(0, USER_COL, user_id)
        recommendation_rows.append(user_recs)

    recommendations_df = (
        pd.concat(recommendation_rows, ignore_index=True)
        if recommendation_rows
        else pd.DataFrame()
    )

    output_cols = [
        USER_COL,
        "rank",
        ITEM_COL,
        "cold_case2_score",
        "bayesian_score",
        "item_avg_rating",
        "item_rating_count",
        "is_relevant",
    ]
    for col in METADATA_COLS:
        if col in recommendations_df.columns:
            output_cols.append(col)

    return evaluate_ranked_recommendations_by_user(
        recommendations_df=recommendations_df,
        relevant_items_by_user=relevant_items_by_user,
        user_col=USER_COL,
        item_col=ITEM_COL,
        k=top_k,
        output_cols=output_cols,
    )


def build_empty_summary(
    config_name,
    top_k,
    relevance_threshold,
    bayesian_m,
    global_avg_rating,
):
    return {
        "case": "cold_case2_unseen_user_seen_item",
        "config_name": config_name,
        "top_k": int(top_k),
        "relevance_threshold": float(relevance_threshold),
        "bayesian_m": int(bayesian_m),
        "global_avg_rating": float(global_avg_rating),
        "num_case2_rows": 0,
        "num_case2_users": 0,
        "num_candidate_items": 0,
        "num_eval_users": 0,
        f"Precision@{top_k}": 0.0,
        f"Recall@{top_k}": 0.0,
        f"NDCG@{top_k}": 0.0,
        "RMSE": 0.0,
        "num_rmse_items": 0,
    }


def evaluate_cold_case2_batch(
    train_df,
    test_df,
    top_k=10,
    relevance_threshold=4.0,
    bayesian_m=20,
    config_name="config",
):
    train_df = normalize_df(train_df, "train_df")
    test_df = normalize_df(test_df, "test_df")

    global_avg_rating = float(train_df[RATING_COL].mean()) if not train_df.empty else 0.0
    case2_df = build_cold_case2_df(train_df, test_df)
    relevant_items_by_user = build_relevant_items_by_user(case2_df, relevance_threshold)

    if case2_df.empty:
        summary = build_empty_summary(
            config_name=config_name,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
            bayesian_m=bayesian_m,
            global_avg_rating=global_avg_rating,
        )
        return summary, case2_df, pd.DataFrame(), pd.DataFrame()

    candidate_items_df = build_candidate_items_df(case2_df, train_df)
    item_stats_df, global_avg_rating = build_item_popularity_stats(train_df, bayesian_m)
    case2_df = add_case2_predictions(case2_df, item_stats_df, global_avg_rating)
    ranked_candidates_df = build_ranked_candidates(candidate_items_df, item_stats_df)
    topk_df = topk_items_from_ranked_candidates(ranked_candidates_df, top_k)

    user_ids = sorted(case2_df[USER_COL].dropna().astype(str).unique().tolist())
    recommendations_df, user_metrics_df = build_recommendations_and_metrics(
        topk_df=topk_df,
        user_ids=user_ids,
        relevant_items_by_user=relevant_items_by_user,
        top_k=top_k,
    )
    recommendations_df["predicted_rating"] = pd.to_numeric(
        recommendations_df["cold_case2_score"],
        errors="coerce",
    ).clip(lower=1.0, upper=5.0)
    recommendations_df, rmse_user_metrics_df = attach_rating_rmse(
        recommendations_df=recommendations_df,
        truth_df=case2_df,
        prediction_col="predicted_rating",
        user_col=USER_COL,
        item_col=ITEM_COL,
        rating_col=RATING_COL,
    )
    user_metrics_df = merge_rmse_user_metrics(
        user_metrics_df=user_metrics_df,
        rmse_user_metrics_df=rmse_user_metrics_df,
        user_col=USER_COL,
    )
    rmse, num_rmse_items = rmse_from_prediction_columns(
        case2_df,
        prediction_col="predicted_rating",
        rating_col=RATING_COL,
    )

    summary = {
        "case": "cold_case2_unseen_user_seen_item",
        "config_name": config_name,
        "top_k": int(top_k),
        "relevance_threshold": float(relevance_threshold),
        "bayesian_m": int(bayesian_m),
        "global_avg_rating": float(global_avg_rating),
        "num_case2_rows": int(len(case2_df)),
        "num_case2_users": int(case2_df[USER_COL].nunique()),
        "num_candidate_items": int(len(candidate_items_df)),
        "num_eval_users": int(len(user_metrics_df)),
        f"Precision@{top_k}": (
            float(user_metrics_df[f"Precision@{top_k}"].mean())
            if not user_metrics_df.empty
            else 0.0
        ),
        f"Recall@{top_k}": (
            float(user_metrics_df[f"Recall@{top_k}"].mean())
            if not user_metrics_df.empty
            else 0.0
        ),
        f"NDCG@{top_k}": (
            float(user_metrics_df[f"NDCG@{top_k}"].mean())
            if not user_metrics_df.empty
            else 0.0
        ),
        "RMSE": rmse,
        "num_rmse_items": num_rmse_items,
    }

    return summary, case2_df, recommendations_df, user_metrics_df


def _metric_column(summary, metric_name):
    prefix = f"{metric_name}@"
    for col in summary:
        if col.startswith(prefix):
            return col
    raise ValueError(f"No metric column found for {metric_name!r}")


def select_best_result(results, metric_name):
    if not results:
        raise ValueError("No tuning results were produced")

    best_result = None
    best_score = -np.inf
    for result in results:
        summary = result["summary"]
        metric_col = _metric_column(summary, metric_name)
        score = summary.get(metric_col, 0.0)
        if pd.isna(score):
            score = 0.0
        if float(score) > best_score:
            best_score = float(score)
            best_result = result

    return best_result


def add_summary_metadata(summary, selection_stage, train_source, eval_source):
    summary.update(
        {
            "selection_stage": selection_stage,
            "train_source": train_source,
            "eval_source": eval_source,
            "val_ratio": float(VAL_RATIO),
        }
    )
    return summary


def evaluate_configs(train_df, eval_df, bayesian_m_values, selection_stage, train_source, eval_source):
    results = []
    for bayesian_m in bayesian_m_values:
        config_name = f"bayesian_m_{bayesian_m}"
        print(f"\nRunning {selection_stage} config: {config_name}")
        summary, case2_df, recommendations_df, user_metrics_df = evaluate_cold_case2_batch(
            train_df=train_df,
            test_df=eval_df,
            top_k=TOP_K,
            relevance_threshold=RELEVANCE_THRESHOLD,
            bayesian_m=bayesian_m,
            config_name=config_name,
        )
        add_summary_metadata(
            summary,
            selection_stage=selection_stage,
            train_source=train_source,
            eval_source=eval_source,
        )
        results.append(
            {
                "summary": summary,
                "case2_df": case2_df,
                "recommendations_df": recommendations_df,
                "user_metrics_df": user_metrics_df,
            }
        )
    return results


def validate_non_empty_case(case_df, stage_name):
    if case_df.empty:
        raise ValueError(
            f"{stage_name} produced no cold-case-2 rows. "
            "Cannot tune hyperparameters on an empty validation slice."
        )


def run_from_config():
    full_train_df = load_df(TRAIN_FILE)
    test_df = load_df(TEST_FILE)
    tune_train_df, validation_df = temporal_split_raw(full_train_df, VAL_RATIO)

    validation_results = evaluate_configs(
        train_df=tune_train_df,
        eval_df=validation_df,
        bayesian_m_values=BAYESIAN_M_CONFIGS,
        selection_stage="validation",
        train_source="temporal_train_prefix",
        eval_source="temporal_train_tail",
    )
    validate_non_empty_case(validation_results[0]["case2_df"], "Validation tuning")

    validation_best_result = select_best_result(validation_results, BEST_METRIC)
    selected_bayesian_m = int(validation_best_result["summary"]["bayesian_m"])

    final_results = evaluate_configs(
        train_df=full_train_df,
        eval_df=test_df,
        bayesian_m_values=[selected_bayesian_m],
        selection_stage="test",
        train_source="full_train",
        eval_source="test",
    )
    final_result = final_results[0]

    summary_df = pd.DataFrame([final_result["summary"]])

    final_result["case2_df"].to_csv(CASE2_RECORDS_OUT, index=False)
    final_result["recommendations_df"].to_csv(RECOMMENDATIONS_OUT, index=False)
    final_result["user_metrics_df"].to_csv(USER_METRICS_OUT, index=False)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    print("\nCold case 2 validation tuning results:")
    print(pd.DataFrame([result["summary"] for result in validation_results]).to_string(index=False))
    print(
        f"\nBest validation config by {BEST_METRIC}: "
        f"{validation_best_result['summary']['config_name']}"
    )
    print("\nFinal test summary with validation-selected config:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved cold-case records to: {CASE2_RECORDS_OUT}")
    print(f"Saved final ranked recommendations to: {RECOMMENDATIONS_OUT}")
    print(f"Saved final user metrics to: {USER_METRICS_OUT}")
    print(f"Saved final summary to: {SUMMARY_OUT}")


if __name__ == "__main__":
    run_from_config()
