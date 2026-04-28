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

STYLE_PRIOR_M = 100
BREWER_PRIOR_M = 50
ABV_PRIOR_M = 100
ABV_BINS = [0, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, np.inf]

HYPERPARAMETER_CONFIGS = [
    {"config_name": "balanced", "style_weight": 0.40, "brewer_weight": 0.40, "abv_weight": 0.20},
    {"config_name": "style_heavy", "style_weight": 0.60, "brewer_weight": 0.25, "abv_weight": 0.15},
    {"config_name": "brewer_heavy", "style_weight": 0.25, "brewer_weight": 0.60, "abv_weight": 0.15},
    {"config_name": "style_brewer_only", "style_weight": 0.50, "brewer_weight": 0.50, "abv_weight": 0.00},
    {
        "config_name": "style_abv",
        "style_weight": 0.60,
        "brewer_weight": 0.00,
        "abv_weight": 0.40,
    },
]

# Tuning grid is active in HYPERPARAMETER_CONFIGS above.
# Keep this historical list for reference when editing the sweep.
# TUNING_GRID_CONFIGS = [
#     {"config_name": "balanced", "style_weight": 0.40, "brewer_weight": 0.40, "abv_weight": 0.20},
#     {"config_name": "style_heavy", "style_weight": 0.60, "brewer_weight": 0.25, "abv_weight": 0.15},
#     {"config_name": "brewer_heavy", "style_weight": 0.25, "brewer_weight": 0.60, "abv_weight": 0.15},
#     {"config_name": "style_brewer_only", "style_weight": 0.50, "brewer_weight": 0.50, "abv_weight": 0.00},
#     {"config_name": "style_abv", "style_weight": 0.60, "brewer_weight": 0.00, "abv_weight": 0.40},
# ]
BEST_METRIC = "NDCG"

CASE3_RECORDS_OUT = "cold_case3_records.csv"
RECOMMENDATIONS_OUT = "cold_case3_ranked_recommendations.csv"
USER_METRICS_OUT = "cold_case3_user_metrics.csv"
SUMMARY_OUT = "cold_case3_summary.csv"


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


def build_cold_case3_df(train_df, test_df):
    seen_user_ids = set(train_df[USER_COL].unique())
    seen_item_ids = set(train_df[ITEM_COL].unique())

    return test_df[
        (~test_df[USER_COL].isin(seen_user_ids))
        & (~test_df[ITEM_COL].isin(seen_item_ids))
    ].copy()


def build_candidate_items_df(case3_df):
    keep_cols = [ITEM_COL]
    for col in METADATA_COLS:
        if col in case3_df.columns:
            keep_cols.append(col)

    return (
        case3_df[keep_cols]
        .copy()
        .drop_duplicates(subset=[ITEM_COL])
        .reset_index(drop=True)
    )


def _abv_bin_codes(series):
    return pd.cut(
        to_numeric(series),
        bins=ABV_BINS,
        right=False,
        include_lowest=True,
    ).astype("string")


def _build_prior_table(train_df, group_col, prior_m, global_avg_rating):
    prior_df = (
        train_df.dropna(subset=[group_col])
        .groupby(group_col, sort=False)[RATING_COL]
        .agg(prior_avg_rating="mean", prior_count="size")
        .reset_index()
    )

    prior_df["prior_count"] = prior_df["prior_count"].astype(np.int64)
    prior_df["prior_score"] = (
        (
            prior_df["prior_count"] * prior_df["prior_avg_rating"]
            + float(prior_m) * global_avg_rating
        )
        / (prior_df["prior_count"] + float(prior_m))
    )

    return prior_df[[group_col, "prior_score", "prior_count"]]


def build_metadata_priors(train_df):
    train_df = train_df.copy()
    global_avg_rating = float(train_df[RATING_COL].mean())

    train_df["_style_key"] = train_df[STYLE_COL].astype("string")
    train_df["_brewer_key"] = train_df[BREWER_COL].astype("string")
    train_df["_abv_bin"] = _abv_bin_codes(train_df[ABV_COL])

    style_prior_df = _build_prior_table(
        train_df=train_df,
        group_col="_style_key",
        prior_m=STYLE_PRIOR_M,
        global_avg_rating=global_avg_rating,
    ).rename(
        columns={
            "_style_key": STYLE_COL,
            "prior_score": "style_prior_score",
            "prior_count": "style_prior_count",
        }
    )

    brewer_prior_df = _build_prior_table(
        train_df=train_df,
        group_col="_brewer_key",
        prior_m=BREWER_PRIOR_M,
        global_avg_rating=global_avg_rating,
    ).rename(
        columns={
            "_brewer_key": BREWER_COL,
            "prior_score": "brewer_prior_score",
            "prior_count": "brewer_prior_count",
        }
    )

    abv_prior_df = _build_prior_table(
        train_df=train_df,
        group_col="_abv_bin",
        prior_m=ABV_PRIOR_M,
        global_avg_rating=global_avg_rating,
    ).rename(
        columns={
            "prior_score": "abv_prior_score",
            "prior_count": "abv_prior_count",
        }
    )

    return {
        "global_avg_rating": global_avg_rating,
        "style_prior_df": style_prior_df,
        "brewer_prior_df": brewer_prior_df,
        "abv_prior_df": abv_prior_df,
    }


def score_candidate_items(candidate_items_df, priors, style_weight, brewer_weight, abv_weight):
    scored_df = candidate_items_df.copy()
    scored_df["_style_key"] = scored_df[STYLE_COL].astype("string")
    scored_df["_brewer_key"] = scored_df[BREWER_COL].astype("string")
    scored_df["_abv_bin"] = _abv_bin_codes(scored_df[ABV_COL])

    scored_df = scored_df.merge(
        priors["style_prior_df"],
        left_on="_style_key",
        right_on=STYLE_COL,
        how="left",
        suffixes=("", "_style_prior"),
    )
    if f"{STYLE_COL}_style_prior" in scored_df.columns:
        scored_df = scored_df.drop(columns=[f"{STYLE_COL}_style_prior"])

    scored_df = scored_df.merge(
        priors["brewer_prior_df"],
        left_on="_brewer_key",
        right_on=BREWER_COL,
        how="left",
        suffixes=("", "_brewer_prior"),
    )
    if f"{BREWER_COL}_brewer_prior" in scored_df.columns:
        scored_df = scored_df.drop(columns=[f"{BREWER_COL}_brewer_prior"])

    scored_df = scored_df.merge(
        priors["abv_prior_df"],
        on="_abv_bin",
        how="left",
    )

    global_avg_rating = priors["global_avg_rating"]
    prior_cols = ["style_prior_score", "brewer_prior_score", "abv_prior_score"]
    count_cols = ["style_prior_count", "brewer_prior_count", "abv_prior_count"]

    for col in prior_cols:
        scored_df[col] = scored_df[col].fillna(global_avg_rating).astype(float)

    for col in count_cols:
        scored_df[col] = scored_df[col].fillna(0).astype(np.int64)

    scored_df["cold_case3_score"] = (
        scored_df["style_prior_score"] * float(style_weight)
        + scored_df["brewer_prior_score"] * float(brewer_weight)
        + scored_df["abv_prior_score"] * float(abv_weight)
    )

    scored_df = scored_df.sort_values(
        [
            "cold_case3_score",
            "style_prior_count",
            "brewer_prior_count",
            ITEM_COL,
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return scored_df.drop(columns=["_style_key", "_brewer_key", "_abv_bin"])


def add_case3_predictions(case3_df, scored_candidates_df):
    if case3_df.empty:
        df = case3_df.copy()
        df["predicted_rating"] = pd.Series(dtype="float64")
        return df

    prediction_df = scored_candidates_df[[ITEM_COL, "cold_case3_score"]].copy()
    prediction_df[ITEM_COL] = prediction_df[ITEM_COL].astype(str)
    df = case3_df.copy()
    df[ITEM_COL] = df[ITEM_COL].astype(str)
    df = df.merge(prediction_df, on=ITEM_COL, how="left")
    df["predicted_rating"] = pd.to_numeric(
        df["cold_case3_score"],
        errors="coerce",
    ).clip(lower=1.0, upper=5.0)
    return df.drop(columns=["cold_case3_score"])


def topk_items_from_ranked_candidates(ranked_candidates_df, top_k):
    cols = [
        ITEM_COL,
        "cold_case3_score",
        "style_prior_score",
        "brewer_prior_score",
        "abv_prior_score",
        "style_prior_count",
        "brewer_prior_count",
        "abv_prior_count",
    ]
    for col in METADATA_COLS:
        if col in ranked_candidates_df.columns:
            cols.append(col)

    topk_df = ranked_candidates_df[cols].head(top_k).copy()
    topk_df["rank"] = np.arange(1, len(topk_df) + 1, dtype=np.int32)
    return topk_df


def build_relevant_items_by_user(case3_df, relevance_threshold):
    positives = case3_df[case3_df[RATING_COL] >= relevance_threshold].copy()
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
        "cold_case3_score",
        "style_prior_score",
        "brewer_prior_score",
        "abv_prior_score",
        "style_prior_count",
        "brewer_prior_count",
        "abv_prior_count",
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


def build_summary(
    case3_df,
    candidate_items_df,
    user_metrics_df,
    config_name,
    top_k,
    relevance_threshold,
    style_weight,
    brewer_weight,
    abv_weight,
    global_avg_rating,
):
    return {
        "case": "cold_case3_unseen_user_unseen_item",
        "config_name": config_name,
        "top_k": int(top_k),
        "relevance_threshold": float(relevance_threshold),
        "style_weight": float(style_weight),
        "brewer_weight": float(brewer_weight),
        "abv_weight": float(abv_weight),
        "style_prior_m": int(STYLE_PRIOR_M),
        "brewer_prior_m": int(BREWER_PRIOR_M),
        "abv_prior_m": int(ABV_PRIOR_M),
        "global_avg_rating": float(global_avg_rating),
        "num_case3_rows": int(len(case3_df)),
        "num_case3_users": int(case3_df[USER_COL].nunique()) if not case3_df.empty else 0,
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
        "RMSE": (
            float(np.sqrt(
                pd.to_numeric(user_metrics_df["RMSE"], errors="coerce")
                .dropna()
                .pow(2)
                .mean()
            ))
            if not user_metrics_df.empty
            and "RMSE" in user_metrics_df.columns
            and pd.to_numeric(user_metrics_df["RMSE"], errors="coerce").notna().any()
            else 0.0
        ),
        "num_rmse_items": (
            int(pd.to_numeric(user_metrics_df["num_rmse_items"], errors="coerce").fillna(0).sum())
            if not user_metrics_df.empty and "num_rmse_items" in user_metrics_df.columns
            else 0
        ),
    }


def evaluate_cold_case3_batch(
    train_df,
    test_df,
    top_k=10,
    relevance_threshold=4.0,
    style_weight=0.40,
    brewer_weight=0.40,
    abv_weight=0.20,
    config_name="config",
):
    train_df = normalize_df(train_df, "train_df")
    test_df = normalize_df(test_df, "test_df")

    priors = build_metadata_priors(train_df)
    case3_df = build_cold_case3_df(train_df, test_df)
    relevant_items_by_user = build_relevant_items_by_user(case3_df, relevance_threshold)

    if case3_df.empty:
        summary = build_summary(
            case3_df=case3_df,
            candidate_items_df=pd.DataFrame(),
            user_metrics_df=pd.DataFrame(),
            config_name=config_name,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
            style_weight=style_weight,
            brewer_weight=brewer_weight,
            abv_weight=abv_weight,
            global_avg_rating=priors["global_avg_rating"],
        )
        return summary, case3_df, pd.DataFrame(), pd.DataFrame()

    candidate_items_df = build_candidate_items_df(case3_df)
    ranked_candidates_df = score_candidate_items(
        candidate_items_df=candidate_items_df,
        priors=priors,
        style_weight=style_weight,
        brewer_weight=brewer_weight,
        abv_weight=abv_weight,
    )
    case3_df = add_case3_predictions(case3_df, ranked_candidates_df)
    topk_df = topk_items_from_ranked_candidates(ranked_candidates_df, top_k)

    user_ids = sorted(case3_df[USER_COL].dropna().astype(str).unique().tolist())
    recommendations_df, user_metrics_df = build_recommendations_and_metrics(
        topk_df=topk_df,
        user_ids=user_ids,
        relevant_items_by_user=relevant_items_by_user,
        top_k=top_k,
    )
    recommendations_df["predicted_rating"] = pd.to_numeric(
        recommendations_df["cold_case3_score"],
        errors="coerce",
    ).clip(lower=1.0, upper=5.0)
    recommendations_df, rmse_user_metrics_df = attach_rating_rmse(
        recommendations_df=recommendations_df,
        truth_df=case3_df,
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
        case3_df,
        prediction_col="predicted_rating",
        rating_col=RATING_COL,
    )

    summary = build_summary(
        case3_df=case3_df,
        candidate_items_df=candidate_items_df,
        user_metrics_df=user_metrics_df,
        config_name=config_name,
        top_k=top_k,
        relevance_threshold=relevance_threshold,
        style_weight=style_weight,
        brewer_weight=brewer_weight,
        abv_weight=abv_weight,
        global_avg_rating=priors["global_avg_rating"],
    )
    summary["RMSE"] = rmse
    summary["num_rmse_items"] = num_rmse_items

    return summary, case3_df, recommendations_df, user_metrics_df


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


def evaluate_configs(train_df, eval_df, configs, selection_stage, train_source, eval_source):
    results = []
    for config in configs:
        print(f"\nRunning {selection_stage} config: {config['config_name']}")
        summary, case3_df, recommendations_df, user_metrics_df = evaluate_cold_case3_batch(
            train_df=train_df,
            test_df=eval_df,
            top_k=TOP_K,
            relevance_threshold=RELEVANCE_THRESHOLD,
            style_weight=config["style_weight"],
            brewer_weight=config["brewer_weight"],
            abv_weight=config["abv_weight"],
            config_name=config["config_name"],
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
                "case3_df": case3_df,
                "recommendations_df": recommendations_df,
                "user_metrics_df": user_metrics_df,
            }
        )
    return results


def validate_non_empty_case(case_df, stage_name):
    if case_df.empty:
        raise ValueError(
            f"{stage_name} produced no cold-case-3 rows. "
            "Cannot tune hyperparameters on an empty validation slice."
        )


def run_from_config():
    full_train_df = load_df(TRAIN_FILE)
    test_df = load_df(TEST_FILE)
    tune_train_df, validation_df = temporal_split_raw(full_train_df, VAL_RATIO)

    validation_results = evaluate_configs(
        train_df=tune_train_df,
        eval_df=validation_df,
        configs=HYPERPARAMETER_CONFIGS,
        selection_stage="validation",
        train_source="temporal_train_prefix",
        eval_source="temporal_train_tail",
    )
    validate_non_empty_case(validation_results[0]["case3_df"], "Validation tuning")

    validation_best_result = select_best_result(validation_results, BEST_METRIC)
    selected_config = {
        "config_name": validation_best_result["summary"]["config_name"],
        "style_weight": validation_best_result["summary"]["style_weight"],
        "brewer_weight": validation_best_result["summary"]["brewer_weight"],
        "abv_weight": validation_best_result["summary"]["abv_weight"],
    }

    final_results = evaluate_configs(
        train_df=full_train_df,
        eval_df=test_df,
        configs=[selected_config],
        selection_stage="test",
        train_source="full_train",
        eval_source="test",
    )
    final_result = final_results[0]

    summary_df = pd.DataFrame([final_result["summary"]])

    final_result["case3_df"].to_csv(CASE3_RECORDS_OUT, index=False)
    final_result["recommendations_df"].to_csv(RECOMMENDATIONS_OUT, index=False)
    final_result["user_metrics_df"].to_csv(USER_METRICS_OUT, index=False)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    print("\nCold case 3 validation tuning results:")
    print(pd.DataFrame([result["summary"] for result in validation_results]).to_string(index=False))
    print(
        f"\nBest validation config by {BEST_METRIC}: "
        f"{validation_best_result['summary']['config_name']}"
    )
    print("\nFinal test summary with validation-selected config:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved cold-case records to: {CASE3_RECORDS_OUT}")
    print(f"Saved final ranked recommendations to: {RECOMMENDATIONS_OUT}")
    print(f"Saved final user metrics to: {USER_METRICS_OUT}")
    print(f"Saved final summary to: {SUMMARY_OUT}")


if __name__ == "__main__":
    run_from_config()
