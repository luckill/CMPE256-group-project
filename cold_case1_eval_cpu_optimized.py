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

HYPERPARAMETER_CONFIGS = [
    {"config_name": "baseline", "style_weight": 0.45, "abv_weight": 0.35, "brewer_weight": 0.20, "abv_scale": 1.5},
    {"config_name": "style_heavy", "style_weight": 0.60, "abv_weight": 0.25, "brewer_weight": 0.15, "abv_scale": 1.5},
    {"config_name": "abv_heavy", "style_weight": 0.30, "abv_weight": 0.55, "brewer_weight": 0.15, "abv_scale": 1.5},
    {"config_name": "brewer_heavy", "style_weight": 0.35, "abv_weight": 0.25, "brewer_weight": 0.40, "abv_scale": 1.5},
    {"config_name": "abv_tight", "style_weight": 0.45, "abv_weight": 0.35, "brewer_weight": 0.20, "abv_scale": 1.0},
    {"config_name": "abv_loose", "style_weight": 0.45, "abv_weight": 0.35, "brewer_weight": 0.20, "abv_scale": 2.5},
    {
        "config_name": "brewer_50_style30_abv20",
        "style_weight": 0.30,
        "abv_weight": 0.20,
        "brewer_weight": 0.50,
        "abv_scale": 1.5,
    },
    {"config_name": "brewer_50_style25_abv25", "style_weight": 0.25, "abv_weight": 0.25, "brewer_weight": 0.50, "abv_scale": 1.5},
    {"config_name": "brewer_40_style40_abv20", "style_weight": 0.40, "abv_weight": 0.20, "brewer_weight": 0.40, "abv_scale": 1.5},
    {"config_name": "brewer_40_style30_abv30", "style_weight": 0.30, "abv_weight": 0.30, "brewer_weight": 0.40, "abv_scale": 1.5},
    {"config_name": "brewer_heavy_abv_scale_2_0", "style_weight": 0.35, "abv_weight": 0.25, "brewer_weight": 0.40, "abv_scale": 2.0},
    {"config_name": "brewer_heavy_abv_scale_2_5", "style_weight": 0.35, "abv_weight": 0.25, "brewer_weight": 0.40, "abv_scale": 2.5},
    {"config_name": "brewer_heavy_abv_scale_3_0", "style_weight": 0.35, "abv_weight": 0.25, "brewer_weight": 0.40, "abv_scale": 3.0},
]

# Tuning grid is active in HYPERPARAMETER_CONFIGS above.
# Keep this historical list for reference when editing the sweep.
# TUNING_GRID_CONFIGS = [
#     {"config_name": "baseline", "style_weight": 0.45, "abv_weight": 0.35, "brewer_weight": 0.20, "abv_scale": 1.5},
#     {"config_name": "style_heavy", "style_weight": 0.60, "abv_weight": 0.25, "brewer_weight": 0.15, "abv_scale": 1.5},
#     {"config_name": "abv_heavy", "style_weight": 0.30, "abv_weight": 0.55, "brewer_weight": 0.15, "abv_scale": 1.5},
#     {"config_name": "brewer_heavy", "style_weight": 0.35, "abv_weight": 0.25, "brewer_weight": 0.40, "abv_scale": 1.5},
#     {"config_name": "abv_tight", "style_weight": 0.45, "abv_weight": 0.35, "brewer_weight": 0.20, "abv_scale": 1.0},
#     {"config_name": "abv_loose", "style_weight": 0.45, "abv_weight": 0.35, "brewer_weight": 0.20, "abv_scale": 2.5},
#     {"config_name": "brewer_50_style30_abv20", "style_weight": 0.30, "abv_weight": 0.20, "brewer_weight": 0.50, "abv_scale": 1.5},
#     {"config_name": "brewer_50_style25_abv25", "style_weight": 0.25, "abv_weight": 0.25, "brewer_weight": 0.50, "abv_scale": 1.5},
#     {"config_name": "brewer_40_style40_abv20", "style_weight": 0.40, "abv_weight": 0.20, "brewer_weight": 0.40, "abv_scale": 1.5},
#     {"config_name": "brewer_40_style30_abv30", "style_weight": 0.30, "abv_weight": 0.30, "brewer_weight": 0.40, "abv_scale": 1.5},
#     {"config_name": "brewer_heavy_abv_scale_2_0", "style_weight": 0.35, "abv_weight": 0.25, "brewer_weight": 0.40, "abv_scale": 2.0},
#     {"config_name": "brewer_heavy_abv_scale_2_5", "style_weight": 0.35, "abv_weight": 0.25, "brewer_weight": 0.40, "abv_scale": 2.5},
#     {"config_name": "brewer_heavy_abv_scale_3_0", "style_weight": 0.35, "abv_weight": 0.25, "brewer_weight": 0.40, "abv_scale": 3.0},
# ]

BEST_METRIC = "NDCG"

CASE1_RECORDS_OUT = "cold_case1_records.csv"
RECOMMENDATIONS_OUT = "cold_case1_ranked_recommendations.csv"
USER_METRICS_OUT = "cold_case1_user_metrics.csv"
SUMMARY_OUT = "cold_case1_summary.csv"


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


def build_cold_case1_df(train_df, test_df):
    seen_user_ids = set(train_df[USER_COL].unique())
    seen_item_ids = set(train_df[ITEM_COL].unique())

    return test_df[
        test_df[USER_COL].isin(seen_user_ids)
        & (~test_df[ITEM_COL].isin(seen_item_ids))
    ].copy()


def build_candidate_items_df(case1_df):
    keep_cols = [ITEM_COL]
    for col in METADATA_COLS:
        if col in case1_df.columns:
            keep_cols.append(col)

    return (
        case1_df[keep_cols]
        .copy()
        .drop_duplicates(subset=[ITEM_COL])
        .reset_index(drop=True)
    )


def _preference_dict(count_df, value_col):
    pref = {}
    for user_id, group in count_df.groupby(USER_COL, sort=False):
        total = group["cnt"].sum()
        if total > 0:
            pref[user_id] = dict(
                zip(group[value_col].astype(str), (group["cnt"] / total).astype(float))
            )
    return pref


def build_user_profiles(train_df, user_ids, positive_threshold=4.0):
    train_subset = train_df[train_df[USER_COL].isin(user_ids)].copy()
    if train_subset.empty:
        return {}

    all_style_pref = {}
    all_brewer_pref = {}
    all_mean_abv = {}

    if STYLE_COL in train_subset.columns:
        style_counts_all = (
            train_subset.dropna(subset=[STYLE_COL])
            .groupby([USER_COL, STYLE_COL], sort=False)
            .size()
            .rename("cnt")
            .reset_index()
        )
        all_style_pref = _preference_dict(style_counts_all, STYLE_COL)

    if BREWER_COL in train_subset.columns:
        brewer_counts_all = (
            train_subset.dropna(subset=[BREWER_COL])
            .groupby([USER_COL, BREWER_COL], sort=False)
            .size()
            .rename("cnt")
            .reset_index()
        )
        all_brewer_pref = _preference_dict(brewer_counts_all, BREWER_COL)

    if ABV_COL in train_subset.columns:
        all_mean_abv = train_subset.groupby(USER_COL, sort=False)[ABV_COL].mean().to_dict()

    liked = train_subset[train_subset[RATING_COL] >= positive_threshold].copy()

    pos_style_pref = {}
    pos_brewer_pref = {}
    pos_mean_abv = {}

    if not liked.empty and STYLE_COL in liked.columns:
        style_counts_pos = (
            liked.dropna(subset=[STYLE_COL])
            .groupby([USER_COL, STYLE_COL], sort=False)
            .size()
            .rename("cnt")
            .reset_index()
        )
        pos_style_pref = _preference_dict(style_counts_pos, STYLE_COL)

    if not liked.empty and BREWER_COL in liked.columns:
        brewer_counts_pos = (
            liked.dropna(subset=[BREWER_COL])
            .groupby([USER_COL, BREWER_COL], sort=False)
            .size()
            .rename("cnt")
            .reset_index()
        )
        pos_brewer_pref = _preference_dict(brewer_counts_pos, BREWER_COL)

    if not liked.empty and ABV_COL in liked.columns:
        pos_mean_abv = liked.groupby(USER_COL, sort=False)[ABV_COL].mean().to_dict()

    profiles = {}
    for user_id in user_ids:
        profiles[user_id] = {
            "style_pref": pos_style_pref.get(user_id, all_style_pref.get(user_id, {})),
            "brewer_pref": pos_brewer_pref.get(user_id, all_brewer_pref.get(user_id, {})),
            "mean_abv": pos_mean_abv.get(user_id, all_mean_abv.get(user_id, np.nan)),
        }

    return profiles


def _string_array_without_missing(series):
    values = series.astype("string").to_numpy(dtype=object)
    missing = pd.isna(values)
    values[missing] = None
    return values


def _build_codes(values):
    lookup = {}
    codes = np.full(len(values), -1, dtype=np.int32)

    for idx, value in enumerate(values):
        if value is None or pd.isna(value):
            continue
        value = str(value)
        if value not in lookup:
            lookup[value] = len(lookup)
        codes[idx] = lookup[value]

    return lookup, codes


def prepare_candidate_arrays(candidate_items_df):
    item_ids = candidate_items_df[ITEM_COL].astype(str).to_numpy()
    item_names = (
        _string_array_without_missing(candidate_items_df[NAME_COL])
        if NAME_COL in candidate_items_df.columns
        else None
    )
    item_styles = (
        _string_array_without_missing(candidate_items_df[STYLE_COL])
        if STYLE_COL in candidate_items_df.columns
        else None
    )
    item_brewers = (
        _string_array_without_missing(candidate_items_df[BREWER_COL])
        if BREWER_COL in candidate_items_df.columns
        else None
    )
    item_abv = (
        candidate_items_df[ABV_COL].to_numpy(dtype=float)
        if ABV_COL in candidate_items_df.columns
        else None
    )

    style_lookup = None
    style_codes = None
    if item_styles is not None:
        style_lookup, style_codes = _build_codes(item_styles)

    brewer_lookup = None
    brewer_codes = None
    if item_brewers is not None:
        brewer_lookup, brewer_codes = _build_codes(item_brewers)

    return {
        "item_ids": item_ids,
        "item_names": item_names,
        "item_styles": item_styles,
        "item_brewers": item_brewers,
        "item_abv": item_abv,
        "style_lookup": style_lookup,
        "style_codes": style_codes,
        "brewer_lookup": brewer_lookup,
        "brewer_codes": brewer_codes,
    }


def make_pref_vector(lookup, pref_dict):
    vec = np.zeros(len(lookup), dtype=np.float32)
    for key, value in pref_dict.items():
        idx = lookup.get(str(key))
        if idx is not None:
            vec[idx] = float(value)
    return vec


def score_candidates_fast(
    candidate_arrays,
    user_pref,
    style_weight=0.45,
    abv_weight=0.35,
    brewer_weight=0.20,
    abv_scale=1.5,
):
    n_items = len(candidate_arrays["item_ids"])

    style_score = np.zeros(n_items, dtype=np.float32)
    abv_score = np.zeros(n_items, dtype=np.float32)
    brewer_score = np.zeros(n_items, dtype=np.float32)

    style_available = np.zeros(n_items, dtype=np.float32)
    abv_available = np.zeros(n_items, dtype=np.float32)
    brewer_available = np.zeros(n_items, dtype=np.float32)

    style_lookup = candidate_arrays["style_lookup"]
    style_codes = candidate_arrays["style_codes"]
    if style_lookup is not None and style_codes is not None:
        pref = user_pref.get("style_pref", {})
        if pref:
            pref_vec = make_pref_vector(style_lookup, pref)
            valid = style_codes >= 0
            style_score[valid] = pref_vec[style_codes[valid]]
            style_available[valid] = 1.0

    item_abv = candidate_arrays["item_abv"]
    mean_abv = user_pref.get("mean_abv", np.nan)
    if item_abv is not None and not pd.isna(mean_abv):
        valid = ~np.isnan(item_abv)
        abv_score[valid] = np.exp(
            -(np.abs(item_abv[valid] - float(mean_abv)) / abv_scale)
        ).astype(np.float32)
        abv_available[valid] = 1.0

    brewer_lookup = candidate_arrays["brewer_lookup"]
    brewer_codes = candidate_arrays["brewer_codes"]
    if brewer_lookup is not None and brewer_codes is not None:
        pref = user_pref.get("brewer_pref", {})
        if pref:
            pref_vec = make_pref_vector(brewer_lookup, pref)
            valid = brewer_codes >= 0
            brewer_score[valid] = pref_vec[brewer_codes[valid]]
            brewer_available[valid] = 1.0

    weight_sum = (
        style_available * style_weight
        + abv_available * abv_weight
        + brewer_available * brewer_weight
    )

    weighted_sum = (
        style_score * style_weight
        + abv_score * abv_weight
        + brewer_score * brewer_weight
    )

    scores = np.zeros(n_items, dtype=np.float32)
    valid = weight_sum > 0
    scores[valid] = weighted_sum[valid] / weight_sum[valid]
    return scores, style_score, abv_score, brewer_score


def topk_indices(scores, k):
    n = scores.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)

    k = min(k, n)
    if k <= 0:
        return np.array([], dtype=np.int64)

    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="mergesort")]
    return idx.astype(np.int64)


def build_relevant_items_by_user(case1_df, relevance_threshold):
    positives = case1_df[case1_df[RATING_COL] >= relevance_threshold].copy()
    if positives.empty:
        return {}

    return (
        positives.groupby(USER_COL, sort=False)[ITEM_COL]
        .apply(lambda s: set(s.dropna().astype(str)))
        .to_dict()
    )


def _candidate_values(candidate_arrays, key, idx):
    values = candidate_arrays[key]
    if values is None:
        return None
    return values[idx]


def add_case1_predictions(case1_df, candidate_arrays, user_profiles, style_weight, abv_weight, brewer_weight, abv_scale):
    if case1_df.empty:
        df = case1_df.copy()
        df["predicted_rating"] = pd.Series(dtype="float64")
        return df

    item_ids = candidate_arrays["item_ids"]
    item_position_by_id = {str(item_id): idx for idx, item_id in enumerate(item_ids)}
    df = case1_df.copy()
    predicted_ratings = np.full(len(df), np.nan, dtype=np.float64)

    row_positions_by_user = {}
    for pos, user_id in enumerate(df[USER_COL].astype(str).to_numpy()):
        row_positions_by_user.setdefault(user_id, []).append(pos)

    item_ids_for_rows = df[ITEM_COL].astype(str).to_numpy()

    for user_id, row_positions in row_positions_by_user.items():
        user_pref = user_profiles.get(user_id, {})
        scores, _, _, _ = score_candidates_fast(
            candidate_arrays=candidate_arrays,
            user_pref=user_pref,
            style_weight=style_weight,
            abv_weight=abv_weight,
            brewer_weight=brewer_weight,
            abv_scale=abv_scale,
        )
        for row_pos in row_positions:
            item_pos = item_position_by_id.get(item_ids_for_rows[row_pos])
            if item_pos is not None:
                predicted_ratings[row_pos] = 1.0 + 4.0 * float(scores[item_pos])

    df["predicted_rating"] = pd.Series(predicted_ratings, index=df.index).clip(lower=1.0, upper=5.0)
    return df


def evaluate_cold_case1_batch(
    train_df,
    test_df,
    top_k=10,
    relevance_threshold=4.0,
    style_weight=0.45,
    abv_weight=0.35,
    brewer_weight=0.20,
    abv_scale=1.5,
    config_name="config",
):
    train_df = normalize_df(train_df, "train_df")
    test_df = normalize_df(test_df, "test_df")

    case1_df = build_cold_case1_df(train_df, test_df)
    relevant_items_by_user = build_relevant_items_by_user(case1_df, relevance_threshold)

    if case1_df.empty:
        summary = {
            "case": "cold_case1_seen_user_unseen_item",
            "config_name": config_name,
            "top_k": int(top_k),
            "relevance_threshold": float(relevance_threshold),
            "style_weight": float(style_weight),
            "abv_weight": float(abv_weight),
            "brewer_weight": float(brewer_weight),
            "abv_scale": float(abv_scale),
            "num_case1_rows": 0,
            "num_case1_users": 0,
            "num_candidate_items": 0,
            "num_eval_users": 0,
            f"Precision@{top_k}": 0.0,
            f"Recall@{top_k}": 0.0,
            f"NDCG@{top_k}": 0.0,
            "RMSE": 0.0,
            "num_rmse_items": 0,
        }
        return summary, case1_df, pd.DataFrame(), pd.DataFrame()

    candidate_items_df = build_candidate_items_df(case1_df)
    candidate_arrays = prepare_candidate_arrays(candidate_items_df)

    user_ids = sorted(case1_df[USER_COL].dropna().astype(str).unique().tolist())
    user_profiles = build_user_profiles(
        train_df=train_df,
        user_ids=user_ids,
        positive_threshold=relevance_threshold,
    )
    case1_df = add_case1_predictions(
        case1_df=case1_df,
        candidate_arrays=candidate_arrays,
        user_profiles=user_profiles,
        style_weight=style_weight,
        abv_weight=abv_weight,
        brewer_weight=brewer_weight,
        abv_scale=abv_scale,
    )

    item_ids = candidate_arrays["item_ids"]
    recommendation_rows = []

    for user_id in user_ids:
        user_pref = user_profiles.get(user_id, {})
        scores, style_score, abv_score, brewer_score = score_candidates_fast(
            candidate_arrays=candidate_arrays,
            user_pref=user_pref,
            style_weight=style_weight,
            abv_weight=abv_weight,
            brewer_weight=brewer_weight,
            abv_scale=abv_scale,
        )

        idx = topk_indices(scores, top_k)
        rows = {
            USER_COL: [user_id] * len(idx),
            "rank": np.arange(1, len(idx) + 1, dtype=np.int32),
            ITEM_COL: item_ids[idx],
            "cold_case1_score": scores[idx],
            "style_score": style_score[idx],
            "abv_score": abv_score[idx],
            "brewer_score": brewer_score[idx],
        }

        metadata_map = {
            NAME_COL: "item_names",
            BREWER_COL: "item_brewers",
            ABV_COL: "item_abv",
            STYLE_COL: "item_styles",
        }
        for output_col, array_key in metadata_map.items():
            values = _candidate_values(candidate_arrays, array_key, idx)
            if values is not None:
                rows[output_col] = values

        recommendation_rows.append(pd.DataFrame(rows))

    recommendations_df = (
        pd.concat(recommendation_rows, ignore_index=True)
        if recommendation_rows
        else pd.DataFrame()
    )
    recommendations_df, user_metrics_df = evaluate_ranked_recommendations_by_user(
        recommendations_df=recommendations_df,
        relevant_items_by_user=relevant_items_by_user,
        user_col=USER_COL,
        item_col=ITEM_COL,
        k=top_k,
    )
    recommendations_df["predicted_rating"] = (
        1.0 + 4.0 * pd.to_numeric(recommendations_df["cold_case1_score"], errors="coerce")
    ).clip(lower=1.0, upper=5.0)
    recommendations_df, rmse_user_metrics_df = attach_rating_rmse(
        recommendations_df=recommendations_df,
        truth_df=case1_df,
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
        case1_df,
        prediction_col="predicted_rating",
        rating_col=RATING_COL,
    )

    summary = {
        "case": "cold_case1_seen_user_unseen_item",
        "config_name": config_name,
        "top_k": int(top_k),
        "relevance_threshold": float(relevance_threshold),
        "style_weight": float(style_weight),
        "abv_weight": float(abv_weight),
        "brewer_weight": float(brewer_weight),
        "abv_scale": float(abv_scale),
        "num_case1_rows": int(len(case1_df)),
        "num_case1_users": int(case1_df[USER_COL].nunique()),
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

    return summary, case1_df, recommendations_df, user_metrics_df


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
        summary, case1_df, recommendations_df, user_metrics_df = evaluate_cold_case1_batch(
            train_df=train_df,
            test_df=eval_df,
            top_k=TOP_K,
            relevance_threshold=RELEVANCE_THRESHOLD,
            style_weight=config["style_weight"],
            abv_weight=config["abv_weight"],
            brewer_weight=config["brewer_weight"],
            abv_scale=config["abv_scale"],
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
                "case1_df": case1_df,
                "recommendations_df": recommendations_df,
                "user_metrics_df": user_metrics_df,
            }
        )
    return results


def validate_non_empty_case(case_df, stage_name):
    if case_df.empty:
        raise ValueError(
            f"{stage_name} produced no cold-case-1 rows. "
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
    validate_non_empty_case(validation_results[0]["case1_df"], "Validation tuning")

    validation_best_result = select_best_result(validation_results, BEST_METRIC)
    selected_config = {
        "config_name": validation_best_result["summary"]["config_name"],
        "style_weight": validation_best_result["summary"]["style_weight"],
        "abv_weight": validation_best_result["summary"]["abv_weight"],
        "brewer_weight": validation_best_result["summary"]["brewer_weight"],
        "abv_scale": validation_best_result["summary"]["abv_scale"],
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

    final_result["case1_df"].to_csv(CASE1_RECORDS_OUT, index=False)
    final_result["recommendations_df"].to_csv(RECOMMENDATIONS_OUT, index=False)
    final_result["user_metrics_df"].to_csv(USER_METRICS_OUT, index=False)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    print("\nCold case 1 validation tuning results:")
    print(pd.DataFrame([result["summary"] for result in validation_results]).to_string(index=False))
    print(
        f"\nBest validation config by {BEST_METRIC}: "
        f"{validation_best_result['summary']['config_name']}"
    )
    print("\nFinal test summary with validation-selected config:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved cold-case records to: {CASE1_RECORDS_OUT}")
    print(f"Saved final ranked recommendations to: {RECOMMENDATIONS_OUT}")
    print(f"Saved final user metrics to: {USER_METRICS_OUT}")
    print(f"Saved final summary to: {SUMMARY_OUT}")


if __name__ == "__main__":
    run_from_config()
