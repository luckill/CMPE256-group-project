import numpy as np
import pandas as pd
import sys
import gc
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bandit.bandit_data import load_bandit_table, load_ground_truth
from bandit.bandit_features import zscore_fit_stats, get_feature_cols
from bandit.bandit_model import LinUCB
from evaluation import evaluate_ncf_top10, run_online_bandit_split
from utils import set_seed


RUN_BANDIT_TUNING = True

BEST_FEATURE_SET = "no_emb_prod"
BEST_FEATURE_EXCLUDE_PREFIXES = ("emb_prod_",)
BEST_ALPHA = 0.15
BEST_LAMBDA_NCF = 0.30

FEATURE_SET_CONFIGS = [
    ("all_features", ()),
    ("no_style", ("style_",)),
    ("no_emb_prod", ("emb_prod_",)),
    ("no_style_no_emb_prod", ("style_", "emb_prod_")),
    ("compact", ("user_emb_", "beer_emb_", "emb_prod_", "style_")),
]

FEATURE_TUNING_ALPHA = BEST_ALPHA
FEATURE_TUNING_LAMBDA_NCF = BEST_LAMBDA_NCF


def select_feature_cols(val_bandit_df, exclude_prefixes):
    feature_cols = get_feature_cols(
        val_bandit_df,
        exclude_prefixes=exclude_prefixes,
    )
    mean, std = zscore_fit_stats(val_bandit_df, feature_cols)
    return feature_cols, mean, std


def evaluate_bandit_config(
    val_bandit_df,
    val_relevant_items,
    val_reward_map,
    val_user_order,
    feature_cols,
    mean,
    std,
    top_k,
    feature_set,
    alpha,
    lambda_ncf,
):
    metrics, _, _ = run_online_bandit_split(
        bandit_df=val_bandit_df,
        relevant_items_by_user=val_relevant_items,
        reward_by_user_item=val_reward_map,
        user_order=val_user_order,
        feature_cols=feature_cols,
        mean=mean,
        std=std,
        alpha=alpha,
        top_k=top_k,
        lambda_ncf=lambda_ncf,
        bandit=None,
        update_model=True,
        method_name=f"linucb_hybrid_val_top{top_k}_{feature_set}",
    )
    return metrics


def main():
    set_seed(42)

    relevance_threshold = 4.0
    top_k = 10

    VAL_CANDIDATE_PATH = "top100_candidates_val_warm.parquet"
    VAL_USER_EMB_PATH = "user_embeddings_val_warm.parquet"
    VAL_ITEM_EMB_PATH = "item_embeddings_val_warm.parquet"
    VAL_USER_STATS_PATH = "user_stats_val_warm.parquet"
    VAL_ITEM_STATS_PATH = "item_stats_val_warm.parquet"
    VAL_GT_PATH = "advanced_calibration_temporal_warm.parquet"

    TEST_CANDIDATE_PATH = "top100_candidates_with_scores_warm.parquet"
    TEST_USER_EMB_PATH = "user_embeddings_warm.parquet"
    TEST_ITEM_EMB_PATH = "item_embeddings_warm.parquet"
    TEST_USER_STATS_PATH = "user_stats_warm.parquet"
    TEST_ITEM_STATS_PATH = "item_stats_warm.parquet"
    TEST_GT_PATH = "advanced_test_temporal.parquet"

    # first round
    # alpha_grid = [0.0, 0.001, 0.005, 0.01, 0.05]
    # lambda_grid = [0.70, 0.80, 0.85, 0.90, 0.95]

    # no_emb_prod retune grid
    alpha_grid = [0.05, 0.10, 0.15, 0.20, 0.30]
    lambda_grid = [0.20, 0.25, 0.30, 0.35, 0.40]

    print("Loading validation candidate/features table...")
    val_bandit_df = load_bandit_table(
        candidate_path=VAL_CANDIDATE_PATH,
        user_emb_path=VAL_USER_EMB_PATH,
        item_emb_path=VAL_ITEM_EMB_PATH,
        user_stats_path=VAL_USER_STATS_PATH,
        item_stats_path=VAL_ITEM_STATS_PATH,
    )

    print("Loading validation ground truth...")
    _, val_relevant_items, val_reward_map, val_user_order = load_ground_truth(
        VAL_GT_PATH,
        relevance_threshold=relevance_threshold,
    )

    # Keep only users that have labels
    val_valid_users = sorted(set(val_bandit_df["user_id"].astype(str)) & set(val_relevant_items.keys()))

    val_bandit_df = val_bandit_df[val_bandit_df["user_id"].isin(val_valid_users)].copy()

    print("Validation table shape:", val_bandit_df.shape)
    print("Num validation users:", len(val_valid_users))
    print(f"Run bandit tuning: {RUN_BANDIT_TUNING}")

    if RUN_BANDIT_TUNING:
        print(f"Feature tuning alpha: {FEATURE_TUNING_ALPHA}")
        print(f"Feature tuning lambda NCF: {FEATURE_TUNING_LAMBDA_NCF}")

        feature_tuning_rows = []
        best_feature_set = None
        best_feature_exclude_prefixes = None
        best_feature_val_ndcg = -1.0

        for feature_set, exclude_prefixes in FEATURE_SET_CONFIGS:
            feature_cols, mean, std = select_feature_cols(
                val_bandit_df=val_bandit_df,
                exclude_prefixes=exclude_prefixes,
            )
            val_metrics = evaluate_bandit_config(
                val_bandit_df=val_bandit_df,
                val_relevant_items=val_relevant_items,
                val_reward_map=val_reward_map,
                val_user_order=val_user_order,
                feature_cols=feature_cols,
                mean=mean,
                std=std,
                top_k=top_k,
                feature_set=feature_set,
                alpha=FEATURE_TUNING_ALPHA,
                lambda_ncf=FEATURE_TUNING_LAMBDA_NCF,
            )

            row = {
                "feature_set": feature_set,
                "num_features": len(feature_cols),
                "alpha": FEATURE_TUNING_ALPHA,
                "lambda_ncf": FEATURE_TUNING_LAMBDA_NCF,
                **val_metrics,
            }
            feature_tuning_rows.append(row)

            if val_metrics["NDCG@10"] > best_feature_val_ndcg:
                best_feature_val_ndcg = val_metrics["NDCG@10"]
                best_feature_set = feature_set
                best_feature_exclude_prefixes = exclude_prefixes

        feature_tuning_summary_df = pd.DataFrame(feature_tuning_rows)
        print("\nFeature tuning summary:")
        print(feature_tuning_summary_df.sort_values("NDCG@10", ascending=False).to_string(index=False))

        feature_cols, mean, std = select_feature_cols(
            val_bandit_df=val_bandit_df,
            exclude_prefixes=best_feature_exclude_prefixes,
        )

        alpha_lambda_rows = []
        best_alpha = None
        best_lambda = None
        best_val_ndcg = -1.0

        for alpha in alpha_grid:
            for lambda_ncf in lambda_grid:
                val_metrics = evaluate_bandit_config(
                    val_bandit_df=val_bandit_df,
                    val_relevant_items=val_relevant_items,
                    val_reward_map=val_reward_map,
                    val_user_order=val_user_order,
                    feature_cols=feature_cols,
                    mean=mean,
                    std=std,
                    top_k=top_k,
                    feature_set=best_feature_set,
                    alpha=alpha,
                    lambda_ncf=lambda_ncf,
                )

                row = {
                    "feature_set": best_feature_set,
                    "num_features": len(feature_cols),
                    "alpha": alpha,
                    "lambda_ncf": lambda_ncf,
                    **val_metrics,
                }
                alpha_lambda_rows.append(row)

                if val_metrics["NDCG@10"] > best_val_ndcg:
                    best_val_ndcg = val_metrics["NDCG@10"]
                    best_alpha = alpha
                    best_lambda = lambda_ncf

        alpha_lambda_summary_df = pd.DataFrame(alpha_lambda_rows)
        print("\nAlpha/lambda tuning summary:")
        print(alpha_lambda_summary_df.sort_values("NDCG@10", ascending=False).to_string(index=False))
    else:
        best_feature_set = BEST_FEATURE_SET
        best_feature_exclude_prefixes = BEST_FEATURE_EXCLUDE_PREFIXES
        best_alpha = BEST_ALPHA
        best_lambda = BEST_LAMBDA_NCF
        best_val_ndcg = np.nan

        feature_cols, mean, std = select_feature_cols(
            val_bandit_df=val_bandit_df,
            exclude_prefixes=best_feature_exclude_prefixes,
        )

        print("\nSkipping tuning. Using fixed bandit config.")

    print("\nSelected bandit config:")
    print(f"Best feature set: {best_feature_set}")
    print(f"Best feature exclude prefixes: {best_feature_exclude_prefixes}")
    print(f"Best alpha: {best_alpha}")
    print(f"Best lambda_ncf: {best_lambda}")
    print(f"Best validation NDCG@10: {best_val_ndcg}")
    print(f"Best num features: {len(feature_cols)}")
    if RUN_BANDIT_TUNING:
        print("\nCopy these values into the BEST_* constants before setting RUN_BANDIT_TUNING = False.")

    # Warm-start the bandit by replaying validation stream using the best alpha/lambda pair.
    warm_bandit = LinUCB(d=len(feature_cols), alpha=best_alpha)
    _, _, warm_bandit = run_online_bandit_split(
        bandit_df=val_bandit_df,
        relevant_items_by_user=val_relevant_items,
        reward_by_user_item=val_reward_map,
        user_order=val_user_order,
        feature_cols=feature_cols,
        mean=mean,
        std=std,
        alpha=best_alpha,
        top_k=top_k,
        lambda_ncf=best_lambda,
        bandit=warm_bandit,
        update_model=True,
        method_name=f"linucb_hybrid_warmup_top{top_k}_{best_feature_set}",
    )

    del val_bandit_df
    del val_relevant_items
    del val_reward_map
    del val_user_order
    gc.collect()

    print("\nLoading test candidate/features table...")
    test_bandit_df = load_bandit_table(
        candidate_path=TEST_CANDIDATE_PATH,
        user_emb_path=TEST_USER_EMB_PATH,
        item_emb_path=TEST_ITEM_EMB_PATH,
        user_stats_path=TEST_USER_STATS_PATH,
        item_stats_path=TEST_ITEM_STATS_PATH,
    )

    print("Loading test ground truth...")
    _, test_relevant_items, test_reward_map, test_user_order = load_ground_truth(
        TEST_GT_PATH,
        relevance_threshold=relevance_threshold,
    )

    test_valid_users = sorted(set(test_bandit_df["user_id"].astype(str)) & set(test_relevant_items.keys()))
    test_bandit_df = test_bandit_df[test_bandit_df["user_id"].isin(test_valid_users)].copy()

    missing_feature_cols = [col for col in feature_cols if col not in test_bandit_df.columns]
    if missing_feature_cols:
        missing_features_df = pd.DataFrame(
            0.0,
            index=test_bandit_df.index,
            columns=missing_feature_cols,
        )
        test_bandit_df = pd.concat([test_bandit_df, missing_features_df], axis=1).copy()

    print("Test table shape:", test_bandit_df.shape)
    print("Num test users:", len(test_valid_users))

    # Baseline NCF on test
    ncf_test_metrics, ncf_test_recs_df = evaluate_ncf_top10(
        bandit_df=test_bandit_df,
        relevant_items_by_user=test_relevant_items,
        user_order=test_user_order,
        top_k=top_k,
    )

    # Final test evaluation once
    bandit_test_metrics, bandit_test_recs_df, _ = run_online_bandit_split(
        bandit_df=test_bandit_df,
        relevant_items_by_user=test_relevant_items,
        reward_by_user_item=test_reward_map,
        user_order=test_user_order,
        feature_cols=feature_cols,
        mean=mean,
        std=std,
        alpha=best_alpha,
        top_k=top_k,
        lambda_ncf=best_lambda,
        bandit=warm_bandit,
        update_model=True,
        method_name=f"linucb_hybrid_top{top_k}_{best_feature_set}",
    )

    final_summary_df = pd.DataFrame([
        {
            "method": f"ncf_top{top_k}",
            "feature_set": "ncf_only",
            "alpha": np.nan,
            "lambda_ncf": 1.0,
            **ncf_test_metrics,
        },
        {
            "method": f"linucb_hybrid_top{top_k}",
            "feature_set": best_feature_set,
            "alpha": best_alpha,
            "lambda_ncf": best_lambda,
            **bandit_test_metrics,
        },
    ])

    print("\nFinal test summary:")
    print(final_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
