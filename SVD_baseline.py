import time

import pandas as pd
import numpy as np
import heapq
import math

from joblib import Parallel, delayed
from surprise import Dataset, Reader, SVD, accuracy

train_df = pd.read_parquet("basic_train_temporal.parquet")
test_df = pd.read_parquet("basic_test_temporal.parquet")

# clean types
train_df["user_id"] = train_df["user_id"].astype(str)
train_df["item_id"] = train_df["item_id"].astype(str)
train_df["rating"] = pd.to_numeric(train_df["rating"], errors="coerce")

test_df["user_id"] = test_df["user_id"].astype(str)
test_df["item_id"] = test_df["item_id"].astype(str)
test_df["rating"] = pd.to_numeric(test_df["rating"], errors="coerce")

# optional filtering
user_counts = train_df["user_id"].value_counts()
item_counts = train_df["item_id"].value_counts()

print("DataFrame length(before):", len(train_df))
train_df = train_df[train_df["user_id"].isin(user_counts[user_counts >= 2].index)]
train_df = train_df[train_df["item_id"].isin(item_counts[item_counts >= 2].index)]
print("DataFrame length(after):", len(train_df))

# build Surprise dataset
reader = Reader(rating_scale=(1, 5))
train_dataset = Dataset.load_from_df(train_df[["user_id", "item_id", "rating"]], reader)
trainset = train_dataset.build_full_trainset()
# train SVD directly with fixed parameters
algo = SVD(
    n_epochs=50,
    lr_all=0.01,
    reg_all=0.05,
    n_factors=100
)
algo.fit(trainset)

testset = list(test_df[["user_id", "item_id", "rating"]].itertuples(index=False, name=None))
predictions = algo.test(testset)
test_rmse = accuracy.rmse(predictions, verbose=True)

def build_user_anti_testset(trainset, raw_uid, fill=0.0):
    inner_uid = trainset.to_inner_uid(raw_uid)
    seen = {iid for iid, _ in trainset.ur[inner_uid]}
    return [
        (raw_uid, trainset.to_raw_iid(inner_iid), fill)
        for inner_iid in trainset.all_items()
        if inner_iid not in seen
    ]


def evaluate_ranking_at_k_sampled(
    algo,
    trainset,
    test_df,
    k=10,
    threshold=4.0,
    max_users=1000,
    num_negatives=1000,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    train_users = {trainset.to_raw_uid(u) for u in trainset.all_users()}
    train_items = {trainset.to_raw_iid(i) for i in trainset.all_items()}

    # keep only positive test interactions on known users/items
    positive_test = test_df[
        (test_df["user_id"].isin(train_users)) &
        (test_df["item_id"].isin(train_items)) &
        (test_df["rating"] >= threshold)
    ].copy()

    relevant_items_per_user = (
        positive_test.groupby("user_id")["item_id"].apply(set).to_dict()
    )

    eval_users = list(relevant_items_per_user.keys())
    if max_users is not None and len(eval_users) > max_users:
        eval_users = rng.choice(eval_users, size=max_users, replace=False).tolist()

    precisions = []
    recalls = []
    ndcgs = []

    for user in eval_users:
        try:
            inner_uid = trainset.to_inner_uid(user)
        except ValueError:
            continue

        seen_items = {
            trainset.to_raw_iid(inner_iid)
            for inner_iid, _ in trainset.ur[inner_uid]
        }

        relevant_items = relevant_items_per_user[user] - seen_items
        if not relevant_items:
            continue

        negative_pool = list(train_items - seen_items - relevant_items)
        if len(negative_pool) == 0:
            continue

        sampled_size = min(num_negatives, len(negative_pool))
        sampled_negatives = rng.choice(
            negative_pool,
            size=sampled_size,
            replace=False
        ).tolist()

        candidate_items = list(relevant_items) + sampled_negatives
        user_testset = [(user, item, 0.0) for item in candidate_items]

        preds = algo.test(user_testset)

        # faster than sorting the whole list
        top_k_preds = heapq.nlargest(k, preds, key=lambda p: p.est)
        top_k_items = [p.iid for p in top_k_preds]

        hits = [1 if item in relevant_items else 0 for item in top_k_items]

        precisions.append(sum(hits) / k)
        recalls.append(sum(hits) / len(relevant_items))

        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
        ideal_hits = min(len(relevant_items), k)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "num_eval_users": len(precisions),
    }

ranking_start = time.perf_counter()

ranking_metrics = evaluate_ranking_at_k_sampled(
    algo=algo,
    trainset=trainset,
    test_df=test_df,
    k=10,
    threshold=4.0,
    max_users=10000,      # try 500 first
    num_negatives=10000,  # try 500 or 1000
    random_state=42,
)

ranking_end = time.perf_counter()

print(ranking_metrics)
print(f"Ranking evaluation time: {ranking_end - ranking_start:.2f} seconds")


