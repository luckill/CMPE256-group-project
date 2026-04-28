import math
import numpy as np
import pandas as pd
import torch

from bandit.bandit_features import zscore_apply
from bandit.bandit_model import LinUCB, select_hybrid_top_k
from utils import rating_to_reward


def denormalize_rating(r):
    return 1.0 + 4.0 * r


def evaluate_rmse(model, loader, device):
    model.eval()
    squared_error = 0.0
    total_count = 0
    use_cuda = device.type == "cuda"

    with torch.inference_mode():
        for users, items, ratings in loader:
            users = users.to(device, non_blocking=use_cuda)
            items = items.to(device, non_blocking=use_cuda)
            ratings = ratings.to(device, non_blocking=use_cuda)

            preds = model(users, items).float()
            preds = denormalize_rating(preds)
            trues = denormalize_rating(ratings.float())

            squared_error += torch.sum((preds - trues) ** 2).item()
            total_count += ratings.numel()

    if total_count == 0:
        return 0.0

    return math.sqrt(squared_error / total_count)


def _select_users(users, max_users=None, seed=42):
    users = np.array(list(users), dtype=np.int64)

    if max_users is None or len(users) <= max_users:
        return users.tolist()

    rng = np.random.default_rng(seed)
    chosen = rng.choice(users, size=max_users, replace=False)
    return chosen.tolist()


def get_topk_metrics(
    model,
    train_df,
    test_df,
    num_items,
    device,
    k=10,
    relevance_threshold=4.0,
    max_users=None,
    seed=42,
):
    model.eval()

    train_items_per_user = (
        train_df.groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )

    test_relevant_per_user = (
        test_df[test_df["rating"] >= relevance_threshold]
        .groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )

    users = _select_users(
        users=test_relevant_per_user.keys(),
        max_users=max_users,
        seed=seed,
    )

    precisions = []
    recalls = []
    ndcgs = []

    if num_items == 0 or not users:
        return {
            "Precision@K": 0.0,
            "Recall@K": 0.0,
            "NDCG@K": 0.0,
            "num_eval_users": 0,
        }

    all_items = torch.arange(num_items, dtype=torch.long, device=device)
    use_cuda = device.type == "cuda"

    with torch.inference_mode():
        for user in users:
            relevant_items = test_relevant_per_user.get(user, set())
            if not relevant_items:
                continue

            seen_items = train_items_per_user.get(user, set())
            if len(seen_items) >= num_items:
                continue

            user_tensor = torch.full(
                (num_items,),
                fill_value=int(user),
                dtype=torch.long,
                device=device,
            )

            scores = model(user_tensor, all_items).float()

            if seen_items:
                seen_idx = torch.tensor(
                    list(seen_items),
                    dtype=torch.long,
                    device=device,
                )
                scores[seen_idx] = -torch.inf

            num_available_items = num_items - len(seen_items)
            topk_size = min(k, num_available_items)
            if topk_size <= 0:
                continue

            topk_indices = torch.topk(scores, k=topk_size).indices
            topk_items = topk_indices.detach().cpu().numpy()

            relevant_array = np.fromiter(relevant_items, dtype=np.int64)
            hits = np.isin(topk_items, relevant_array).astype(np.float32)
            num_hits = float(hits.sum())

            precision = num_hits / k
            recall = num_hits / len(relevant_items)

            if num_hits > 0:
                hit_positions = np.where(hits > 0)[0]
                dcg = float(np.sum(1.0 / np.log2(hit_positions + 2)))
            else:
                dcg = 0.0

            ideal_hits = min(len(relevant_items), k)
            idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_hits + 2))))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)

    return {
        "Precision@K": float(np.mean(precisions)) if precisions else 0.0,
        "Recall@K": float(np.mean(recalls)) if recalls else 0.0,
        "NDCG@K": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "num_eval_users": len(precisions),
    }

def precision_recall_ndcg_at_k(recommended_items, relevant_items, k=10):
    recommended_items = list(recommended_items)[:k]
    relevant_items = set(relevant_items)

    hits = [1 if item in relevant_items else 0 for item in recommended_items]
    num_hits = sum(hits)

    precision = num_hits / k if k > 0 else 0.0
    recall = 0.0 if len(relevant_items) == 0 else num_hits / len(relevant_items)

    dcg = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = 0.0 if idcg == 0 else dcg / idcg

    return precision, recall, ndcg


def evaluate_ranked_recommendations_by_user(
    recommendations_df,
    relevant_items_by_user,
    user_col="user_id",
    item_col="item_id",
    k=10,
    output_cols=None,
    include_users_without_relevant=False,
):
    if recommendations_df.empty:
        return recommendations_df, pd.DataFrame()

    df = recommendations_df.copy()
    df[user_col] = df[user_col].astype(str)
    df[item_col] = df[item_col].astype(str)
    df = df.sort_values([user_col, "rank"], ascending=[True, True]).reset_index(drop=True)

    metric_rows = []
    is_relevant_values = []

    for user_id, user_recs in df.groupby(user_col, sort=False):
        relevant_items = set(relevant_items_by_user.get(user_id, set()))
        recommended_items = user_recs[item_col].astype(str).tolist()
        user_is_relevant = [1 if item_id in relevant_items else 0 for item_id in recommended_items]
        is_relevant_values.extend(user_is_relevant)

        if not relevant_items and not include_users_without_relevant:
            continue

        precision, recall, ndcg = precision_recall_ndcg_at_k(
            recommended_items,
            relevant_items,
            k=k,
        )
        metric_rows.append(
            {
                user_col: user_id,
                "num_relevant_items": int(len(relevant_items)),
                f"Precision@{k}": float(precision),
                f"Recall@{k}": float(recall),
                f"NDCG@{k}": float(ndcg),
            }
        )

    df["is_relevant"] = np.asarray(is_relevant_values, dtype=np.int32)
    if output_cols is not None:
        existing_output_cols = [col for col in output_cols if col in df.columns]
        df = df[existing_output_cols]

    return df, pd.DataFrame(metric_rows)


def attach_rating_rmse(
    recommendations_df,
    truth_df,
    prediction_col,
    user_col="user_id",
    item_col="item_id",
    rating_col="rating",
    route_col=None,
    actual_col="actual_rating",
    squared_error_col="squared_error",
    min_rating=1.0,
    max_rating=5.0,
):
    if recommendations_df.empty:
        group_cols = [user_col]
        if route_col is not None:
            group_cols.insert(0, route_col)
        return recommendations_df, pd.DataFrame(columns=group_cols + ["num_rmse_items", "RMSE"])

    df = recommendations_df.copy()
    df[user_col] = df[user_col].astype(str)
    df[item_col] = df[item_col].astype(str)
    df = df.drop(
        columns=[col for col in [actual_col, squared_error_col] if col in df.columns],
    )
    df[prediction_col] = (
        pd.to_numeric(df[prediction_col], errors="coerce")
        .clip(lower=min_rating, upper=max_rating)
    )

    join_cols = [user_col, item_col]
    if route_col is not None and route_col in df.columns and route_col in truth_df.columns:
        join_cols = [route_col, user_col, item_col]
        df[route_col] = df[route_col].astype(str)

    truth = truth_df[join_cols + [rating_col]].copy()
    truth[user_col] = truth[user_col].astype(str)
    truth[item_col] = truth[item_col].astype(str)
    if route_col is not None and route_col in truth.columns:
        truth[route_col] = truth[route_col].astype(str)
    truth[rating_col] = pd.to_numeric(truth[rating_col], errors="coerce")
    truth = truth.dropna(subset=join_cols + [rating_col]).copy()
    truth = (
        truth.groupby(join_cols, sort=False, as_index=False)[rating_col]
        .mean()
        .rename(columns={rating_col: actual_col})
    )

    df = df.merge(truth, on=join_cols, how="left")
    valid = df[prediction_col].notna() & df[actual_col].notna()
    df[squared_error_col] = np.nan
    df.loc[valid, squared_error_col] = (
        df.loc[valid, prediction_col] - df.loc[valid, actual_col]
    ) ** 2

    group_cols = [user_col]
    if route_col is not None and route_col in df.columns:
        group_cols = [route_col, user_col]

    if not valid.any():
        return df, pd.DataFrame(columns=group_cols + ["num_rmse_items", "RMSE"])

    rmse_df = (
        df.loc[valid, group_cols + [squared_error_col]]
        .groupby(group_cols, sort=False)
        .agg(
            num_rmse_items=(squared_error_col, "size"),
            mean_squared_error=(squared_error_col, "mean"),
        )
        .reset_index()
    )
    rmse_df["RMSE"] = np.sqrt(rmse_df["mean_squared_error"])
    rmse_df = rmse_df.drop(columns=["mean_squared_error"])

    return df, rmse_df


def merge_rmse_user_metrics(
    user_metrics_df,
    rmse_user_metrics_df,
    user_col="user_id",
    route_col=None,
):
    if user_metrics_df.empty:
        return user_metrics_df
    if rmse_user_metrics_df.empty:
        result = user_metrics_df.copy()
        result["num_rmse_items"] = 0
        result["RMSE"] = np.nan
        return result

    merge_cols = [user_col]
    if route_col is not None and route_col in user_metrics_df.columns and route_col in rmse_user_metrics_df.columns:
        merge_cols = [route_col, user_col]

    result = user_metrics_df.merge(rmse_user_metrics_df, on=merge_cols, how="left")
    result["num_rmse_items"] = result["num_rmse_items"].fillna(0).astype(np.int64)
    return result


def rmse_from_squared_errors(recommendations_df, squared_error_col="squared_error"):
    if recommendations_df.empty or squared_error_col not in recommendations_df.columns:
        return 0.0, 0

    squared_errors = pd.to_numeric(
        recommendations_df[squared_error_col],
        errors="coerce",
    ).dropna()
    if squared_errors.empty:
        return 0.0, 0

    return float(np.sqrt(squared_errors.mean())), int(len(squared_errors))


def rmse_from_prediction_columns(
    df,
    prediction_col,
    rating_col="rating",
    min_rating=1.0,
    max_rating=5.0,
):
    if df.empty or prediction_col not in df.columns or rating_col not in df.columns:
        return 0.0, 0

    predictions = pd.to_numeric(df[prediction_col], errors="coerce").clip(
        lower=min_rating,
        upper=max_rating,
    )
    actuals = pd.to_numeric(df[rating_col], errors="coerce")
    valid = predictions.notna() & actuals.notna()
    if not valid.any():
        return 0.0, 0

    squared_errors = (predictions[valid] - actuals[valid]) ** 2
    return float(np.sqrt(squared_errors.mean())), int(valid.sum())


def evaluate_recommendations_at_k(
    recommendations_df,
    relevant_items_by_user,
    user_col="user_id",
    item_col="item_id",
    route_col="route",
    k=10,
):
    if recommendations_df.empty:
        return recommendations_df, pd.DataFrame()

    df = recommendations_df.copy()
    df[user_col] = df[user_col].astype(str)
    df[item_col] = df[item_col].astype(str)
    df = df.sort_values([user_col, "rank"], ascending=[True, True]).reset_index(drop=True)

    metric_rows = []
    is_relevant_values = []

    for user_id, user_recs in df.groupby(user_col, sort=False):
        relevant_items = set(relevant_items_by_user.get(user_id, set()))
        recommended_items = user_recs[item_col].astype(str).tolist()
        user_is_relevant = [1 if item_id in relevant_items else 0 for item_id in recommended_items]
        is_relevant_values.extend(user_is_relevant)

        precision, recall, ndcg = precision_recall_ndcg_at_k(
            recommended_items,
            relevant_items,
            k=k,
        )
        metric_rows.append(
            {
                user_col: user_id,
                route_col: str(user_recs[route_col].iloc[0]),
                "num_recommended_items": int(len(recommended_items)),
                "num_relevant_items": int(len(relevant_items)),
                f"Precision@{k}": float(precision),
                f"Recall@{k}": float(recall),
                f"NDCG@{k}": float(ndcg),
            }
        )

    df["is_relevant"] = is_relevant_values
    return df, pd.DataFrame(metric_rows)


def build_hybrid_summary(
    user_metrics_df,
    recommendations_df,
    train_df,
    test_df,
    routes,
    top_k,
    relevance_threshold,
    alpha,
    lambda_ncf,
    rmse_source_df=None,
    user_col="user_id",
    item_col="item_id",
    route_col="route",
):
    if rmse_source_df is None:
        rmse_source_df = recommendations_df

    summary = {
        "system": "hybrid_router",
        "top_k": int(top_k),
        "relevance_threshold": float(relevance_threshold),
        "alpha": float(alpha),
        "lambda_ncf": float(lambda_ncf),
        "num_train_rows": int(len(train_df)),
        "num_test_rows": int(len(test_df)),
        "num_train_users": int(train_df[user_col].nunique()),
        "num_train_items": int(train_df[item_col].nunique()),
        "num_test_users": int(test_df[user_col].nunique()),
        "num_recommended_users": int(recommendations_df[user_col].nunique()) if not recommendations_df.empty else 0,
        "num_recommendation_rows": int(len(recommendations_df)),
        "num_eval_users": int(len(user_metrics_df)),
    }

    for metric in [
        f"Precision@{top_k}",
        f"Recall@{top_k}",
        f"NDCG@{top_k}",
    ]:
        summary[metric] = float(user_metrics_df[metric].mean()) if not user_metrics_df.empty else 0.0

    summary["RMSE"], summary["num_rmse_items"] = rmse_from_squared_errors(rmse_source_df)

    route_counts = (
        user_metrics_df[route_col].value_counts().to_dict()
        if not user_metrics_df.empty
        else {}
    )
    for route in routes:
        summary[f"{route}_users"] = int(route_counts.get(route, 0))
        route_rmse, route_rmse_items = rmse_from_squared_errors(
            rmse_source_df[rmse_source_df[route_col] == route]
            if route_col in rmse_source_df.columns and not rmse_source_df.empty
            else pd.DataFrame()
        )
        summary[f"{route}_RMSE"] = route_rmse
        summary[f"{route}_rmse_items"] = route_rmse_items

    return summary


def evaluate_ncf_top10(
    bandit_df,
    relevant_items_by_user,
    user_order,
    top_k=10,
):
    rows = []
    metric_suffix = f"@{top_k}"
    method_name = f"ncf_top{top_k}"

    valid_user_set = set(bandit_df["user_id"].astype(str)) & set(relevant_items_by_user.keys())
    users = [u for u in user_order if u in valid_user_set]

    if not users:
        return {
            f"Precision{metric_suffix}": 0.0,
            f"Recall{metric_suffix}": 0.0,
            f"NDCG{metric_suffix}": 0.0,
            "num_eval_events": 0,
        }, pd.DataFrame()

    total_precision = 0.0
    total_recall = 0.0
    total_ndcg = 0.0

    for user_id in users:
        event_df = (
            bandit_df[bandit_df["user_id"] == user_id]
            .sort_values(["ncf_score", "rank"], ascending=[False, True])
            .head(top_k)
            .copy()
        )

        recommended_items = event_df["item_id"].astype(str).tolist()
        relevant_items = relevant_items_by_user[user_id]

        precision, recall, ndcg = precision_recall_ndcg_at_k(
            recommended_items,
            relevant_items,
            k=top_k,
        )

        total_precision += precision
        total_recall += recall
        total_ndcg += ndcg

        rows.append({
            "user_id": user_id,
            "method": method_name,
            "recommended_items": "|".join(recommended_items),
            f"precision{metric_suffix}": precision,
            f"recall{metric_suffix}": recall,
            f"ndcg{metric_suffix}": ndcg,
        })

    n = len(rows)
    metrics = {
        f"Precision{metric_suffix}": total_precision / n if n else 0.0,
        f"Recall{metric_suffix}": total_recall / n if n else 0.0,
        f"NDCG{metric_suffix}": total_ndcg / n if n else 0.0,
        "num_eval_events": n,
    }

    return metrics, pd.DataFrame(rows)

def run_online_bandit_split(
    bandit_df,
    relevant_items_by_user,
    reward_by_user_item,
    user_order,
    feature_cols,
    mean,
    std,
    alpha=0.01,
    top_k=10,
    lambda_ncf=0.85,
    bandit=None,
    update_model=True,
    method_name=None,
):
    if bandit is None:
        bandit = LinUCB(d=len(feature_cols), alpha=alpha)
    if method_name is None:
        method_name = f"linucb_hybrid_top{top_k}"

    metric_suffix = f"@{top_k}"

    rows = []

    valid_user_set = set(bandit_df["user_id"].astype(str)) & set(relevant_items_by_user.keys())
    users = [u for u in user_order if u in valid_user_set]

    total_precision = 0.0
    total_recall = 0.0
    total_ndcg = 0.0
    cumulative_reward = 0.0

    for user_id in users:
        event_df = bandit_df[bandit_df["user_id"] == user_id].copy()
        event_df = event_df.sort_values(["rank", "item_id"]).reset_index(drop=True)

        if len(event_df) == 0:
            continue

        X_event = zscore_apply(event_df, feature_cols, mean, std)
        top_idx, bandit_scores, final_scores = select_hybrid_top_k(
            event_df=event_df,
            X_event=X_event,
            bandit=bandit,
            top_k=top_k,
            lambda_ncf=lambda_ncf,
        )

        chosen_df = event_df.iloc[top_idx].copy()
        chosen_df["bandit_score"] = bandit_scores[top_idx]
        chosen_df["final_score"] = final_scores[top_idx]

        recommended_items = chosen_df["item_id"].astype(str).tolist()
        relevant_items = relevant_items_by_user[user_id]

        rewards_selected = np.array(
            [rating_to_reward(reward_by_user_item.get((user_id, item_id))) for item_id in recommended_items],
            dtype=np.float64,
        )

        precision, recall, ndcg = precision_recall_ndcg_at_k(
            recommended_items,
            relevant_items,
            k=top_k,
        )

        total_precision += precision
        total_recall += recall
        total_ndcg += ndcg
        cumulative_reward += rewards_selected.sum()

        if update_model:
            bandit.update(X_event[top_idx], rewards_selected)

        rows.append({
            "user_id": user_id,
            "method": method_name,
            "recommended_items": "|".join(recommended_items),
            f"precision{metric_suffix}": precision,
            f"recall{metric_suffix}": recall,
            f"ndcg{metric_suffix}": ndcg,
            "reward_sum": float(rewards_selected.sum()),
        })

    n = len(rows)
    metrics = {
        f"Precision{metric_suffix}": total_precision / n if n else 0.0,
        f"Recall{metric_suffix}": total_recall / n if n else 0.0,
        f"NDCG{metric_suffix}": total_ndcg / n if n else 0.0,
        "Cumulative reward": cumulative_reward,
        "num_eval_events": n,
    }

    return metrics, pd.DataFrame(rows), bandit


def get_topk_metrics_from_ranker(
    train_df,
    test_df,
    ranker_fn,
    user_col="user_id",
    item_col="item_id",
    rating_col="rating",
    k=10,
    relevance_threshold=4.0,
    max_users=None,
    seed=42,
):
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[user_col] = train_df[user_col].astype(str)
    train_df[item_col] = train_df[item_col].astype(str)
    test_df[user_col] = test_df[user_col].astype(str)
    test_df[item_col] = test_df[item_col].astype(str)
    test_df[rating_col] = pd.to_numeric(test_df[rating_col], errors="coerce")

    train_items_per_user = (
        train_df.groupby(user_col)[item_col]
        .apply(set)
        .to_dict()
    )

    test_relevant_per_user = (
        test_df[test_df[rating_col] >= relevance_threshold]
        .groupby(user_col)[item_col]
        .apply(set)
        .to_dict()
    )

    users = list(test_relevant_per_user.keys())
    if max_users is not None:
        rng = np.random.default_rng(seed)
        if len(users) > max_users:
            users = list(rng.choice(users, size=max_users, replace=False))

    precisions = []
    recalls = []
    ndcgs = []

    for user in users:
        relevant_items = test_relevant_per_user.get(user, set())
        if not relevant_items:
            continue

        pred_items = ranker_fn(user, k)
        if pred_items is None:
            continue

        pred_items = [str(x) for x in pred_items[:k]]

        hits = np.isin(pred_items, list(relevant_items)).astype(np.float32)
        num_hits = float(hits.sum())

        precision = num_hits / k
        recall = num_hits / len(relevant_items)

        if num_hits > 0:
            hit_positions = np.where(hits > 0)[0]
            dcg = float(np.sum(1.0 / np.log2(hit_positions + 2)))
        else:
            dcg = 0.0

        ideal_hits = min(len(relevant_items), k)
        idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_hits + 2))))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)

    return {
        "Precision@K": float(np.mean(precisions)) if precisions else 0.0,
        "Recall@K": float(np.mean(recalls)) if recalls else 0.0,
        "NDCG@K": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "num_eval_users": len(precisions),
    }
