import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from NCF.ncf_io import unwrap_model
from utils import invert_map


TOP_N_CANDIDATE_COLUMNS = [
    "user_idx",
    "user_id",
    "item_idx",
    "item_id",
    "rank",
    "ncf_score",
    "ncf_predicted_rating",
]


def _denormalize_rating(scores):
    return 1.0 + 4.0 * scores


def export_top_n_candidates_dataframe(
    model,
    train_df,
    eval_df,
    num_items,
    device,
    top_n=50,
    user_inv_map=None,
    item_inv_map=None,
    rating_model=None,
    score_batch_size=20000,
):
    model.eval()
    if rating_model is not None:
        rating_model.eval()

    eval_users = sorted(eval_df["user_idx"].unique().tolist())
    seen_by_user = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    all_items = np.arange(num_items, dtype=np.int64)

    rows = []

    with torch.inference_mode():
        user_bar = tqdm(eval_users, desc=f"export top-{top_n}", leave=False)

        for user_idx in user_bar:
            seen_items = seen_by_user.get(user_idx, set())

            candidate_mask = np.ones(num_items, dtype=bool)
            if seen_items:
                candidate_mask[list(seen_items)] = False

            candidate_items = all_items[candidate_mask]

            if len(candidate_items) == 0:
                continue

            score_chunks = []
            predicted_rating_chunks = []
            for start in range(0, len(candidate_items), score_batch_size):
                batch_items = candidate_items[start:start + score_batch_size]

                user_tensor = torch.full(
                    (len(batch_items),),
                    int(user_idx),
                    dtype=torch.long,
                    device=device,
                )
                item_tensor = torch.as_tensor(
                    batch_items,
                    dtype=torch.long,
                    device=device,
                )

                batch_scores = (
                    model(user_tensor, item_tensor)
                    .reshape(-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                score_chunks.append(batch_scores)
                if rating_model is not None:
                    batch_predicted_ratings = (
                        _denormalize_rating(rating_model(user_tensor, item_tensor).float())
                        .reshape(-1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    predicted_rating_chunks.append(batch_predicted_ratings)

            scores = np.concatenate(score_chunks, axis=0)
            predicted_ratings = (
                np.concatenate(predicted_rating_chunks, axis=0)
                if predicted_rating_chunks
                else np.full(len(scores), np.nan, dtype=np.float32)
            )

            top_k = min(top_n, len(scores))
            top_idx = np.argpartition(-scores, kth=top_k - 1)[:top_k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

            top_items = candidate_items[top_idx]
            top_scores = scores[top_idx]
            top_predicted_ratings = predicted_ratings[top_idx]

            for rank, (item_idx, score, predicted_rating) in enumerate(
                zip(top_items, top_scores, top_predicted_ratings),
                start=1,
            ):
                row = {
                    "user_idx": int(user_idx),
                    "item_idx": int(item_idx),
                    "rank": int(rank),
                    "ncf_score": float(score),
                    "ncf_predicted_rating": float(predicted_rating),
                }

                if user_inv_map is not None:
                    row["user_id"] = str(user_inv_map[int(user_idx)])

                if item_inv_map is not None:
                    row["item_id"] = str(item_inv_map[int(item_idx)])

                rows.append(row)

    df = pd.DataFrame(rows)

    base_cols = TOP_N_CANDIDATE_COLUMNS
    for col in base_cols:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")

    return df[base_cols]


def export_top_n_candidates_parquet(
    model,
    train_df,
    eval_df,
    num_items,
    device,
    output_path,
    top_n=50,
    user_inv_map=None,
    item_inv_map=None,
    rating_model=None,
    score_batch_size=20000,
    flush_rows=100000,
):
    model.eval()
    if rating_model is not None:
        rating_model.eval()

    eval_users = sorted(eval_df["user_idx"].unique().tolist())
    seen_by_user = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    all_items = np.arange(num_items, dtype=np.int64)

    writer = None
    rows = []
    row_count = 0

    def flush():
        nonlocal writer, rows, row_count
        if not rows:
            return

        df = pd.DataFrame(rows, columns=TOP_N_CANDIDATE_COLUMNS)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        row_count += len(rows)
        rows = []

    try:
        with torch.inference_mode():
            user_bar = tqdm(eval_users, desc=f"export top-{top_n}", leave=False)

            for user_idx in user_bar:
                seen_items = seen_by_user.get(user_idx, set())

                candidate_mask = np.ones(num_items, dtype=bool)
                if seen_items:
                    candidate_mask[list(seen_items)] = False

                candidate_items = all_items[candidate_mask]

                if len(candidate_items) == 0:
                    continue

                score_chunks = []
                predicted_rating_chunks = []
                for start in range(0, len(candidate_items), score_batch_size):
                    batch_items = candidate_items[start:start + score_batch_size]

                    user_tensor = torch.full(
                        (len(batch_items),),
                        int(user_idx),
                        dtype=torch.long,
                        device=device,
                    )
                    item_tensor = torch.as_tensor(
                        batch_items,
                        dtype=torch.long,
                        device=device,
                    )

                    batch_scores = (
                        model(user_tensor, item_tensor)
                        .reshape(-1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    score_chunks.append(batch_scores)
                    if rating_model is not None:
                        batch_predicted_ratings = (
                            _denormalize_rating(rating_model(user_tensor, item_tensor).float())
                            .reshape(-1)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        predicted_rating_chunks.append(batch_predicted_ratings)

                scores = np.concatenate(score_chunks, axis=0)
                predicted_ratings = (
                    np.concatenate(predicted_rating_chunks, axis=0)
                    if predicted_rating_chunks
                    else np.full(len(scores), np.nan, dtype=np.float32)
                )

                top_k = min(top_n, len(scores))
                top_idx = np.argpartition(-scores, kth=top_k - 1)[:top_k]
                top_idx = top_idx[np.argsort(-scores[top_idx])]

                top_items = candidate_items[top_idx]
                top_scores = scores[top_idx]
                top_predicted_ratings = predicted_ratings[top_idx]

                for rank, (item_idx, score, predicted_rating) in enumerate(
                    zip(top_items, top_scores, top_predicted_ratings),
                    start=1,
                ):
                    rows.append(
                        {
                            "user_idx": int(user_idx),
                            "user_id": str(user_inv_map[int(user_idx)]) if user_inv_map is not None else "",
                            "item_idx": int(item_idx),
                            "item_id": str(item_inv_map[int(item_idx)]) if item_inv_map is not None else "",
                            "rank": int(rank),
                            "ncf_score": float(score),
                            "ncf_predicted_rating": float(predicted_rating),
                        }
                    )

                if len(rows) >= flush_rows:
                    flush()

        flush()
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        empty_df = pd.DataFrame(columns=TOP_N_CANDIDATE_COLUMNS)
        empty_df.to_parquet(output_path, index=False)

    return row_count


def export_user_embeddings_dataframe(model, user_map):
    base_model = unwrap_model(model)

    user_inv_map = invert_map(user_map)
    num_users = len(user_map)

    with torch.inference_mode():
        user_tensor = torch.arange(num_users, dtype=torch.long, device=base_model.users.weight.device)
        user_emb = base_model.users(user_tensor).detach().cpu().numpy()

    rows = []
    emb_dim = user_emb.shape[1]

    for user_idx in range(num_users):
        row = {
            "user_idx": int(user_idx),
            "user_id": str(user_inv_map[user_idx]),
        }
        for j in range(emb_dim):
            row[f"user_emb_{j}"] = float(user_emb[user_idx, j])
        rows.append(row)

    cols = ["user_idx", "user_id"] + [f"user_emb_{j}" for j in range(emb_dim)]
    return pd.DataFrame(rows)[cols]


def export_item_embeddings_dataframe(model, item_map):
    base_model = unwrap_model(model)

    item_inv_map = invert_map(item_map)
    num_items = len(item_map)

    with torch.inference_mode():
        item_tensor = torch.arange(num_items, dtype=torch.long, device=base_model.items.weight.device)
        item_emb = base_model.items(item_tensor).detach().cpu().numpy()

    rows = []
    emb_dim = item_emb.shape[1]

    for item_idx in range(num_items):
        row = {
            "item_idx": int(item_idx),
            "item_id": str(item_inv_map[item_idx]),
        }
        for j in range(emb_dim):
            row[f"beer_emb_{j}"] = float(item_emb[item_idx, j])
        rows.append(row)

    cols = ["item_idx", "item_id"] + [f"beer_emb_{j}" for j in range(emb_dim)]
    return pd.DataFrame(rows)[cols]
