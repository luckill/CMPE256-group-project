import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from preprocessed import preprocess_df


def map_eval_df(eval_df, user_map, item_map):
    eval_df = eval_df.copy()
    eval_df["user_idx"] = eval_df["user_id"].astype(str).map(user_map)
    eval_df["item_idx"] = eval_df["item_id"].astype(str).map(item_map)
    eval_df = eval_df.dropna(subset=["user_idx", "item_idx", "rating"])
    eval_df["user_idx"] = eval_df["user_idx"].astype(int)
    eval_df["item_idx"] = eval_df["item_idx"].astype(int)
    return eval_df


def temporal_split(df, val_ratio_within_train=0.125):
    df = df.copy()

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    elif "review/time" in df.columns:
        df = df.sort_values("review/time").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    split_idx = int(len(df) * (1.0 - val_ratio_within_train))
    train_part = df.iloc[:split_idx].copy()
    val_part = df.iloc[split_idx:].copy()
    return train_part, val_part


def prepare_split(raw_train_df, raw_eval_df):
    train_df, user_map, item_map = preprocess_df(raw_train_df)
    eval_df = prepare_eval_df(raw_eval_df, user_map, item_map)

    return {
        "train_df": train_df,
        "eval_df": eval_df,
        "user_map": user_map,
        "item_map": item_map,
        "num_users": len(user_map),
        "num_items": len(item_map),
    }


def build_stats_from_raw_train(raw_train_df):
    train_df = raw_train_df.copy()
    train_df["beer/ABV"] = pd.to_numeric(train_df["beer/ABV"], errors="coerce")

    item_stats = train_df.groupby("item_id").agg(
        item_popularity=("rating", "count"),
        item_avg_overall=("rating", "mean"),
        item_avg_aroma=("review/aroma", "mean"),
        item_avg_appearance=("review/appearance", "mean"),
        item_avg_palate=("review/palate", "mean"),
        item_avg_taste=("review/taste", "mean"),
        beer_abv=("beer/ABV", lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan),
        style=("beer/style", "first"),
    ).reset_index()

    user_stats = train_df.groupby("user_id").agg(
        user_avg_rating=("rating", "mean"),
        user_pref_abv=("beer/ABV", "mean"),
        user_avg_aroma=("review/aroma", "mean"),
        user_avg_appearance=("review/appearance", "mean"),
        user_avg_palate=("review/palate", "mean"),
        user_avg_taste=("review/taste", "mean"),
    ).reset_index()

    item_abv_median = item_stats["beer_abv"].median()
    if pd.isna(item_abv_median):
        item_abv_median = 0.0
    item_stats["beer_abv"] = item_stats["beer_abv"].fillna(item_abv_median)

    user_abv_median = user_stats["user_pref_abv"].median()
    if pd.isna(user_abv_median):
        user_abv_median = 0.0
    user_stats["user_pref_abv"] = user_stats["user_pref_abv"].fillna(user_abv_median)

    return user_stats, item_stats


def make_loader(dataset, batch_size, shuffle, device, num_workers):
    use_cuda = device.type == "cuda"
    if len(dataset) == 0:
        shuffle = False

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 6

    return DataLoader(**loader_kwargs)


def prepare_eval_df(raw_eval_df, user_map, item_map):
    eval_df = map_eval_df(raw_eval_df, user_map, item_map)
    eval_df["rating_norm"] = (eval_df["rating"] - 1.0) / 4.0
    return eval_df
