import numpy as np
import pandas as pd

from utils import parse_to_float, pick_first_existing, drop_unnamed


def load_ground_truth(split_path, relevance_threshold=4.0):
    df = pd.read_parquet(split_path).copy()

    user_col = pick_first_existing(df, ["user_id", "review/profileName"])
    item_col = pick_first_existing(df, ["item_id", "beer/beerId"])
    rating_col = pick_first_existing(df, ["rating", "review/overall"])
    time_col = pick_first_existing(df, ["timestamp", "review/time"])

    if user_col is None or item_col is None or rating_col is None:
        raise ValueError(f"Missing user/item/rating columns in {split_path}")

    gt = pd.DataFrame({
        "user_id": df[user_col].astype(str),
        "item_id": df[item_col].astype(str),
        "rating": df[rating_col].apply(parse_to_float),
    })

    if time_col is not None:
        gt["event_time"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        gt["event_time"] = np.nan

    gt = gt.dropna(subset=["user_id", "item_id", "rating"]).copy()
    gt["is_relevant"] = (gt["rating"] >= relevance_threshold).astype(int)

    relevant_items_by_user = (
        gt[gt["is_relevant"] == 1]
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    reward_by_user_item = (
        gt.groupby(["user_id", "item_id"])["rating"]
        .max()
        .to_dict()
    )

    if gt["event_time"].notna().any():
        user_order = (
            gt.groupby("user_id")["event_time"]
            .min()
            .sort_values()
            .index.astype(str)
            .tolist()
        )
    else:
        user_order = sorted(gt["user_id"].astype(str).unique().tolist())

    return gt, relevant_items_by_user, reward_by_user_item, user_order


def load_bandit_table(
    candidate_path,
    user_emb_path,
    item_emb_path,
    user_stats_path,
    item_stats_path,
):
    def read_table(path):
        path_str = str(path)
        if path_str.endswith(".parquet"):
            return pd.read_parquet(path)
        if path_str.endswith(".csv"):
            return pd.read_csv(path)
        raise ValueError(f"Unsupported table format for {path}")

    def safe_numeric_series(df, col, fill_value=0.0):
        if col not in df.columns:
            return pd.Series(fill_value, index=df.index, dtype=np.float64)
        s = pd.to_numeric(df[col], errors="coerce")
        median_val = s.median()
        if pd.isna(median_val):
            median_val = fill_value
        return s.fillna(median_val).astype(np.float64)

    def safe_gap(df, left_col, right_col):
        if left_col in df.columns and right_col in df.columns:
            left = pd.to_numeric(df[left_col], errors="coerce")
            right = pd.to_numeric(df[right_col], errors="coerce")
            return (left - right).abs().fillna(0.0).astype(np.float64)
        return pd.Series(0.0, index=df.index, dtype=np.float64)

    top_df = drop_unnamed(read_table(candidate_path))
    user_emb_df = drop_unnamed(read_table(user_emb_path))
    item_emb_df = drop_unnamed(read_table(item_emb_path))
    user_stats_df = drop_unnamed(read_table(user_stats_path))
    item_stats_df = drop_unnamed(read_table(item_stats_path))

    top_df["user_id"] = top_df["user_id"].astype(str)
    top_df["item_id"] = top_df["item_id"].astype(str)
    user_emb_df["user_id"] = user_emb_df["user_id"].astype(str)
    item_emb_df["item_id"] = item_emb_df["item_id"].astype(str)
    user_stats_df["user_id"] = user_stats_df["user_id"].astype(str)
    item_stats_df["item_id"] = item_stats_df["item_id"].astype(str)

    bandit_df = top_df.merge(
        user_emb_df,
        on=["user_idx", "user_id"],
        how="left",
    )

    bandit_df = bandit_df.merge(
        item_emb_df,
        on=["item_idx", "item_id"],
        how="left",
    )

    bandit_df = bandit_df.merge(
        user_stats_df,
        on="user_id",
        how="left",
    )

    bandit_df = bandit_df.merge(
        item_stats_df,
        on="item_id",
        how="left",
    ).copy()

    # Clean numeric columns first
    numeric_cols = [
        "ncf_score",
        "beer_abv",
        "item_popularity",
        "item_avg_overall",
        "item_avg_aroma",
        "item_avg_appearance",
        "item_avg_palate",
        "item_avg_taste",
        "user_avg_rating",
        "user_pref_abv",
        "user_avg_aroma",
        "user_avg_appearance",
        "user_avg_palate",
        "user_avg_taste",
    ]
    numeric_cols += [c for c in bandit_df.columns if c.startswith("user_emb_")]
    numeric_cols += [c for c in bandit_df.columns if c.startswith("beer_emb_")]

    for col in numeric_cols:
        if col in bandit_df.columns:
            bandit_df[col] = safe_numeric_series(bandit_df, col)

    extra_blocks = []

    # Style one-hot
    if "style" in bandit_df.columns:
        style_series = bandit_df["style"].fillna("UNKNOWN").astype(str)
        style_dummies = pd.get_dummies(
            style_series,
            prefix="style",
            dtype=np.float64,
        )
        extra_blocks.append(style_dummies)

    # Derived cross features built in one block
    derived_df = pd.DataFrame(
        {
            "abv_gap": safe_gap(bandit_df, "user_pref_abv", "beer_abv"),
            "overall_gap": safe_gap(bandit_df, "user_avg_rating", "item_avg_overall"),
            "aroma_gap": safe_gap(bandit_df, "user_avg_aroma", "item_avg_aroma"),
            "appearance_gap": safe_gap(bandit_df, "user_avg_appearance", "item_avg_appearance"),
            "palate_gap": safe_gap(bandit_df, "user_avg_palate", "item_avg_palate"),
            "taste_gap": safe_gap(bandit_df, "user_avg_taste", "item_avg_taste"),
            "log_item_popularity": (
                np.log1p(
                    pd.to_numeric(
                        bandit_df["item_popularity"], errors="coerce"
                    ).fillna(0.0)
                ).astype(np.float64)
                if "item_popularity" in bandit_df.columns
                else pd.Series(0.0, index=bandit_df.index, dtype=np.float64)
            ),
        },
        index=bandit_df.index,
    )
    extra_blocks.append(derived_df)

    # Embedding interaction features built in one block
    user_emb_cols = [c for c in bandit_df.columns if c.startswith("user_emb_")]
    item_emb_cols = [c for c in bandit_df.columns if c.startswith("beer_emb_")]

    if user_emb_cols and item_emb_cols and len(user_emb_cols) == len(item_emb_cols):
        U = bandit_df[user_emb_cols].to_numpy(dtype=np.float64)
        V = bandit_df[item_emb_cols].to_numpy(dtype=np.float64)

        emb_dot = np.einsum("ij,ij->i", U, V)
        u_norm = np.linalg.norm(U, axis=1)
        v_norm = np.linalg.norm(V, axis=1)
        emb_cosine = emb_dot / (u_norm * v_norm + 1e-8)

        emb_df = pd.DataFrame(
            {
                "emb_dot": emb_dot,
                "emb_cosine": emb_cosine,
            },
            index=bandit_df.index,
        )

        prod = U * V
        prod_cols = [f"emb_prod_{i}" for i in range(prod.shape[1])]
        prod_df = pd.DataFrame(prod, columns=prod_cols, index=bandit_df.index)

        extra_blocks.append(emb_df)
        extra_blocks.append(prod_df)
    else:
        emb_df = pd.DataFrame(
            {
                "emb_dot": np.zeros(len(bandit_df), dtype=np.float64),
                "emb_cosine": np.zeros(len(bandit_df), dtype=np.float64),
            },
            index=bandit_df.index,
        )
        extra_blocks.append(emb_df)

    # Concatenate all new feature blocks at once
    bandit_df = pd.concat([bandit_df] + extra_blocks, axis=1).copy()

    # Final cleanup
    final_numeric_cols = [
        "abv_gap",
        "overall_gap",
        "aroma_gap",
        "appearance_gap",
        "palate_gap",
        "taste_gap",
        "log_item_popularity",
        "emb_dot",
        "emb_cosine",
    ]
    final_numeric_cols += [c for c in bandit_df.columns if c.startswith("emb_prod_")]
    final_numeric_cols += [c for c in bandit_df.columns if c.startswith("style_")]

    for col in final_numeric_cols:
        if col in bandit_df.columns:
            bandit_df[col] = pd.to_numeric(bandit_df[col], errors="coerce").fillna(0.0)

    return bandit_df
