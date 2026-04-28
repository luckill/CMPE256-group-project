import pandas as pd

def normalize_rating(r):
    return (r - 1.0) / 4.0

def denormalize_rating(r):
    return 1.0 + 4.0 * r

def preprocess_df(df):
    df = df.copy()

    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "rating"])

    user_map = {u: i for i, u in enumerate(df["user_id"].unique())}
    item_map = {it: i for i, it in enumerate(df["item_id"].unique())}

    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["item_id"].map(item_map)
    df["rating_norm"] = df["rating"].apply(normalize_rating)

    return df, user_map, item_map
