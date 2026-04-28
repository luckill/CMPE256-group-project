import numpy as np


def get_feature_cols(df, exclude_prefixes=None):
    exclude_prefixes = tuple(exclude_prefixes or ())
    feature_cols = [
        "ncf_score",
        "beer_abv",
        "item_popularity",
        "log_item_popularity",
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
        "abv_gap",
        "overall_gap",
        "aroma_gap",
        "appearance_gap",
        "palate_gap",
        "taste_gap",
        "emb_dot",
        "emb_cosine",
    ]
    feature_cols += [c for c in df.columns if c.startswith("user_emb_")]
    feature_cols += [c for c in df.columns if c.startswith("beer_emb_")]
    feature_cols += [c for c in df.columns if c.startswith("emb_prod_")]
    feature_cols += [c for c in df.columns if c.startswith("style_")]

    feature_cols = [c for c in feature_cols if c in df.columns]
    if exclude_prefixes:
        feature_cols = [
            c for c in feature_cols
            if not c.startswith(exclude_prefixes)
        ]
    return feature_cols


def zscore_fit_stats(df, feature_cols):
    feature_df = df[feature_cols]
    mean = feature_df.mean(axis=0, skipna=True).to_numpy(dtype=np.float64)
    std = feature_df.std(axis=0, skipna=True, ddof=0).to_numpy(dtype=np.float64)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where((np.isfinite(std)) & (std > 0), std, 1.0)
    return mean, std


def zscore_fit_transform(df, feature_cols):
    X = df[feature_cols].to_numpy(dtype=np.float64)
    mean, std = zscore_fit_stats(df, feature_cols)
    X_scaled = (X - mean) / std
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return X_scaled, mean, std


def zscore_apply(df, feature_cols, mean, std):
    X = df[feature_cols].to_numpy(dtype=np.float64)
    X = (X - mean) / std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X
