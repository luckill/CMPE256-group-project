import random

import numpy as np
import torch
from torch.utils.data import Dataset


class BPRDataset(Dataset):
    def __init__(self, df, num_items, positive_threshold=4.0, num_negatives=1):
        self.num_items = int(num_items)
        self.num_negatives = int(num_negatives)

        df = df[["user_idx", "item_idx", "rating"]].dropna().copy()
        df["user_idx"] = df["user_idx"].astype(np.int64)
        df["item_idx"] = df["item_idx"].astype(np.int64)

        positive_df = df[
            (df["rating"] >= positive_threshold)
            & (df["item_idx"] >= 0)
            & (df["item_idx"] < self.num_items)
        ].copy()

        self.user_pos_items = (
            positive_df.groupby("user_idx")["item_idx"]
            .apply(set)
            .to_dict()
        )

        users_with_negatives = {
            user_idx
            for user_idx, pos_items in self.user_pos_items.items()
            if len(pos_items) < self.num_items
        }
        positive_df = positive_df[positive_df["user_idx"].isin(users_with_negatives)]

        self.user_indices = positive_df["user_idx"].to_numpy(dtype=np.int64)
        self.pos_item_indices = positive_df["item_idx"].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.user_indices) * self.num_negatives

    def __getitem__(self, idx):
        pair_idx = idx // self.num_negatives
        user_idx = int(self.user_indices[pair_idx])
        pos_item_idx = int(self.pos_item_indices[pair_idx])
        neg_item_idx = self.sample_negative_item(user_idx)

        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(pos_item_idx, dtype=torch.long),
            torch.tensor(neg_item_idx, dtype=torch.long),
        )

    def sample_negative_item(self, user_idx):
        pos_items = self.user_pos_items[user_idx]

        for _ in range(100):
            item_idx = random.randrange(self.num_items)
            if item_idx not in pos_items:
                return item_idx

        for item_idx in range(self.num_items):
            if item_idx not in pos_items:
                return item_idx

        raise RuntimeError(f"User {user_idx} has no available negative items.")
