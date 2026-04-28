import torch
from torch.utils.data import Dataset


class RatingsDataset(Dataset):
    """
    Simple PyTorch dataset for explicit-feedback recommendation.

    Expected columns in the input DataFrame:
      - user_idx
      - item_idx
      - rating_norm (preferred) or rating

    Returns:
      (user_idx, item_idx, rating)
    """

    def __init__(self, df, rating_col=None):
        if "user_idx" not in df.columns:
            raise ValueError("Missing required column: user_idx")
        if "item_idx" not in df.columns:
            raise ValueError("Missing required column: item_idx")

        if rating_col is None:
            if "rating_norm" in df.columns:
                rating_col = "rating_norm"
            elif "rating" in df.columns:
                rating_col = "rating"
            else:
                raise ValueError("Missing required rating column: rating_norm or rating")

        self.users = torch.tensor(df["user_idx"].to_numpy(), dtype=torch.long)
        self.items = torch.tensor(df["item_idx"].to_numpy(), dtype=torch.long)
        self.ratings = torch.tensor(df[rating_col].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]
