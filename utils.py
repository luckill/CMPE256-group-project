import json
import os
import random
import numpy as np
import pandas as pd
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def set_seed_x(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def drop_unnamed(df):
    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df

def pick_first_existing(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def parse_to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def rating_to_reward(rating):
    if rating is None or pd.isna(rating):
        return 0.0

    rating = float(rating)
    if rating < 4.0:
        return 0.0
    if rating < 4.5:
        return 0.5
    if rating < 5.0:
        return 0.75
    return 1.0


def file_fingerprint(path):
    return {
        "path": path,
        "size": os.path.getsize(path),
        "mtime": os.path.getmtime(path),
    }


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, sort_keys=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def all_exist(paths):
    return all(os.path.exists(p) for p in paths)


def same_metadata(meta_path, current_meta):
    if not os.path.exists(meta_path):
        return False
    try:
        old_meta = load_json(meta_path)
        return old_meta == current_meta
    except Exception:
        return False
    
def build_stage_metadata(stage_name, config, train_file, test_file=None, val_ratio=0.125, top_n=50):
    meta = {
        "stage": stage_name,
        "config": dict(config),
        "val_ratio_within_train": float(val_ratio),
        "top_n": int(top_n),
        "train_file": file_fingerprint(train_file),
    }
    if test_file is not None:
        meta["test_file"] = file_fingerprint(test_file)
    return meta

def configure_runtime(device):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def get_num_workers():
    return 16


def invert_map(mapping):
    return {v: k for k, v in mapping.items()}

def zscore_1d(arr):
    arr = np.asarray(arr, dtype=np.float64)
    s = arr.std()
    if s <= 1e-12:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / s