import json
import os
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
import torch


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_date(date_str):
    dt = datetime.fromisoformat(date_str)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def to_ms(dt):
    return int(dt.timestamp() * 1000)


def timeframe_to_seconds(timeframe):
    mapping = {"m": 60, "h": 3600, "d": 86400, "w": 604800, "M": 2592000}
    try:
        return int(timeframe[:-1]) * mapping[timeframe[-1]]
    except (ValueError, KeyError):
        raise ValueError(f"unsupported timeframe {timeframe}")


def annualization_factor(timeframe, days_per_year=365):
    seconds = timeframe_to_seconds(timeframe)
    if seconds <= 0:
        raise ValueError(f"invalid timeframe {timeframe}")
    return (days_per_year * 86400) / seconds


def resolve_data_sources(cfg):
    d = cfg.get("data", {})

    def to_list(x):
        return list(x) if isinstance(x, (list, tuple)) else [x]

    # 优先取 symbols/timeframes (复数)，否则回退到 symbol/timeframe (单数)
    syms = to_list(d.get("symbols") or d["symbol"])
    tfs = to_list(d.get("timeframes") or d["timeframe"])
    return [(s, t) for s in syms for t in tfs]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def rolling_zscore(series, window):
    r = series.rolling(window=window, min_periods=window)
    return (series - r.mean()) / (r.std(ddof=0) + 1e-12)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def split_by_time(df, train_end, val_end, test_end):
    # 使用 .copy() 避免 SettingWithCopyWarning
    mask_train = df["datetime"] <= train_end
    mask_val = (df["datetime"] > train_end) & (df["datetime"] <= val_end)
    mask_test = (df["datetime"] > val_end) & (df["datetime"] <= test_end)
    return df[mask_train].copy(), df[mask_val].copy(), df[mask_test].copy()
