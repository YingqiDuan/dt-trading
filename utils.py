import json
import os
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_date(date_str):
    dt = datetime.fromisoformat(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def to_ms(dt):
    return int(dt.timestamp() * 1000)


def timeframe_to_seconds(timeframe):
    unit = timeframe[-1]
    amount = int(timeframe[:-1])
    if unit == "m":
        return amount * 60
    if unit == "h":
        return amount * 3600
    if unit == "d":
        return amount * 86_400
    if unit == "w":
        return amount * 7 * 86_400
    if unit == "M":
        return amount * 30 * 86_400
    raise ValueError(f"unsupported timeframe {timeframe}")


def annualization_factor(timeframe, days_per_year=365):
    seconds_per_bar = timeframe_to_seconds(timeframe)
    if seconds_per_bar <= 0:
        raise ValueError(f"invalid timeframe {timeframe}")
    return (days_per_year * 24 * 3600) / seconds_per_bar


def resolve_data_sources(cfg):
    data_cfg = cfg.get("data", {})
    symbols = data_cfg.get("symbols", None)
    timeframes = data_cfg.get("timeframes", None)
    if symbols is None and timeframes is None:
        return [(data_cfg["symbol"], data_cfg["timeframe"])]

    def normalize(value, fallback):
        if value is None:
            return [fallback]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    symbols = normalize(symbols, data_cfg["symbol"])
    timeframes = normalize(timeframes, data_cfg["timeframe"])
    sources = []
    for symbol in symbols:
        for timeframe in timeframes:
            sources.append((symbol, timeframe))
    return sources


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def rolling_zscore(series, window):
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    return (series - mean) / (std + 1e-12)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def split_by_time(df, train_end, val_end, test_end):
    train = df[df["datetime"] <= train_end].copy()
    val = df[(df["datetime"] > train_end) & (df["datetime"] <= val_end)].copy()
    test = df[(df["datetime"] > val_end) & (df["datetime"] <= test_end)].copy()
    return train, val, test
