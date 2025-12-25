import json
import os
import random
from datetime import datetime, timezone

import numpy as np
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def rolling_zscore(series, window):
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    return (series - mean) / (std + 1e-12)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _valid_starts_from_traj(traj_id, seq_len):
    boundaries = np.where(np.diff(traj_id) != 0)[0] + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(traj_id)]))
    valid = []
    for start, end in zip(starts, ends):
        if end - start >= seq_len:
            valid.extend(range(start, end - seq_len + 1))
    return np.array(valid, dtype=np.int64)


def rtg_quantile_from_dataset(path, quantile, seq_len, scope="window_start"):
    data = np.load(path, allow_pickle=True)
    rtg = data["rtg"].astype(np.float32)
    traj_id = data["traj_id"].astype(np.int64)
    if rtg.size == 0:
        raise ValueError("empty rtg dataset")

    scope = str(scope).lower()
    if scope in ("trajectory_start", "episode_start"):
        starts = np.concatenate(([0], np.where(np.diff(traj_id) != 0)[0] + 1))
        sample = rtg[starts]
    elif scope in ("all", "all_steps"):
        sample = rtg
    else:
        valid_starts = _valid_starts_from_traj(traj_id, seq_len)
        if valid_starts.size == 0:
            raise ValueError("no valid window starts for rtg quantile")
        sample = rtg[valid_starts]

    return float(np.quantile(sample, quantile))


def resolve_inference_rtg(cfg):
    value = cfg["backtest"].get("inference_rtg", "auto")
    if isinstance(value, str) and value.lower() == "auto":
        quantile = float(cfg["backtest"].get("inference_rtg_quantile", 0.9))
        scope = cfg["backtest"].get("rtg_quantile_scope", "window_start")
        train_path = os.path.join(cfg["data"]["dataset_dir"], "train_dataset.npz")
        return rtg_quantile_from_dataset(
            train_path, quantile, cfg["dataset"]["seq_len"], scope
        )
    if value is None:
        quantile = float(cfg["backtest"].get("inference_rtg_quantile", 0.9))
        scope = cfg["backtest"].get("rtg_quantile_scope", "window_start")
        train_path = os.path.join(cfg["data"]["dataset_dir"], "train_dataset.npz")
        return rtg_quantile_from_dataset(
            train_path, quantile, cfg["dataset"]["seq_len"], scope
        )
    return float(value)
