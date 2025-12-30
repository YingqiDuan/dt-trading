import os

import numpy as np
import pandas as pd

from utils import rolling_zscore


def compute_rsi(close, window):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_ema(series, window):
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def compute_bollinger(close, window, n_std):
    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    return mid, upper, lower, std


def compute_macd(close, fast, slow, signal):
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def resolve_state_cols(cfg):
    log_return_cum_windows = [int(w) for w in cfg.get("log_return_cum_windows", [])]
    ema_windows = [int(w) for w in cfg.get("ema_windows", [])]

    feature_cols = []
    if cfg.get("include_log_return", True):
        feature_cols.append("log_return")
    include_log_return_cum = cfg.get("include_log_return_cum", bool(log_return_cum_windows))
    if include_log_return_cum:
        for window in log_return_cum_windows:
            if window > 0:
                feature_cols.append(f"log_return_cum_{window}")
    if cfg.get("include_volatility", True):
        feature_cols.append("volatility")
    if cfg.get("include_volume", True):
        feature_cols.append("volume_z")

    include_price_to_ma = cfg.get("include_price_to_ma", True)
    if include_price_to_ma:
        feature_cols.append("price_to_ma")
    include_rsi = cfg.get("include_rsi", True)
    if include_rsi:
        feature_cols.append("rsi")

    if cfg.get("include_ema", bool(ema_windows)):
        for window in ema_windows:
            feature_cols.append(f"ema_{window}")

    include_boll = cfg.get("include_boll", False)
    if include_boll:
        feature_cols.extend(["boll_z", "boll_width"])
    include_macd = cfg.get("include_macd", False)
    if include_macd:
        feature_cols.extend(["macd_line", "macd_signal", "macd_hist"])

    skip_zscore = set(cfg.get("skip_zscore", []))
    skip_zscore.add("volume_z")
    for window in log_return_cum_windows:
        if window > 0:
            skip_zscore.add(f"log_return_cum_{window}")

    state_cols = []
    for col in feature_cols:
        if col in skip_zscore:
            state_cols.append(col)
        else:
            state_cols.append(f"{col}_z")
    return state_cols


def load_feature_cache(cfg, symbol=None, timeframe=None):
    feature_dir = cfg.get("data", {}).get("feature_dir")
    if not feature_dir:
        return None, None

    symbol = (symbol or cfg["data"]["symbol"]).replace("/", "")
    tf = timeframe or cfg["data"]["timeframe"]
    feature_path = os.path.join(feature_dir, f"{symbol}_{tf}_features.csv")
    if not os.path.exists(feature_path):
        return None, None

    state_cols = resolve_state_cols(cfg["features"])
    df = pd.read_csv(feature_path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    elif "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    required_cols = ["close", "datetime"]
    missing = [col for col in required_cols + state_cols if col not in df.columns]
    if missing:
        return None, None

    return df, state_cols


def build_features(df, cfg):
    out = df.copy()
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    out["volatility"] = (
        out["log_return"].rolling(window=cfg["volatility_window"], min_periods=cfg["volatility_window"]).std(ddof=0)
    )
    out["volume_z"] = rolling_zscore(out["volume"], cfg["volume_z_window"])

    log_return_cum_windows = [int(w) for w in cfg.get("log_return_cum_windows", [])]
    for window in log_return_cum_windows:
        if window <= 0:
            continue
        out[f"log_return_cum_{window}"] = (
            out["log_return"].rolling(window=window, min_periods=window).sum()
        )

    ema_windows = [int(w) for w in cfg.get("ema_windows", [])]
    for window in ema_windows:
        out[f"ema_{window}"] = compute_ema(out["close"], window)

    include_price_to_ma = cfg.get("include_price_to_ma", True)
    if include_price_to_ma:
        ma_window = int(cfg.get("ma_window", 24))
        out["ma"] = out["close"].rolling(window=ma_window, min_periods=ma_window).mean()
        out["price_to_ma"] = out["close"] / out["ma"] - 1.0

    include_rsi = cfg.get("include_rsi", True)
    if include_rsi:
        out["rsi"] = compute_rsi(out["close"], cfg["rsi_window"])

    include_boll = cfg.get("include_boll", False)
    if include_boll:
        boll_window = int(cfg.get("boll_window", 20))
        boll_n_std = float(cfg.get("boll_n_std", 2.0))
        mid, upper, lower, std = compute_bollinger(out["close"], boll_window, boll_n_std)
        out["boll_z"] = (out["close"] - mid) / (std + 1e-12)
        out["boll_width"] = (upper - lower) / (mid + 1e-12)

    include_macd = cfg.get("include_macd", False)
    if include_macd:
        macd_fast = int(cfg.get("macd_fast", 12))
        macd_slow = int(cfg.get("macd_slow", 26))
        macd_signal = int(cfg.get("macd_signal", 9))
        macd_line, macd_sig, macd_hist = compute_macd(
            out["close"], macd_fast, macd_slow, macd_signal
        )
        out["macd_line"] = macd_line
        out["macd_signal"] = macd_sig
        out["macd_hist"] = macd_hist

    feature_cols = []
    if cfg.get("include_log_return", True):
        feature_cols.append("log_return")
    include_log_return_cum = cfg.get("include_log_return_cum", bool(log_return_cum_windows))
    if include_log_return_cum:
        for window in log_return_cum_windows:
            if window > 0:
                feature_cols.append(f"log_return_cum_{window}")
    if cfg.get("include_volatility", True):
        feature_cols.append("volatility")
    if cfg.get("include_volume", True):
        feature_cols.append("volume_z")
    if include_price_to_ma:
        feature_cols.append("price_to_ma")
    if include_rsi:
        feature_cols.append("rsi")
    if cfg.get("include_ema", bool(ema_windows)):
        for window in ema_windows:
            feature_cols.append(f"ema_{window}")
    if include_boll:
        feature_cols.extend(["boll_z", "boll_width"])
    if include_macd:
        feature_cols.extend(["macd_line", "macd_signal", "macd_hist"])

    skip_zscore = set(cfg.get("skip_zscore", []))
    skip_zscore.add("volume_z")
    for window in log_return_cum_windows:
        if window > 0:
            skip_zscore.add(f"log_return_cum_{window}")
    state_cols = []
    for col in feature_cols:
        if col in skip_zscore:
            state_cols.append(col)
            continue
        z = rolling_zscore(out[col], cfg["zscore_window"])
        name = f"{col}_z"
        out[name] = z
        state_cols.append(name)

    return out, state_cols
