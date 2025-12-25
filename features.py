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


def build_features(df, cfg):
    out = df.copy()
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    out["volatility"] = (
        out["log_return"].rolling(window=cfg["volatility_window"], min_periods=cfg["volatility_window"]).std(ddof=0)
    )
    out["ma"] = out["close"].rolling(window=cfg["ma_window"], min_periods=cfg["ma_window"]).mean()
    out["price_to_ma"] = out["close"] / out["ma"] - 1.0
    out["volume_z"] = rolling_zscore(out["volume"], cfg["volume_z_window"])
    out["rsi"] = compute_rsi(out["close"], cfg["rsi_window"])

    feature_cols = []
    if cfg.get("include_log_return", True):
        feature_cols.append("log_return")
    if cfg.get("include_volatility", True):
        feature_cols.append("volatility")
    if cfg.get("include_volume", True):
        feature_cols.append("volume_z")
    if cfg.get("include_price_to_ma", True):
        feature_cols.append("price_to_ma")
    if cfg.get("include_rsi", True):
        feature_cols.append("rsi")

    state_cols = []
    for col in feature_cols:
        if col == "volume_z":
            state_cols.append(col)
            continue
        z = rolling_zscore(out[col], cfg["zscore_window"])
        name = f"{col}_z"
        out[name] = z
        state_cols.append(name)

    return out, state_cols
