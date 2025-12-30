import os
import numpy as np
import pandas as pd
from utils import rolling_zscore


def compute_rsi(close, window):
    delta = close.diff()
    roll = lambda s: s.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    gain = roll(delta.clip(lower=0.0))
    loss = roll(-delta.clip(upper=0.0))
    return 100.0 - (100.0 / (1.0 + gain / (loss + 1e-12)))


def compute_ema(series, window):
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def compute_bollinger(close, window, n_std):
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    return mid, mid + n_std * std, mid - n_std * std, std


def compute_macd(close, fast, slow, signal):
    fast_ma, slow_ma = compute_ema(close, fast), compute_ema(close, slow)
    macd = fast_ma - slow_ma
    sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd, sig, macd - sig


def _resolve_cols(cfg):
    log_cum_wins = [int(w) for w in cfg.get("log_return_cum_windows", [])]
    ema_wins = [int(w) for w in cfg.get("ema_windows", [])]

    # 1. 确定启用的特征列
    # 辅助检查函数，默认值为 True
    check = lambda k, d=True: cfg.get(k, d)

    cols = []
    if check("include_log_return"):
        cols.append("log_return")
    if cfg.get("include_log_return_cum", bool(log_cum_wins)):
        cols.extend(f"log_return_cum_{w}" for w in log_cum_wins if w > 0)
    if check("include_volatility"):
        cols.append("volatility")
    if check("include_volume"):
        cols.append("volume_z")
    if check("include_price_to_ma"):
        cols.append("price_to_ma")
    if check("include_rsi"):
        cols.append("rsi")
    if check("include_ema", bool(ema_wins)):
        cols.extend(f"ema_{w}" for w in ema_wins)
    if check("include_boll", False):
        cols.extend(["boll_z", "boll_width"])
    if check("include_macd", False):
        cols.extend(["macd_line", "macd_signal", "macd_hist"])

    # 2. 确定不进行 Z-Score 标准化的列
    skip = set(cfg.get("skip_zscore", [])) | {"volume_z", "boll_z", "boll_width"}
    skip.update(f"log_return_cum_{w}" for w in log_cum_wins if w > 0)

    return cols, skip


def resolve_state_cols(cfg):
    cols, skip = _resolve_cols(cfg)
    return [c if c in skip else f"{c}_z" for c in cols]


def load_feature_cache(cfg, symbol=None, timeframe=None):
    f_dir = cfg.get("data", {}).get("feature_dir")
    if not f_dir:
        return None, None

    sym = (symbol or cfg["data"]["symbol"]).replace("/", "")
    tf = timeframe or cfg["data"]["timeframe"]
    path = os.path.join(f_dir, f"{sym}_{tf}_features.csv")

    if not os.path.exists(path):
        return None, None

    df = pd.read_csv(path)
    # 兼容 timestamp (ms) 和 datetime 格式
    ts_col = "datetime" if "datetime" in df.columns else "timestamp"
    unit = "ms" if ts_col == "timestamp" else None
    df["datetime"] = pd.to_datetime(df[ts_col], unit=unit, utc=True)

    # 检查完整性
    state_cols = resolve_state_cols(cfg["features"])
    if any(c not in df.columns for c in ["close", "datetime"] + state_cols):
        return None, None

    return df, state_cols


def build_features(df, cfg):
    out = df.copy()
    close = out["close"]

    # --- 基础特征计算 ---
    out["log_return"] = np.log(close / close.shift(1))

    vol_win = cfg["volatility_window"]
    out["volatility"] = (
        out["log_return"].rolling(vol_win, min_periods=vol_win).std(ddof=0)
    )

    out["volume_z"] = rolling_zscore(out["volume"], cfg["volume_z_window"])

    for w in [int(x) for x in cfg.get("log_return_cum_windows", []) if x > 0]:
        out[f"log_return_cum_{w}"] = out["log_return"].rolling(w, min_periods=w).sum()

    for w in [int(x) for x in cfg.get("ema_windows", [])]:
        out[f"ema_{w}"] = compute_ema(close, w)

    if cfg.get("include_price_to_ma", False):
        mw = int(cfg.get("ma_window", 24))
        out["price_to_ma"] = close / close.rolling(mw, min_periods=mw).mean() - 1.0

    if cfg.get("include_rsi", False):
        out["rsi"] = compute_rsi(close, cfg["rsi_window"])

    if cfg.get("include_boll", False):
        mid, up, low, std = compute_bollinger(
            close, int(cfg.get("boll_window", 20)), float(cfg.get("boll_n_std", 2.0))
        )
        out["boll_z"] = (close - mid) / (std + 1e-12)
        out["boll_width"] = (up - low) / (mid + 1e-12)

    if cfg.get("include_macd", False):
        m_l, m_s, m_h = compute_macd(
            close,
            int(cfg.get("macd_fast", 12)),
            int(cfg.get("macd_slow", 26)),
            int(cfg.get("macd_signal", 9)),
        )
        out["macd_line"], out["macd_signal"], out["macd_hist"] = m_l, m_s, m_h

    # --- 特征筛选与标准化 ---
    cols, skip = _resolve_cols(cfg)
    final_cols = []

    for c in cols:
        if c in skip:
            final_cols.append(c)
        else:
            # 仅对未在 skip 列表中的特征进行 Z-Score
            z_name = f"{c}_z"
            out[z_name] = rolling_zscore(out[c], cfg["zscore_window"])
            final_cols.append(z_name)

    return out, final_cols
