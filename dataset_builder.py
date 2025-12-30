import argparse
import os
import time

import ccxt
import numpy as np
import pandas as pd

from features import build_features
from utils import ensure_dir, load_config, parse_date, to_ms
from tqdm import tqdm


def fetch_ohlcv(exchange_id, symbol, timeframe, since, until, limit):
    """从交易所获取 OHLCV 数据"""
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    since_ms, until_ms = to_ms(parse_date(since)), to_ms(parse_date(until))

    all_rows = []
    pbar = tqdm(
        total=(until_ms - since_ms) // tf_ms, desc=f"fetch {symbol}", unit="bar"
    )

    while since_ms < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        since_ms = batch[-1][0] + tf_ms
        pbar.update(len(batch))
        if len(batch) < limit:
            break
        if exchange.rateLimit:
            time.sleep(exchange.rateLimit / 1000.0)

    pbar.close()

    df = pd.DataFrame(
        all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df[df["timestamp"] <= until_ms].reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def timeframe_to_millis(tf):
    """将时间周期字符串转换为毫秒"""
    unit, value = tf[-1], int(tf[:-1])
    lookup = {"m": 60, "h": 3600, "d": 86400, "w": 604800, "M": 2592000}
    if unit not in lookup:
        raise ValueError(f"unsupported timeframe {tf}")
    return value * lookup[unit] * 1000


def validate_ohlcv(df, timeframe, cfg):
    """检查数据完整性并处理缺口"""
    report = {
        "duplicate_rows": 0,
        "invalid_timestamp_rows": 0,
        "missing_bars": 0,
        "filled_bars": 0,
    }
    if df is None or df.empty:
        return df, report

    # 1. 清洗时间戳
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    invalid_count = df["timestamp"].isna().sum()
    if invalid_count > 0:
        df = df.dropna(subset=["timestamp"])
        report["invalid_timestamp_rows"] = int(invalid_count)

    # 2. 去重与排序
    orig_len = len(df)
    df = (
        df.astype({"timestamp": np.int64})
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    report["duplicate_rows"] = orig_len - len(df)

    integrity = cfg.get("data", {}).get("integrity", {})
    if not integrity.get("enabled", True) or len(df) < 2:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df, report

    # 3. 缺口检测
    tf_ms = timeframe_to_millis(timeframe)
    ts = df["timestamp"].values
    gaps = np.diff(ts)
    gap_indices = np.where(gaps > tf_ms)[0]

    missing_counts = (gaps[gap_indices] // tf_ms) - 1
    report["missing_bars"] = (
        int(np.sum(missing_counts)) if len(missing_counts) > 0 else 0
    )

    gap_policy = str(integrity.get("gap_policy", "warn")).lower()
    if report["missing_bars"] > 0:
        if gap_policy == "error":
            raise ValueError(f"found {report['missing_bars']} missing bars")

        # 4. 缺口填充 (仅当 policy=fill)
        if gap_policy == "fill":
            max_gap = int(integrity.get("max_gap_bars", 0))
            new_rows = []
            for idx, count in zip(gap_indices, missing_counts):
                if max_gap > 0 and count > max_gap:
                    raise ValueError(
                        f"gap of {count} bars exceeds max_gap_bars={max_gap}"
                    )

                # 用前一个 bar 的收盘价填充 OHLC，Vol=0
                prev = df.iloc[idx]
                fill_val = prev["close"]
                base_ts = prev["timestamp"]
                for step in range(1, int(count) + 1):
                    row = {
                        "timestamp": base_ts + tf_ms * step,
                        "open": fill_val,
                        "high": fill_val,
                        "low": fill_val,
                        "close": fill_val,
                        "volume": 0.0,
                    }
                    new_rows.append(row)

            if new_rows:
                df = (
                    pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
                report["filled_bars"] = len(new_rows)

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df, report


def load_or_fetch(cfg, force=False, symbol=None, timeframe=None):
    """加载缓存或下载新数据"""
    raw_dir = cfg["data"]["raw_dir"]
    ensure_dir(raw_dir)
    sym, tf = symbol or cfg["data"]["symbol"], timeframe or cfg["data"]["timeframe"]
    path = os.path.join(raw_dir, f"{sym.replace('/', '')}_{tf}.csv")

    if os.path.exists(path) and not force:
        df = pd.read_csv(path)
        from_cache = True
    else:
        df = fetch_ohlcv(
            cfg["data"]["exchange"],
            sym,
            tf,
            cfg["data"]["since"],
            cfg["data"]["until"],
            cfg["data"]["limit"],
        )
        from_cache = False

    df, r = validate_ohlcv(df, tf, cfg)

    # 如果数据有变动（清洗过或新下载），则保存
    changed = any(
        r[k] > 0 for k in ["duplicate_rows", "invalid_timestamp_rows", "filled_bars"]
    )
    if not from_cache or changed:
        df.to_csv(path, index=False)
        if from_cache and changed:
            print(f"saved cleaned OHLCV to {path}")

    if r["missing_bars"]:
        print(
            f"Integrity [{sym} {tf}]: missing={r['missing_bars']} filled={r['filled_bars']} policy={cfg.get('data',{}).get('integrity',{}).get('gap_policy')}"
        )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # 标准化 symbols 和 timeframes 为列表
    def to_list(x):
        return x if isinstance(x, (list, tuple)) else [x]

    symbols = to_list(cfg.get("data", {}).get("symbols") or cfg["data"]["symbol"])
    timeframes = to_list(
        cfg.get("data", {}).get("timeframes") or cfg["data"]["timeframe"]
    )

    feat_dir = cfg["data"]["feature_dir"]
    ensure_dir(feat_dir)

    for sym in symbols:
        for tf in timeframes:
            raw_df = load_or_fetch(cfg, force=args.force, symbol=sym, timeframe=tf)
            feat_df, cols = build_features(raw_df, cfg["features"])

            # 删除因指标计算产生的 NaN 行 (如 MA 的前 N 行)
            feat_df = feat_df.dropna(subset=cols).reset_index(drop=True)

            out_path = os.path.join(
                feat_dir, f"{sym.replace('/', '')}_{tf}_features.csv"
            )
            feat_df.to_csv(out_path, index=False)
            print(f"saved features to {out_path} ({len(feat_df)} rows)")


if __name__ == "__main__":
    main()
