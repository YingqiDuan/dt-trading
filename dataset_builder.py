import argparse
import os
import time

import ccxt
import pandas as pd

from features import build_features
from utils import ensure_dir, load_config, to_ms


def fetch_ohlcv(exchange_id, symbol, timeframe, since, until, limit):
    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})
    timeframe_ms = int(exchange.parse_timeframe(timeframe) * 1000)

    since_ms = to_ms(parse_date(since))
    until_ms = to_ms(parse_date(until))
    all_rows = []

    while since_ms < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        since_ms = batch[-1][0] + timeframe_ms
        if len(batch) < limit:
            break
        if exchange.rateLimit:
            time.sleep(exchange.rateLimit / 1000.0)

    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df[df["timestamp"] <= until_ms].reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def load_or_fetch(cfg, force=False):
    raw_dir = cfg["data"]["raw_dir"]
    ensure_dir(raw_dir)
    symbol = cfg["data"]["symbol"].replace("/", "")
    tf = cfg["data"]["timeframe"]
    path = os.path.join(raw_dir, f"{symbol}_{tf}.csv")

    if os.path.exists(path) and not force:
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    df = fetch_ohlcv(
        cfg["data"]["exchange"],
        cfg["data"]["symbol"],
        cfg["data"]["timeframe"],
        cfg["data"]["since"],
        cfg["data"]["until"],
        cfg["data"]["limit"],
    )
    df.to_csv(path, index=False)
    return df


def split_by_time(df, train_end, val_end, test_end):
    train = df[df["datetime"] <= train_end].copy()
    val = df[(df["datetime"] > train_end) & (df["datetime"] <= val_end)].copy()
    test = df[(df["datetime"] > val_end) & (df["datetime"] <= test_end)].copy()
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_df = load_or_fetch(cfg, force=args.force)
    feat_df, state_cols = build_features(raw_df, cfg["features"])
    feat_df = feat_df.dropna(subset=state_cols).reset_index(drop=True)

    feature_dir = cfg["data"]["feature_dir"]
    ensure_dir(feature_dir)
    symbol = cfg["data"]["symbol"].replace("/", "")
    tf = cfg["data"]["timeframe"]
    feature_path = os.path.join(feature_dir, f"{symbol}_{tf}_features.csv")
    feat_df.to_csv(feature_path, index=False)
    print(f"saved feature CSV to {feature_path} ({len(feat_df)} rows)")


if __name__ == "__main__":
    main()
