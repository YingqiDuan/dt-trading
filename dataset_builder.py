import argparse
import os
import time

import ccxt
import numpy as np
import pandas as pd

from features import build_features
from utils import ensure_dir, load_config, parse_date, to_ms
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def fetch_ohlcv(exchange_id, symbol, timeframe, since, until, limit):
    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})
    timeframe_ms = int(exchange.parse_timeframe(timeframe) * 1000)

    since_ms = to_ms(parse_date(since))
    until_ms = to_ms(parse_date(until))
    all_rows = []
    total = None
    if timeframe_ms > 0:
        total = max(0, (until_ms - since_ms) // timeframe_ms)
    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=total,
            desc=f"fetch {symbol} {timeframe}",
            unit="bar",
            dynamic_ncols=True,
            leave=False,
        )

    while since_ms < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        since_ms = batch[-1][0] + timeframe_ms
        if pbar is not None:
            pbar.update(len(batch))
        if len(batch) < limit:
            break
        if exchange.rateLimit:
            time.sleep(exchange.rateLimit / 1000.0)
    if pbar is not None:
        pbar.close()

    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df[df["timestamp"] <= until_ms].reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def timeframe_to_millis(timeframe):
    unit = timeframe[-1]
    amount = int(timeframe[:-1])
    if unit == "m":
        return amount * 60_000
    if unit == "h":
        return amount * 3_600_000
    if unit == "d":
        return amount * 86_400_000
    if unit == "w":
        return amount * 7 * 86_400_000
    if unit == "M":
        return amount * 30 * 86_400_000
    raise ValueError(f"unsupported timeframe {timeframe}")


def validate_ohlcv(df, timeframe, cfg):
    report = {
        "duplicate_rows": 0,
        "invalid_timestamp_rows": 0,
        "missing_bars": 0,
        "filled_bars": 0,
        "irregular_gaps": 0,
    }
    if df is None or df.empty:
        return df, report

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(f"missing required columns: {missing_cols}")

    out = df.copy()
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    invalid_ts = int(out["timestamp"].isna().sum())
    if invalid_ts:
        out = out.dropna(subset=["timestamp"]).copy()
    report["invalid_timestamp_rows"] = invalid_ts
    if out.empty:
        return out, report

    out["timestamp"] = out["timestamp"].astype(np.int64)
    before = len(out)
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    report["duplicate_rows"] = int(before - len(out))

    integrity_cfg = cfg.get("data", {}).get("integrity", {})
    if not bool(integrity_cfg.get("enabled", True)):
        out["datetime"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
        return out, report

    gap_policy = str(integrity_cfg.get("gap_policy", "warn")).lower()
    if gap_policy not in ("fill", "warn", "error"):
        raise ValueError(f"unsupported gap_policy {gap_policy}")

    timeframe_ms = timeframe_to_millis(timeframe)
    ts = out["timestamp"].to_numpy()
    if len(ts) < 2:
        out["datetime"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
        return out, report

    diffs = np.diff(ts)
    gap_indices = np.where(diffs > timeframe_ms)[0]
    missing_total = 0
    irregular = 0
    gap_sizes = []
    for idx in gap_indices:
        gap = int(diffs[idx])
        if gap % timeframe_ms != 0:
            irregular += 1
        missing = gap // timeframe_ms - 1
        if missing > 0:
            gap_sizes.append(missing)
            missing_total += missing

    report["missing_bars"] = int(missing_total)
    report["irregular_gaps"] = int(irregular)

    if report["irregular_gaps"] > 0 and gap_policy in ("fill", "error"):
        raise ValueError("found irregular timestamp gaps; check timeframe alignment")
    if report["missing_bars"] > 0 and gap_policy == "error":
        raise ValueError(f"found {report['missing_bars']} missing bars")

    if report["missing_bars"] > 0 and gap_policy == "fill":
        max_gap_bars = int(integrity_cfg.get("max_gap_bars", 0))
        missing_rows = []
        for idx in gap_indices:
            gap = int(diffs[idx])
            missing = gap // timeframe_ms - 1
            if missing <= 0:
                continue
            if max_gap_bars > 0 and missing > max_gap_bars:
                raise ValueError(
                    f"gap of {missing} bars exceeds max_gap_bars={max_gap_bars}"
                )
            prev_row = out.iloc[idx].to_dict()
            prev_close = float(prev_row["close"])
            for col in ("open", "high", "low", "close"):
                if col in prev_row:
                    prev_row[col] = prev_close
            if "volume" in prev_row:
                prev_row["volume"] = 0.0
            for step in range(1, missing + 1):
                row = prev_row.copy()
                row["timestamp"] = int(ts[idx] + timeframe_ms * step)
                missing_rows.append(row)
        if missing_rows:
            fill_df = pd.DataFrame(missing_rows)
            out = pd.concat([out, fill_df], ignore_index=True)
            out = out.sort_values("timestamp").reset_index(drop=True)
            report["filled_bars"] = int(len(missing_rows))

    out["datetime"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    return out, report


def load_or_fetch(cfg, force=False):
    raw_dir = cfg["data"]["raw_dir"]
    ensure_dir(raw_dir)
    symbol = cfg["data"]["symbol"].replace("/", "")
    tf = cfg["data"]["timeframe"]
    path = os.path.join(raw_dir, f"{symbol}_{tf}.csv")

    from_cache = os.path.exists(path) and not force
    if from_cache:
        df = pd.read_csv(path)
    else:
        df = fetch_ohlcv(
            cfg["data"]["exchange"],
            cfg["data"]["symbol"],
            cfg["data"]["timeframe"],
            cfg["data"]["since"],
            cfg["data"]["until"],
            cfg["data"]["limit"],
        )

    df, report = validate_ohlcv(df, tf, cfg)
    changed = any(
        report[key] > 0
        for key in ("duplicate_rows", "invalid_timestamp_rows", "filled_bars")
    )
    if not from_cache or changed:
        df.to_csv(path, index=False)
        if from_cache and changed:
            print(f"saved cleaned OHLCV to {path}")
    if report["missing_bars"] > 0:
        print(
            f"data integrity: missing_bars={report['missing_bars']} filled_bars={report['filled_bars']} "
            f"gap_policy={cfg.get('data', {}).get('integrity', {}).get('gap_policy', 'warn')}"
        )
    if report["duplicate_rows"] > 0 or report["invalid_timestamp_rows"] > 0:
        print(
            f"data integrity: duplicate_rows={report['duplicate_rows']} "
            f"invalid_timestamp_rows={report['invalid_timestamp_rows']}"
        )
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
