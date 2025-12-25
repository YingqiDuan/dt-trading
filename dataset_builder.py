import argparse
import os
import time

import ccxt
import numpy as np
import pandas as pd

from features import build_features
from utils import ensure_dir, load_config, parse_date, set_seed, to_ms


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


def policy_ma(df, deadzone):
    ratio = df["close"] / df["ma"] - 1.0
    action = np.where(ratio > deadzone, 1, np.where(ratio < -deadzone, -1, 0))
    return action.astype(np.int64)


def policy_random(length, stay_prob, rng):
    actions = np.zeros(length, dtype=np.int64)
    for idx in range(1, length):
        if rng.rand() < stay_prob:
            actions[idx] = actions[idx - 1]
        else:
            actions[idx] = rng.choice([-1, 0, 1])
    return actions


def compute_rewards(close, actions, fee, slip):
    returns = close[1:] / close[:-1] - 1.0
    delta = np.diff(np.concatenate(([0], actions)))
    reward = np.zeros_like(actions, dtype=np.float32)
    reward[:-1] = actions[:-1] * returns - (fee + slip) * np.abs(delta[:-1])
    return reward


def compute_rtg(rewards):
    return np.flip(np.cumsum(np.flip(rewards))).astype(np.float32)


def build_dataset(df, state_cols, actions, rewards_cfg, traj_id, rtg_scale):
    states = df[state_cols].to_numpy(dtype=np.float32)
    close = df["close"].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].to_numpy(dtype=np.int64)

    rewards = compute_rewards(close, actions, rewards_cfg["fee"], rewards_cfg["slip"])
    rtg = compute_rtg(rewards) * rtg_scale

    traj_ids = np.full(len(df), traj_id, dtype=np.int64)
    steps = np.arange(len(df), dtype=np.int64)

    return {
        "states": states,
        "actions": actions.astype(np.int64),
        "rewards": rewards,
        "rtg": rtg,
        "timestamps": timestamps,
        "traj_id": traj_ids,
        "step_idx": steps,
    }


def concat_datasets(datasets):
    out = {}
    for key in datasets[0].keys():
        out[key] = np.concatenate([d[key] for d in datasets], axis=0)
    return out


def save_dataset(path, data, state_cols):
    np.savez(
        path,
        states=data["states"],
        actions=data["actions"],
        rewards=data["rewards"],
        rtg=data["rtg"],
        timestamps=data["timestamps"],
        traj_id=data["traj_id"],
        step_idx=data["step_idx"],
        state_cols=np.array(state_cols, dtype=object),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["behavior_policies"].get("seed", 42))

    raw_df = load_or_fetch(cfg, force=args.force)
    feat_df, state_cols = build_features(raw_df, cfg["features"])
    feat_df = feat_df.dropna(subset=state_cols + ["ma"]).reset_index(drop=True)

    train_end = parse_date(cfg["data"]["train_end"])
    val_end = parse_date(cfg["data"].get("val_end", cfg["data"]["train_end"]))
    test_end = parse_date(cfg["data"]["test_end"])

    train_df, val_df, test_df = split_by_time(feat_df, train_end, val_end, test_end)

    feature_dir = cfg["data"]["feature_dir"]
    ensure_dir(feature_dir)
    symbol = cfg["data"]["symbol"].replace("/", "")
    tf = cfg["data"]["timeframe"]
    feature_path = os.path.join(feature_dir, f"{symbol}_{tf}_features.csv")
    feat_df.to_csv(feature_path, index=False)

    datasets = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    dataset_dir = cfg["data"]["dataset_dir"]
    ensure_dir(dataset_dir)

    deadzone = cfg["behavior_policies"]["ma_deadzone"]
    stay_prob = cfg["behavior_policies"]["random_stay_prob"]
    mix_ratio = float(cfg["behavior_policies"].get("mix_ratio", 0.5))
    mix_ratio = max(0.0, min(1.0, mix_ratio))
    mix_scale = 10
    rng = np.random.RandomState(cfg["behavior_policies"].get("seed", 42))

    for split_name, split_df in datasets.items():
        if split_df.empty:
            print(f"split {split_name} has no data")
            continue

        actions_a = policy_ma(split_df, deadzone)

        a_copies = max(1, int(round(mix_ratio * mix_scale))) if mix_ratio > 0 else 0
        b_copies = max(1, mix_scale - a_copies) if mix_ratio < 1 else 0

        datasets_list = []
        traj_id = 0
        for _ in range(a_copies):
            datasets_list.append(
                build_dataset(
                    split_df,
                    state_cols,
                    actions_a,
                    cfg["rewards"],
                    traj_id,
                    cfg["dataset"]["rtg_scale"],
                )
            )
            traj_id += 1
        for _ in range(b_copies):
            actions_b = policy_random(len(split_df), stay_prob, rng)
            datasets_list.append(
                build_dataset(
                    split_df,
                    state_cols,
                    actions_b,
                    cfg["rewards"],
                    traj_id,
                    cfg["dataset"]["rtg_scale"],
                )
            )
            traj_id += 1

        combined = concat_datasets(datasets_list)

        out_path = os.path.join(dataset_dir, f"{split_name}_dataset.npz")
        save_dataset(out_path, combined, state_cols)
        print(f"saved {split_name} dataset to {out_path} ({len(combined['actions'])} steps)")


if __name__ == "__main__":
    main()
