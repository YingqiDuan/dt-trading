import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from dataset_builder import load_or_fetch
from dt_model import DecisionTransformer
from features import build_features
from utils import ensure_dir, load_config, parse_date, resolve_inference_rtg, save_json


def action_to_index(actions):
    return actions + 1


def index_to_action(indices):
    return indices - 1


def find_latest_checkpoint(ckpt_dir):
    paths = glob.glob(os.path.join(ckpt_dir, "dt_best_*.pt"))
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)




def simulate(actions, close, fee, slip, initial_cash):
    actions = actions.astype(np.int64)
    delta = np.diff(np.concatenate(([0], actions)))
    returns = close[1:] / close[:-1] - 1.0
    step_returns = np.zeros_like(actions, dtype=np.float32)
    step_returns[:-1] = actions[:-1] * returns - (fee + slip) * np.abs(delta[:-1])

    equity = np.empty_like(step_returns, dtype=np.float64)
    equity[0] = initial_cash
    for idx in range(len(step_returns) - 1):
        equity[idx + 1] = equity[idx] * (1.0 + step_returns[idx])

    trade_count = int(np.sum(np.abs(delta) > 0))
    turnover = float(np.mean(np.abs(delta)))
    return equity, step_returns, trade_count, turnover


def compute_metrics(equity, step_returns, trade_count, turnover, annual_factor):
    total_return = equity[-1] / equity[0] - 1.0
    ret_mean = np.mean(step_returns)
    ret_std = np.std(step_returns)
    sharpe = 0.0 if ret_std == 0 else (ret_mean / ret_std) * np.sqrt(annual_factor)

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = float(drawdown.min())

    gains = step_returns[step_returns > 0].sum()
    losses = -step_returns[step_returns < 0].sum()
    profit_factor = float(gains / losses) if losses > 0 else float("inf")

    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "trade_count": trade_count,
        "turnover": turnover,
    }


def action_distribution(actions):
    counts = np.bincount(actions + 1, minlength=3)
    total = counts.sum()
    if total == 0:
        return {"counts": {"short": 0, "flat": 0, "long": 0}, "proportions": {}}
    return {
        "counts": {
            "short": int(counts[0]),
            "flat": int(counts[1]),
            "long": int(counts[2]),
        },
        "proportions": {
            "short": float(counts[0] / total),
            "flat": float(counts[1] / total),
            "long": float(counts[2] / total),
        },
    }


def run_model_backtest(cfg, model, df, state_cols, device, inference_rtg_override=None):
    seq_len = cfg["dataset"]["seq_len"]
    inference_rtg = (
        float(inference_rtg_override)
        if inference_rtg_override is not None
        else resolve_inference_rtg(cfg)
    )
    fee = cfg["backtest"]["fee"]
    slip = cfg["backtest"]["slip"]

    states = df[state_cols].to_numpy(dtype=np.float32)
    close = df["close"].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].to_numpy(dtype=np.int64)

    actions = np.zeros(len(df), dtype=np.int64)
    rtg_hist = np.zeros(len(df), dtype=np.float32)
    rtg_hist[0] = inference_rtg
    prev_action = 0
    for idx in range(len(df)):
        if idx < seq_len:
            actions[idx] = 0
        else:
            state_window = states[idx - seq_len + 1 : idx + 1]
            actions_window = actions[idx - seq_len + 1 : idx + 1]
            actions_in = np.zeros(seq_len, dtype=np.int64)
            window_prev_action = actions[idx - seq_len] if idx - seq_len >= 0 else 0
            actions_in[0] = window_prev_action
            actions_in[1:] = actions_window[:-1]
            rtg_window = rtg_hist[idx - seq_len + 1 : idx + 1]

            with torch.no_grad():
                s = torch.tensor(state_window, device=device).unsqueeze(0)
                a = torch.tensor(action_to_index(actions_in), device=device).unsqueeze(0)
                rtg = torch.tensor(rtg_window, device=device).unsqueeze(0)
                logits = model(s, a, rtg)
                action_idx = int(torch.argmax(logits[0, -1]).item())
                actions[idx] = index_to_action(action_idx)

        if idx < len(df) - 1:
            ret = close[idx + 1] / close[idx] - 1.0
            trade_cost = (fee + slip) * abs(actions[idx] - prev_action)
            reward = actions[idx] * ret - trade_cost
            rtg_hist[idx + 1] = rtg_hist[idx] - reward
        prev_action = actions[idx]

    equity, step_returns, trade_count, turnover = simulate(
        actions,
        close,
        cfg["backtest"]["fee"],
        cfg["backtest"]["slip"],
        cfg["backtest"]["initial_cash"],
    )

    curve = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": close,
            "action": actions,
            "rtg": rtg_hist,
            "equity": equity,
            "step_return": step_returns,
        }
    )

    return curve, equity, step_returns, trade_count, turnover, inference_rtg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_df = load_or_fetch(cfg)
    ema_window = int(cfg["behavior_policies"].get("ema_window", 18))
    ema_col = f"ema_{ema_window}"
    feat_df, state_cols = build_features(raw_df, cfg["features"])
    feat_df = feat_df.dropna(subset=state_cols + [ema_col]).reset_index(drop=True)

    train_end = parse_date(cfg["data"]["train_end"])
    val_end = parse_date(cfg["data"].get("val_end", cfg["data"]["train_end"]))
    test_end = parse_date(cfg["data"]["test_end"])
    test_df = feat_df[(feat_df["datetime"] > val_end) & (feat_df["datetime"] <= test_end)].copy()

    ckpt_path = args.ckpt or find_latest_checkpoint(cfg["train"]["checkpoint_dir"])
    if not ckpt_path:
        raise FileNotFoundError("no checkpoint found")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = ckpt["model_config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecisionTransformer(
        state_dim=model_cfg["state_dim"],
        act_dim=model_cfg["act_dim"],
        seq_len=model_cfg["seq_len"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        dropout=model_cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    curve, equity, step_returns, trade_count, turnover, inference_rtg = run_model_backtest(
        cfg, model, test_df, state_cols, device
    )

    annual_factor = 24 * 365
    dt_metrics = compute_metrics(equity, step_returns, trade_count, turnover, annual_factor)
    dt_metrics["action_distribution"] = action_distribution(curve["action"].to_numpy(dtype=np.int64))
    dt_metrics["inference_rtg"] = float(inference_rtg)
    metrics = {"decision_transformer": dt_metrics}
    print(f"decision_transformer action_distribution: {dt_metrics['action_distribution']}")

    close = test_df["close"].to_numpy(dtype=np.float32)
    ema_actions = np.where(
        test_df["close"] / test_df[ema_col] - 1.0 > cfg["behavior_policies"]["ma_deadzone"],
        1,
        np.where(
            test_df["close"] / test_df[ema_col] - 1.0 < -cfg["behavior_policies"]["ma_deadzone"],
            -1,
            0,
        ),
    ).astype(np.int64)

    buy_hold = np.ones(len(test_df), dtype=np.int64)
    rng = np.random.RandomState(cfg["behavior_policies"].get("seed", 42))
    rand_actions = np.zeros(len(test_df), dtype=np.int64)
    for idx in range(1, len(test_df)):
        if rng.rand() < cfg["behavior_policies"]["random_stay_prob"]:
            rand_actions[idx] = rand_actions[idx - 1]
        else:
            rand_actions[idx] = rng.choice([-1, 0, 1])

    for name, actions in [
        ("buy_hold", buy_hold),
        ("ema_trend", ema_actions),
        ("random", rand_actions),
    ]:
        eq, ret, trades, turnover = simulate(
            actions,
            close,
            cfg["backtest"]["fee"],
            cfg["backtest"]["slip"],
            cfg["backtest"]["initial_cash"],
        )
        metrics[name] = compute_metrics(eq, ret, trades, turnover, annual_factor)

    ensure_dir(cfg["backtest"]["output_dir"])
    curve_path = os.path.join(cfg["backtest"]["output_dir"], "equity_curve.csv")
    metrics_path = os.path.join(cfg["backtest"]["output_dir"], "metrics.json")

    curve.to_csv(curve_path, index=False)
    save_json(metrics_path, metrics)

    print(f"saved backtest curve to {curve_path}")
    print(f"saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
