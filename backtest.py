import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from dataset_builder import load_or_fetch
from dt_model import DecisionTransformer
from features import build_features
from utils import ensure_dir, load_config, parse_date, save_json


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
    actions = np.asarray(actions, dtype=np.float32).reshape(-1)
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


def run_model_backtest(cfg, model, df, state_cols, device):
    seq_len = cfg["dataset"]["seq_len"]
    fee = cfg["backtest"]["fee"]
    slip = cfg["backtest"]["slip"]
    action_mode = getattr(model, "action_mode", "discrete")

    states = df[state_cols].to_numpy(dtype=np.float32)
    close = df["close"].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].to_numpy(dtype=np.int64)

    if action_mode == "continuous":
        actions = np.zeros(len(df), dtype=np.float32)
    else:
        actions = np.zeros(len(df), dtype=np.int64)
    rewards_hist = np.zeros(len(df), dtype=np.float32)
    step_rewards = np.zeros(len(df), dtype=np.float32)
    prev_action = 0.0
    for idx in range(len(df)):
        if idx < seq_len:
            action = 0.0 if action_mode == "continuous" else 0
        else:
            state_window = states[idx - seq_len + 1 : idx + 1]
            actions_window = actions[idx - seq_len + 1 : idx + 1]
            reward_window = rewards_hist[idx - seq_len + 1 : idx + 1]

            with torch.no_grad():
                s = torch.tensor(state_window, device=device).unsqueeze(0)
                r = torch.tensor(reward_window, device=device).unsqueeze(0)
                if action_mode == "continuous":
                    actions_in = np.zeros((seq_len, 1), dtype=np.float32)
                    window_prev_action = actions[idx - seq_len] if idx - seq_len >= 0 else 0.0
                    actions_in[0, 0] = float(window_prev_action)
                    actions_in[1:, 0] = actions_window[:-1]
                    a = torch.tensor(actions_in, device=device).unsqueeze(0)
                    logits = model(s, a, r)
                    mean = logits[0, -1]
                    action = float(torch.tanh(mean).cpu().numpy().item())
                else:
                    actions_in = np.zeros(seq_len, dtype=np.int64)
                    window_prev_action = actions[idx - seq_len] if idx - seq_len >= 0 else 0
                    actions_in[0] = int(window_prev_action)
                    actions_in[1:] = actions_window[:-1]
                    a = torch.tensor(action_to_index(actions_in), device=device).unsqueeze(0)
                    logits = model(s, a, r)
                    action_idx = int(torch.argmax(logits[0, -1]).item())
                    action = int(index_to_action(action_idx))

        actions[idx] = action
        if idx < len(df) - 1:
            ret = close[idx + 1] / close[idx] - 1.0
            trade_cost = (fee + slip) * abs(action - prev_action)
            reward = action * ret - trade_cost
            rewards_hist[idx + 1] = reward
            step_rewards[idx] = reward
        prev_action = action

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
            "reward": step_rewards,
            "equity": equity,
            "step_return": step_returns,
        }
    )
    return curve, equity, step_returns, trade_count, turnover


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_df = load_or_fetch(cfg)
    feat_df, state_cols = build_features(raw_df, cfg["features"])
    feat_df = feat_df.dropna(subset=state_cols).reset_index(drop=True)

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
        action_mode=model_cfg.get("action_mode", "discrete"),
        use_value_head=bool(model_cfg.get("use_value_head", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    curve, equity, step_returns, trade_count, turnover = run_model_backtest(
        cfg, model, test_df, state_cols, device
    )

    annual_factor = 24 * 365
    dt_metrics = compute_metrics(equity, step_returns, trade_count, turnover, annual_factor)
    if model.action_mode == "continuous":
        actions = curve["action"].to_numpy(dtype=np.float32)
        dt_metrics["action_stats"] = {
            "mean_position": float(np.mean(actions)),
            "mean_abs_position": float(np.mean(np.abs(actions))),
        }
    else:
        dt_metrics["action_distribution"] = action_distribution(
            curve["action"].to_numpy(dtype=np.int64)
        )
        print(f"decision_transformer action_distribution: {dt_metrics['action_distribution']}")
    metrics = {"decision_transformer": dt_metrics}

    close = test_df["close"].to_numpy(dtype=np.float32)
    buy_hold = np.ones(len(test_df), dtype=np.int64)
    rng = np.random.RandomState(int(cfg["backtest"].get("baseline_seed", 42)))
    rand_actions = np.zeros(len(test_df), dtype=np.int64)
    for idx in range(1, len(test_df)):
        if rng.rand() < cfg["backtest"].get("random_stay_prob", 0.5):
            rand_actions[idx] = rand_actions[idx - 1]
        else:
            rand_actions[idx] = rng.choice([-1, 0, 1])

    baselines = [("buy_hold", buy_hold), ("random", rand_actions)]
    ema_windows = [int(w) for w in cfg["features"].get("ema_windows", [])]
    ema_window = max(ema_windows) if ema_windows else None
    ema_col = f"ema_{ema_window}" if ema_window else None
    if ema_col and ema_col in test_df.columns:
        deadzone = float(cfg["backtest"].get("ema_deadzone", 0.001))
        ema_actions = np.where(
            test_df["close"] / test_df[ema_col] - 1.0 > deadzone,
            1,
            np.where(
                test_df["close"] / test_df[ema_col] - 1.0 < -deadzone,
                -1,
                0,
            ),
        ).astype(np.int64)
        baselines.append(("ema_trend", ema_actions))

    for name, actions in baselines:
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
