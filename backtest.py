import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from dataset_builder import load_or_fetch
from dt_model import DecisionTransformer
from dt_utils import compute_step_reward, update_rtg
from features import build_features
from utils import annualization_factor, ensure_dir, load_config, parse_date, save_json


def action_to_index(actions):
    return actions + 1


def index_to_action(indices):
    return indices - 1


def find_latest_checkpoint(ckpt_dir):
    paths = glob.glob(os.path.join(ckpt_dir, "dt_best_*.pt"))
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)




def simulate(
    actions,
    close,
    fee,
    slip,
    initial_cash,
    open_prices=None,
    high_prices=None,
    low_prices=None,
    price_mode="close",
):
    actions = np.asarray(actions, dtype=np.float32).reshape(-1)
    close = np.asarray(close, dtype=np.float32).reshape(-1)
    if open_prices is not None:
        open_prices = np.asarray(open_prices, dtype=np.float32).reshape(-1)
    if high_prices is not None:
        high_prices = np.asarray(high_prices, dtype=np.float32).reshape(-1)
    if low_prices is not None:
        low_prices = np.asarray(low_prices, dtype=np.float32).reshape(-1)

    mode = str(price_mode or "close").lower()
    price = close
    if open_prices is not None and high_prices is not None and low_prices is not None:
        if mode == "open":
            price = open_prices
        elif mode in ("hl2", "hl"):
            price = 0.5 * (high_prices + low_prices)
        elif mode == "oc2":
            price = 0.5 * (open_prices + close)
        elif mode == "ohlc4":
            price = 0.25 * (open_prices + high_prices + low_prices + close)
        elif mode == "typical":
            price = (high_prices + low_prices + close) / 3.0
        else:
            price = close

    delta = np.diff(np.concatenate(([0], actions)))
    returns = price[1:] / price[:-1] - 1.0
    step_returns = np.zeros_like(actions, dtype=np.float32)
    step_returns[:-1] = actions[:-1] * returns - (fee + slip) * np.abs(delta[:-1])

    equity = np.empty_like(step_returns, dtype=np.float64)
    equity[0] = initial_cash
    for idx in range(len(step_returns) - 1):
        equity[idx + 1] = equity[idx] * (1.0 + step_returns[idx])

    trade_count = int(np.sum(np.abs(delta) > 0))
    turnover = float(np.mean(np.abs(delta)))
    return equity, step_returns, trade_count, turnover


def compute_metrics(equity, step_returns, trade_count, turnover, annual_factor, actions=None, risk_free=0.0):
    equity = np.asarray(equity, dtype=np.float64).reshape(-1)
    step_returns = np.asarray(step_returns, dtype=np.float64).reshape(-1)
    if equity.size == 0 or step_returns.size == 0 or equity[0] == 0:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "exposure": 0.0,
            "trade_count": int(trade_count),
            "turnover": float(turnover),
        }

    total_return = float(equity[-1] / equity[0] - 1.0)
    n_steps = max(1, step_returns.size - 1)
    if equity[0] > 0 and equity[-1] > 0:
        annual_return = float((equity[-1] / equity[0]) ** (annual_factor / n_steps) - 1.0)
    else:
        annual_return = 0.0

    rf_per_step = float(risk_free) / annual_factor if annual_factor else 0.0
    excess = step_returns - rf_per_step
    ret_mean = float(np.mean(excess))
    ret_std = float(np.std(excess))
    sharpe = 0.0 if ret_std == 0 else (ret_mean / ret_std) * np.sqrt(annual_factor)

    downside = excess[excess < 0]
    downside_std = float(np.std(downside)) if downside.size else 0.0
    if downside_std == 0.0:
        sortino = float("inf") if ret_mean > 0 else 0.0
    else:
        sortino = (ret_mean / downside_std) * np.sqrt(annual_factor)

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = float(drawdown.min())
    if max_drawdown == 0.0:
        calmar = float("inf") if annual_return > 0 else 0.0
    else:
        calmar = float(annual_return / abs(max_drawdown))

    gains = float(step_returns[step_returns > 0].sum())
    losses = float(-step_returns[step_returns < 0].sum())
    profit_factor = float(gains / losses) if losses > 0 else float("inf")

    win_rate = 0.0
    exposure = 0.0
    if actions is not None:
        actions = np.asarray(actions, dtype=np.float64).reshape(-1)
        n = min(actions.size, step_returns.size)
        if n > 0:
            actions = actions[:n]
            step_returns_slice = step_returns[:n]
            active = np.abs(actions) > 1e-8
            exposure = float(np.mean(active))
            active_returns = step_returns_slice[active]
            if active_returns.size:
                win_rate = float(np.mean(active_returns > 0))

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "exposure": exposure,
        "trade_count": int(trade_count),
        "turnover": float(turnover),
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
    fee = cfg["rewards"]["fee"]
    slip = cfg["rewards"]["slip"]
    action_mode = getattr(model, "action_mode", "discrete")
    reward_scale = float(cfg.get("rl", {}).get("reward_scale", 1.0))
    turnover_penalty = float(cfg.get("rl", {}).get("turnover_penalty", 0.0))
    position_penalty = float(cfg.get("rl", {}).get("position_penalty", 0.0))
    drawdown_penalty = float(cfg.get("rl", {}).get("drawdown_penalty", 0.0))
    price_mode = cfg["rewards"].get("price_mode", "close")
    range_penalty = float(cfg["rewards"].get("range_penalty", 0.0))

    dataset_cfg = cfg.get("dataset", {})
    target_return = float(dataset_cfg.get("target_return", 0.0))
    rtg_scale = float(dataset_cfg.get("rtg_scale", 1.0))
    if rtg_scale <= 0:
        rtg_scale = 1.0
    gamma = float(dataset_cfg.get("rtg_gamma", cfg.get("rl", {}).get("gamma", 1.0)))

    states = df[state_cols].to_numpy(dtype=np.float32)
    close = df["close"].to_numpy(dtype=np.float32)
    open_prices = df["open"].to_numpy(dtype=np.float32) if "open" in df.columns else None
    high_prices = df["high"].to_numpy(dtype=np.float32) if "high" in df.columns else None
    low_prices = df["low"].to_numpy(dtype=np.float32) if "low" in df.columns else None
    timestamps = df["timestamp"].to_numpy(dtype=np.int64)

    if action_mode == "continuous":
        actions = np.zeros(len(df), dtype=np.float32)
    else:
        actions = np.zeros(len(df), dtype=np.int64)
    rtg_hist = np.zeros(len(df), dtype=np.float32)
    rtg_hist[0] = target_return
    step_rewards = np.zeros(len(df), dtype=np.float32)
    prev_action = 0.0
    current_rtg = float(target_return)
    equity = 1.0
    max_equity = 1.0
    for idx in range(len(df)):
        if idx < seq_len:
            action = 0.0 if action_mode == "continuous" else 0
        else:
            state_window = states[idx - seq_len + 1 : idx + 1]
            actions_window = actions[idx - seq_len + 1 : idx + 1]
            rtg_window = rtg_hist[idx - seq_len + 1 : idx + 1] / rtg_scale

            with torch.no_grad():
                s = torch.tensor(state_window, device=device).unsqueeze(0)
                r = torch.tensor(rtg_window, device=device).unsqueeze(0)
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
            reward, _, equity, max_equity = compute_step_reward(
                float(action),
                float(prev_action),
                float(close[idx]),
                float(close[idx + 1]),
                fee,
                slip,
                reward_scale=reward_scale,
                turnover_penalty=turnover_penalty,
                position_penalty=position_penalty,
                drawdown_penalty=drawdown_penalty,
                equity=equity,
                max_equity=max_equity,
                price_mode=price_mode,
                open_t=float(open_prices[idx]) if open_prices is not None else None,
                high_t=float(high_prices[idx]) if high_prices is not None else None,
                low_t=float(low_prices[idx]) if low_prices is not None else None,
                open_t1=float(open_prices[idx + 1]) if open_prices is not None else None,
                high_t1=float(high_prices[idx + 1]) if high_prices is not None else None,
                low_t1=float(low_prices[idx + 1]) if low_prices is not None else None,
                range_penalty=range_penalty,
            )
            current_rtg = update_rtg(current_rtg, reward, gamma)
            rtg_hist[idx + 1] = current_rtg
            step_rewards[idx] = reward
        prev_action = action

    equity, step_returns, trade_count, turnover = simulate(
        actions,
        close,
        cfg["rewards"]["fee"],
        cfg["rewards"]["slip"],
        cfg["backtest"]["initial_cash"],
        open_prices=open_prices,
        high_prices=high_prices,
        low_prices=low_prices,
        price_mode=price_mode,
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

    annual_factor = annualization_factor(cfg["data"]["timeframe"])
    risk_free = float(cfg["backtest"].get("risk_free", 0.0))
    dt_metrics = compute_metrics(
        equity,
        step_returns,
        trade_count,
        turnover,
        annual_factor,
        actions=curve["action"].to_numpy(),
        risk_free=risk_free,
    )
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
            cfg["rewards"]["fee"],
            cfg["rewards"]["slip"],
            cfg["backtest"]["initial_cash"],
            open_prices=test_df["open"].to_numpy(dtype=np.float32)
            if "open" in test_df.columns
            else None,
            high_prices=test_df["high"].to_numpy(dtype=np.float32)
            if "high" in test_df.columns
            else None,
            low_prices=test_df["low"].to_numpy(dtype=np.float32)
            if "low" in test_df.columns
            else None,
            price_mode=cfg.get("rewards", {}).get("price_mode", "close"),
        )
        metrics[name] = compute_metrics(
            eq,
            ret,
            trades,
            turnover,
            annual_factor,
            actions=actions,
            risk_free=risk_free,
        )

    ensure_dir(cfg["backtest"]["output_dir"])
    curve_path = os.path.join(cfg["backtest"]["output_dir"], "equity_curve.csv")
    metrics_path = os.path.join(cfg["backtest"]["output_dir"], "metrics.json")

    curve.to_csv(curve_path, index=False)
    save_json(metrics_path, metrics)

    print(f"saved backtest curve to {curve_path}")
    print(f"saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
