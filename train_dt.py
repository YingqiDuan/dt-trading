import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data import DataLoader, Dataset
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from backtest import compute_metrics, simulate
from dataset_builder import load_or_fetch, split_by_time
from dt_model import DecisionTransformer
from dt_utils import (
    compute_rtg,
    compute_step_reward,
    compute_trajectory_rewards,
    update_rtg,
)
from features import (
    build_features,
    compute_bollinger,
    compute_ema,
    compute_macd,
    compute_rsi,
    load_feature_cache,
)
from market_env import MarketEnv
from utils import (
    annualization_factor,
    ensure_dir,
    load_config,
    parse_date,
    resolve_data_sources,
    save_json,
    set_seed,
)


def select_device(pref):
    if pref == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def make_run_id(cfg):
    base = time.strftime("%Y%m%d_%H%M%S")
    run_name = str(cfg.get("train", {}).get("run_name", "")).strip()
    if not run_name:
        return base
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in run_name)
    return f"{safe}_{base}"


def write_csv_log(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_run_config(path, cfg, args, run_meta):
    payload = {"config": cfg, "args": vars(args), "run_meta": run_meta}
    save_json(path, payload)


def infer_metric_mode(metric_name):
    if not metric_name:
        return "max"
    name = str(metric_name).lower()
    if "loss" in name or "drawdown" in name:
        return "min"
    return "max"


def get_metric_value(metric_name, train_loss=None, val_metrics=None, extra_metrics=None):
    if not metric_name:
        return None
    name = str(metric_name)
    if extra_metrics and name in extra_metrics:
        return extra_metrics[name]
    if val_metrics:
        if name in val_metrics:
            return val_metrics[name]
        if name.startswith("val_"):
            key = name[4:]
            if key in val_metrics:
                return val_metrics[key]
    if name == "train_loss":
        return train_loss
    return None


def init_lr_scheduler(optimizer, cfg, default_metric=None):
    sched_cfg = cfg.get("train", {}).get("lr_scheduler", {})
    sched_type = str(sched_cfg.get("type", "none")).lower()
    if sched_type in ("none", "", "null"):
        return None, None

    metric = sched_cfg.get("metric", default_metric)
    mode = str(sched_cfg.get("mode", infer_metric_mode(metric))).lower()
    if sched_type == "plateau":
        factor = float(sched_cfg.get("factor", 0.5))
        patience = int(sched_cfg.get("patience", 5))
        min_lr = float(sched_cfg.get("min_lr", 1e-6))
        threshold = float(sched_cfg.get("threshold", 1e-4))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            threshold=threshold,
        )
        return scheduler, {"type": "plateau", "metric": metric}

    if sched_type == "cosine":
        t_max = int(sched_cfg.get("t_max", cfg["train"]["epochs"]))
        min_lr = float(sched_cfg.get("min_lr", 0.0))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=min_lr
        )
        return scheduler, {"type": "cosine"}

    if sched_type == "step":
        step_size = int(sched_cfg.get("step_size", 10))
        gamma = float(sched_cfg.get("gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
        return scheduler, {"type": "step"}

    raise ValueError(f"unsupported lr_scheduler type {sched_type}")


def step_lr_scheduler(scheduler, scheduler_info, metric_value):
    if scheduler is None or not scheduler_info:
        return
    if scheduler_info["type"] == "plateau":
        if metric_value is None:
            return
        scheduler.step(metric_value)
    else:
        scheduler.step()


def init_early_stopping(cfg, default_metric=None):
    early_cfg = cfg.get("train", {}).get("early_stopping", {})
    if not early_cfg.get("enabled", False):
        return None
    metric = early_cfg.get("metric", default_metric)
    mode = str(early_cfg.get("mode", infer_metric_mode(metric))).lower()
    return {
        "metric": metric,
        "mode": mode,
        "patience": int(early_cfg.get("patience", 10)),
        "min_delta": float(early_cfg.get("min_delta", 0.0)),
        "best": None,
        "bad_epochs": 0,
    }


def update_early_stopping(state, metric_value):
    if state is None or metric_value is None:
        return False
    if state["best"] is None:
        state["best"] = float(metric_value)
        state["bad_epochs"] = 0
        return False

    best = state["best"]
    if state["mode"] == "min":
        improved = metric_value < best - state["min_delta"]
    else:
        improved = metric_value > best + state["min_delta"]

    if improved:
        state["best"] = float(metric_value)
        state["bad_epochs"] = 0
        return False

    state["bad_epochs"] += 1
    return state["bad_epochs"] >= state["patience"]


def aggregate_metrics(metrics_list, weights=None):
    if not metrics_list:
        return None
    if weights is None:
        weights = [1.0] * len(metrics_list)
    weights = np.asarray(weights, dtype=np.float64)
    if np.sum(weights) <= 0:
        weights = np.ones_like(weights)

    keys = set()
    for metrics in metrics_list:
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                keys.add(key)

    summary = {}
    for key in sorted(keys):
        vals = []
        wts = []
        for metrics, weight in zip(metrics_list, weights):
            value = metrics.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                vals.append(float(value))
                wts.append(float(weight))
        if wts:
            summary[key] = float(np.average(vals, weights=wts))
    summary["datasets"] = len(metrics_list)
    return summary


def action_to_index(actions):
    return actions + 1


def index_to_action(indices):
    return indices - 1


def atanh(x):
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


BEHAVIOR_POLICIES = (
    "ema3_6_cross",
    "ema9_12_cross",
    "ema15_18_cross",
    "boll_ema12_expansion",
    "macd_signal_reversal",
    "rsi_extremes",
)


def build_behavior_actions(df, cfg, action_mode, rng, policy_override=None):
    dataset_cfg = cfg.get("dataset", {})
    policy = str(policy_override or dataset_cfg.get("behavior_policy", "ema3_6_cross")).lower()
    stop_loss = float(dataset_cfg.get("behavior_stop_loss", 0.1))

    close = df["close"].to_numpy(dtype=np.float64)

    def ema_series(window):
        col = f"ema_{window}"
        if col in df.columns:
            return df[col].astype(float).to_numpy()
        return compute_ema(df["close"], window).to_numpy(dtype=np.float64)

    def rsi_series(window):
        if "rsi" in df.columns:
            return df["rsi"].astype(float).to_numpy()
        return compute_rsi(df["close"], window).to_numpy(dtype=np.float64)

    def macd_signal_series():
        if "macd_signal" in df.columns:
            return df["macd_signal"].astype(float).to_numpy()
        macd_fast = int(cfg.get("features", {}).get("macd_fast", 12))
        macd_slow = int(cfg.get("features", {}).get("macd_slow", 26))
        macd_signal = int(cfg.get("features", {}).get("macd_signal", 9))
        _, signal, _ = compute_macd(df["close"], macd_fast, macd_slow, macd_signal)
        return signal.to_numpy(dtype=np.float64)

    def boll_series():
        boll_window = int(cfg.get("features", {}).get("boll_window", 20))
        boll_n_std = float(cfg.get("features", {}).get("boll_n_std", 2.0))
        mid, upper, lower, _ = compute_bollinger(df["close"], boll_window, boll_n_std)
        width = (upper - lower) / (mid + 1e-12)
        return (
            mid.to_numpy(dtype=np.float64),
            width.to_numpy(dtype=np.float64),
        )

    def apply_stop_loss(position, entry_price, price):
        if position == 0.0 or entry_price <= 0:
            return False
        pnl = position * (price / entry_price - 1.0)
        return pnl <= -stop_loss

    def run_ema_cross(fast, slow):
        fast = np.asarray(fast, dtype=np.float64)
        slow = np.asarray(slow, dtype=np.float64)
        actions = np.zeros(len(close), dtype=np.float64)
        position = 0.0
        entry_price = 0.0
        for idx in range(len(close)):
            if np.isnan(fast[idx]) or np.isnan(slow[idx]):
                actions[idx] = 0.0 if position == 0.0 else position
                continue
            if position != 0.0 and apply_stop_loss(position, entry_price, close[idx]):
                position = 0.0
                entry_price = 0.0
                actions[idx] = 0.0
                continue
            if position == 0.0 and idx > 0:
                if not np.isnan(fast[idx - 1]) and not np.isnan(slow[idx - 1]):
                    if fast[idx - 1] <= slow[idx - 1] and fast[idx] > slow[idx]:
                        position = 1.0
                        entry_price = close[idx]
                    elif fast[idx - 1] >= slow[idx - 1] and fast[idx] < slow[idx]:
                        position = -1.0
                        entry_price = close[idx]
            elif position == 1.0 and idx > 0:
                if fast[idx] < fast[idx - 1]:
                    position = 0.0
                    entry_price = 0.0
            elif position == -1.0 and idx > 0:
                if fast[idx] > fast[idx - 1]:
                    position = 0.0
                    entry_price = 0.0
            actions[idx] = position
        return actions

    def run_boll_ema12():
        ema12 = ema_series(12)
        mid, width = boll_series()
        actions = np.zeros(len(close), dtype=np.float64)
        position = 0.0
        entry_price = 0.0
        for idx in range(len(close)):
            if np.isnan(ema12[idx]) or np.isnan(mid[idx]) or np.isnan(width[idx]):
                actions[idx] = 0.0 if position == 0.0 else position
                continue
            if position != 0.0 and apply_stop_loss(position, entry_price, close[idx]):
                position = 0.0
                entry_price = 0.0
                actions[idx] = 0.0
                continue
            if idx == 0 or np.isnan(width[idx - 1]):
                actions[idx] = position
                continue
            expanding = width[idx] > width[idx - 1]
            contracting = width[idx] < width[idx - 1]
            if position == 0.0:
                if expanding:
                    if ema12[idx] > mid[idx]:
                        position = 1.0
                        entry_price = close[idx]
                    elif ema12[idx] < mid[idx]:
                        position = -1.0
                        entry_price = close[idx]
            else:
                if contracting:
                    position = 0.0
                    entry_price = 0.0
            actions[idx] = position
        return actions

    def run_macd_signal():
        signal = macd_signal_series()
        actions = np.zeros(len(close), dtype=np.float64)
        position = 0.0
        entry_price = 0.0
        for idx in range(len(close)):
            if np.isnan(signal[idx]):
                actions[idx] = 0.0 if position == 0.0 else position
                continue
            if position != 0.0 and apply_stop_loss(position, entry_price, close[idx]):
                position = 0.0
                entry_price = 0.0
                actions[idx] = 0.0
                continue
            if position == 0.0 and idx > 0 and not np.isnan(signal[idx - 1]):
                if signal[idx - 1] <= 0.0 and signal[idx] > 0.0:
                    position = 1.0
                    entry_price = close[idx]
                elif signal[idx - 1] >= 0.0 and signal[idx] < 0.0:
                    position = -1.0
                    entry_price = close[idx]
            elif position == 1.0 and idx > 1:
                if signal[idx - 1] > signal[idx - 2] and signal[idx] < signal[idx - 1]:
                    position = 0.0
                    entry_price = 0.0
            elif position == -1.0 and idx > 1:
                if signal[idx - 1] < signal[idx - 2] and signal[idx] > signal[idx - 1]:
                    position = 0.0
                    entry_price = 0.0
            actions[idx] = position
        return actions

    def run_rsi_extremes():
        rsi = rsi_series(int(cfg.get("features", {}).get("rsi_window", 14)))
        actions = np.zeros(len(close), dtype=np.float64)
        position = 0.0
        entry_price = 0.0
        for idx in range(len(close)):
            if np.isnan(rsi[idx]):
                actions[idx] = 0.0 if position == 0.0 else position
                continue
            if position != 0.0 and apply_stop_loss(position, entry_price, close[idx]):
                position = 0.0
                entry_price = 0.0
                actions[idx] = 0.0
                continue
            if position == 0.0:
                if rsi[idx] > 90.0:
                    position = -1.0
                    entry_price = close[idx]
                elif rsi[idx] < 10.0:
                    position = 1.0
                    entry_price = close[idx]
            elif position == 1.0:
                if rsi[idx] > 70.0:
                    position = 0.0
                    entry_price = 0.0
            elif position == -1.0:
                if rsi[idx] < 30.0:
                    position = 0.0
                    entry_price = 0.0
            actions[idx] = position
        return actions

    if policy == "ema3_6_cross":
        actions = run_ema_cross(ema_series(3), ema_series(6))
    elif policy == "ema9_12_cross":
        actions = run_ema_cross(ema_series(9), ema_series(12))
    elif policy == "ema15_18_cross":
        actions = run_ema_cross(ema_series(15), ema_series(18))
    elif policy == "boll_ema12_expansion":
        actions = run_boll_ema12()
    elif policy == "macd_signal_reversal":
        actions = run_macd_signal()
    elif policy == "rsi_extremes":
        actions = run_rsi_extremes()
    else:
        raise ValueError(f"unsupported behavior_policy {policy}")

    if action_mode == "continuous":
        actions = np.clip(actions.astype(np.float32), -1.0, 1.0)
    else:
        actions = actions.astype(np.int64)

    return actions


def build_trajectories(df, state_cols, cfg, action_mode, act_dim, rng):
    seq_len = int(cfg["dataset"]["seq_len"])
    dataset_cfg = cfg.get("dataset", {})
    policy = str(dataset_cfg.get("behavior_policy", "mixed")).lower()
    episode_len = dataset_cfg.get("episode_len", None)
    if episode_len is None:
        episode_len = cfg.get("rl", {}).get("episode_len", None)
    if episode_len is None:
        episode_len = len(df)
    else:
        episode_len = int(episode_len)
    if episode_len <= 0:
        episode_len = len(df)

    gamma = float(dataset_cfg.get("rtg_gamma", cfg.get("rl", {}).get("gamma", 1.0)))
    rtg_scale = float(dataset_cfg.get("rtg_scale", 1.0))
    if rtg_scale <= 0:
        rtg_scale = 1.0

    reward_scale = float(cfg.get("rl", {}).get("reward_scale", 1.0))
    turnover_penalty = float(cfg.get("rl", {}).get("turnover_penalty", 0.0))
    position_penalty = float(cfg.get("rl", {}).get("position_penalty", 0.0))
    drawdown_penalty = float(cfg.get("rl", {}).get("drawdown_penalty", 0.0))
    price_mode = cfg["rewards"].get("price_mode", "close")
    range_penalty = float(cfg["rewards"].get("range_penalty", 0.0))
    fee = float(cfg["rewards"]["fee"])
    slip = float(cfg["rewards"]["slip"])

    if action_mode == "continuous" and act_dim != 1:
        raise ValueError("continuous action_mode currently supports action_dim=1")

    if policy == "mixed":
        mix_cfg = dataset_cfg.get("behavior_mix", None)
        if mix_cfg is None:
            policies = list(BEHAVIOR_POLICIES)
            weights = np.ones(len(policies), dtype=np.float64)
        elif isinstance(mix_cfg, dict):
            policies = []
            weights = []
            for name, weight in mix_cfg.items():
                policy_name = str(name).lower()
                if policy_name not in BEHAVIOR_POLICIES:
                    raise ValueError(f"unsupported behavior policy in mix: {policy_name}")
                weight_val = float(weight)
                if weight_val <= 0:
                    raise ValueError(f"behavior_mix weight must be > 0 for {policy_name}")
                policies.append(policy_name)
                weights.append(weight_val)
            weights = np.asarray(weights, dtype=np.float64)
        elif isinstance(mix_cfg, (list, tuple)):
            policies = [str(name).lower() for name in mix_cfg]
            if not policies:
                raise ValueError("behavior_mix list is empty")
            for name in policies:
                if name not in BEHAVIOR_POLICIES:
                    raise ValueError(f"unsupported behavior policy in mix: {name}")
            weights = np.ones(len(policies), dtype=np.float64)
        else:
            raise ValueError("behavior_mix must be a list or dict of weights")

        weights = weights / weights.sum()
        actions_by_policy = {
            name: build_behavior_actions(df, cfg, action_mode, rng, policy_override=name)
            for name in policies
        }
    else:
        actions = build_behavior_actions(df, cfg, action_mode, rng, policy_override=policy)
    states = df[state_cols].to_numpy(dtype=np.float32)
    close = df["close"].to_numpy(dtype=np.float32)
    open_prices = df["open"].to_numpy(dtype=np.float32) if "open" in df.columns else None
    high_prices = df["high"].to_numpy(dtype=np.float32) if "high" in df.columns else None
    low_prices = df["low"].to_numpy(dtype=np.float32) if "low" in df.columns else None

    trajectories = []
    start = 0
    total_len = len(df)
    while start < total_len:
        end = min(start + episode_len, total_len)
        if end - start < seq_len:
            break
        if policy == "mixed":
            chosen = str(rng.choice(policies, p=weights)).lower()
            actions_ep = actions_by_policy[chosen][start:end]
        else:
            actions_ep = actions[start:end]
        close_ep = close[start:end]
        rewards_ep = compute_trajectory_rewards(
            actions_ep,
            close_ep,
            fee,
            slip,
            reward_scale=reward_scale,
            turnover_penalty=turnover_penalty,
            position_penalty=position_penalty,
            drawdown_penalty=drawdown_penalty,
            open_=open_prices[start:end] if open_prices is not None else None,
            high=high_prices[start:end] if high_prices is not None else None,
            low=low_prices[start:end] if low_prices is not None else None,
            price_mode=price_mode,
            range_penalty=range_penalty,
        )
        rtg_ep = compute_rtg(rewards_ep, gamma) / rtg_scale
        trajectories.append(
            {
                "states": states[start:end],
                "actions": actions_ep,
                "rtg": rtg_ep,
            }
        )
        start = end

    return trajectories


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, seq_len, action_mode, act_dim):
        self.trajectories = trajectories
        self.seq_len = seq_len
        self.action_mode = action_mode
        self.act_dim = act_dim
        self.window_counts = [
            max(0, len(traj["states"]) - seq_len + 1) for traj in trajectories
        ]
        self.cum_windows = (
            np.cumsum(self.window_counts) if self.window_counts else np.array([])
        )

    def __len__(self):
        return int(self.cum_windows[-1]) if len(self.cum_windows) else 0

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("trajectory index out of range")
        traj_idx = int(np.searchsorted(self.cum_windows, idx, side="right"))
        prev_cum = int(self.cum_windows[traj_idx - 1]) if traj_idx > 0 else 0
        start = int(idx - prev_cum)
        traj = self.trajectories[traj_idx]

        states = traj["states"][start : start + self.seq_len]
        actions = traj["actions"][start : start + self.seq_len]
        rtg = traj["rtg"][start : start + self.seq_len]

        if self.action_mode == "continuous":
            actions_in = np.zeros((self.seq_len, self.act_dim), dtype=np.float32)
            prev_action = traj["actions"][start - 1] if start > 0 else 0.0
            actions_in[0, 0] = float(prev_action)
            actions_in[1:, 0] = np.asarray(actions[:-1], dtype=np.float32)
            targets = np.asarray(actions, dtype=np.float32)
        else:
            actions_in = np.zeros(self.seq_len, dtype=np.int64)
            prev_action = traj["actions"][start - 1] if start > 0 else 0
            actions_in[0] = int(prev_action)
            actions_in[1:] = np.asarray(actions[:-1], dtype=np.int64)
            targets = np.asarray(actions, dtype=np.int64)

        return states, actions_in, rtg, targets


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    action_mode,
    act_dim,
    grad_clip=None,
    progress=False,
    progress_desc=None,
    progress_position=0,
):
    model.train()
    total_loss = 0.0
    total_batches = 0

    train_iter = loader
    pbar = None
    if progress and tqdm is not None:
        pbar = tqdm(
            total=len(loader),
            desc=progress_desc or "train",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            position=progress_position,
        )

    for states, actions_in, rtg, targets in train_iter:
        states = states.to(device)
        rtg = rtg.to(device)

        if action_mode == "continuous":
            actions_in = actions_in.to(device)
            targets = targets.to(device)
            logits = model(states, actions_in, rtg)
            target_clipped = torch.clamp(targets, -0.999, 0.999)
            target_pre = atanh(target_clipped)
            if target_pre.dim() == 2:
                target_pre = target_pre.unsqueeze(-1)
            loss = F.mse_loss(logits, target_pre)
        else:
            actions_in = action_to_index(actions_in).long().to(device)
            targets = action_to_index(targets).long().to(device)
            logits = model(states, actions_in, rtg)
            loss = F.cross_entropy(logits.reshape(-1, act_dim), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    if total_batches == 0:
        return 0.0
    return total_loss / total_batches


def evaluate_policy(
    cfg,
    model,
    df,
    state_cols,
    device,
    action_mode,
    act_dim,
    timeframe=None,
    progress=False,
    progress_desc=None,
    progress_position=0,
):
    if df is None or df.empty:
        return None

    was_training = model.training
    model.eval()

    seq_len = cfg["dataset"]["seq_len"]
    fee = float(cfg["rewards"]["fee"])
    slip = float(cfg["rewards"]["slip"])
    reward_scale = float(cfg.get("rl", {}).get("reward_scale", 1.0))
    turnover_penalty = float(cfg.get("rl", {}).get("turnover_penalty", 0.0))
    position_penalty = float(cfg.get("rl", {}).get("position_penalty", 0.0))
    drawdown_penalty = float(cfg.get("rl", {}).get("drawdown_penalty", 0.0))
    price_mode = cfg["rewards"].get("price_mode", "close")
    range_penalty = float(cfg["rewards"].get("range_penalty", 0.0))
    risk_free = float(cfg.get("backtest", {}).get("risk_free", 0.0))

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

    eval_cfg = cfg.get("rl", {})
    eval_mode = str(eval_cfg.get("eval_mode", "deterministic")).lower()
    if eval_mode not in ("deterministic", "sample"):
        eval_mode = "deterministic"
    eval_samples = int(eval_cfg.get("eval_samples", 1))
    if eval_mode == "deterministic":
        eval_samples = 1
    else:
        eval_samples = max(1, eval_samples)
    eval_seed = eval_cfg.get("eval_seed", None)
    try:
        eval_seed = int(eval_seed) if eval_seed is not None else None
    except (TypeError, ValueError):
        eval_seed = None

    def run_once(rng, pbar):
        if action_mode == "continuous":
            actions = np.zeros(len(df), dtype=np.float32)
        else:
            actions = np.zeros(len(df), dtype=np.int64)

        rtg_hist = np.zeros(len(df), dtype=np.float32)
        rtg_hist[0] = target_return
        current_rtg = float(target_return)
        prev_action = 0.0
        equity = 1.0
        max_equity = 1.0

        for idx in range(len(df)):
            if idx < seq_len:
                action = 0.0 if action_mode == "continuous" else 0
            else:
                state_window = states[idx - seq_len + 1 : idx + 1]
                rtg_window = rtg_hist[idx - seq_len + 1 : idx + 1] / rtg_scale

                if action_mode == "continuous":
                    actions_in = np.zeros((seq_len, act_dim), dtype=np.float32)
                    window_prev = actions[idx - seq_len] if idx - seq_len >= 0 else 0.0
                    actions_in[0, 0] = float(window_prev)
                    actions_in[1:, 0] = actions[idx - seq_len + 1 : idx]
                    a = torch.tensor(actions_in, device=device).unsqueeze(0)
                else:
                    actions_in = np.zeros(seq_len, dtype=np.int64)
                    window_prev = actions[idx - seq_len] if idx - seq_len >= 0 else 0
                    actions_in[0] = int(window_prev)
                    actions_in[1:] = actions[idx - seq_len + 1 : idx]
                    a = torch.tensor(action_to_index(actions_in), device=device).unsqueeze(0)

                with torch.no_grad():
                    s = torch.tensor(state_window, device=device).unsqueeze(0)
                    r = torch.tensor(rtg_window, device=device).unsqueeze(0)
                    logits = model(s, a, r)
                    if action_mode == "continuous":
                        mean = logits[0, -1].detach().cpu().numpy()
                        if eval_mode == "sample":
                            std = model.log_std.exp().detach().cpu().numpy()
                            sample = mean + rng.normal(scale=std, size=mean.shape)
                            action = float(np.tanh(sample).squeeze())
                        else:
                            action = float(torch.tanh(logits[0, -1]).cpu().numpy().item())
                    else:
                        if eval_mode == "sample":
                            probs = torch.softmax(logits[0, -1], dim=-1).cpu().numpy()
                            action_idx = int(rng.choice(len(probs), p=probs))
                        else:
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
            prev_action = action
            if pbar is not None:
                pbar.update(1)

        equity_curve, step_returns, trade_count, turnover = simulate(
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
        annual_factor = annualization_factor(timeframe or cfg["data"]["timeframe"])
        return compute_metrics(
            equity_curve,
            step_returns,
            trade_count,
            turnover,
            annual_factor,
            actions=actions,
            risk_free=risk_free,
        )

    total_steps = len(df) * eval_samples
    pbar = None
    if progress and tqdm is not None:
        pbar = tqdm(
            total=total_steps,
            desc=progress_desc or "eval",
            unit="step",
            dynamic_ncols=True,
            leave=False,
            position=progress_position,
        )

    metrics_list = []
    for sample_idx in range(eval_samples):
        if eval_seed is None:
            rng = np.random
        else:
            rng = np.random.RandomState(eval_seed + sample_idx)
        metrics_list.append(run_once(rng, pbar))

    if pbar is not None:
        pbar.close()

    if len(metrics_list) == 1:
        metrics = metrics_list[0]
    else:
        metric_keys = [
            "total_return",
            "sharpe",
            "max_drawdown",
            "profit_factor",
            "trade_count",
            "turnover",
        ]
        metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metric_keys}
        metrics["eval_samples"] = eval_samples
        metrics["eval_mode"] = eval_mode

    metrics["timestamps"] = [int(timestamps[0]), int(timestamps[-1])]
    if was_training:
        model.train()
    return metrics


def build_context(states_hist, actions_hist, rewards_hist, seq_len, action_mode, act_dim):
    start = max(0, len(states_hist) - seq_len)
    actual_len = len(states_hist) - start

    state_window = np.asarray(states_hist[start:], dtype=np.float32)
    reward_window = np.asarray(rewards_hist[start:], dtype=np.float32)

    if action_mode == "continuous":
        actions_in = np.zeros((actual_len, act_dim), dtype=np.float32)
        if start > 0:
            actions_in[0, 0] = float(actions_hist[start - 1])
        if actual_len > 1:
            actions_in[1:, 0] = np.asarray(
                actions_hist[start : start + actual_len - 1], dtype=np.float32
            )
        action_ctx = np.zeros((seq_len, act_dim), dtype=np.float32)
    else:
        actions_in = np.zeros(actual_len, dtype=np.int64)
        if start > 0:
            actions_in[0] = int(actions_hist[start - 1])
        if actual_len > 1:
            actions_in[1:] = np.asarray(
                actions_hist[start : start + actual_len - 1], dtype=np.int64
            )
        action_ctx = np.zeros(seq_len, dtype=np.int64)

    state_ctx = np.zeros((seq_len, state_window.shape[1]), dtype=np.float32)
    reward_ctx = np.zeros(seq_len, dtype=np.float32)
    state_ctx[-actual_len:] = state_window
    reward_ctx[-actual_len:] = reward_window
    action_ctx[-actual_len:] = actions_in

    return state_ctx, action_ctx, reward_ctx


def policy_step(model, state_ctx, action_ctx, reward_ctx, device, action_mode):
    with torch.no_grad():
        states = torch.tensor(state_ctx, device=device).unsqueeze(0)
        rewards = torch.tensor(reward_ctx, device=device).unsqueeze(0)
        if action_mode == "continuous":
            actions_in = torch.tensor(action_ctx, device=device).unsqueeze(0)
            mean, values = model(states, actions_in, rewards, return_values=True)
            mean = mean[:, -1]
            values = values[:, -1]
            log_std = model.log_std.view(1, -1)
            std = log_std.exp()
            normal = Normal(mean, std)
            z = normal.rsample()
            action = torch.tanh(z)
            log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(-1)
            return float(action.squeeze(0).item()), None, float(log_prob.item()), float(values.item())

        actions_in = torch.tensor(action_to_index(action_ctx), device=device).unsqueeze(0)
        logits, values = model(states, actions_in, rewards, return_values=True)
        logits = logits[:, -1]
        values = values[:, -1]
        dist = Categorical(logits=logits)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        action_val = int(index_to_action(action_idx.item()))
        return action_val, int(action_idx.item()), float(log_prob.item()), float(values.item())


def predict_value(model, state_ctx, action_ctx, reward_ctx, device, action_mode):
    with torch.no_grad():
        states = torch.tensor(state_ctx, device=device).unsqueeze(0)
        rewards = torch.tensor(reward_ctx, device=device).unsqueeze(0)
        if action_mode == "continuous":
            actions_in = torch.tensor(action_ctx, device=device).unsqueeze(0)
        else:
            actions_in = torch.tensor(action_to_index(action_ctx), device=device).unsqueeze(0)
        _, values = model(states, actions_in, rewards, return_values=True)
        return float(values[0, -1].item())


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = 0.0
    for idx in reversed(range(len(rewards))):
        if idx == len(rewards) - 1:
            next_non_terminal = 1.0 - float(dones[idx])
            next_value = last_value
        else:
            next_non_terminal = 1.0 - float(dones[idx])
            next_value = values[idx + 1]
        delta = rewards[idx] + gamma * next_value * next_non_terminal - values[idx]
        last_adv = delta + gamma * gae_lambda * next_non_terminal * last_adv
        advantages[idx] = last_adv
    returns = advantages + values
    return advantages, returns


def collect_rollout(
    env,
    model,
    device,
    seq_len,
    action_mode,
    act_dim,
    rollout_steps,
    gamma,
    gae_lambda,
    progress=False,
    progress_desc=None,
    progress_position=0,
):
    states_hist = [env.reset()]
    actions_hist = []
    rewards_hist = [0.0]

    ctx_states = []
    ctx_actions = []
    ctx_rewards = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []

    ep_returns = []
    ep_lengths = []
    ep_return = 0.0
    ep_len = 0

    step_iter = range(rollout_steps)
    if progress and tqdm is not None:
        step_iter = tqdm(
            step_iter,
            desc=progress_desc or "rollout",
            unit="step",
            dynamic_ncols=True,
            leave=False,
            position=progress_position,
        )
    for _ in step_iter:
        state_ctx, action_ctx, reward_ctx = build_context(
            states_hist, actions_hist, rewards_hist, seq_len, action_mode, act_dim
        )
        action, action_idx, log_prob, value = policy_step(
            model, state_ctx, action_ctx, reward_ctx, device, action_mode
        )

        next_state, reward, done, _ = env.step(action)

        ctx_states.append(state_ctx)
        ctx_actions.append(action_ctx)
        ctx_rewards.append(reward_ctx)
        actions.append(action_idx if action_mode == "discrete" else action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        dones.append(done)

        actions_hist.append(action)
        rewards_hist.append(reward)
        states_hist.append(next_state)

        ep_return += reward
        ep_len += 1
        if done:
            ep_returns.append(ep_return)
            ep_lengths.append(ep_len)
            states_hist = [env.reset()]
            actions_hist = []
            rewards_hist = [0.0]
            ep_return = 0.0
            ep_len = 0

    last_value = 0.0
    if not dones[-1]:
        state_ctx, action_ctx, reward_ctx = build_context(
            states_hist, actions_hist, rewards_hist, seq_len, action_mode, act_dim
        )
        last_value = predict_value(model, state_ctx, action_ctx, reward_ctx, device, action_mode)

    adv, ret = compute_gae(
        np.asarray(rewards, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
        np.asarray(dones, dtype=np.bool_),
        last_value,
        gamma,
        gae_lambda,
    )

    return {
        "states": np.asarray(ctx_states, dtype=np.float32),
        "actions_in": np.asarray(ctx_actions),
        "rewards_in": np.asarray(ctx_rewards, dtype=np.float32),
        "actions": np.asarray(actions),
        "log_probs": np.asarray(log_probs, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "advantages": adv,
        "returns": ret,
        "ep_returns": ep_returns,
        "ep_lengths": ep_lengths,
    }


def ppo_update(
    model,
    optimizer,
    buffer,
    device,
    cfg,
    action_mode,
    progress=False,
    progress_desc=None,
    progress_position=0,
):
    clip_range = float(cfg["rl"].get("clip_range", 0.2))
    value_coef = float(cfg["rl"].get("value_coef", 0.5))
    entropy_coef = float(cfg["rl"].get("entropy_coef", 0.01))
    ppo_epochs = int(cfg["rl"].get("ppo_epochs", 4))
    minibatch_size = int(cfg["rl"].get("minibatch_size", 256))
    grad_clip = cfg["train"].get("grad_clip", None)

    states = torch.tensor(buffer["states"], device=device)
    rewards_in = torch.tensor(buffer["rewards_in"], device=device)
    actions_in = torch.tensor(buffer["actions_in"], device=device)
    old_log_probs = torch.tensor(buffer["log_probs"], device=device)
    advantages = torch.tensor(buffer["advantages"], device=device)
    returns = torch.tensor(buffer["returns"], device=device)

    if action_mode == "discrete":
        actions = torch.tensor(buffer["actions"], device=device, dtype=torch.long)
        actions_in = action_to_index(actions_in).long()
    else:
        actions = torch.tensor(buffer["actions"], device=device, dtype=torch.float32)
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_clip = 0.0
    total_batches = 0

    batch_size = states.shape[0]
    total_steps = 0
    update_pbar = None
    if progress and tqdm is not None:
        steps_per_epoch = int(np.ceil(batch_size / minibatch_size))
        total_steps = ppo_epochs * steps_per_epoch
        update_pbar = tqdm(
            total=total_steps,
            desc=progress_desc or "update",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            position=progress_position,
        )

    for _ in range(ppo_epochs):
        perm = np.random.permutation(batch_size)
        for start in range(0, batch_size, minibatch_size):
            idx = perm[start : start + minibatch_size]
            logits, values = model(
                states[idx],
                actions_in[idx],
                rewards_in[idx],
                return_values=True,
            )
            logits = logits[:, -1]
            values = values[:, -1]

            if action_mode == "continuous":
                mean = logits
                log_std = model.log_std.view(1, -1)
                std = log_std.exp()
                normal = Normal(mean, std)
                clipped_actions = torch.clamp(actions[idx], -0.999, 0.999)
                z = atanh(clipped_actions)
                new_log_prob = normal.log_prob(z) - torch.log(1.0 - clipped_actions.pow(2) + 1e-6)
                new_log_prob = new_log_prob.sum(-1)
                entropy = normal.entropy().sum(-1)
            else:
                dist = Categorical(logits=logits)
                new_log_prob = dist.log_prob(actions[idx])
                entropy = dist.entropy()

            log_ratio = new_log_prob - old_log_probs[idx]
            ratio = torch.exp(log_ratio)
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns[idx] - values).pow(2).mean()
            entropy_loss = -entropy_coef * entropy.mean()
            loss = policy_loss + value_coef * value_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs[idx] - new_log_prob).mean()
                clip_frac = (torch.abs(ratio - 1.0) > clip_range).float().mean()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy.mean().item())
            total_kl += float(approx_kl.item())
            total_clip += float(clip_frac.item())
            total_batches += 1
            if update_pbar is not None:
                update_pbar.update(1)

    if update_pbar is not None:
        update_pbar.close()

    if total_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    return (
        total_policy_loss / total_batches,
        total_value_loss / total_batches,
        total_entropy / total_batches,
        total_kl / total_batches,
        total_clip / total_batches,
    )


def evaluate_policy_ppo(
    cfg,
    model,
    df,
    state_cols,
    device,
    action_mode,
    act_dim,
    timeframe=None,
    progress=False,
    progress_desc=None,
    progress_position=0,
):
    if df is None or df.empty:
        return None

    was_training = model.training
    model.eval()

    seq_len = cfg["dataset"]["seq_len"]
    fee = float(cfg["rewards"]["fee"])
    slip = float(cfg["rewards"]["slip"])
    reward_scale = float(cfg.get("rl", {}).get("reward_scale", 1.0))
    turnover_penalty = float(cfg.get("rl", {}).get("turnover_penalty", 0.0))
    position_penalty = float(cfg.get("rl", {}).get("position_penalty", 0.0))
    drawdown_penalty = float(cfg.get("rl", {}).get("drawdown_penalty", 0.0))
    price_mode = cfg["rewards"].get("price_mode", "close")
    range_penalty = float(cfg["rewards"].get("range_penalty", 0.0))
    risk_free = float(cfg.get("backtest", {}).get("risk_free", 0.0))

    states = df[state_cols].to_numpy(dtype=np.float32)
    close = df["close"].to_numpy(dtype=np.float32)
    open_prices = df["open"].to_numpy(dtype=np.float32) if "open" in df.columns else None
    high_prices = df["high"].to_numpy(dtype=np.float32) if "high" in df.columns else None
    low_prices = df["low"].to_numpy(dtype=np.float32) if "low" in df.columns else None
    timestamps = df["timestamp"].to_numpy(dtype=np.int64)

    eval_cfg = cfg.get("rl", {})
    eval_mode = str(eval_cfg.get("eval_mode", "deterministic")).lower()
    if eval_mode not in ("deterministic", "sample"):
        eval_mode = "deterministic"
    eval_samples = int(eval_cfg.get("eval_samples", 1))
    if eval_mode == "deterministic":
        eval_samples = 1
    else:
        eval_samples = max(1, eval_samples)
    eval_seed = eval_cfg.get("eval_seed", None)
    try:
        eval_seed = int(eval_seed) if eval_seed is not None else None
    except (TypeError, ValueError):
        eval_seed = None

    def run_once(rng, pbar):
        if action_mode == "continuous":
            actions = np.zeros(len(df), dtype=np.float32)
        else:
            actions = np.zeros(len(df), dtype=np.int64)

        rewards_hist = np.zeros(len(df), dtype=np.float32)
        prev_action = 0.0

        for idx in range(len(df)):
            if idx < seq_len:
                action = 0.0 if action_mode == "continuous" else 0
            else:
                state_window = states[idx - seq_len + 1 : idx + 1]
                reward_window = rewards_hist[idx - seq_len + 1 : idx + 1]
                action_window = actions[idx - seq_len + 1 : idx + 1]

                if action_mode == "continuous":
                    actions_in = np.zeros((seq_len, act_dim), dtype=np.float32)
                    window_prev = actions[idx - seq_len] if idx - seq_len >= 0 else 0.0
                    actions_in[0, 0] = float(window_prev)
                    actions_in[1:, 0] = action_window[:-1]
                    a = torch.tensor(actions_in, device=device).unsqueeze(0)
                else:
                    actions_in = np.zeros(seq_len, dtype=np.int64)
                    window_prev = actions[idx - seq_len] if idx - seq_len >= 0 else 0
                    actions_in[0] = int(window_prev)
                    actions_in[1:] = action_window[:-1]
                    a = torch.tensor(action_to_index(actions_in), device=device).unsqueeze(0)

                with torch.no_grad():
                    s = torch.tensor(state_window, device=device).unsqueeze(0)
                    r = torch.tensor(reward_window, device=device).unsqueeze(0)
                    logits, _ = model(s, a, r, return_values=True)
                    if action_mode == "continuous":
                        mean = logits[0, -1].detach().cpu().numpy()
                        if eval_mode == "sample":
                            std = model.log_std.exp().detach().cpu().numpy()
                            sample = mean + rng.normal(scale=std, size=mean.shape)
                            action = float(np.tanh(sample).squeeze())
                        else:
                            action = float(torch.tanh(logits[0, -1]).cpu().numpy().item())
                    else:
                        if eval_mode == "sample":
                            probs = torch.softmax(logits[0, -1], dim=-1).cpu().numpy()
                            action_idx = int(rng.choice(len(probs), p=probs))
                        else:
                            action_idx = int(torch.argmax(logits[0, -1]).item())
                        action = int(index_to_action(action_idx))

            actions[idx] = action
            if idx < len(df) - 1:
                reward, _, _, _ = compute_step_reward(
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
                    price_mode=price_mode,
                    open_t=float(open_prices[idx]) if open_prices is not None else None,
                    high_t=float(high_prices[idx]) if high_prices is not None else None,
                    low_t=float(low_prices[idx]) if low_prices is not None else None,
                    open_t1=float(open_prices[idx + 1]) if open_prices is not None else None,
                    high_t1=float(high_prices[idx + 1]) if high_prices is not None else None,
                    low_t1=float(low_prices[idx + 1]) if low_prices is not None else None,
                    range_penalty=range_penalty,
                )
                rewards_hist[idx + 1] = reward
            prev_action = action
            if pbar is not None:
                pbar.update(1)

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
        annual_factor = annualization_factor(timeframe or cfg["data"]["timeframe"])
        return compute_metrics(
            equity,
            step_returns,
            trade_count,
            turnover,
            annual_factor,
            actions=actions,
            risk_free=risk_free,
        )

    total_steps = len(df) * eval_samples
    pbar = None
    if progress and tqdm is not None:
        pbar = tqdm(
            total=total_steps,
            desc=progress_desc or "eval",
            unit="step",
            dynamic_ncols=True,
            leave=False,
            position=progress_position,
        )

    metrics_list = []
    for sample_idx in range(eval_samples):
        if eval_seed is None:
            rng = np.random
        else:
            rng = np.random.RandomState(eval_seed + sample_idx)
        metrics_list.append(run_once(rng, pbar))

    if pbar is not None:
        pbar.close()

    if len(metrics_list) == 1:
        metrics = metrics_list[0]
    else:
        metric_keys = [
            "total_return",
            "sharpe",
            "max_drawdown",
            "profit_factor",
            "trade_count",
            "turnover",
        ]
        metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metric_keys}
        metrics["eval_samples"] = eval_samples
        metrics["eval_mode"] = eval_mode

    metrics["timestamps"] = [int(timestamps[0]), int(timestamps[-1])]
    if was_training:
        model.train()
    return metrics


def prepare_splits(cfg):
    sources = resolve_data_sources(cfg)
    if not sources:
        raise ValueError("no data sources configured")

    splits = []
    state_cols = None
    train_end = parse_date(cfg["data"]["train_end"])
    val_end = parse_date(cfg["data"].get("val_end", cfg["data"]["train_end"]))
    test_end = parse_date(cfg["data"]["test_end"])

    for symbol, timeframe in sources:
        feat_df, local_state_cols = load_feature_cache(cfg, symbol=symbol, timeframe=timeframe)
        if feat_df is None:
            raw_df = load_or_fetch(cfg, symbol=symbol, timeframe=timeframe)
            feat_df, local_state_cols = build_features(raw_df, cfg["features"])
        feat_df = feat_df.dropna(subset=local_state_cols).reset_index(drop=True)

        if state_cols is None:
            state_cols = local_state_cols
        elif state_cols != local_state_cols:
            raise ValueError("state_cols mismatch across data sources")

        train_df, val_df, _ = split_by_time(feat_df, train_end, val_end, test_end)
        if len(train_df) < 2:
            raise ValueError(f"train split too small for {symbol} {timeframe}")

        splits.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "train": train_df,
                "val": val_df,
            }
        )

    return splits, state_cols


def train_offline(cfg, args):
    seed = int(cfg.get("rl", {}).get("seed", 42))
    set_seed(seed)

    splits, state_cols = prepare_splits(cfg)

    device = select_device(cfg["train"]["device"])
    action_mode = str(cfg.get("rl", {}).get("action_mode", "discrete")).lower()
    act_dim = int(cfg.get("rl", {}).get("action_dim", 1)) if action_mode == "continuous" else 3
    if action_mode == "continuous" and act_dim != 1:
        raise ValueError("continuous action_mode currently supports action_dim=1")

    model = DecisionTransformer(
        state_dim=len(state_cols),
        act_dim=act_dim,
        seq_len=cfg["dataset"]["seq_len"],
        d_model=cfg["train"]["d_model"],
        n_layers=cfg["train"]["n_layers"],
        n_heads=cfg["train"]["n_heads"],
        dropout=cfg["train"]["dropout"],
        action_mode=action_mode,
        use_value_head=False,
    ).to(device)
    model.condition_mode = "rtg"

    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"loaded init checkpoint weights from {args.init_ckpt}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    rng = np.random.RandomState(seed)
    trajectories = []
    for split in splits:
        trajectories.extend(
            build_trajectories(
                split["train"], state_cols, cfg, action_mode, act_dim, rng
            )
        )
    dataset = TrajectoryDataset(trajectories, cfg["dataset"]["seq_len"], action_mode, act_dim)
    if len(dataset) == 0:
        raise ValueError("no training windows available; check seq_len/episode_len")

    batch_size = int(cfg["train"]["batch_size"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    ensure_dir(cfg["train"]["log_dir"])
    ensure_dir(cfg["train"]["checkpoint_dir"])

    log_rows = []
    best_val = -float("inf")
    best_loss = float("inf")
    run_id = make_run_id(cfg)
    run_meta = {
        "run_id": run_id,
        "mode": "offline",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    train_cfg = cfg.get("train", {})
    if train_cfg.get("save_config", True):
        config_path = os.path.join(cfg["train"]["log_dir"], f"run_config_{run_id}.json")
        save_run_config(config_path, cfg, args, run_meta)

    eval_every = int(cfg.get("rl", {}).get("eval_every", 1))
    has_val = any(split["val"] is not None and not split["val"].empty for split in splits)

    epochs = int(cfg["train"]["epochs"])
    train_progress = bool(train_cfg.get("train_progress", train_cfg.get("update_progress", False)))
    eval_progress = bool(train_cfg.get("eval_progress", False))
    epoch_iter = range(1, epochs + 1)
    progress_position = 0

    dataset_cfg = cfg.get("dataset", {})
    ckpt_config = {
        "state_dim": len(state_cols),
        "act_dim": act_dim,
        "seq_len": cfg["dataset"]["seq_len"],
        "d_model": cfg["train"]["d_model"],
        "n_layers": cfg["train"]["n_layers"],
        "n_heads": cfg["train"]["n_heads"],
        "dropout": cfg["train"]["dropout"],
        "action_mode": action_mode,
        "use_value_head": False,
        "condition_mode": "rtg",
        "rtg_gamma": float(dataset_cfg.get("rtg_gamma", cfg.get("rl", {}).get("gamma", 1.0))),
        "rtg_scale": float(dataset_cfg.get("rtg_scale", 1.0)),
        "target_return": float(dataset_cfg.get("target_return", 0.0)),
    }

    default_metric = "val_total_return" if has_val else "train_loss"
    scheduler, scheduler_info = init_lr_scheduler(optimizer, cfg, default_metric=default_metric)
    early_stop = init_early_stopping(cfg, default_metric=default_metric)

    for epoch in epoch_iter:
        train_loss = train_epoch(
            model,
            loader,
            optimizer,
            device,
            action_mode,
            act_dim,
            grad_clip=train_cfg.get("grad_clip", None),
            progress=train_progress,
            progress_desc=f"train {epoch}",
            progress_position=progress_position,
        )

        val_metrics = None
        if has_val and eval_every > 0 and epoch % eval_every == 0:
            metrics_list = []
            weights = []
            for split in splits:
                val_df = split["val"]
                if val_df is None or val_df.empty:
                    continue
                metrics = evaluate_policy(
                    cfg,
                    model,
                    val_df,
                    state_cols,
                    device,
                    action_mode,
                    act_dim,
                    timeframe=split["timeframe"],
                    progress=eval_progress,
                    progress_desc=f"eval {epoch} {split['symbol']} {split['timeframe']}",
                    progress_position=progress_position,
                )
                if metrics is not None:
                    metrics_list.append(metrics)
                    weights.append(len(val_df))
            val_metrics = aggregate_metrics(metrics_list, weights)
            if val_metrics is not None and val_metrics.get("total_return", -float("inf")) > best_val:
                best_val = val_metrics["total_return"]
                ckpt = {"model_state": model.state_dict(), "model_config": ckpt_config}
                best_path = os.path.join(
                    cfg["train"]["checkpoint_dir"], f"dt_best_{run_id}.pt"
                )
                torch.save(ckpt, best_path)
        elif not has_val and train_loss < best_loss:
            best_loss = train_loss
            ckpt = {"model_state": model.state_dict(), "model_config": ckpt_config}
            best_path = os.path.join(
                cfg["train"]["checkpoint_dir"], f"dt_best_{run_id}.pt"
            )
            torch.save(ckpt, best_path)

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if val_metrics is not None:
            for key in (
                "total_return",
                "annual_return",
                "sharpe",
                "sortino",
                "max_drawdown",
                "calmar",
                "profit_factor",
                "win_rate",
                "exposure",
                "trade_count",
                "turnover",
            ):
                if key in val_metrics:
                    log_row[f"val_{key}"] = val_metrics[key]
            log_row["val_datasets"] = val_metrics.get("datasets", 0)

        log_rows.append(log_row)

        val_str = ""
        if val_metrics is not None:
            val_str = f" val_total_return={val_metrics['total_return']:.4f}"
        print(f"epoch {epoch}: train_loss={train_loss:.6f}{val_str}")

        sched_metric = get_metric_value(
            scheduler_info.get("metric") if scheduler_info else None,
            train_loss=train_loss,
            val_metrics=val_metrics,
        )
        step_lr_scheduler(scheduler, scheduler_info, sched_metric)

        stop_metric = get_metric_value(
            early_stop["metric"] if early_stop else None,
            train_loss=train_loss,
            val_metrics=val_metrics,
        )
        if update_early_stopping(early_stop, stop_metric):
            print("early stopping triggered")
            break

    log_path = os.path.join(cfg["train"]["log_dir"], f"training_log_{run_id}.json")
    save_json(log_path, log_rows)
    if train_cfg.get("track_csv", True):
        csv_path = os.path.join(cfg["train"]["log_dir"], f"training_log_{run_id}.csv")
        write_csv_log(csv_path, log_rows)


def train_ppo(cfg, args):
    seed = int(cfg.get("rl", {}).get("seed", 42))
    set_seed(seed)

    splits, state_cols = prepare_splits(cfg)
    if len(splits) != 1:
        raise ValueError("PPO mode does not support multi-dataset pooling")
    train_df = splits[0]["train"]
    val_df = splits[0]["val"]

    device = select_device(cfg["train"]["device"])
    action_mode = str(cfg.get("rl", {}).get("action_mode", "discrete")).lower()
    act_dim = int(cfg.get("rl", {}).get("action_dim", 1)) if action_mode == "continuous" else 3
    if action_mode == "continuous" and act_dim != 1:
        raise ValueError("continuous action_mode currently supports action_dim=1")

    model = DecisionTransformer(
        state_dim=len(state_cols),
        act_dim=act_dim,
        seq_len=cfg["dataset"]["seq_len"],
        d_model=cfg["train"]["d_model"],
        n_layers=cfg["train"]["n_layers"],
        n_heads=cfg["train"]["n_heads"],
        dropout=cfg["train"]["dropout"],
        action_mode=action_mode,
        use_value_head=True,
    ).to(device)
    model.condition_mode = "reward"

    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"loaded init checkpoint weights from {args.init_ckpt}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    episode_len = cfg.get("rl", {}).get("episode_len", cfg.get("dataset", {}).get("episode_len", None))
    if episode_len is not None:
        episode_len = int(episode_len)
        if episode_len >= len(train_df):
            episode_len = len(train_df) - 1
        if episode_len <= 0:
            raise ValueError("episode_len too small for PPO training")

    train_env = MarketEnv(
        states=train_df[state_cols].to_numpy(dtype=np.float32),
        close=train_df["close"].to_numpy(dtype=np.float32),
        timestamps=train_df["timestamp"].to_numpy(dtype=np.int64),
        fee=cfg["rewards"]["fee"],
        slip=cfg["rewards"]["slip"],
        episode_len=episode_len,
        reward_scale=cfg["rl"].get("reward_scale", 1.0),
        turnover_penalty=cfg["rl"].get("turnover_penalty", 0.0),
        position_penalty=cfg["rl"].get("position_penalty", 0.0),
        drawdown_penalty=cfg["rl"].get("drawdown_penalty", 0.0),
        action_mode=action_mode,
        rng=np.random.RandomState(seed),
        open_prices=train_df["open"].to_numpy(dtype=np.float32)
        if "open" in train_df.columns
        else None,
        high_prices=train_df["high"].to_numpy(dtype=np.float32)
        if "high" in train_df.columns
        else None,
        low_prices=train_df["low"].to_numpy(dtype=np.float32)
        if "low" in train_df.columns
        else None,
        price_mode=cfg.get("rewards", {}).get("price_mode", "close"),
        range_penalty=cfg.get("rewards", {}).get("range_penalty", 0.0),
    )

    ensure_dir(cfg["train"]["log_dir"])
    ensure_dir(cfg["train"]["checkpoint_dir"])

    log_rows = []
    best_val = -float("inf")
    run_id = make_run_id(cfg)
    run_meta = {
        "run_id": run_id,
        "mode": "ppo",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    train_cfg = cfg.get("train", {})
    if train_cfg.get("save_config", True):
        config_path = os.path.join(cfg["train"]["log_dir"], f"run_config_{run_id}.json")
        save_run_config(config_path, cfg, args, run_meta)

    rollout_steps = int(cfg["rl"].get("rollout_steps", 2048))
    gamma = float(cfg["rl"].get("gamma", 0.99))
    gae_lambda = float(cfg["rl"].get("gae_lambda", 0.95))
    eval_every = int(cfg["rl"].get("eval_every", 1))
    has_val = val_df is not None and not val_df.empty

    epochs = int(cfg["train"]["epochs"])
    rollout_progress = bool(train_cfg.get("rollout_progress", False))
    update_progress = bool(train_cfg.get("update_progress", False))
    eval_progress = bool(train_cfg.get("eval_progress", False))
    epoch_iter = range(1, epochs + 1)
    progress_position = 0

    ckpt_config = {
        "state_dim": len(state_cols),
        "act_dim": act_dim,
        "seq_len": cfg["dataset"]["seq_len"],
        "d_model": cfg["train"]["d_model"],
        "n_layers": cfg["train"]["n_layers"],
        "n_heads": cfg["train"]["n_heads"],
        "dropout": cfg["train"]["dropout"],
        "action_mode": action_mode,
        "use_value_head": True,
        "condition_mode": "reward",
    }

    default_metric = "val_total_return" if has_val else "mean_episode_return"
    scheduler, scheduler_info = init_lr_scheduler(optimizer, cfg, default_metric=default_metric)
    early_stop = init_early_stopping(cfg, default_metric=default_metric)

    for epoch in epoch_iter:
        buffer = collect_rollout(
            train_env,
            model,
            device,
            cfg["dataset"]["seq_len"],
            action_mode,
            act_dim,
            rollout_steps,
            gamma,
            gae_lambda,
            progress=rollout_progress,
            progress_desc=f"train {epoch}",
            progress_position=progress_position,
        )

        policy_loss, value_loss, entropy, approx_kl, clip_frac = ppo_update(
            model,
            optimizer,
            buffer,
            device,
            cfg,
            action_mode,
            progress=update_progress,
            progress_desc=f"update {epoch}",
            progress_position=progress_position,
        )

        mean_ep_return = float(np.mean(buffer["ep_returns"])) if buffer["ep_returns"] else 0.0
        mean_ep_len = float(np.mean(buffer["ep_lengths"])) if buffer["ep_lengths"] else 0.0

        val_metrics = None
        if has_val and eval_every > 0 and epoch % eval_every == 0:
            val_metrics = evaluate_policy_ppo(
                cfg,
                model,
                val_df,
                state_cols,
                device,
                action_mode,
                act_dim,
                progress=eval_progress,
                progress_desc=f"eval {epoch}",
                progress_position=progress_position,
            )
            if val_metrics is not None and val_metrics["total_return"] > best_val:
                best_val = val_metrics["total_return"]
                ckpt = {"model_state": model.state_dict(), "model_config": ckpt_config}
                best_path = os.path.join(
                    cfg["train"]["checkpoint_dir"], f"dt_best_{run_id}.pt"
                )
                torch.save(ckpt, best_path)
        elif not has_val and mean_ep_return > best_val:
            best_val = mean_ep_return
            ckpt = {"model_state": model.state_dict(), "model_config": ckpt_config}
            best_path = os.path.join(
                cfg["train"]["checkpoint_dir"], f"dt_best_{run_id}.pt"
            )
            torch.save(ckpt, best_path)

        log_row = {
            "epoch": epoch,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "mean_episode_return": mean_ep_return,
            "mean_episode_len": mean_ep_len,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if val_metrics is not None:
            for key in (
                "total_return",
                "annual_return",
                "sharpe",
                "sortino",
                "max_drawdown",
                "calmar",
                "profit_factor",
                "win_rate",
                "exposure",
                "trade_count",
                "turnover",
            ):
                if key in val_metrics:
                    log_row[f"val_{key}"] = val_metrics[key]

        log_rows.append(log_row)

        val_str = ""
        if val_metrics is not None:
            val_str = f" val_total_return={val_metrics['total_return']:.4f}"
        print(
            f"epoch {epoch}: policy_loss={policy_loss:.4f} value_loss={value_loss:.4f} "
            f"entropy={entropy:.4f} approx_kl={approx_kl:.4f} clip_frac={clip_frac:.3f} "
            f"mean_ep_return={mean_ep_return:.4f}{val_str}"
        )

        extra_metrics = {
            "mean_episode_return": mean_ep_return,
            "mean_episode_len": mean_ep_len,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
        }
        sched_metric = get_metric_value(
            scheduler_info.get("metric") if scheduler_info else None,
            val_metrics=val_metrics,
            extra_metrics=extra_metrics,
        )
        step_lr_scheduler(scheduler, scheduler_info, sched_metric)

        stop_metric = get_metric_value(
            early_stop["metric"] if early_stop else None,
            val_metrics=val_metrics,
            extra_metrics=extra_metrics,
        )
        if update_early_stopping(early_stop, stop_metric):
            print("early stopping triggered")
            break

    log_path = os.path.join(cfg["train"]["log_dir"], f"training_log_{run_id}.json")
    save_json(log_path, log_rows)
    if train_cfg.get("track_csv", True):
        csv_path = os.path.join(cfg["train"]["log_dir"], f"training_log_{run_id}.csv")
        write_csv_log(csv_path, log_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--init_ckpt", default=None)
    parser.add_argument("--mode", choices=["offline", "ppo"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get("walk_forward", {}).get("enabled", False):
        from walk_forward import run_walk_forward

        run_walk_forward(cfg)
        return
    mode = args.mode or cfg.get("train", {}).get("mode", "offline")
    mode = str(mode).lower()
    if mode not in ("offline", "ppo"):
        raise ValueError(f"unsupported mode {mode}")

    if mode == "ppo":
        train_ppo(cfg, args)
    else:
        train_offline(cfg, args)


if __name__ == "__main__":
    main()
