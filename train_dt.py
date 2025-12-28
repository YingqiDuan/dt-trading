import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
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
from features import build_features
from utils import ensure_dir, load_config, parse_date, save_json, set_seed


def select_device(pref):
    if pref == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def action_to_index(actions):
    return actions + 1


def index_to_action(indices):
    return indices - 1


def atanh(x):
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def build_behavior_actions(df, cfg, action_mode, rng):
    dataset_cfg = cfg.get("dataset", {})
    policy = str(dataset_cfg.get("behavior_policy", "ema_trend")).lower()

    if policy == "ema_trend":
        ema_windows = [int(w) for w in cfg.get("features", {}).get("ema_windows", [])]
        ema_window = max(ema_windows) if ema_windows else None
        ema_col = f"ema_{ema_window}" if ema_window else None
        if not ema_col or ema_col not in df.columns:
            print("behavior_policy=ema_trend but ema feature is missing; using flat actions")
            actions = np.zeros(len(df), dtype=np.float32)
        else:
            deadzone = float(cfg.get("backtest", {}).get("ema_deadzone", 0.001))
            close = df["close"].to_numpy(dtype=np.float32)
            ema = df[ema_col].to_numpy(dtype=np.float32)
            signal = close / ema - 1.0
            actions = np.where(
                signal > deadzone,
                1.0,
                np.where(signal < -deadzone, -1.0, 0.0),
            )
    elif policy == "buy_hold":
        actions = np.ones(len(df), dtype=np.float32)
    elif policy == "flat":
        actions = np.zeros(len(df), dtype=np.float32)
    elif policy == "random":
        if action_mode == "continuous":
            actions = rng.uniform(-1.0, 1.0, size=len(df)).astype(np.float32)
        else:
            stay_prob = float(cfg.get("backtest", {}).get("random_stay_prob", 0.5))
            actions = np.zeros(len(df), dtype=np.int64)
            for idx in range(1, len(df)):
                if rng.rand() < stay_prob:
                    actions[idx] = actions[idx - 1]
                else:
                    actions[idx] = int(rng.choice([-1, 0, 1]))
    else:
        raise ValueError(f"unsupported behavior_policy {policy}")

    if action_mode == "continuous":
        if actions.dtype != np.float32:
            actions = actions.astype(np.float32)
        actions = np.clip(actions, -1.0, 1.0)
    else:
        actions = actions.astype(np.int64)

    return actions


def build_trajectories(df, state_cols, cfg, action_mode, act_dim, rng):
    seq_len = int(cfg["dataset"]["seq_len"])
    dataset_cfg = cfg.get("dataset", {})
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
    price_mode = cfg["rewards"].get("price_mode", "close")
    range_penalty = float(cfg["rewards"].get("range_penalty", 0.0))

    if action_mode == "continuous" and act_dim != 1:
        raise ValueError("continuous action_mode currently supports action_dim=1")

    actions = build_behavior_actions(df, cfg, action_mode, rng)
    states = df[state_cols].to_numpy(dtype=np.float32)
    close = df["close"].to_numpy(dtype=np.float32)
    open_prices = df["open"].to_numpy(dtype=np.float32) if "open" in df.columns else None
    high_prices = df["high"].to_numpy(dtype=np.float32) if "high" in df.columns else None
    low_prices = df["low"].to_numpy(dtype=np.float32) if "low" in df.columns else None
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

    dataset_cfg = cfg.get("dataset", {})
    target_return = float(dataset_cfg.get("target_return", 0.0))
    rtg_scale = float(dataset_cfg.get("rtg_scale", 1.0))
    if rtg_scale <= 0:
        rtg_scale = 1.0
    gamma = float(dataset_cfg.get("rtg_gamma", cfg.get("rl", {}).get("gamma", 1.0)))

    states = df[state_cols].to_numpy(dtype=np.float32)
    close = df["close"].to_numpy(dtype=np.float32)
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
            cfg["backtest"]["fee"],
            cfg["backtest"]["slip"],
            cfg["backtest"]["initial_cash"],
            open_prices=open_prices,
            high_prices=high_prices,
            low_prices=low_prices,
            price_mode=price_mode,
        )
        annual_factor = 24 * 365
        return compute_metrics(equity_curve, step_returns, trade_count, turnover, annual_factor)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--init_ckpt", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("rl", {}).get("seed", 42))
    set_seed(seed)

    raw_df = load_or_fetch(cfg)
    feat_df, state_cols = build_features(raw_df, cfg["features"])
    feat_df = feat_df.dropna(subset=state_cols).reset_index(drop=True)

    train_end = parse_date(cfg["data"]["train_end"])
    val_end = parse_date(cfg["data"].get("val_end", cfg["data"]["train_end"]))
    test_end = parse_date(cfg["data"]["test_end"])
    train_df, val_df, _ = split_by_time(feat_df, train_end, val_end, test_end)

    if len(train_df) < 2:
        raise ValueError("train split too small")

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
    trajectories = build_trajectories(train_df, state_cols, cfg, action_mode, act_dim, rng)
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
    run_id = time.strftime("%Y%m%d_%H%M%S")

    eval_every = int(cfg.get("rl", {}).get("eval_every", 1))
    has_val = val_df is not None and not val_df.empty

    epochs = int(cfg["train"]["epochs"])
    train_cfg = cfg.get("train", {})
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
            val_metrics = evaluate_policy(
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
        elif not has_val and train_loss < best_loss:
            best_loss = train_loss
            ckpt = {"model_state": model.state_dict(), "model_config": ckpt_config}
            best_path = os.path.join(
                cfg["train"]["checkpoint_dir"], f"dt_best_{run_id}.pt"
            )
            torch.save(ckpt, best_path)

        log_row = {"epoch": epoch, "train_loss": train_loss}
        if val_metrics is not None:
            log_row.update(
                {
                    "val_total_return": val_metrics["total_return"],
                    "val_sharpe": val_metrics["sharpe"],
                    "val_max_drawdown": val_metrics["max_drawdown"],
                }
            )

        log_rows.append(log_row)

        val_str = ""
        if val_metrics is not None:
            val_str = f" val_total_return={val_metrics['total_return']:.4f}"
        print(f"epoch {epoch}: train_loss={train_loss:.6f}{val_str}")

    log_path = os.path.join(cfg["train"]["log_dir"], f"training_log_{run_id}.json")
    save_json(log_path, log_rows)


if __name__ == "__main__":
    main()
