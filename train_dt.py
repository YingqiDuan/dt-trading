import argparse
import os
import time

import numpy as np
import torch
from torch.distributions import Categorical, Normal

from dataset_builder import load_or_fetch, split_by_time
from dt_model import DecisionTransformer
from features import build_features
from market_env import MarketEnv
from utils import ensure_dir, load_config, parse_date, save_json, set_seed
from backtest import compute_metrics, simulate


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
            actions_in[1:] = np.asarray(actions_hist[start : start + actual_len - 1], dtype=np.int64)
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


def collect_rollout(env, model, device, seq_len, action_mode, act_dim, rollout_steps, gamma, gae_lambda):
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

    for _ in range(rollout_steps):
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


def ppo_update(model, optimizer, buffer, device, cfg, action_mode):
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

    if total_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    return (
        total_policy_loss / total_batches,
        total_value_loss / total_batches,
        total_entropy / total_batches,
        total_kl / total_batches,
        total_clip / total_batches,
    )


def evaluate_policy(cfg, model, df, state_cols, device, action_mode, act_dim):
    if df is None or df.empty:
        return None

    seq_len = cfg["dataset"]["seq_len"]
    fee = cfg["rewards"]["fee"]
    slip = cfg["rewards"]["slip"]

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

    def run_once(rng):
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
                ret = close[idx + 1] / close[idx] - 1.0
                trade_cost = (fee + slip) * abs(action - prev_action)
                reward = action * ret - trade_cost
                rewards_hist[idx + 1] = reward
            prev_action = action

        equity, step_returns, trade_count, turnover = simulate(
            actions,
            close,
            cfg["backtest"]["fee"],
            cfg["backtest"]["slip"],
            cfg["backtest"]["initial_cash"],
        )
        annual_factor = 24 * 365
        return compute_metrics(equity, step_returns, trade_count, turnover, annual_factor)

    metrics_list = []
    for sample_idx in range(eval_samples):
        if eval_seed is None:
            rng = np.random
        else:
            rng = np.random.RandomState(eval_seed + sample_idx)
        metrics_list.append(run_once(rng))

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
    action_mode = str(cfg["rl"].get("action_mode", "discrete")).lower()
    act_dim = int(cfg["rl"].get("action_dim", 1)) if action_mode == "continuous" else 3
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

    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"loaded init checkpoint weights from {args.init_ckpt}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    train_env = MarketEnv(
        states=train_df[state_cols].to_numpy(dtype=np.float32),
        close=train_df["close"].to_numpy(dtype=np.float32),
        timestamps=train_df["timestamp"].to_numpy(dtype=np.int64),
        fee=cfg["rewards"]["fee"],
        slip=cfg["rewards"]["slip"],
        episode_len=cfg["rl"].get("episode_len", None),
        reward_scale=cfg["rl"].get("reward_scale", 1.0),
        turnover_penalty=cfg["rl"].get("turnover_penalty", 0.0),
        position_penalty=cfg["rl"].get("position_penalty", 0.0),
        drawdown_penalty=cfg["rl"].get("drawdown_penalty", 0.0),
        action_mode=action_mode,
        rng=np.random.RandomState(seed),
    )

    ensure_dir(cfg["train"]["log_dir"])
    ensure_dir(cfg["train"]["checkpoint_dir"])

    log_rows = []
    best_val = -float("inf")
    run_id = time.strftime("%Y%m%d_%H%M%S")

    rollout_steps = int(cfg["rl"].get("rollout_steps", 2048))
    gamma = float(cfg["rl"].get("gamma", 0.99))
    gae_lambda = float(cfg["rl"].get("gae_lambda", 0.95))
    eval_every = int(cfg["rl"].get("eval_every", 1))
    has_val = val_df is not None and not val_df.empty

    for epoch in range(1, cfg["train"]["epochs"] + 1):
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
        )

        policy_loss, value_loss, entropy, approx_kl, clip_frac = ppo_update(
            model, optimizer, buffer, device, cfg, action_mode
        )

        mean_ep_return = float(np.mean(buffer["ep_returns"])) if buffer["ep_returns"] else 0.0
        mean_ep_len = float(np.mean(buffer["ep_lengths"])) if buffer["ep_lengths"] else 0.0

        val_metrics = None
        if has_val and eval_every > 0 and epoch % eval_every == 0:
            val_metrics = evaluate_policy(cfg, model, val_df, state_cols, device, action_mode, act_dim)
            if val_metrics is not None and val_metrics["total_return"] > best_val:
                best_val = val_metrics["total_return"]
                ckpt = {
                    "model_state": model.state_dict(),
                    "model_config": {
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
                    },
                }
                best_path = os.path.join(
                    cfg["train"]["checkpoint_dir"], f"dt_best_{run_id}.pt"
                )
                torch.save(ckpt, best_path)
        elif not has_val and mean_ep_return > best_val:
            best_val = mean_ep_return
            ckpt = {
                "model_state": model.state_dict(),
                "model_config": {
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
                },
            }
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
        }
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
        print(
            f"epoch {epoch}: policy_loss={policy_loss:.4f} value_loss={value_loss:.4f} "
            f"entropy={entropy:.4f} approx_kl={approx_kl:.4f} clip_frac={clip_frac:.3f} "
            f"mean_ep_return={mean_ep_return:.4f}{val_str}"
        )

    log_path = os.path.join(cfg["train"]["log_dir"], f"training_log_{run_id}.json")
    save_json(log_path, log_rows)


if __name__ == "__main__":
    main()
