import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from backtest import action_distribution, compute_metrics, run_model_backtest
from dataset_builder import load_or_fetch
from dt_model import DecisionTransformer
from features import build_features, load_feature_cache
from train_dt import (
    TrajectoryDataset,
    build_trajectories,
    evaluate_policy,
    select_device,
    train_epoch,
)
from utils import ensure_dir, load_config, save_json, set_seed


def train_for_fold_offline(cfg, train_df, val_df, state_cols, fold_dir, fold_seed):
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    rng = np.random.RandomState(fold_seed)
    trajectories = build_trajectories(train_df, state_cols, cfg, action_mode, act_dim, rng)
    dataset = TrajectoryDataset(trajectories, cfg["dataset"]["seq_len"], action_mode, act_dim)
    if len(dataset) == 0:
        raise ValueError("no training windows available; check seq_len/episode_len")

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        drop_last=False,
    )

    eval_every = int(cfg.get("rl", {}).get("eval_every", 1))
    has_val = eval_every > 0 and not val_df.empty

    log_rows = []
    best_val = -float("inf")
    best_loss = float("inf")
    best_path = os.path.join(fold_dir, "dt_best.pt")

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
        improved = False
        if has_val and epoch % eval_every == 0:
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
            if val_metrics and val_metrics["total_return"] > best_val:
                best_val = val_metrics["total_return"]
                improved = True
        elif not has_val and train_loss < best_loss:
            best_loss = train_loss
            improved = True

        if improved:
            ckpt = {"model_state": model.state_dict(), "model_config": ckpt_config}
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

    log_path = os.path.join(fold_dir, "training_log.json")
    save_json(log_path, log_rows)

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    model.condition_mode = "rtg"
    return model, best_path, log_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    wf_cfg = cfg.get("walk_forward", {})
    if not wf_cfg.get("enabled", False):
        raise SystemExit("walk_forward.enabled is false; enable it in config.yaml")

    train_bars = int(wf_cfg.get("train_bars", 0))
    val_bars = int(wf_cfg.get("val_bars", 0))
    test_bars = int(wf_cfg.get("test_bars", 0))
    step_bars = int(wf_cfg.get("step_bars", 0))
    max_folds = int(wf_cfg.get("max_folds", 0))

    if train_bars <= 0 or test_bars <= 0 or step_bars <= 0:
        raise ValueError("train_bars/test_bars/step_bars must be > 0")
    if val_bars <= 0:
        raise ValueError("val_bars must be > 0 for walk-forward training")

    output_dir = wf_cfg.get("output_dir", "outputs/walk_forward")
    ensure_dir(output_dir)

    seed = int(cfg.get("rl", {}).get("seed", 42))
    set_seed(seed)

    feat_df, state_cols = load_feature_cache(cfg)
    if feat_df is None:
        raw_df = load_or_fetch(cfg)
        feat_df, state_cols = build_features(raw_df, cfg["features"])
    feat_df = feat_df.dropna(subset=state_cols).reset_index(drop=True)

    total_len = len(feat_df)
    start_idx = 0
    fold_idx = 0
    summary = {"folds": [], "mode": "dt"}

    while True:
        train_end = start_idx + train_bars
        val_end = train_end + val_bars
        test_end = val_end + test_bars
        if test_end > total_len:
            break

        fold_dir = os.path.join(output_dir, f"fold_{fold_idx:02d}")
        ensure_dir(fold_dir)

        train_df = feat_df.iloc[start_idx:train_end].copy()
        val_df = feat_df.iloc[train_end:val_end].copy()
        test_df = feat_df.iloc[val_end:test_end].copy()
        if len(train_df) < cfg["dataset"]["seq_len"] or len(val_df) < cfg["dataset"]["seq_len"]:
            raise ValueError("train/val window shorter than seq_len")

        print(f"fold {fold_idx}: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        fold_seed = seed + fold_idx
        model, ckpt_path, _ = train_for_fold_offline(
            cfg, train_df, val_df, state_cols, fold_dir, fold_seed
        )

        device = next(model.parameters()).device
        curve, equity, step_returns, trade_count, turnover = run_model_backtest(
            cfg, model, test_df, state_cols, device
        )

        annual_factor = 24 * 365
        dt_metrics = compute_metrics(
            equity, step_returns, trade_count, turnover, annual_factor
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

        metrics_path = os.path.join(fold_dir, "metrics.json")
        curve_path = os.path.join(fold_dir, "equity_curve.csv")
        save_json(metrics_path, {"decision_transformer": dt_metrics})
        curve.to_csv(curve_path, index=False)

        fold_summary = {
            "fold": fold_idx,
            "train_start": str(train_df["datetime"].iloc[0]),
            "train_end": str(train_df["datetime"].iloc[-1]),
            "val_start": str(val_df["datetime"].iloc[0]),
            "val_end": str(val_df["datetime"].iloc[-1]),
            "test_start": str(test_df["datetime"].iloc[0]),
            "test_end": str(test_df["datetime"].iloc[-1]),
            "checkpoint": ckpt_path,
            "metrics": dt_metrics,
        }
        summary["folds"].append(fold_summary)

        fold_idx += 1
        if max_folds and fold_idx >= max_folds:
            break
        start_idx += step_bars

    summary_path = os.path.join(output_dir, "summary.json")
    save_json(summary_path, summary)
    print(f"walk-forward summary saved to {summary_path}")


if __name__ == "__main__":
    main()
