import argparse
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from backtest import action_distribution, compute_metrics, run_model_backtest
from dataset_builder import (
    build_dataset,
    concat_datasets,
    load_or_fetch,
    policy_ema,
    policy_random,
    save_dataset,
)
from dt_model import DecisionTransformer
from features import build_features
from train_dt import (
    TrajectoryDataset,
    compute_class_weights,
    eval_epoch,
    select_device,
    train_epoch,
)
from utils import ensure_dir, load_config, rtg_quantile_from_dataset, save_json, set_seed


def resolve_fold_inference_rtg(cfg, train_path):
    value = cfg["backtest"].get("inference_rtg", "auto")
    if isinstance(value, str) and value.lower() == "auto":
        quantile = float(cfg["backtest"].get("inference_rtg_quantile", 0.9))
        scope = cfg["backtest"].get("rtg_quantile_scope", "window_start")
        return rtg_quantile_from_dataset(
            train_path, quantile, cfg["dataset"]["seq_len"], scope
        )
    if value is None:
        quantile = float(cfg["backtest"].get("inference_rtg_quantile", 0.9))
        scope = cfg["backtest"].get("rtg_quantile_scope", "window_start")
        return rtg_quantile_from_dataset(
            train_path, quantile, cfg["dataset"]["seq_len"], scope
        )
    return float(value)


def build_mixed_dataset(split_df, state_cols, cfg, rng):
    deadzone = cfg["behavior_policies"]["ma_deadzone"]
    ema_window = int(cfg["behavior_policies"].get("ema_window", 18))
    stay_prob = cfg["behavior_policies"]["random_stay_prob"]
    mix_ratio = float(cfg["behavior_policies"].get("mix_ratio", 0.5))
    mix_ratio = max(0.0, min(1.0, mix_ratio))
    mix_scale = 10

    actions_a = policy_ema(split_df, deadzone, ema_window=ema_window)
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

    return concat_datasets(datasets_list)


def train_for_fold(cfg, train_path, val_path, fold_dir):
    train_ds = TrajectoryDataset(train_path, cfg["dataset"]["seq_len"])
    val_ds = TrajectoryDataset(val_path, cfg["dataset"]["seq_len"])

    device = select_device(cfg["train"]["device"])
    state_dim = train_ds.states.shape[1]
    act_dim = 3

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        seq_len=cfg["dataset"]["seq_len"],
        d_model=cfg["train"]["d_model"],
        n_layers=cfg["train"]["n_layers"],
        n_heads=cfg["train"]["n_heads"],
        dropout=cfg["train"]["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    weights, counts = compute_class_weights(train_ds.actions)
    use_class_weights = bool(cfg["train"].get("use_class_weights", True))
    if use_class_weights:
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        train_criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        train_criterion = nn.CrossEntropyLoss()
    val_criterion = nn.CrossEntropyLoss()

    if use_class_weights:
        print(
            "class_counts short/flat/long="
            f"{int(counts[0])}/{int(counts[1])}/{int(counts[2])}, "
            f"class_weights={weights.round(3).tolist()}"
        )
    else:
        print(
            "class_counts short/flat/long="
            f"{int(counts[0])}/{int(counts[1])}/{int(counts[2])}, "
            "class_weights=disabled"
        )

    use_sampling = bool(cfg["train"].get("use_sampling", True))
    sampling = str(cfg["train"].get("sampling", "uniform")).lower()
    sampling_mode = None
    if sampling in ("rtg", "rtg_start"):
        sampling_mode = "rtg"
    elif sampling in ("episode_return", "episode"):
        sampling_mode = "episode_return"

    sampler = None
    if use_sampling and sampling_mode:
        power = float(cfg["train"].get("sampling_power", 1.0))
        epsilon = float(cfg["train"].get("sampling_epsilon", 1e-3))
        samp_weights = train_ds.sampling_weights(
            sampling_mode, power=power, epsilon=epsilon
        )
        sampler = WeightedRandomSampler(
            samp_weights, num_samples=len(samp_weights), replacement=True
        )
        print(
            "sampling=weighted "
            f"mode={sampling_mode} power={power} epsilon={epsilon} "
            f"weight_stats(min/mean/max)={samp_weights.min():.3g}/{samp_weights.mean():.3g}/{samp_weights.max():.3g}"
        )
    elif use_sampling:
        print("sampling=uniform")
    else:
        print("sampling=disabled")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    log_rows = []
    best_val = float("inf")
    best_path = os.path.join(fold_dir, "dt_best.pt")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss, train_acc, train_f1, train_recalls = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train_criterion,
            cfg["train"]["grad_clip"],
        )
        val_loss, val_acc, val_f1, val_recalls = eval_epoch(
            model, val_loader, device, val_criterion
        )

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_macro_f1": train_f1,
                "train_recall_short": train_recalls[0],
                "train_recall_flat": train_recalls[1],
                "train_recall_long": train_recalls[2],
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "val_recall_short": val_recalls[0],
                "val_recall_flat": val_recalls[1],
                "val_recall_long": val_recalls[2],
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "model_config": {
                    "state_dim": state_dim,
                    "act_dim": act_dim,
                    "seq_len": cfg["dataset"]["seq_len"],
                    "d_model": cfg["train"]["d_model"],
                    "n_layers": cfg["train"]["n_layers"],
                    "n_heads": cfg["train"]["n_heads"],
                    "dropout": cfg["train"]["dropout"],
                },
            }
            torch.save(ckpt, best_path)

        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
            f"train_f1={train_f1:.3f} val_f1={val_f1:.3f}"
        )

    log_path = os.path.join(fold_dir, "training_log.json")
    save_json(log_path, log_rows)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
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

    set_seed(cfg["behavior_policies"].get("seed", 42))

    ema_window = int(cfg["behavior_policies"].get("ema_window", 18))
    ema_col = f"ema_{ema_window}"
    raw_df = load_or_fetch(cfg)
    feat_df, state_cols = build_features(raw_df, cfg["features"])
    feat_df = feat_df.dropna(subset=state_cols + [ema_col]).reset_index(drop=True)

    total_len = len(feat_df)
    start_idx = 0
    fold_idx = 0
    summary = {"folds": []}

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

        rng = np.random.RandomState(cfg["behavior_policies"].get("seed", 42) + fold_idx)
        train_data = build_mixed_dataset(train_df, state_cols, cfg, rng)
        val_data = build_mixed_dataset(val_df, state_cols, cfg, rng)

        train_path = os.path.join(fold_dir, "train_dataset.npz")
        val_path = os.path.join(fold_dir, "val_dataset.npz")
        save_dataset(train_path, train_data, state_cols)
        save_dataset(val_path, val_data, state_cols)

        print(f"fold {fold_idx}: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        model, ckpt_path, _ = train_for_fold(cfg, train_path, val_path, fold_dir)

        inference_rtg = resolve_fold_inference_rtg(cfg, train_path)
        device = next(model.parameters()).device
        curve, equity, step_returns, trade_count, turnover, _ = run_model_backtest(
            cfg, model, test_df, state_cols, device, inference_rtg
        )

        annual_factor = 24 * 365
        dt_metrics = compute_metrics(
            equity, step_returns, trade_count, turnover, annual_factor
        )
        dt_metrics["action_distribution"] = action_distribution(
            curve["action"].to_numpy(dtype=np.int64)
        )
        dt_metrics["inference_rtg"] = float(inference_rtg)

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
