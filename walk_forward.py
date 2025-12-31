import argparse, csv, json, os, numpy as np, torch
from torch.utils.data import DataLoader
try: from tqdm import tqdm
except ImportError: tqdm = None

from backtest import action_distribution, compute_metrics, run_model_backtest
from dataset_builder import load_or_fetch
from dt_model import DecisionTransformer
from features import build_features, load_feature_cache
from train_dt import TrajectoryDataset, build_trajectories, evaluate_policy, select_device, train_epoch
from utils import annualization_factor, ensure_dir, load_config, save_json, set_seed

def summarize_metrics(folds):
    data = {}
    for f in folds:
        for k, v in f.get("metrics", {}).items():
            if isinstance(v, (int, float, np.number)) and np.isfinite(v):
                data.setdefault(k, []).append(float(v))
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "min": float(np.min(v)), "max": float(np.max(v))} for k, v in data.items()}

def serialize_metric_value(value):
    return float(value) if isinstance(value, (np.number, float, int)) else (json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value)

def write_summary_csv(path, folds):
    if not folds: return
    base = ["fold", "train_start", "train_end", "val_start", "val_end", "test_start", "test_end", "checkpoint"]
    m_keys = sorted({k for f in folds for k in f.get("metrics", {})})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base + m_keys)
        writer.writeheader()
        for fold in folds:
            row = {k: fold.get(k, "") for k in base}
            row.update({k: serialize_metric_value(fold.get("metrics", {}).get(k, "")) for k in m_keys})
            writer.writerow(row)

def write_summary_md(path, summary, fold_count):
    lines = ["# Walk-Forward Summary", "", f"- folds: {fold_count}", ""]
    if not summary: lines.append("No metrics available.")
    else:
        lines.extend(["| metric | mean | std | min | max |", "| --- | --- | --- | --- | --- |"])
        for k in sorted(summary.keys()):
            s = summary[k]
            lines.append(f"| {k} | {s['mean']:.6f} | {s['std']:.6f} | {s['min']:.6f} | {s['max']:.6f} |")
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(lines))

def train_for_fold_offline(cfg, train_df, val_df, state_cols, fold_dir, fold_seed):
    device, train_cfg = select_device(cfg["train"]["device"]), cfg["train"]
    act_mode = str(cfg.get("rl", {}).get("action_mode", "discrete")).lower()
    act_dim = int(cfg.get("rl", {}).get("action_dim", 1)) if act_mode == "continuous" else 3
    if act_mode == "continuous" and act_dim != 1: raise ValueError("continuous action_mode supports action_dim=1")

    model = DecisionTransformer(len(state_cols), act_dim, cfg["dataset"]["seq_len"], cfg["train"]["d_model"],
        cfg["train"]["n_layers"], cfg["train"]["n_heads"], cfg["train"]["dropout"], action_mode=act_mode, use_value_head=False).to(device)
    model.condition_mode = "rtg"
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    ds = TrajectoryDataset(build_trajectories(train_df, state_cols, cfg, act_mode, act_dim, np.random.RandomState(fold_seed)), 
                           cfg["dataset"]["seq_len"], act_mode, act_dim)
    if len(ds) == 0: raise ValueError("no training windows available")
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True)

    best_val, best_loss, best_path = -float("inf"), float("inf"), os.path.join(fold_dir, "dt_best.pt")
    eval_every, has_val = int(cfg.get("rl", {}).get("eval_every", 1)), not val_df.empty
    log_rows, ds_cfg = [], cfg.get("dataset", {})
    
    ckpt_cfg = {"state_dim": len(state_cols), "act_dim": act_dim, "seq_len": cfg["dataset"]["seq_len"],
        "d_model": cfg["train"]["d_model"], "n_layers": cfg["train"]["n_layers"], "n_heads": cfg["train"]["n_heads"],
        "dropout": cfg["train"]["dropout"], "action_mode": act_mode, "use_value_head": False, "condition_mode": "rtg",
        "rtg_gamma": float(ds_cfg.get("rtg_gamma", cfg.get("rl", {}).get("gamma", 1.0))),
        "rtg_scale": float(ds_cfg.get("rtg_scale", 1.0)), "target_return": float(ds_cfg.get("target_return", 0.0))}

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        loss = train_epoch(model, loader, optimizer, device, act_mode, act_dim, grad_clip=train_cfg.get("grad_clip"),
            progress=bool(train_cfg.get("train_progress", train_cfg.get("update_progress", False))), progress_desc=f"train {epoch}")
        
        val_met, improved = None, False
        if has_val and eval_every > 0 and epoch % eval_every == 0:
            val_met = evaluate_policy(cfg, model, val_df, state_cols, device, act_mode, act_dim, progress=bool(train_cfg.get("eval_progress")), progress_desc=f"eval {epoch}")
            if val_met and val_met["total_return"] > best_val: best_val, improved = val_met["total_return"], True
        elif not has_val and loss < best_loss: best_loss, improved = loss, True

        if improved: torch.save({"model_state": model.state_dict(), "model_config": ckpt_cfg}, best_path)
        log_rows.append({"epoch": epoch, "train_loss": loss, **({"val_total_return": val_met["total_return"], "val_sharpe": val_met["sharpe"], "val_max_drawdown": val_met["max_drawdown"]} if val_met else {})})
        print(f"epoch {epoch}: train_loss={loss:.6f}{f' val_total_return={val_met['total_return']:.4f}' if val_met else ''}")

    save_json(os.path.join(fold_dir, "training_log.json"), log_rows)
    if os.path.exists(best_path): model.load_state_dict(torch.load(best_path, map_location=device)["model_state"], strict=False)
    model.eval(); model.condition_mode = "rtg"
    return model, best_path

def run_walk_forward(cfg):
    wf = cfg.get("walk_forward", {})
    if not wf.get("enabled", False): raise SystemExit("walk_forward.enabled is false")
    
    train_bars, val_bars, test_bars, step_bars = int(wf.get("train_bars", 0)), int(wf.get("val_bars", 0)), int(wf.get("test_bars", 0)), int(wf.get("step_bars", 0))
    if any(x <= 0 for x in [train_bars, val_bars, test_bars, step_bars]): raise ValueError("invalid bars config")
    
    out_dir = wf.get("output_dir", "outputs/walk_forward"); ensure_dir(out_dir)
    set_seed(int(cfg.get("rl", {}).get("seed", 42)))

    feat_df, cols = load_feature_cache(cfg)
    if feat_df is None: feat_df, cols = build_features(load_or_fetch(cfg), cfg["features"])
    feat_df = feat_df.dropna(subset=cols).reset_index(drop=True)

    start_idx, fold_idx, summary = 0, 0, {"folds": [], "mode": "dt"}
    while start_idx + train_bars + val_bars + test_bars <= len(feat_df):
        te, ve, tse = start_idx + train_bars, start_idx + train_bars + val_bars, start_idx + train_bars + val_bars + test_bars
        fold_dir = os.path.join(out_dir, f"fold_{fold_idx:02d}"); ensure_dir(fold_dir)
        train_df, val_df, test_df = feat_df.iloc[start_idx:te].copy(), feat_df.iloc[te:ve].copy(), feat_df.iloc[ve:tse].copy()
        
        if len(train_df) < cfg["dataset"]["seq_len"] or len(val_df) < cfg["dataset"]["seq_len"]: raise ValueError("window < seq_len")
        print(f"fold {fold_idx}: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        
        model, ckpt, _ = train_for_fold_offline(cfg, train_df, val_df, cols, fold_dir, int(cfg.get("rl", {}).get("seed", 42)) + fold_idx)
        curve, eq, ret, trd, turnover = run_model_backtest(cfg, model, test_df, cols, next(model.parameters()).device)
        
        met = compute_metrics(eq, ret, trd, turnover, annualization_factor(cfg["data"]["timeframe"]), curve["action"].to_numpy(), float(cfg.get("backtest", {}).get("risk_free", 0.0)))
        if model.action_mode == "continuous": met["action_stats"] = {"mean_position": float(np.mean(curve["action"])), "mean_abs_position": float(np.mean(np.abs(curve["action"])))}
        else: met["action_distribution"] = action_distribution(curve["action"].to_numpy(dtype=np.int64))

        save_json(os.path.join(fold_dir, "metrics.json"), {"decision_transformer": met})
        curve.to_csv(os.path.join(fold_dir, "equity_curve.csv"), index=False)
        
        summary["folds"].append({"fold": fold_idx, "train_start": str(train_df["datetime"].iloc[0]), "train_end": str(train_df["datetime"].iloc[-1]),
            "val_start": str(val_df["datetime"].iloc[0]), "val_end": str(val_df["datetime"].iloc[-1]), "test_start": str(test_df["datetime"].iloc[0]),
            "test_end": str(test_df["datetime"].iloc[-1]), "checkpoint": ckpt, "metrics": met})
        
        fold_idx += 1; start_idx += step_bars
        if (mf := int(wf.get("max_folds", 0))) and fold_idx >= mf: break

    summary["metric_summary"] = summarize_metrics(summary["folds"])
    save_json(os.path.join(out_dir, "summary.json"), summary)
    write_summary_csv(os.path.join(out_dir, "summary.csv"), summary["folds"])
    write_summary_md(os.path.join(out_dir, "summary.md"), summary["metric_summary"], len(summary["folds"]))
    print(f"walk-forward summary saved to {out_dir}")

if __name__ == "__main__":
    args = argparse.ArgumentParser(); args.add_argument("--config", default="config.yaml")
    run_walk_forward(load_config(args.parse_args().config))