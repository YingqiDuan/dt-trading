import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_cmd(cmd, dry_run=False):
    print(">>", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def find_latest_checkpoint(ckpt_dir):
    paths = glob.glob(os.path.join(ckpt_dir, "dt_best_*.pt"))
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def parse_run_id(ckpt_path):
    base = os.path.basename(ckpt_path)
    if base.startswith("dt_best_") and base.endswith(".pt"):
        return base[len("dt_best_") : -len(".pt")]
    return time.strftime("%Y%m%d_%H%M%S")


def compute_config_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_metrics(metrics_path):
    if not metrics_path or not os.path.exists(metrics_path):
        return {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("decision_transformer", payload)


def append_registry(registry_path, record):
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_manifest(deploy_dir, record):
    os.makedirs(deploy_dir, exist_ok=True)
    manifest_path = os.path.join(deploy_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-register", action="store_true")
    parser.add_argument("--skip-deploy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=["offline", "ppo"], default=None)
    parser.add_argument("--init-ckpt", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--registry-path", default=None)
    parser.add_argument("--deploy-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline_cfg = cfg.get("pipeline", {})
    registry_path = args.registry_path or pipeline_cfg.get(
        "registry_path", "outputs/models/registry.jsonl"
    )
    deploy_dir = args.deploy_dir or pipeline_cfg.get("deploy_dir", "outputs/deploy")

    if not args.skip_data:
        cmd = [sys.executable, "dataset_builder.py", "--config", args.config]
        if args.force_data:
            cmd.append("--force")
        run_cmd(cmd, dry_run=args.dry_run)

    if not args.skip_train:
        cmd = [sys.executable, "train_dt.py", "--config", args.config]
        if args.init_ckpt:
            cmd += ["--init_ckpt", args.init_ckpt]
        if args.mode:
            cmd += ["--mode", args.mode]
        run_cmd(cmd, dry_run=args.dry_run)

    ckpt_path = args.ckpt
    if not ckpt_path:
        ckpt_path = find_latest_checkpoint(cfg["train"]["checkpoint_dir"])
    if not ckpt_path and not args.skip_backtest:
        raise FileNotFoundError("no checkpoint found for backtest/deploy")

    if not args.skip_backtest:
        cmd = [sys.executable, "backtest.py", "--config", args.config, "--ckpt", ckpt_path]
        run_cmd(cmd, dry_run=args.dry_run)

    metrics_path = os.path.join(cfg["backtest"]["output_dir"], "metrics.json")
    if not args.skip_register:
        record = {
            "run_id": parse_run_id(ckpt_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_path": args.config,
            "config_hash": compute_config_hash(args.config),
            "checkpoint": ckpt_path,
            "metrics_path": metrics_path,
            "metrics": load_metrics(metrics_path),
            "symbol": cfg["data"]["symbol"],
            "timeframe": cfg["data"]["timeframe"],
            "mode": args.mode or cfg.get("train", {}).get("mode", "offline"),
        }
        append_registry(registry_path, record)

    if not args.skip_deploy:
        if not ckpt_path:
            raise FileNotFoundError("no checkpoint found for deploy")
        os.makedirs(deploy_dir, exist_ok=True)
        target_path = os.path.join(deploy_dir, "current.pt")
        if not args.dry_run:
            shutil.copy2(ckpt_path, target_path)
        record = {
            "checkpoint": ckpt_path,
            "deployed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_path": args.config,
            "config_hash": compute_config_hash(args.config),
        }
        write_manifest(deploy_dir, record)
        print(f"deployed checkpoint to {target_path}")


if __name__ == "__main__":
    main()
