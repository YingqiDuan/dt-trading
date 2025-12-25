import argparse
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from dt_model import DecisionTransformer
from utils import ensure_dir, load_config, save_json


class TrajectoryDataset(Dataset):
    def __init__(self, npz_path, seq_len):
        data = np.load(npz_path, allow_pickle=True)
        self.states = data["states"].astype(np.float32)
        self.actions = data["actions"].astype(np.int64)
        self.rtg = data["rtg"].astype(np.float32)
        self.traj_id = data["traj_id"].astype(np.int64)
        self.seq_len = seq_len

        boundaries = np.where(np.diff(self.traj_id) != 0)[0] + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(self.traj_id)]))

        valid = []
        valid_traj_start = []
        for start, end in zip(starts, ends):
            if end - start >= seq_len:
                count = end - start - seq_len + 1
                valid.extend(range(start, end - seq_len + 1))
                valid_traj_start.extend([start] * count)
        self.valid_starts = np.array(valid, dtype=np.int64)
        self.valid_traj_start = np.array(valid_traj_start, dtype=np.int64)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.seq_len
        states = self.states[start:end]
        actions = self.actions[start:end]
        rtg = self.rtg[start:end]
        traj_start = self.valid_traj_start[idx]
        prev_action = 0
        if start > traj_start:
            prev_action = int(self.actions[start - 1])
        return states, actions, rtg, prev_action

    def sampling_weights(self, mode, power=1.0, epsilon=1e-3):
        if mode == "episode_return":
            scores = self.rtg[self.valid_traj_start]
        else:
            scores = self.rtg[self.valid_starts]
        if scores.size == 0:
            return np.ones(len(self.valid_starts), dtype=np.float64)
        scores = scores - scores.min()
        weights = (scores + epsilon) ** power
        if not np.isfinite(weights).all() or weights.sum() == 0:
            return np.ones(len(self.valid_starts), dtype=np.float64)
        return weights.astype(np.float64)


def select_device(pref):
    if pref == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def action_to_index(actions):
    return actions + 1


def compute_class_weights(actions):
    counts = np.bincount(actions + 1, minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return weights, counts


def update_confusion(confusion, preds, targets, num_classes):
    preds = preds.reshape(-1).cpu().numpy()
    targets = targets.reshape(-1).cpu().numpy()
    idx = targets * num_classes + preds
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    confusion += counts.reshape(num_classes, num_classes)
    return confusion


def compute_macro_f1_and_recall(confusion):
    num_classes = confusion.shape[0]
    recalls = []
    f1s = []
    for cls in range(num_classes):
        tp = confusion[cls, cls]
        fp = confusion[:, cls].sum() - tp
        fn = confusion[cls, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        recalls.append(float(recall))
        f1s.append(float(f1))
    return float(np.mean(f1s)), recalls


def train_epoch(model, loader, optimizer, device, criterion, grad_clip):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    confusion = np.zeros((3, 3), dtype=np.int64)

    for states, actions, rtg, prev_actions in tqdm(loader, desc="train", leave=False):
        states = states.to(device)
        actions = actions.to(device)
        rtg = rtg.to(device)
        prev_actions = prev_actions.to(device).long()

        actions_in = torch.zeros_like(actions)
        actions_in[:, 0] = prev_actions
        actions_in[:, 1:] = actions[:, :-1]

        logits = model(states, action_to_index(actions_in), rtg)
        targets = action_to_index(actions)

        loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * states.size(0)
        preds = logits.argmax(dim=-1)
        confusion = update_confusion(confusion, preds, targets, 3)
        total_correct += (preds == targets).sum().item()
        total_count += targets.numel()

    avg_loss = total_loss / len(loader.dataset)
    acc = total_correct / max(1, total_count)
    macro_f1, recalls = compute_macro_f1_and_recall(confusion)
    return avg_loss, acc, macro_f1, recalls


def eval_epoch(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    confusion = np.zeros((3, 3), dtype=np.int64)
    with torch.no_grad():
        for states, actions, rtg, prev_actions in tqdm(loader, desc="val", leave=False):
            states = states.to(device)
            actions = actions.to(device)
            rtg = rtg.to(device)
            prev_actions = prev_actions.to(device).long()

            actions_in = torch.zeros_like(actions)
            actions_in[:, 0] = prev_actions
            actions_in[:, 1:] = actions[:, :-1]

            logits = model(states, action_to_index(actions_in), rtg)
            targets = action_to_index(actions)

            loss = criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            total_loss += loss.item() * states.size(0)

            preds = logits.argmax(dim=-1)
            confusion = update_confusion(confusion, preds, targets, 3)
            total_correct += (preds == targets).sum().item()
            total_count += targets.numel()

    avg_loss = total_loss / len(loader.dataset)
    acc = total_correct / max(1, total_count)
    macro_f1, recalls = compute_macro_f1_and_recall(confusion)
    return avg_loss, acc, macro_f1, recalls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train", default=None)
    parser.add_argument("--val", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_path = args.train or os.path.join(
        cfg["data"]["dataset_dir"], "train_dataset.npz"
    )
    val_path = args.val or os.path.join(cfg["data"]["dataset_dir"], "val_dataset.npz")

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
        weights = train_ds.sampling_weights(sampling_mode, power=power, epsilon=epsilon)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        print(
            "sampling=weighted "
            f"mode={sampling_mode} power={power} epsilon={epsilon} "
            f"weight_stats(min/mean/max)={weights.min():.3g}/{weights.mean():.3g}/{weights.max():.3g}"
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

    ensure_dir(cfg["train"]["log_dir"])
    ensure_dir(cfg["train"]["checkpoint_dir"])

    log_rows = []
    best_val = float("inf")
    run_id = time.strftime("%Y%m%d_%H%M%S")

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
            best_path = os.path.join(
                cfg["train"]["checkpoint_dir"], f"dt_best_{run_id}.pt"
            )
            torch.save(ckpt, best_path)

        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
            f"train_f1={train_f1:.3f} val_f1={val_f1:.3f}"
        )

    log_path = os.path.join(cfg["train"]["log_dir"], f"training_log_{run_id}.json")
    save_json(log_path, log_rows)


if __name__ == "__main__":
    main()
