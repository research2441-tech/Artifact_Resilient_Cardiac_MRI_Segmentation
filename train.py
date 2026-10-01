from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from data import build_loader
from losses import CombinedDiceCELoss
from models import build_model


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def fast_dice_from_logits(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    pred = logits.argmax(1)
    scores = []
    for c in range(1, num_classes):
        p = pred == c
        t = target == c
        inter = (p & t).sum().float()
        denom = p.sum().float() + t.sum().float()
        scores.append(torch.where(denom > 0, 2 * inter / denom, torch.ones_like(denom)))
    return float(torch.stack(scores).mean().item())


def _scheduler(optimizer, cfg: Dict[str, Any]):
    tc = cfg["training"]
    name = str(tc.get("scheduler", "cosine")).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(tc["epochs"])))
    if name in {"none", "off"}:
        return None
    raise ValueError(f"Unsupported scheduler: {name}")


def run_training(cfg: Dict[str, Any], run_name: str = "full_model") -> str:
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = build_loader(cfg, "train", shuffle=True)
    val_loader = build_loader(cfg, "val", shuffle=False)
    model = build_model(cfg).to(device)
    criterion = CombinedDiceCELoss(cfg)
    tc = cfg["training"]

    if str(tc.get("optimizer", "SGD")).upper() != "SGD":
        raise ValueError("This reproducibility implementation currently supports SGD, matching the manuscript protocol.")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(tc["learning_rate"]),
        momentum=float(tc.get("momentum", 0.9)),
        weight_decay=float(tc.get("weight_decay", 1e-4)),
        nesterov=bool(tc.get("nesterov", True)),
    )
    scheduler = _scheduler(optimizer, cfg)
    amp_enabled = bool(tc.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    grad_clip = float(tc.get("grad_clip_norm", 0.0))
    epochs = int(tc["epochs"])
    patience = int(tc.get("early_stopping_patience", 20))
    num_classes = int(cfg["data"]["num_classes"])

    best_score = -float("inf")
    stale = 0
    history = []
    checkpoint_path = f"{run_name}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_dice, n_train = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"{run_name} epoch {epoch}/{epochs}", leave=False)
        for images, masks, _ in pbar:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            bs = images.size(0)
            train_loss += float(loss.item()) * bs
            train_dice += fast_dice_from_logits(logits.detach(), masks, num_classes) * bs
            n_train += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_loss, val_dice, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(images)
                    loss = criterion(logits, masks)
                bs = images.size(0)
                val_loss += float(loss.item()) * bs
                val_dice += fast_dice_from_logits(logits, masks, num_classes) * bs
                n_val += bs

        if scheduler is not None:
            scheduler.step()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss / max(1, n_train),
            "train_dsc": train_dice / max(1, n_train),
            "val_loss": val_loss / max(1, n_val),
            "val_dsc": val_dice / max(1, n_val),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(f"{run_name}_history.csv", index=False)

        score = row["val_dsc"]
        if score > best_score + 1e-6:
            best_score = score
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "run_name": run_name,
                    "epoch": epoch,
                    "best_val_dsc": best_score,
                    "torch_version": torch.__version__,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= patience:
                break

    return checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run-name", default="full_model")
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = run_training(cfg, args.run_name)
    print(path)


if __name__ == "__main__":
    main()
