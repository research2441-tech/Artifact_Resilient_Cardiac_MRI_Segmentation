from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data import build_loader
from metrics import bootstrap_ci, multiclass_metrics
from models import build_model
from train import load_config, seed_everything


def _to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _accumulate_predictions(model, loader, device, amp_enabled: bool):
    store = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for images, masks, meta in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
            pred = logits.argmax(1).cpu().numpy()
            true = masks.numpy()
            subject_ids = list(meta["subject_id"])
            slice_indices = _to_list(meta["slice_index"])
            phases = list(meta["phase"])
            conditions = list(meta["condition"])
            for i, sid in enumerate(subject_ids):
                key = (str(sid), str(phases[i]), str(conditions[i]))
                store[key].append((int(slice_indices[i]), pred[i], true[i], images[i, 0].detach().cpu().numpy()))
    return store


def _subject_rows(store, cfg: Dict[str, Any]):
    rows = []
    num_classes = int(cfg["data"]["num_classes"])
    class_names = cfg["data"].get("class_names")
    spacing = cfg.get("evaluation", {}).get("hd95_spacing")
    for (sid, phase, condition), slices in store.items():
        slices.sort(key=lambda x: x[0])
        pred_v = np.stack([x[1] for x in slices], axis=0)
        true_v = np.stack([x[2] for x in slices], axis=0)
        m = multiclass_metrics(pred_v, true_v, num_classes, class_names, False, spacing)
        rows.append({"subject_id": sid, "phase": phase, "condition": condition, **m})
    return pd.DataFrame(rows)


def _summary(subject_df: pd.DataFrame, cfg: Dict[str, Any]):
    ec = cfg.get("evaluation", {})
    n_boot = int(ec.get("bootstrap_samples", 2000))
    conf = float(ec.get("confidence_level", 0.95))
    seed = int(cfg.get("seed", 42))
    rows = []
    for metric in ["dsc_macro", "iou_macro", "hd95_macro", "precision_macro", "recall_macro"]:
        vals = subject_df[metric].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(vals, n_boot=n_boot, confidence=conf, seed=seed)
        rows.append(
            {
                "metric": metric,
                "n_subject_phase_records": int(np.isfinite(vals).sum()),
                "mean": float(np.nanmean(vals)),
                "sd": float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1 else 0.0,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows)


def _save_qualitative(store, prefix: str, max_cases: int = 4):
    keys = list(store.keys())[:max_cases]
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), 3, figsize=(10, 3.2 * len(keys)), squeeze=False)
    for r, key in enumerate(keys):
        slices = sorted(store[key], key=lambda x: x[0])
        center = slices[len(slices) // 2]
        _, pred, true, image = center
        axes[r, 0].imshow(image, cmap="gray")
        axes[r, 0].set_title(f"{key[0]} input")
        axes[r, 1].imshow(true, vmin=0, vmax=max(1, int(true.max())))
        axes[r, 1].set_title("Ground truth")
        axes[r, 2].imshow(pred, vmin=0, vmax=max(1, int(pred.max())))
        axes[r, 2].set_title("Prediction")
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{prefix}_qualitative.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_evaluation(cfg: Dict[str, Any], checkpoint: str, split: str = "test", prefix: str = "full_model") -> Dict[str, float]:
    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state"] if "model_state" in payload else payload)
    loader = build_loader(cfg, split, shuffle=False)
    store = _accumulate_predictions(model, loader, device, bool(cfg["training"].get("amp", True)) and device.type == "cuda")
    subject_df = _subject_rows(store, cfg)
    summary_df = _summary(subject_df, cfg)
    subject_df.to_csv(f"{prefix}_subject_metrics.csv", index=False)
    summary_df.to_csv(f"{prefix}_summary.csv", index=False)
    _save_qualitative(store, prefix, int(cfg.get("evaluation", {}).get("save_qualitative_cases", 4)))
    return {row.metric: float(row.mean) for row in summary_df.itertuples()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--prefix", default="full_model")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = run_evaluation(cfg, args.checkpoint, args.split, args.prefix)
    print(result)


if __name__ == "__main__":
    main()
