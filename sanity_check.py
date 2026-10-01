from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

from data import REQUIRED_MANIFEST_COLUMNS, _load_array
from train import load_config


def run_checks(cfg: Dict[str, Any]) -> None:
    manifest_path = Path(cfg["data"]["manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    df["split"] = df["split"].astype(str).str.lower()
    allowed = {"train", "val", "test"}
    unknown = sorted(set(df["split"]) - allowed)
    if unknown:
        raise ValueError(f"Unknown split values: {unknown}")

    subject_splits = df.groupby("subject_id")["split"].nunique()
    leaked = subject_splits[subject_splits > 1]
    if len(leaked):
        raise ValueError(f"Subject leakage detected: {leaked.index.astype(str).tolist()}")

    expected_labels: Set[int] = set(int(k) for k in cfg["data"].get("label_map", {}).keys())
    observed: Set[int] = set()
    class_presence: Dict[int, int] = {x: 0 for x in expected_labels}
    problems: List[str] = []

    for row in df.itertuples(index=False):
        image_path = Path(str(row.image)); mask_path = Path(str(row.mask))
        if not image_path.exists():
            problems.append(f"Missing image: {image_path}"); continue
        if not mask_path.exists():
            problems.append(f"Missing mask: {mask_path}"); continue
        image, _ = _load_array(str(image_path))
        mask, _ = _load_array(str(mask_path))
        if image.shape != mask.shape:
            problems.append(f"Shape mismatch {row.subject_id}: {image.shape} vs {mask.shape}")
        if image.ndim != 3:
            problems.append(f"Expected 3-D image for {row.subject_id}, got {image.ndim}D")
        if not np.isfinite(image).all():
            problems.append(f"NaN/Inf found in image for {row.subject_id}")
        labels = set(np.unique(np.rint(mask).astype(int)).tolist())
        observed |= labels
        for label in labels:
            class_presence[label] = class_presence.get(label, 0) + 1

    unexpected = observed - expected_labels
    if unexpected:
        problems.append(f"Unexpected mask labels: {sorted(unexpected)}")

    num_classes = int(cfg["data"]["num_classes"])
    model_classes = int(cfg["model"]["num_classes"])
    if num_classes != model_classes:
        problems.append(f"data.num_classes={num_classes} but model.num_classes={model_classes}")

    print("Split summary:")
    print(df.groupby("split").agg(subjects=("subject_id", "nunique"), volumes=("subject_id", "size")).to_string())
    print(f"Observed raw labels: {sorted(observed)}")
    print(f"Expected raw labels: {sorted(expected_labels)}")

    if problems:
        raise RuntimeError("Sanity check failed:\n- " + "\n- ".join(problems))
    print("Sanity check passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_checks(load_config(args.config))


if __name__ == "__main__":
    main()
