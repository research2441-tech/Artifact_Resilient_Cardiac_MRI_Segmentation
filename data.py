from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset

from augmentations import build_train_augmentation


REQUIRED_MANIFEST_COLUMNS = {"image", "mask", "subject_id", "split"}


def _load_array(path: str) -> Tuple[np.ndarray, Optional[Tuple[float, ...]]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix == ".npy":
        return np.load(p), None
    if p.name.endswith(".nii") or p.name.endswith(".nii.gz"):
        nii = nib.load(str(p))
        nii = nib.as_closest_canonical(nii)
        arr = np.asarray(nii.get_fdata(dtype=np.float32))
        spacing = tuple(float(x) for x in nii.header.get_zooms()[: arr.ndim])
        return arr, spacing
    raise ValueError(f"Unsupported file type: {p}")


def _resample_volume(
    image: np.ndarray,
    mask: np.ndarray,
    source_spacing: Optional[Sequence[float]],
    target_spacing: Optional[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    if target_spacing is None or source_spacing is None:
        return image, mask
    if len(source_spacing) < image.ndim or len(target_spacing) < image.ndim:
        raise ValueError("Spacing dimensionality does not match image dimensionality.")
    factors = [float(source_spacing[i]) / float(target_spacing[i]) for i in range(image.ndim)]
    image_r = zoom(image, factors, order=1, mode="nearest")
    mask_r = zoom(mask, factors, order=0, mode="nearest")
    return image_r.astype(np.float32), mask_r.astype(np.int64)


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    finite = np.isfinite(volume)
    if not finite.all():
        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    support = np.abs(volume) > 1e-8
    values = volume[support] if support.any() else volume.reshape(-1)
    mean = float(values.mean())
    std = float(values.std())
    std = max(std, 1e-6)
    return (volume - mean) / std


def remap_labels(mask: np.ndarray, label_map: Mapping[int, int]) -> np.ndarray:
    out = np.full(mask.shape, -1, dtype=np.int64)
    for src, dst in label_map.items():
        out[mask == int(src)] = int(dst)
    unknown = np.unique(mask[out < 0])
    if unknown.size:
        raise ValueError(f"Mask contains labels not present in label_map: {unknown.tolist()}")
    return out


def resize_slice(image: np.ndarray, mask: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    image_t = torch.from_numpy(image).float()[None, None]
    mask_t = torch.from_numpy(mask.astype(np.float32))[None, None]
    image_r = F.interpolate(image_t, size=(size, size), mode="bilinear", align_corners=False)[0, 0]
    mask_r = F.interpolate(mask_t, size=(size, size), mode="nearest")[0, 0]
    return image_r.numpy().astype(np.float32), mask_r.numpy().astype(np.int64)


class _VolumeCache:
    def __init__(self, max_items: int = 8):
        self.max_items = max(0, int(max_items))
        self.cache: OrderedDict[str, Tuple[np.ndarray, np.ndarray, Optional[Tuple[float, ...]]]] = OrderedDict()

    def get(self, key: str):
        if key not in self.cache:
            return None
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def put(self, key: str, value):
        if self.max_items <= 0:
            return
        if key in self.cache:
            self.cache.pop(key)
        self.cache[key] = value
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)


class CardiacSliceDataset(Dataset):
    """Slice dataset backed by subject-level 3-D cardiac MR volumes."""

    def __init__(self, cfg: Dict[str, Any], split: str):
        self.cfg = cfg
        self.split = split
        data_cfg = cfg["data"]
        self.image_size = int(data_cfg["image_size"])
        self.slice_axis = int(data_cfg.get("slice_axis", 2))
        self.target_spacing = data_cfg.get("target_spacing")
        self.label_map = {int(k): int(v) for k, v in data_cfg.get("label_map", {}).items()}
        self.min_foreground_pixels = int(data_cfg.get("min_foreground_pixels", 0))
        self.cache = _VolumeCache(int(data_cfg.get("cache_volumes", 8)))
        self.augmentation = build_train_augmentation(cfg.get("augmentation", {})) if split == "train" else None

        manifest = pd.read_csv(data_cfg["manifest"])
        missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        manifest["split"] = manifest["split"].astype(str).str.lower()
        self.rows = manifest[manifest["split"] == split.lower()].reset_index(drop=True)
        if len(self.rows) == 0:
            raise ValueError(f"No rows found for split='{split}'.")

        self.index: List[Tuple[int, int]] = []
        for row_idx, row in self.rows.iterrows():
            image, _ = _load_array(str(row["image"]))
            if image.ndim != 3:
                raise ValueError(f"Expected a 3-D volume for {row['image']}, got shape {image.shape}")
            axis = self.slice_axis % 3
            for s in range(image.shape[axis]):
                self.index.append((row_idx, s))

    def __len__(self) -> int:
        return len(self.index)

    def _volume_pair(self, row_idx: int):
        row = self.rows.iloc[row_idx]
        key = f"{row['image']}|{row['mask']}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        image, spacing = _load_array(str(row["image"]))
        mask, mask_spacing = _load_array(str(row["mask"]))
        if image.shape != mask.shape:
            raise ValueError(f"Shape mismatch for subject {row['subject_id']}: {image.shape} vs {mask.shape}")
        source_spacing = spacing if spacing is not None else mask_spacing
        image, mask = _resample_volume(image, mask, source_spacing, self.target_spacing)
        image = zscore_normalize(image)
        mask = remap_labels(np.rint(mask).astype(np.int64), self.label_map)
        value = (image, mask, source_spacing)
        self.cache.put(key, value)
        return value

    def __getitem__(self, idx: int):
        row_idx, slice_idx = self.index[idx]
        row = self.rows.iloc[row_idx]
        image_v, mask_v, _ = self._volume_pair(row_idx)
        axis = self.slice_axis % 3
        image = np.take(image_v, slice_idx, axis=axis)
        mask = np.take(mask_v, slice_idx, axis=axis)
        image, mask = resize_slice(image, mask, self.image_size)

        if self.augmentation is not None:
            aug = self.augmentation(image=image, mask=mask)
            image, mask = aug["image"].astype(np.float32), aug["mask"].astype(np.int64)

        image_t = torch.from_numpy(image).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask).long()
        meta = {
            "subject_id": str(row["subject_id"]),
            "slice_index": int(slice_idx),
            "phase": str(row["phase"]) if "phase" in row.index and pd.notna(row["phase"]) else "",
            "condition": str(row["condition"]) if "condition" in row.index and pd.notna(row["condition"]) else "",
            "image_path": str(row["image"]),
        }
        return image_t, mask_t, meta


def build_loader(cfg: Dict[str, Any], split: str, shuffle: Optional[bool] = None) -> DataLoader:
    ds = CardiacSliceDataset(cfg, split)
    if shuffle is None:
        shuffle = split == "train"
    section = cfg["training"] if split in {"train", "val"} else cfg["evaluation"]
    return DataLoader(
        ds,
        batch_size=int(section.get("batch_size", cfg["training"]["batch_size"])),
        shuffle=bool(shuffle),
        num_workers=int(section.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=int(section.get("num_workers", 4)) > 0,
    )
