from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def _binary_stats(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float, float, float]:
    pred = pred.astype(bool)
    true = true.astype(bool)
    tp = np.logical_and(pred, true).sum(dtype=np.float64)
    fp = np.logical_and(pred, ~true).sum(dtype=np.float64)
    fn = np.logical_and(~pred, true).sum(dtype=np.float64)
    denom_d = 2.0 * tp + fp + fn
    denom_i = tp + fp + fn
    dice = 1.0 if denom_d == 0 else (2.0 * tp) / denom_d
    iou = 1.0 if denom_i == 0 else tp / denom_i
    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    return float(dice), float(iou), float(precision), float(recall)


def hd95(pred: np.ndarray, true: np.ndarray, spacing: Optional[Sequence[float]] = None) -> float:
    pred = pred.astype(bool)
    true = true.astype(bool)
    if not pred.any() and not true.any():
        return 0.0
    if not pred.any() or not true.any():
        return float("nan")

    pred_surface = np.logical_xor(pred, binary_erosion(pred, border_value=0))
    true_surface = np.logical_xor(true, binary_erosion(true, border_value=0))
    sampling = tuple(spacing) if spacing is not None else None
    dt_true = distance_transform_edt(~true_surface, sampling=sampling)
    dt_pred = distance_transform_edt(~pred_surface, sampling=sampling)
    d1 = dt_true[pred_surface]
    d2 = dt_pred[true_surface]
    distances = np.concatenate([d1, d2])
    return float(np.percentile(distances, 95)) if distances.size else 0.0


def multiclass_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    num_classes: int,
    class_names: Optional[Sequence[str]] = None,
    include_background: bool = False,
    spacing: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    class_names = list(class_names or [str(i) for i in range(num_classes)])
    start = 0 if include_background else 1
    out: Dict[str, float] = {}
    ds, js, hs, ps, rs = [], [], [], [], []
    for c in range(start, num_classes):
        name = class_names[c] if c < len(class_names) else str(c)
        d, j, p, r = _binary_stats(pred == c, true == c)
        h = hd95(pred == c, true == c, spacing=spacing)
        out[f"dsc_{name}"] = d
        out[f"iou_{name}"] = j
        out[f"hd95_{name}"] = h
        out[f"precision_{name}"] = p
        out[f"recall_{name}"] = r
        ds.append(d); js.append(j); hs.append(h); ps.append(p); rs.append(r)
    out["dsc_macro"] = float(np.nanmean(ds))
    out["iou_macro"] = float(np.nanmean(js))
    out["hd95_macro"] = float(np.nanmean(hs))
    out["precision_macro"] = float(np.nanmean(ps))
    out["recall_macro"] = float(np.nanmean(rs))
    return out


def bootstrap_ci(values: Iterable[float], n_boot: int = 2000, confidence: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=x.size, replace=True).mean()
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))
