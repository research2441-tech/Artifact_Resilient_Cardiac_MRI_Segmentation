from __future__ import annotations

from typing import Any, Dict

import albumentations as A


def build_train_augmentation(cfg: Dict[str, Any]) -> A.Compose | None:
    """Build paired image/mask augmentation approximating respiratory-motion variation."""
    if not cfg.get("enabled", True):
        return None

    transforms = [
        A.Affine(
            scale=(1.0 - float(cfg.get("scale_limit", 0.10)), 1.0 + float(cfg.get("scale_limit", 0.10))),
            translate_percent={"x": (-float(cfg.get("shift_limit", 0.05)), float(cfg.get("shift_limit", 0.05))),
                               "y": (-float(cfg.get("shift_limit", 0.05)), float(cfg.get("shift_limit", 0.05)))},
            rotate=(-float(cfg.get("rotate_limit_deg", 10)), float(cfg.get("rotate_limit_deg", 10))),
            shear=(-3.0, 3.0),
            interpolation=1,
            mask_interpolation=0,
            fit_output=False,
            keep_ratio=True,
            p=float(cfg.get("affine_p", 0.50)),
        ),
        A.ElasticTransform(
            alpha=float(cfg.get("elastic_alpha", 1.0)),
            sigma=float(cfg.get("elastic_sigma", 50.0)),
            interpolation=1,
            mask_interpolation=0,
            p=float(cfg.get("elastic_p", 0.25)),
        ),
        A.GaussNoise(
            var_limit=tuple(float(v) for v in cfg.get("gaussian_noise_var", [5.0, 25.0])),
            p=float(cfg.get("noise_p", 0.25)),
        ),
        A.RandomGamma(
            gamma_limit=tuple(int(v) for v in cfg.get("gamma_limit", [85, 115])),
            p=float(cfg.get("gamma_p", 0.25)),
        ),
        A.RandomBrightnessContrast(
            brightness_limit=float(cfg.get("brightness_limit", 0.10)),
            contrast_limit=float(cfg.get("contrast_limit", 0.10)),
            p=float(cfg.get("brightness_contrast_p", 0.25)),
        ),
    ]

    hflip = float(cfg.get("horizontal_flip_p", 0.0))
    vflip = float(cfg.get("vertical_flip_p", 0.0))
    if hflip > 0:
        transforms.append(A.HorizontalFlip(p=hflip))
    if vflip > 0:
        transforms.append(A.VerticalFlip(p=vflip))

    return A.Compose(transforms)
