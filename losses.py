from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes: int, include_background: bool = False, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = int(num_classes)
        self.include_background = bool(include_background)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        start = 0 if self.include_background else 1
        probs = probs[:, start:]
        onehot = onehot[:, start:]
        dims = (0, 2, 3)
        intersection = torch.sum(probs * onehot, dims)
        denominator = torch.sum(probs + onehot, dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class CombinedDiceCELoss(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        tc = cfg["training"]
        num_classes = int(cfg["data"]["num_classes"])
        self.dice_weight = float(tc.get("dice_weight", 0.5))
        self.ce_weight = float(tc.get("ce_weight", 0.5))
        self.dice = SoftDiceLoss(num_classes, bool(tc.get("include_background_in_dice", False)))
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.ce_weight * self.ce(logits, target)
