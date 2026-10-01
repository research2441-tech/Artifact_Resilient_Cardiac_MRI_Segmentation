from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1, dropout: float = 0.0):
        pad = dilation * (kernel // 2)
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_ch, out_ch, 3, 1, dropout),
            ConvNormAct(out_ch, out_ch, 3, 1, dropout),
        )
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.proj(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class MultiScaleFusion(nn.Module):
    """Three parallel 3x3 dilated branches (rates 1, 2, 3) followed by fusion."""

    def __init__(self, channels: int):
        super().__init__()
        branch = max(16, channels // 3)
        self.b1 = ConvNormAct(channels, branch, 3, dilation=1)
        self.b2 = ConvNormAct(channels, branch, 3, dilation=2)
        self.b3 = ConvNormAct(channels, branch, 3, dilation=3)
        self.fuse = nn.Sequential(
            nn.Conv2d(branch * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)) + x


class TinyTransformerBottleneck(nn.Module):
    """Transformer bottleneck used by the baseline Transformer U-Net."""

    def __init__(self, channels: int, heads: int = 8, layers: int = 2):
        super().__init__()
        heads = max(1, min(heads, channels))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        enc_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(self.encoder(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class BaselineEncoder(nn.Module):
    def __init__(self, in_ch: int, channels: Sequence[int]):
        super().__init__()
        blocks = []
        current = in_ch
        for i, ch in enumerate(channels):
            stage = [nn.MaxPool2d(2)] if i > 0 else []
            stage.append(ConvBlock(current, ch, dropout=0.1 if ch < 256 else 0.2))
            blocks.append(nn.Sequential(*stage))
            current = ch
        self.blocks = nn.ModuleList(blocks)
        self.channels = list(channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return feats


class EfficientFormerV2Encoder(nn.Module):
    def __init__(self, backbone_name: str, in_ch: int, pretrained: bool):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for EfficientFormerV2. Install requirements.txt") from exc

        try:
            self.backbone = timm.create_model(
                backbone_name,
                features_only=True,
                pretrained=pretrained,
                in_chans=in_ch,
                out_indices=(0, 1, 2, 3),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to create EfficientFormerV2 backbone '{backbone_name}'. "
                "Confirm that the pinned timm version supports this model and that pretrained weights are available."
            ) from exc
        self.channels = list(self.backbone.feature_info.channels())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.backbone(x)
        if len(feats) != 4:
            raise RuntimeError(f"Expected four feature scales, received {len(feats)}")
        return list(feats)


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_attention: bool, use_multiscale: bool, dropout: float):
        super().__init__()
        self.skip_attention = ChannelAttention(skip_ch) if use_attention else nn.Identity()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout=dropout)
        self.multiscale = MultiScaleFusion(out_ch) if use_multiscale else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.skip_attention(skip)
        x = self.conv(torch.cat([x, skip], dim=1))
        return self.multiscale(x)


class ArtifactResilientUNet(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        mc = cfg["model"]
        in_ch = int(mc.get("in_channels", 1))
        num_classes = int(mc.get("num_classes", 4))
        use_efficient = bool(mc.get("use_efficientformerv2", True))
        pretrained = bool(mc.get("pretrained", True)) and use_efficient
        use_attention = bool(mc.get("use_channel_attention", True))
        use_multiscale = bool(mc.get("use_multiscale_fusion", True))
        dropout = float(mc.get("decoder_dropout", 0.10))

        if use_efficient:
            self.encoder = EfficientFormerV2Encoder(mc["backbone"], in_ch, pretrained)
            channels = self.encoder.channels
            self.context = nn.Identity()
        else:
            channels = [int(x) for x in mc.get("baseline_channels", [32, 64, 128, 256])]
            self.encoder = BaselineEncoder(in_ch, channels)
            self.context = TinyTransformerBottleneck(
                channels[-1],
                heads=int(mc.get("transformer_heads", 8)),
                layers=int(mc.get("transformer_layers", 2)),
            )

        stages = []
        current = channels[-1]
        for skip_ch in reversed(channels[:-1]):
            stages.append(
                DecoderStage(
                    current,
                    skip_ch,
                    skip_ch,
                    use_attention=use_attention,
                    use_multiscale=use_multiscale,
                    dropout=dropout,
                )
            )
            current = skip_ch
        self.decoder = nn.ModuleList(stages)
        self.head = nn.Conv2d(current, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        feats = self.encoder(x)
        z = self.context(feats[-1])
        for stage, skip in zip(self.decoder, reversed(feats[:-1])):
            z = stage(z, skip)
        logits = self.head(z)
        if logits.shape[-2:] != input_hw:
            logits = F.interpolate(logits, size=input_hw, mode="bilinear", align_corners=False)
        return logits


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    return ArtifactResilientUNet(cfg)
