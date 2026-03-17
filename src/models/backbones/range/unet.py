import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..spec import BackboneSpec, make_backbone_spec
from .common import ConvBlock


def add_range_unet_backbone_args(parser: argparse.ArgumentParser) -> tuple[str, ...]:
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    return ("base_channels", "depth", "dropout")


class RangeUNetBackbone(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32, depth: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if int(depth) < 2:
            raise ValueError("depth must be >= 2")
        channels = [int(base_channels) * (2**i) for i in range(int(depth))]
        self.encoders = nn.ModuleList()
        prev = int(in_channels)
        for ch in channels:
            self.encoders.append(ConvBlock(prev, ch, dropout=dropout))
            prev = ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, dropout=dropout)

        self.decoders = nn.ModuleList()
        dec_in = channels[-1] * 2
        for skip_ch in reversed(channels):
            self.decoders.append(ConvBlock(dec_in + skip_ch, skip_ch, dropout=dropout))
            dec_in = skip_ch

        self.output_channels = channels[0]

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None, t: torch.Tensor | None = None) -> torch.Tensor:
        _ = cond
        if t is not None:
            raise ValueError("RangeUNetBackbone does not support timestep conditioning.")
        skips: list[torch.Tensor] = []
        for idx, block in enumerate(self.encoders):
            x = block(x)
            skips.append(x)
            if idx != len(self.encoders) - 1:
                x = self.pool(x)

        x = self.pool(x)
        x = self.bottleneck(x)
        for block, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


def _build_unet(**kwargs) -> nn.Module:
    return RangeUNetBackbone(
        in_channels=kwargs["input_channels"],
        base_channels=kwargs.get("base_channels", 32),
        depth=kwargs.get("depth", 4),
        dropout=kwargs.get("dropout", 0.0),
    )


UNET_BACKBONE_SPEC: BackboneSpec = make_backbone_spec(
    supported_behaviors=("supervised",),
    build=_build_unet,
    add_args_with_names=add_range_unet_backbone_args,
)
