import argparse
from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from .crossattn_unet import RangeDiffusionBackbone, add_range_crossattn_backbone_args
from .unet import RangeUNetBackbone, add_range_unet_backbone_args


def noop_backbone_args(_parser: argparse.ArgumentParser) -> None:
    return None


@dataclass(frozen=True)
class RangeBackboneSpec:
    backbone_id: str
    supported_behaviors: tuple[str, ...]
    build: Callable[..., nn.Module]
    add_args: Callable[[argparse.ArgumentParser], None] = noop_backbone_args


RANGE_BACKBONE_SPECS: dict[str, RangeBackboneSpec] = {
    "unet": RangeBackboneSpec(
        backbone_id="unet",
        supported_behaviors=("supervised",),
        build=lambda **kwargs: RangeUNetBackbone(
            in_channels=kwargs["input_channels"],
            base_channels=kwargs.get("base_channels", 32),
            depth=kwargs.get("depth", 4),
            dropout=kwargs.get("dropout", 0.0),
        ),
        add_args=add_range_unet_backbone_args,
    ),
    "crossattn_unet": RangeBackboneSpec(
        backbone_id="crossattn_unet",
        supported_behaviors=("diffusion",),
        build=lambda **kwargs: RangeDiffusionBackbone(
            num_classes=kwargs["num_classes"],
            cond_channels=kwargs["input_channels"],
            base_channels=kwargs.get("base_channels", 32),
            dropout=kwargs.get("dropout", 0.1),
        ),
        add_args=add_range_crossattn_backbone_args,
    ),
}


def build_range_backbone(backbone: str, **kwargs) -> nn.Module:
    key = str(backbone).lower()
    try:
        spec = RANGE_BACKBONE_SPECS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Choose from: {', '.join(sorted(RANGE_BACKBONE_SPECS))}."
        ) from exc
    return spec.build(**kwargs)
