import argparse
from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from .crossattn_unet import RangeDiffusionBackbone, add_range_crossattn_backbone_args
from .unet import RangeUNetBackbone, add_range_unet_backbone_args


def _noop_backbone_args(_parser: argparse.ArgumentParser) -> None:
    return None


def _build_unet(**kwargs) -> nn.Module:
    return RangeUNetBackbone(
        in_channels=kwargs["input_channels"],
        base_channels=kwargs.get("base_channels", 32),
        depth=kwargs.get("depth", 4),
        dropout=kwargs.get("dropout", 0.0),
    )


def _build_crossattn_unet(**kwargs) -> nn.Module:
    return RangeDiffusionBackbone(
        num_classes=kwargs["num_classes"],
        cond_channels=kwargs["input_channels"],
        base_channels=kwargs.get("base_channels", 32),
        dropout=kwargs.get("dropout", 0.1),
    )


@dataclass(frozen=True)
class RangeBackboneSpec:
    supported_behaviors: tuple[str, ...]
    build: Callable[..., nn.Module]
    add_args: Callable[[argparse.ArgumentParser], None] = _noop_backbone_args


RANGE_BACKBONE_SPECS: dict[str, RangeBackboneSpec] = {
    "unet": RangeBackboneSpec(
        supported_behaviors=("supervised",),
        build=_build_unet,
        add_args=add_range_unet_backbone_args,
    ),
    "crossattn_unet": RangeBackboneSpec(
        supported_behaviors=("diffusion",),
        build=_build_crossattn_unet,
        add_args=add_range_crossattn_backbone_args,
    ),
}


def build_range_backbone(backbone: str, **kwargs) -> nn.Module:
    key = str(backbone).lower()
    try:
        return RANGE_BACKBONE_SPECS[key].build(**kwargs)
    except KeyError as exc:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Choose from: {', '.join(sorted(RANGE_BACKBONE_SPECS))}."
        ) from exc
