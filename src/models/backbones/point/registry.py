import argparse
from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from .edgeconv import EdgeConvBackbone, add_edgeconv_backbone_args
from .handcrafted import HandcraftedPointBackbone, add_handcrafted_backbone_args
from .pointnet import PointNetBackbone, add_pointnet_backbone_args
from .pointnetplusplus import PointNetPlusPlusBackbone, add_pointnetplusplus_backbone_args


def noop_backbone_args(_parser: argparse.ArgumentParser) -> None:
    return None


@dataclass(frozen=True)
class PointBackboneSpec:
    backbone_id: str
    supported_behaviors: tuple[str, ...]
    build: Callable[..., nn.Module]
    add_args: Callable[[argparse.ArgumentParser], None] = noop_backbone_args


POINT_BACKBONE_SPECS: dict[str, PointBackboneSpec] = {
    "pointnet": PointBackboneSpec(
        backbone_id="pointnet",
        supported_behaviors=("supervised", "diffusion"),
        build=lambda **kwargs: PointNetBackbone(
            input_dim=kwargs["input_dim"],
            hidden_dim=kwargs.get("hidden_dim", 256),
            depth=kwargs.get("depth", 6),
            dropout=kwargs.get("dropout", 0.1),
            time_dim=kwargs.get("time_dim"),
        ),
        add_args=add_pointnet_backbone_args,
    ),
    "edgeconv": PointBackboneSpec(
        backbone_id="edgeconv",
        supported_behaviors=("supervised", "diffusion"),
        build=lambda **kwargs: EdgeConvBackbone(
            input_dim=kwargs["input_dim"],
            hidden_dim=kwargs.get("hidden_dim", 256),
            depth=kwargs.get("depth", 6),
            dropout=kwargs.get("dropout", 0.1),
            knn_k=kwargs.get("knn_k", 16),
            time_dim=kwargs.get("time_dim"),
        ),
        add_args=add_edgeconv_backbone_args,
    ),
    "pointnetplusplus": PointBackboneSpec(
        backbone_id="pointnetplusplus",
        supported_behaviors=("supervised",),
        build=lambda **kwargs: PointNetPlusPlusBackbone(
            input_dim=kwargs["input_dim"],
            hidden_dim=kwargs.get("hidden_dim", 256),
            dropout=kwargs.get("dropout", 0.1),
        ),
        add_args=add_pointnetplusplus_backbone_args,
    ),
    "handcrafted": PointBackboneSpec(
        backbone_id="handcrafted",
        supported_behaviors=("supervised",),
        build=lambda **kwargs: HandcraftedPointBackbone(
            input_dim=kwargs["input_dim"],
            scales=kwargs.get("knn_scales", (16, 32, 64)),
            knn_support_size=kwargs.get("knn_support_size", 16384),
            knn_query_chunk=kwargs.get("knn_query_chunk", 4096),
            proj_dim=kwargs.get("proj_dim", 0),
            proj_depth=kwargs.get("proj_depth", 0),
            proj_dropout=kwargs.get("proj_dropout", 0.0),
        ),
        add_args=add_handcrafted_backbone_args,
    ),
}


def build_point_backbone(backbone: str, **kwargs) -> nn.Module:
    key = str(backbone).lower()
    try:
        spec = POINT_BACKBONE_SPECS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Choose from: {', '.join(sorted(POINT_BACKBONE_SPECS))}."
        ) from exc
    return spec.build(**kwargs)
