import argparse
from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from .edgeconv import EdgeConvBackbone, add_edgeconv_backbone_args
from .handcrafted import HandcraftedPointBackbone, add_handcrafted_backbone_args
from .pointnet import PointNetBackbone, add_pointnet_backbone_args
from .pointnetplusplus import PointNetPlusPlusBackbone, add_pointnetplusplus_backbone_args


def _noop_backbone_args(_parser: argparse.ArgumentParser) -> None:
    return None


def _build_pointnet(**kwargs) -> nn.Module:
    return PointNetBackbone(
        input_dim=kwargs["input_dim"],
        hidden_dim=kwargs.get("hidden_dim", 256),
        depth=kwargs.get("depth", 6),
        dropout=kwargs.get("dropout", 0.1),
        time_dim=kwargs.get("time_dim"),
    )


def _build_edgeconv(**kwargs) -> nn.Module:
    return EdgeConvBackbone(
        input_dim=kwargs["input_dim"],
        hidden_dim=kwargs.get("hidden_dim", 256),
        depth=kwargs.get("depth", 6),
        dropout=kwargs.get("dropout", 0.1),
        knn_k=kwargs.get("knn_k", 16),
        time_dim=kwargs.get("time_dim"),
    )


def _build_pointnetplusplus(**kwargs) -> nn.Module:
    return PointNetPlusPlusBackbone(
        input_dim=kwargs["input_dim"],
        hidden_dim=kwargs.get("hidden_dim", 256),
        dropout=kwargs.get("dropout", 0.1),
    )


def _build_handcrafted(**kwargs) -> nn.Module:
    return HandcraftedPointBackbone(
        input_dim=kwargs["input_dim"],
        scales=kwargs.get("knn_scales", (16, 32, 64)),
        knn_support_size=kwargs.get("knn_support_size", 16384),
        knn_query_chunk=kwargs.get("knn_query_chunk", 4096),
        hidden_dim=kwargs.get("hidden_dim", 0),
        depth=kwargs.get("depth", 0),
        dropout=kwargs.get("dropout", 0.0),
    )


@dataclass(frozen=True)
class PointBackboneSpec:
    supported_behaviors: tuple[str, ...]
    build: Callable[..., nn.Module]
    add_args: Callable[[argparse.ArgumentParser], None] = _noop_backbone_args


POINT_BACKBONE_SPECS: dict[str, PointBackboneSpec] = {
    "pointnet": PointBackboneSpec(
        supported_behaviors=("supervised", "diffusion"),
        build=_build_pointnet,
        add_args=add_pointnet_backbone_args,
    ),
    "edgeconv": PointBackboneSpec(
        supported_behaviors=("supervised", "diffusion"),
        build=_build_edgeconv,
        add_args=add_edgeconv_backbone_args,
    ),
    "pointnetplusplus": PointBackboneSpec(
        supported_behaviors=("supervised",),
        build=_build_pointnetplusplus,
        add_args=add_pointnetplusplus_backbone_args,
    ),
    "handcrafted": PointBackboneSpec(
        supported_behaviors=("supervised",),
        build=_build_handcrafted,
        add_args=add_handcrafted_backbone_args,
    ),
}


def build_point_backbone(backbone: str, **kwargs) -> nn.Module:
    key = str(backbone).lower()
    try:
        return POINT_BACKBONE_SPECS[key].build(**kwargs)
    except KeyError as exc:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Choose from: {', '.join(sorted(POINT_BACKBONE_SPECS))}."
        ) from exc
