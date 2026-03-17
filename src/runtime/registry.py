import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import lightning as L

from ..models.backbones.point.registry import POINT_BACKBONE_SPECS
from ..models.backbones.range.registry import RANGE_BACKBONE_SPECS


def _parse_scales(raw: str) -> tuple[int, ...]:
    items = [x.strip() for x in str(raw).split(",")]
    values = tuple(int(x) for x in items if x)
    if not values:
        raise ValueError("knn-scales must contain at least one integer")
    return values


class BackboneSpecLike(Protocol):
    supported_behaviors: tuple[str, ...]
    add_args: Callable[[argparse.ArgumentParser], None]


@dataclass(frozen=True)
class RepresentationSpec:
    representation: str
    default_data_dir: Path

    @property
    def module_cls(self) -> type[L.LightningModule]:
        return _representation_module_cls(self.representation)

    def load_from_checkpoint(self, checkpoint_path: str | Path) -> L.LightningModule:
        return self.module_cls.load_from_checkpoint(str(checkpoint_path))


@dataclass(frozen=True)
class ModelSelection:
    representation: str
    backbone: str
    behavior: str
    spec: RepresentationSpec

    @property
    def model_id(self) -> str:
        return "__".join([self.representation, self.backbone, self.behavior])

    def load_from_checkpoint(self, checkpoint_path: str | Path) -> L.LightningModule:
        return self.spec.load_from_checkpoint(checkpoint_path)

    def build_module(self, args: argparse.Namespace) -> L.LightningModule:
        module_cls = _representation_module_cls(self.representation)
        common = dict(
            num_classes=args.num_classes,
            learning_rate=args.lr,
            weight_decay=getattr(args, "weight_decay", 1e-4),
            backbone=self.backbone,
            behavior=self.behavior,
            diffusion_steps=getattr(args, "diffusion_steps", 1000),
            use_balanced_class_weights=not bool(args.no_balanced_class_weights),
            geometry_only=bool(args.geometry_only),
        )
        if self.representation == "point_clouds":
            return module_cls(
                **common,
                hidden_dim=getattr(args, "hidden_dim", 256),
                depth=getattr(args, "depth", 6),
                knn_k=getattr(args, "knn_k", 16),
                dropout=getattr(args, "dropout", 0.1),
                knn_scales=_parse_scales(getattr(args, "knn_scales", "16,32,64")),
                knn_support_size=getattr(args, "knn_support_size", 16384),
                knn_query_chunk=getattr(args, "knn_query_chunk", 4096),
                proj_dim=getattr(args, "proj_dim", 0),
                proj_depth=getattr(args, "proj_depth", 0),
                proj_dropout=getattr(args, "proj_dropout", 0.0),
            )
        return module_cls(
            **common,
            base_channels=getattr(args, "base_channels", 32),
            depth=getattr(args, "depth", 4),
            dropout=getattr(args, "dropout", 0.0),
        )


def add_diffusion_behavior_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--diffusion-steps", type=int, default=1000)


REPRESENTATION_REGISTRY: dict[str, RepresentationSpec] = {
    "point_clouds": RepresentationSpec(
        representation="point_clouds",
        default_data_dir=Path("data/preprocessed/point_clouds"),
    ),
    "range_images": RepresentationSpec(
        representation="range_images",
        default_data_dir=Path("data/preprocessed/range_images"),
    ),
}

BACKBONE_REGISTRY: dict[str, dict[str, BackboneSpecLike]] = {
    "point_clouds": POINT_BACKBONE_SPECS,
    "range_images": RANGE_BACKBONE_SPECS,
}


def _representation_module_cls(representation: str) -> type[L.LightningModule]:
    if representation == "point_clouds":
        from ..models import PointCloudTaskModel

        return PointCloudTaskModel
    if representation == "range_images":
        from ..models import RangeImageTaskModel

        return RangeImageTaskModel
    raise KeyError(f"Unknown representation '{representation}'.")


def _backbone_spec(representation: str, backbone: str) -> BackboneSpecLike:
    try:
        return BACKBONE_REGISTRY[str(representation)][str(backbone)]
    except KeyError as exc:
        raise KeyError(f"Unknown backbone '{backbone}' for representation '{representation}'.") from exc


def representation_choices() -> list[str]:
    return sorted(REPRESENTATION_REGISTRY)


def behavior_choices(_representation: str | None = None) -> list[str]:
    return ["diffusion", "supervised"]


def backbone_choices(representation: str | None = None, behavior: str | None = None) -> list[str]:
    values: list[str] = []
    representations = [representation] if representation is not None else list(BACKBONE_REGISTRY)
    for rep in representations:
        for name, spec in BACKBONE_REGISTRY[str(rep)].items():
            if behavior is not None and behavior not in spec.supported_behaviors:
                continue
            values.append(name)
    return sorted(set(values))


def get_model_selection(representation: str, backbone: str, behavior: str) -> ModelSelection:
    try:
        spec = REPRESENTATION_REGISTRY[str(representation)]
    except KeyError as exc:
        raise KeyError(f"Unknown representation '{representation}'. Choices: {sorted(REPRESENTATION_REGISTRY)}") from exc

    backbone_spec = _backbone_spec(spec.representation, backbone)
    if behavior not in behavior_choices(spec.representation):
        raise ValueError(f"Unsupported behavior '{behavior}'.")
    if behavior not in backbone_spec.supported_behaviors:
        raise ValueError(f"Backbone '{backbone}' does not support behavior '{behavior}'.")
    return ModelSelection(
        representation=spec.representation,
        backbone=str(backbone),
        behavior=str(behavior),
        spec=spec,
    )


def maybe_add_component_train_args(
    parser: argparse.ArgumentParser,
    *,
    representation: str | None,
    backbone: str | None,
    behavior: str | None,
) -> None:
    if representation is None or backbone is None or behavior is None:
        return
    selection = get_model_selection(representation, backbone, behavior)
    _backbone_spec(selection.representation, selection.backbone).add_args(parser)
    if selection.behavior == "diffusion":
        add_diffusion_behavior_args(parser)
