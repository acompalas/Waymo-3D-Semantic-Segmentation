import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

import lightning as L
from ..models.backbones.point.registry import POINT_BACKBONE_SPECS
from ..models.backbones.range.registry import RANGE_BACKBONE_SPECS


def _parse_scales(raw: str) -> tuple[int, ...]:
    items = [x.strip() for x in str(raw).split(",")]
    values = tuple(int(x) for x in items if x)
    if not values:
        raise ValueError("knn-scales must contain at least one integer")
    return values


def _noop_train_args(_parser: argparse.ArgumentParser) -> None:
    return None


class BackboneSpecLike(Protocol):
    backbone_id: str
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
class ComponentSpec:
    component_id: str
    representation: str
    supported_behaviors: tuple[str, ...]
    add_train_args: Callable[[argparse.ArgumentParser], None] = _noop_train_args


@dataclass(frozen=True)
class ModelSelection:
    representation: str
    backbone: str
    head: str
    behavior: str
    spec: RepresentationSpec

    @property
    def model_id(self) -> str:
        return "__".join([self.representation, self.backbone, self.head, self.behavior])

    def load_from_checkpoint(self, checkpoint_path: str | Path) -> L.LightningModule:
        return self.spec.load_from_checkpoint(checkpoint_path)

    def build_module(self, args: argparse.Namespace) -> L.LightningModule:
        module_cls = _representation_module_cls(self.representation)
        common = dict(
            num_classes=args.num_classes,
            learning_rate=args.lr,
            weight_decay=getattr(args, "weight_decay", 1e-4),
            backbone=self.backbone,
            head=self.head,
            behavior=self.behavior,
            diffusion_steps=getattr(args, "diffusion_steps", 1000),
            validation_prediction_mode=getattr(args, "validation_prediction_mode", "cheap"),
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
    parser.add_argument("--validation-prediction-mode", type=str, default="cheap", choices=["cheap", "full"])


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


def _representation_module_cls(representation: str) -> type[L.LightningModule]:
    if representation == "point_clouds":
        from ..models import PointCloudTaskModel

        return PointCloudTaskModel
    if representation == "range_images":
        from ..models import RangeImageTaskModel

        return RangeImageTaskModel
    raise KeyError(f"Unknown representation '{representation}'.")


def _backbone_component_specs(
    representation: str,
    specs: Mapping[str, BackboneSpecLike],
) -> dict[str, ComponentSpec]:
    return {
        spec.backbone_id: ComponentSpec(
            component_id=spec.backbone_id,
            representation=representation,
            supported_behaviors=spec.supported_behaviors,
            add_train_args=spec.add_args,
        )
        for spec in specs.values()
    }

BACKBONE_REGISTRY: dict[str, ComponentSpec] = {
    **_backbone_component_specs("point_clouds", POINT_BACKBONE_SPECS),
    **_backbone_component_specs("range_images", RANGE_BACKBONE_SPECS),
}

HEAD_REGISTRY: dict[str, ComponentSpec] = {
    "mlp": ComponentSpec("mlp", "point_clouds", ("supervised", "diffusion")),
    "segmentation": ComponentSpec("segmentation", "range_images", ("supervised",)),
    "denoising": ComponentSpec("denoising", "range_images", ("diffusion",)),
}

BEHAVIOR_REGISTRY: dict[str, ComponentSpec] = {
    "supervised": ComponentSpec("supervised", "point_clouds", ("supervised",)),
    "diffusion": ComponentSpec("diffusion", "point_clouds", ("diffusion",), add_diffusion_behavior_args),
}


def _union_component_ids(registry: dict[str, ComponentSpec]) -> list[str]:
    return sorted(registry)


def _filtered_component_ids(
    registry: dict[str, ComponentSpec],
    *,
    representation: str | None,
    behavior: str | None = None,
) -> list[str]:
    values: list[str] = []
    for component_id, spec in registry.items():
        if representation is not None and spec.representation != representation:
            continue
        if behavior is not None and behavior not in spec.supported_behaviors:
            continue
        values.append(component_id)
    return sorted(values)


def representation_choices() -> list[str]:
    return sorted(REPRESENTATION_REGISTRY)


def behavior_choices(_representation: str | None = None) -> list[str]:
    return ["diffusion", "supervised"]


def backbone_choices(representation: str | None = None, behavior: str | None = None) -> list[str]:
    values = _filtered_component_ids(BACKBONE_REGISTRY, representation=representation, behavior=behavior)
    return values or _union_component_ids(BACKBONE_REGISTRY)


def head_choices(representation: str | None = None, behavior: str | None = None) -> list[str]:
    values = _filtered_component_ids(HEAD_REGISTRY, representation=representation, behavior=behavior)
    return values or _union_component_ids(HEAD_REGISTRY)


def get_model_selection(representation: str, backbone: str, head: str, behavior: str) -> ModelSelection:
    try:
        spec = REPRESENTATION_REGISTRY[str(representation)]
    except KeyError as exc:
        raise KeyError(f"Unknown representation '{representation}'. Choices: {sorted(REPRESENTATION_REGISTRY)}") from exc

    try:
        backbone_spec = BACKBONE_REGISTRY[str(backbone)]
        head_spec = HEAD_REGISTRY[str(head)]
    except KeyError as exc:
        raise KeyError("Unknown backbone or head selection.") from exc

    if backbone_spec.representation != spec.representation:
        raise ValueError(f"Backbone '{backbone}' is not valid for representation '{representation}'.")
    if head_spec.representation != spec.representation:
        raise ValueError(f"Head '{head}' is not valid for representation '{representation}'.")
    if behavior not in backbone_spec.supported_behaviors:
        raise ValueError(f"Backbone '{backbone}' does not support behavior '{behavior}'.")
    if behavior not in head_spec.supported_behaviors:
        raise ValueError(f"Head '{head}' does not support behavior '{behavior}'.")
    return ModelSelection(
        representation=spec.representation,
        backbone=str(backbone),
        head=str(head),
        behavior=str(behavior),
        spec=spec,
    )


def maybe_add_component_train_args(
    parser: argparse.ArgumentParser,
    *,
    representation: str | None,
    backbone: str | None,
    head: str | None,
    behavior: str | None,
) -> None:
    if representation is None or backbone is None or head is None or behavior is None:
        return
    selection = get_model_selection(representation, backbone, head, behavior)
    BACKBONE_REGISTRY[selection.backbone].add_train_args(parser)
    if selection.behavior == "diffusion":
        add_diffusion_behavior_args(parser)
