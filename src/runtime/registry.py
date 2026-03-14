import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import lightning as L

from ..models import (
    LinearSVMPointClassifier,
    PointCloudDiffusionSegmenter,
    PointCloudSupervisedSegmenter,
    RangeImageDiffusionSegmenter,
    RangeImageUNetSegmenter,
)


def _parse_scales(raw: str) -> tuple[int, ...]:
    items = [x.strip() for x in str(raw).split(",")]
    values = tuple(int(x) for x in items if x)
    if not values:
        raise ValueError("knn-scales must contain at least one integer")
    return values


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    representation: str
    default_data_dir: Path
    module_cls: type[L.LightningModule]
    add_train_args: Callable[[argparse.ArgumentParser], None]
    build_module: Callable[[argparse.Namespace], L.LightningModule]

    def load_from_checkpoint(self, checkpoint_path: str | Path) -> L.LightningModule:
        return self.module_cls.load_from_checkpoint(str(checkpoint_path))


def add_point_svm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--svm-reg", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--knn-scales", type=str, default="16,32,64")
    parser.add_argument("--knn-support-size", type=int, default=16384)
    parser.add_argument("--knn-query-chunk", type=int, default=4096)
    parser.add_argument("--geometry-only", action="store_true")


def add_range_unet_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--unet-base-channels", type=int, default=32)
    parser.add_argument("--unet-depth", type=int, default=4)
    parser.add_argument("--unet-dropout", type=float, default=0.0)
    parser.add_argument("--geometry-only", action="store_true")


def add_range_diffusion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--validation-prediction-mode", type=str, default="cheap", choices=["cheap", "full"])
    parser.add_argument("--geometry-only", action="store_true")


def add_point_diffusion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--backbone", type=str, default="edgeconv", choices=["edgeconv", "pointnet"])
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--validation-prediction-mode", type=str, default="cheap", choices=["cheap", "full"])
    parser.add_argument("--geometry-only", action="store_true")


def add_point_supervised_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--backbone", type=str, default="edgeconv", choices=["edgeconv", "pointnet"])
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--geometry-only", action="store_true")


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "point_svm": ModelSpec(
        model_id="point_svm",
        representation="point_clouds",
        default_data_dir=Path("data/preprocessed/point_clouds"),
        module_cls=LinearSVMPointClassifier,
        add_train_args=add_point_svm_args,
        build_module=lambda args: LinearSVMPointClassifier(
            num_classes=args.num_classes,
            learning_rate=args.lr,
            svm_reg=args.svm_reg,
            margin=args.margin,
            knn_scales=_parse_scales(args.knn_scales),
            knn_support_size=args.knn_support_size,
            knn_query_chunk=args.knn_query_chunk,
            use_balanced_class_weights=not bool(args.no_balanced_class_weights),
            geometry_only=args.geometry_only,
        ),
    ),
    "range_unet": ModelSpec(
        model_id="range_unet",
        representation="range_images",
        default_data_dir=Path("data/preprocessed/range_images"),
        module_cls=RangeImageUNetSegmenter,
        add_train_args=add_range_unet_args,
        build_module=lambda args: RangeImageUNetSegmenter(
            num_classes=args.num_classes,
            base_channels=args.unet_base_channels,
            depth=args.unet_depth,
            dropout=args.unet_dropout,
            learning_rate=args.lr,
            use_balanced_class_weights=not bool(args.no_balanced_class_weights),
            geometry_only=args.geometry_only,
        ),
    ),
    "range_diffusion": ModelSpec(
        model_id="range_diffusion",
        representation="range_images",
        default_data_dir=Path("data/preprocessed/range_images"),
        module_cls=RangeImageDiffusionSegmenter,
        add_train_args=add_range_diffusion_args,
        build_module=lambda args: RangeImageDiffusionSegmenter(
            num_classes=args.num_classes,
            base_channels=args.base_channels,
            learning_rate=args.lr,
            diffusion_steps=args.diffusion_steps,
            use_balanced_class_weights=not bool(args.no_balanced_class_weights),
            validation_prediction_mode=args.validation_prediction_mode,
            geometry_only=args.geometry_only,
        ),
    ),
    "point_diffusion": ModelSpec(
        model_id="point_diffusion",
        representation="point_clouds",
        default_data_dir=Path("data/preprocessed/point_clouds"),
        module_cls=PointCloudDiffusionSegmenter,
        add_train_args=add_point_diffusion_args,
        build_module=lambda args: PointCloudDiffusionSegmenter(
            num_classes=args.num_classes,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            diffusion_steps=args.diffusion_steps,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            backbone=args.backbone,
            knn_k=args.knn_k,
            use_balanced_class_weights=not bool(args.no_balanced_class_weights),
            validation_prediction_mode=args.validation_prediction_mode,
            geometry_only=args.geometry_only,
        ),
    ),
    "point_supervised": ModelSpec(
        model_id="point_supervised",
        representation="point_clouds",
        default_data_dir=Path("data/preprocessed/point_clouds"),
        module_cls=PointCloudSupervisedSegmenter,
        add_train_args=add_point_supervised_args,
        build_module=lambda args: PointCloudSupervisedSegmenter(
            num_classes=args.num_classes,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            backbone=args.backbone,
            knn_k=args.knn_k,
            use_balanced_class_weights=not bool(args.no_balanced_class_weights),
            geometry_only=args.geometry_only,
        ),
    ),
}


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[str(model_id)]
    except KeyError as exc:
        raise KeyError(f"Unknown model '{model_id}'. Choices: {sorted(MODEL_REGISTRY)}") from exc
