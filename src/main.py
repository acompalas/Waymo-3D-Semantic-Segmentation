import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Iterable
import uuid

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import numpy as np
import torch

from .data import PreprocessedPointCloudDataset, PreprocessedRangeImageDataset, WaymoLidarDataModule
from .runtime.common import resolve_runtime_device
from .runtime.registry import (
    backbone_choices,
    behavior_choices,
    get_model_selection,
    maybe_add_component_train_args,
    representation_choices,
)
from .runtime.report_runner import evaluate_and_log_splits, parse_report_splits, prepare_model_for_reporting, setup_datamodule_for_report
from .runtime.wandb_logging import WandbSegmentationCallback
from .tools import label_colors, render_point_cloud, save_gif


def add_torch_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--float32-matmul-precision",
        type=str,
        default="medium",
        choices=("highest", "high", "medium"),
    )


def add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--point-data-dir", type=Path, default=Path("data/preprocessed/point_clouds"))
    parser.add_argument("--range-data-dir", type=Path, default=Path("data/preprocessed/range_images"))
    parser.add_argument("--train-subdirs", type=str, default="training")
    parser.add_argument("--val-subdirs", type=str, default="")
    parser.add_argument("--test-subdirs", type=str, default="validation")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=16384)
    parser.add_argument("--num-classes", type=int, default=23)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--worker-start-method", type=str, default="spawn")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-alpha", type=float, default=1.0)
    parser.add_argument("--focal-loss-gamma", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--train-segment-fraction", type=float, default=1.0)
    parser.add_argument("--train-frame-fraction", type=float, default=1.0)
    parser.add_argument("--max-cached-segments", type=int, default=2)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--wandb-project", type=str, default="ece271b-final-project")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--log-pointcloud-count", type=int, default=4)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--val-samples-per-segment", type=int, default=None)
    parser.add_argument("--auto-evaluate", dest="auto_evaluate", action="store_true")
    parser.add_argument("--no-auto-evaluate", dest="auto_evaluate", action="store_false")
    parser.set_defaults(auto_evaluate=True)
    add_torch_runtime_args(parser)


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--point-data-dir", type=Path, default=Path("data/preprocessed/point_clouds"))
    parser.add_argument("--range-data-dir", type=Path, default=Path("data/preprocessed/range_images"))
    parser.add_argument("--test-subdirs", type=str, default="validation")
    parser.add_argument("--splits", type=str, default="test")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=16384)
    parser.add_argument("--num-classes", type=int, default=23)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--worker-start-method", type=str, default="spawn")
    parser.add_argument("--max-cached-segments", type=int, default=2)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("output/eval"))
    parser.add_argument("--wandb-project", type=str, default="ece271b-final-project")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--log-pointcloud-count", type=int, default=4)
    add_torch_runtime_args(parser)


def add_common_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--point-data-dir", type=Path, default=Path("data/preprocessed/point_clouds"))
    parser.add_argument("--range-data-dir", type=Path, default=Path("data/preprocessed/range_images"))
    parser.add_argument("--source-subdirs", type=str, default="validation")
    parser.add_argument("--segment", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-num-points", type=int, default=16384)
    parser.add_argument("--output-gif", type=Path, default=Path("output/render/predictions.gif"))
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--camera-preset", type=str, default="car_pov", choices=["car_pov", "top", "topdown"])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--point-size", type=float, default=2.5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    add_torch_runtime_args(parser)


def add_component_args(
    parser: argparse.ArgumentParser,
    *,
    representation: str | None = None,
    behavior: str | None = None,
) -> None:
    parser.add_argument("--representation", type=str, required=True, choices=representation_choices())
    parser.add_argument("--behavior", type=str, required=True, choices=behavior_choices(representation))
    parser.add_argument("--backbone", type=str, required=True, choices=backbone_choices(representation, behavior))


def build_parser(
    *,
    command: str | None = None,
    representation: str | None = None,
    behavior: str | None = None,
    backbone: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Componentized LiDAR semantic segmentation CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("train", "evaluate", "render"):
        sub = subparsers.add_parser(name)
        add_component_args(sub, representation=representation if command == name else None, behavior=behavior if command == name else None)
        if name == "train":
            add_common_train_args(sub)
            if command == "train":
                maybe_add_component_train_args(
                    sub,
                    representation=representation,
                    behavior=behavior,
                    backbone=backbone,
                )
        elif name == "evaluate":
            add_common_eval_args(sub)
        else:
            add_common_render_args(sub)

    return parser


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    base_parser = build_parser()
    known, _ = base_parser.parse_known_args(argv)
    full_parser = build_parser(
        command=getattr(known, "command", None),
        representation=getattr(known, "representation", None),
        behavior=getattr(known, "behavior", None),
        backbone=getattr(known, "backbone", None),
    )
    return full_parser.parse_args(argv)


def selection_for(args: argparse.Namespace):
    return get_model_selection(args.representation, args.backbone, args.behavior)


def point_data_dir_for(args: argparse.Namespace) -> Path:
    return Path(args.point_data_dir)


def range_data_dir_for(args: argparse.Namespace) -> Path:
    return Path(args.range_data_dir)


def data_dir_for(args: argparse.Namespace) -> Path:
    selection = selection_for(args)
    if selection.representation == "point_clouds":
        return point_data_dir_for(args)
    return range_data_dir_for(args)


def _wandb_tags(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part for part in (item.strip() for item in str(raw).split(",")) if part]


def _default_wandb_name_prefix(selection) -> str:
    return f"{selection.representation}-{selection.backbone}"


def _generate_wandb_run_id() -> str:
    return uuid.uuid4().hex[:8]


def _wandb_name_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _default_wandb_run_name(selection, timestamp: str) -> str:
    return f"{_default_wandb_name_prefix(selection)}-{str(timestamp)}"


def build_wandb_logger(
    args: argparse.Namespace,
    selection,
    *,
    job_type: str,
) -> WandbLogger:
    run_id = _generate_wandb_run_id()
    run_name = args.wandb_run_name or _default_wandb_run_name(selection, _wandb_name_timestamp())
    return WandbLogger(
        project=str(args.wandb_project),
        entity=args.wandb_entity,
        name=run_name,
        save_dir=str(args.output_dir),
        id=run_id,
        tags=_wandb_tags(args.wandb_tags),
        job_type=str(job_type),
        log_model=False,
    )


def _serialize_cli_arg_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_cli_arg_value(item) for item in value]
    return str(value)


def build_cli_hparams_payload(
    args: argparse.Namespace,
) -> dict[str, object]:
    return {key: _serialize_cli_arg_value(value) for key, value in vars(args).items()}


def log_cli_hyperparams(
    logger,
    args: argparse.Namespace,
) -> None:
    logger.log_hyperparams(build_cli_hparams_payload(args))


def configure_torch_runtime(args: argparse.Namespace) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(getattr(args, "float32_matmul_precision", "medium")))


def build_datamodule(args: argparse.Namespace) -> WaymoLidarDataModule:
    selection = selection_for(args)
    val_samples_per_segment = getattr(args, "val_samples_per_segment", None)
    if val_samples_per_segment is None:
        val_samples_per_segment = 1 if selection.behavior == "diffusion" else 0
    return WaymoLidarDataModule(
        data_dir=data_dir_for(args),
        point_data_dir=point_data_dir_for(args),
        range_data_dir=range_data_dir_for(args),
        representation=selection.representation,
        batch_size=args.batch_size,
        num_points=args.num_points,
        num_classes=args.num_classes,
        train_subdirs=getattr(args, "train_subdirs", "training"),
        val_subdirs=getattr(args, "val_subdirs", ""),
        test_subdirs=getattr(args, "test_subdirs", "validation"),
        val_fraction=getattr(args, "val_fraction", 0.1),
        num_workers=args.num_workers,
        max_cached_segments=getattr(args, "max_cached_segments", 2),
        seed=args.seed,
        worker_start_method=args.worker_start_method,
        class_weight_alpha=float(getattr(args, "class_weight_alpha", 1.0)),
        train_segment_fraction=getattr(args, "train_segment_fraction", 1.0),
        train_frame_fraction=getattr(args, "train_frame_fraction", 1.0),
        val_samples_per_segment=int(val_samples_per_segment),
    )


def configure_trainer(args: argparse.Namespace, logger, callbacks: list) -> L.Trainer:
    return L.Trainer(
        default_root_dir=str(getattr(args, "output_dir", Path("output"))),
        max_epochs=getattr(args, "max_epochs", 1),
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_autolog_hparams=False,
    )


def _validate_loaded_model_selection(model, selection) -> None:
    expected = {
        "backbone": selection.backbone,
        "behavior": selection.behavior,
    }
    for key, value in expected.items():
        loaded = str(getattr(model.hparams, key))
        if loaded != value:
            raise ValueError(f"Checkpoint {key}='{loaded}' does not match CLI selection '{value}'.")


def run_train(args: argparse.Namespace) -> None:
    selection = selection_for(args)
    L.seed_everything(args.seed, workers=True)
    datamodule = build_datamodule(args)
    model = selection.build_module(args)

    logger = build_wandb_logger(args, selection, job_type="train")
    log_cli_hyperparams(logger, args)
    run_dir = Path(logger.save_dir) / selection.model_id / str(getattr(logger, "version", "run"))
    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        monitor="valid_metrics/valid_miou",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:02d}-{valid_metrics/valid_miou:.4f}",
        auto_insert_metric_name=False,
    )
    early_stopping_cb = EarlyStopping(
        monitor="valid_metrics/valid_miou",
        mode="max",
        patience=int(args.early_stopping_patience),
        min_delta=float(args.early_stopping_min_delta),
    )
    wandb_cb = WandbSegmentationCallback(log_pointcloud_count=int(args.log_pointcloud_count))
    trainer = configure_trainer(args, logger, [checkpoint_cb, early_stopping_cb, wandb_cb])
    trainer.fit(model=model, datamodule=datamodule)

    if not bool(args.auto_evaluate):
        logger.experiment.finish()
        return

    checkpoint_path = Path(checkpoint_cb.best_model_path or checkpoint_cb.last_model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Unable to locate checkpoint after training: {checkpoint_path}")

    setup_datamodule_for_report(datamodule, ("train", "val", "test"))
    report_model = selection.load_from_checkpoint(checkpoint_path)
    report_device = resolve_runtime_device(args.accelerator)
    prepare_model_for_reporting(report_model, datamodule, report_device)
    final_eval_epoch = max(0, int(trainer.current_epoch) - 1)
    evaluate_and_log_splits(
        logger,
        report_model,
        datamodule,
        splits=("train", "val", "test"),
        device=report_device,
        class_names=report_model.class_names,
        step=trainer.global_step,
        epoch=final_eval_epoch,
        audit_pointcloud_count=int(args.log_pointcloud_count),
    )
    logger.experiment.finish()


def run_evaluate(args: argparse.Namespace) -> None:
    selection = selection_for(args)
    L.seed_everything(args.seed, workers=True)
    datamodule = build_datamodule(args)
    model = selection.load_from_checkpoint(args.checkpoint)
    _validate_loaded_model_selection(model, selection)
    logger = build_wandb_logger(args, selection, job_type="evaluate")
    log_cli_hyperparams(logger, args)
    requested_splits = parse_report_splits(args.splits)
    setup_datamodule_for_report(datamodule, requested_splits)
    report_device = resolve_runtime_device(args.accelerator)
    prepare_model_for_reporting(model, datamodule, report_device)
    evaluate_and_log_splits(
        logger=logger,
        model=model,
        datamodule=datamodule,
        splits=requested_splits,
        device=report_device,
        class_names=model.class_names,
        step=0,
        epoch=0,
        max_batches=args.max_batches,
        audit_pointcloud_count=int(args.log_pointcloud_count),
    )
    logger.experiment.finish()


def _choose_segment(args: argparse.Namespace, datasets: list) -> str:
    segment_sets = [set(ds.segment_names) for ds in datasets]
    available = sorted(set.intersection(*segment_sets)) if segment_sets else []
    if args.segment is not None:
        if args.segment not in available:
            raise ValueError(f"Requested segment not available in selection: {args.segment}")
        return str(args.segment)
    if not available:
        raise RuntimeError("No overlapping segments available for rendering.")
    rng = np.random.default_rng(args.seed)
    return str(rng.choice(np.asarray(available, dtype=object)))


def _sampled_point_frame(
    dataset: PreprocessedPointCloudDataset,
    *,
    segment: str,
    timestamp: int,
) -> dict:
    dataset_idx = dataset.resolve_dataset_index(segment, timestamp)
    sampled = dataset[dataset_idx]
    return {
        "xyz": sampled["points"].detach().cpu().numpy().astype(np.float32, copy=False),
        "point_features": sampled["point_features"].detach().cpu().numpy().astype(np.float32, copy=False),
        "labels": sampled["labels"].detach().cpu().numpy().astype(np.int64, copy=False),
        "valid_label": sampled["valid_label"].detach().cpu().numpy().astype(bool, copy=False),
        "segment_context_name": str(sampled["segment_context_name"]),
        "frame_timestamp_micros": int(sampled["frame_timestamp_micros"]),
    }


def run_render(args: argparse.Namespace) -> None:
    selection = selection_for(args)
    device = resolve_runtime_device(args.device)
    model = selection.load_from_checkpoint(args.checkpoint).to(device)
    _validate_loaded_model_selection(model, selection)
    model.eval()

    source_subdirs = args.source_subdirs
    point_dataset = PreprocessedPointCloudDataset(
        path=point_data_dir_for(args),
        source_subdirs=source_subdirs,
        num_points=max(1, int(args.render_num_points)),
        deterministic_sampling=True,
        seed=args.seed,
        max_cached_segments=1,
    )
    datasets = [point_dataset]
    range_dataset = None
    if selection.representation == "range_images":
        range_dataset = PreprocessedRangeImageDataset(
            path=range_data_dir_for(args),
            source_subdirs=source_subdirs,
            seed=args.seed,
            max_cached_segments=1,
        )
        datasets.append(range_dataset)
    segment = _choose_segment(args, datasets)
    frames = point_dataset.frames_for_segment(segment)
    if args.max_frames > 0:
        frames = frames[: int(args.max_frames)]

    rendered_frames: list[np.ndarray] = []
    for _, (_, timestamp) in enumerate(frames):
        if selection.representation == "point_clouds":
            point_frame = _sampled_point_frame(point_dataset, segment=segment, timestamp=timestamp)
        else:
            point_frame = point_dataset.get_dense_frame(segment, timestamp)
        range_frame = range_dataset.get_frame_data(segment, timestamp) if range_dataset is not None else None
        prediction = model.predict_segmented_pointcloud(
            point_frame=point_frame,
            range_frame=range_frame,
        )

        colors = label_colors(prediction["pred_labels"], valid_label=prediction["valid_label"])
        rendered_frames.append(
            render_point_cloud(
                prediction["points_xyz"],
                colors,
                width=args.width,
                height=args.height,
                point_size=args.point_size,
                camera_preset=args.camera_preset,
            )
        )

    save_gif(rendered_frames, args.output_gif, args.fps)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    configure_torch_runtime(args)
    if args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        run_render(args)


if __name__ == "__main__":
    main()
