import argparse
from pathlib import Path
import sys
from typing import Iterable

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import torch

from .data import PreprocessedPointCloudDataset, PreprocessedRangeImageDataset, WaymoLidarDataModule
from .runtime import (
    backbone_choices,
    behavior_choices,
    collect_stage_reports,
    get_model_selection,
    head_choices,
    log_final_report_metrics,
    maybe_add_component_train_args,
    parse_report_splits,
    prepare_model_for_reporting,
    representation_choices,
    resolve_runtime_device,
    setup_datamodule_for_report,
    write_report_bundle,
)
from .tools import label_colors, render_point_cloud, save_gif


def resolve_device(device_arg: str) -> torch.device:
    key = str(device_arg).lower()
    if key == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return torch.device("mps" if has_mps else "cpu")
    if key == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=None)
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
    parser.add_argument("--no-balanced-class-weights", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--train-segment-fraction", type=float, default=1.0)
    parser.add_argument("--max-cached-segments", type=int, default=2)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--val-samples-per-segment", type=int, default=None)
    parser.add_argument("--auto-evaluate", dest="auto_evaluate", action="store_true")
    parser.add_argument("--no-auto-evaluate", dest="auto_evaluate", action="store_false")
    parser.set_defaults(auto_evaluate=True)


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
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
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("output/eval"))


def add_common_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--point-data-dir", type=Path, default=Path("data/preprocessed/point_clouds"))
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


def add_component_args(
    parser: argparse.ArgumentParser,
    *,
    representation: str | None = None,
    behavior: str | None = None,
) -> None:
    parser.add_argument("--representation", type=str, required=True, choices=representation_choices())
    parser.add_argument("--behavior", type=str, required=True, choices=behavior_choices(representation))
    parser.add_argument("--backbone", type=str, required=True, choices=backbone_choices(representation, behavior))
    parser.add_argument("--head", type=str, required=True, choices=head_choices(representation, behavior))


def build_parser(
    *,
    command: str | None = None,
    representation: str | None = None,
    behavior: str | None = None,
    backbone: str | None = None,
    head: str | None = None,
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
                    head=head,
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
        head=getattr(known, "head", None),
    )
    return full_parser.parse_args(argv)


def selection_for(args: argparse.Namespace):
    return get_model_selection(args.representation, args.backbone, args.head, args.behavior)


def data_dir_for(args: argparse.Namespace) -> Path:
    selection = selection_for(args)
    return Path(args.data_dir) if args.data_dir is not None else selection.spec.default_data_dir


def build_datamodule(args: argparse.Namespace) -> WaymoLidarDataModule:
    selection = selection_for(args)
    val_samples_per_segment = getattr(args, "val_samples_per_segment", None)
    if val_samples_per_segment is None:
        val_samples_per_segment = 1 if selection.behavior == "diffusion" else 0
    return WaymoLidarDataModule(
        data_dir=data_dir_for(args),
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
        balanced_weights=not bool(getattr(args, "no_balanced_class_weights", False)),
        train_segment_fraction=getattr(args, "train_segment_fraction", 1.0),
        val_samples_per_segment=int(val_samples_per_segment),
    )


def configure_trainer(args: argparse.Namespace, logger: CSVLogger | bool, callbacks: list) -> L.Trainer:
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
    )


def _validate_loaded_model_selection(model, selection) -> None:
    expected = {
        "backbone": selection.backbone,
        "head": selection.head,
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

    logger = CSVLogger(save_dir=str(args.output_dir), name=selection.model_id)
    run_dir = Path(logger.log_dir)
    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        monitor="val_mIoU",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:02d}-{val_mIoU:.4f}",
    )
    early_stopping_cb = EarlyStopping(
        monitor="val_mIoU",
        mode="max",
        patience=int(args.early_stopping_patience),
        min_delta=float(args.early_stopping_min_delta),
    )
    trainer = configure_trainer(args, logger, [checkpoint_cb, early_stopping_cb])
    trainer.fit(model=model, datamodule=datamodule)

    if not bool(args.auto_evaluate):
        return

    checkpoint_path = Path(checkpoint_cb.best_model_path or checkpoint_cb.last_model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Unable to locate checkpoint after training: {checkpoint_path}")

    setup_datamodule_for_report(datamodule, ("train", "val", "test"))
    report_model = selection.load_from_checkpoint(checkpoint_path)
    report_device = resolve_runtime_device(args.accelerator)
    prepare_model_for_reporting(report_model, datamodule, report_device)
    stage_reports = collect_stage_reports(report_model, datamodule, splits=("train", "val", "test"), device=report_device)
    report_payload = write_report_bundle(
        spec=selection,
        checkpoint_path=checkpoint_path,
        stage_reports=stage_reports,
        output_dir=run_dir,
        report_name="final_report",
    )
    log_final_report_metrics(logger, report_payload, step=trainer.global_step)


def run_evaluate(args: argparse.Namespace) -> None:
    selection = selection_for(args)
    L.seed_everything(args.seed, workers=True)
    datamodule = build_datamodule(args)
    model = selection.load_from_checkpoint(args.checkpoint)
    _validate_loaded_model_selection(model, selection)
    requested_splits = parse_report_splits(args.splits)
    setup_datamodule_for_report(datamodule, requested_splits)
    report_device = resolve_runtime_device(args.accelerator)
    prepare_model_for_reporting(model, datamodule, report_device)
    stage_reports = collect_stage_reports(
        model,
        datamodule,
        splits=requested_splits,
        device=report_device,
        max_batches=args.max_batches,
    )
    write_report_bundle(
        spec=selection,
        checkpoint_path=Path(args.checkpoint),
        stage_reports=stage_reports,
        output_dir=Path(args.output_dir),
        report_name=f"{selection.model_id}_report",
    )


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
    device = resolve_device(args.device)
    model = selection.load_from_checkpoint(args.checkpoint).to(device)
    _validate_loaded_model_selection(model, selection)
    model.eval()

    source_subdirs = args.source_subdirs
    point_dataset = PreprocessedPointCloudDataset(
        path=args.point_data_dir,
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
            path=data_dir_for(args),
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
    if args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    else:
        run_render(args)


if __name__ == "__main__":
    main()
