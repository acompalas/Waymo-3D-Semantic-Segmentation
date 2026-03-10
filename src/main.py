import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import torch

from .data import PreprocessedPointCloudDataset, PreprocessedRangeImageDataset, WaymoLidarDataModule
from .runtime import MODEL_REGISTRY, get_model_spec
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=16384)
    parser.add_argument("--num-classes", type=int, default=23)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--worker-start-method", type=str, default="spawn")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-balanced-class-weights", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--train-segment-fraction", type=float, default=1.0)
    parser.add_argument("--max-cached-segments", type=int, default=2)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--test-subdirs", type=str, default="validation")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=16384)
    parser.add_argument("--num-classes", type=int, default=23)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--worker-start-method", type=str, default="spawn")
    parser.add_argument("--max-cached-segments", type=int, default=2)
    parser.add_argument("--sampling-steps", type=int, default=None)
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
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--render-num-points", type=int, default=16384)
    parser.add_argument("--output-gif", type=Path, default=Path("output/render/predictions.gif"))
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--camera-preset", type=str, default="car_pov", choices=["car_pov", "top", "topdown"])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--point-size", type=float, default=2.5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])


def build_parser(model_id: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified LiDAR semantic segmentation CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("train", "evaluate", "render"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--model", type=str, required=True, choices=sorted(MODEL_REGISTRY))
        if command == "train":
            add_common_train_args(sub)
        elif command == "evaluate":
            add_common_eval_args(sub)
        else:
            add_common_render_args(sub)

        if model_id is not None:
            try:
                spec = get_model_spec(model_id)
            except KeyError:
                continue
            spec.add_train_args(sub)

    return parser


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    base_parser = build_parser()
    known, _ = base_parser.parse_known_args(argv)
    full_parser = build_parser(getattr(known, "model", None))
    return full_parser.parse_args(argv)


def data_dir_for(args: argparse.Namespace) -> Path:
    spec = get_model_spec(args.model)
    return Path(args.data_dir) if args.data_dir is not None else spec.default_data_dir


def build_datamodule(args: argparse.Namespace) -> WaymoLidarDataModule:
    spec = get_model_spec(args.model)
    return WaymoLidarDataModule(
        data_dir=data_dir_for(args),
        representation=spec.representation,
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
    )


def configure_trainer(args: argparse.Namespace, logger: CSVLogger | bool, callbacks: list) -> L.Trainer:
    return L.Trainer(
        max_epochs=getattr(args, "max_epochs", 1),
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )


def run_train(args: argparse.Namespace) -> None:
    spec = get_model_spec(args.model)
    L.seed_everything(args.seed, workers=True)
    datamodule = build_datamodule(args)
    model = spec.build_module(args)

    logger = CSVLogger(save_dir=str(args.output_dir), name=args.model)
    checkpoint_cb = ModelCheckpoint(
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
    trainer.test(datamodule=datamodule, ckpt_path="best")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_evaluate(args: argparse.Namespace) -> None:
    spec = get_model_spec(args.model)
    L.seed_everything(args.seed, workers=True)
    datamodule = build_datamodule(args)
    model = spec.load_from_checkpoint(args.checkpoint)
    model.set_sampling_steps(args.sampling_steps)

    trainer = configure_trainer(args, False, [])
    results = trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
    metrics = results[0] if results else {}
    conf = model.get_confusion_matrix("test").detach().cpu().tolist() if hasattr(model, "get_confusion_matrix") else None
    _write_json(
        Path(args.output_dir) / f"{args.model}_metrics.json",
        {
            "model": args.model,
            "checkpoint": str(args.checkpoint),
            "metrics": metrics,
            "confusion_matrix": conf,
        },
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


def run_render(args: argparse.Namespace) -> None:
    spec = get_model_spec(args.model)
    device = resolve_device(args.device)
    model = spec.load_from_checkpoint(args.checkpoint).to(device)
    model.eval()
    model.set_sampling_steps(args.sampling_steps)

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
    if spec.representation == "range_images":
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
        point_frame = point_dataset.get_dense_frame(segment, timestamp)
        range_frame = range_dataset.get_frame_data(segment, timestamp) if range_dataset is not None else None
        prediction = model.predict_segmented_pointcloud(
            point_frame=point_frame,
            range_frame=range_frame,
            sampling_steps=args.sampling_steps,
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
