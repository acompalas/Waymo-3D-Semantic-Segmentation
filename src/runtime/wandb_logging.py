from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lightning.pytorch.callbacks import Callback
import numpy as np

from ..data import PreprocessedPointCloudDataset, PreprocessedRangeImageDataset, WaymoLidarDataModule
from .common import sanitize_metric_name
from ..tools.rendering import PALETTE, label_colors


def _import_wandb():
    import wandb

    return wandb


def _point_data_dir_for_representation(data_dir: Path, representation: str) -> Path:
    if str(representation) == "point_clouds":
        return Path(data_dir)
    candidate = Path(data_dir).parent / "point_clouds"
    if not candidate.exists():
        raise FileNotFoundError(f"Unable to locate paired point-cloud dataset at {candidate}")
    return candidate


def _sequence_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if length == 1:
        return [0] * count
    positions = np.linspace(0, length - 1, num=count)
    return [int(round(pos)) for pos in positions.tolist()]


@dataclass(frozen=True)
class AuditExample:
    segment: str
    timestamp: int


@dataclass
class AuditSource:
    split: str
    point_dataset: PreprocessedPointCloudDataset
    range_dataset: PreprocessedRangeImageDataset | None
    examples: list[AuditExample]


def select_audit_examples(
    dataset: PreprocessedPointCloudDataset | PreprocessedRangeImageDataset,
    *,
    count: int,
) -> list[AuditExample]:
    segments = list(dataset.segment_names)
    if count <= 0 or not segments:
        return []

    segment_indices = _sequence_indices(len(segments), count)
    seen_per_segment: dict[str, int] = {}
    examples: list[AuditExample] = []
    for segment_idx in segment_indices:
        segment = str(segments[segment_idx])
        frames = list(dataset.frames_for_segment(segment))
        if not frames:
            continue
        offset = seen_per_segment.get(segment, 0)
        frame_idx = _sequence_indices(len(frames), count=max(1, offset + 1))[-1]
        _, timestamp = frames[frame_idx]
        seen_per_segment[segment] = offset + 1
        examples.append(AuditExample(segment=segment, timestamp=int(timestamp)))
    return examples


def confusion_table_rows(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    *,
    ignore_class_zero: bool = True,
) -> tuple[list[list[object]], list[str]]:
    matrix = np.asarray(confusion_matrix, dtype=np.int64)
    start = 1 if ignore_class_zero else 0
    display_names = [str(class_names[idx]) for idx in range(start, min(len(class_names), matrix.shape[0]))]
    rows: list[list[object]] = []
    for row_idx, actual_name in enumerate(display_names, start=start):
        for col_idx, predicted_name in enumerate(display_names, start=start):
            rows.append([actual_name, predicted_name, int(matrix[row_idx, col_idx])])
    return rows, display_names


def stage_metric_dict(
    stage: str,
    stage_payload: dict,
    class_names: list[str],
    *,
    prefix: str | None = None,
) -> dict[str, float]:
    metrics = dict(stage_payload.get("metrics", {}))
    output: dict[str, float] = {}
    prefix_parts = [part for part in (prefix, stage) if part]
    metric_prefix = "_".join(prefix_parts)

    for key in ("loss", "acc", "mIoU"):
        if key in metrics:
            output[f"{metric_prefix}_{key}"] = float(metrics[key])

    confusion = np.asarray(stage_payload.get("confusion_matrix", []), dtype=np.int64)
    if confusion.size == 0:
        return output

    for class_idx in range(1, min(len(class_names), confusion.shape[0])):
        metric_key = f"IoU_{sanitize_metric_name(class_names[class_idx])}"
        raw_key = f"IoU_class_{class_idx}"
        if raw_key not in metrics:
            continue
        output[f"{metric_prefix}_{metric_key}"] = float(metrics[raw_key])
    return output


def log_confusion_matrix_to_wandb(
    logger,
    *,
    stage: str,
    confusion_matrix: np.ndarray,
    class_names: list[str],
    step: int,
    title: str,
) -> None:
    wandb = _import_wandb()
    rows, _ = confusion_table_rows(confusion_matrix, class_names)
    table = wandb.Table(columns=["Actual", "Predicted", "nPredictions"], data=rows)
    chart = wandb.plot_table(
        vega_spec_name="wandb/confusion_matrix/v1",
        data_table=table,
        fields={
            "Actual": "Actual",
            "Predicted": "Predicted",
            "nPredictions": "nPredictions",
        },
        string_fields={"title": title},
        split_table=False,
    )
    logger.experiment.log({f"{stage}/confusion_matrix": chart}, step=step)


def log_stage_report_to_wandb(
    logger,
    *,
    stage: str,
    stage_payload: dict,
    class_names: list[str],
    step: int,
    prefix: str | None = None,
) -> None:
    metrics = stage_metric_dict(stage, stage_payload, class_names, prefix=prefix)
    if metrics:
        logger.log_metrics(metrics, step=step)
    confusion = stage_payload.get("confusion_matrix")
    if confusion is None:
        return
    chart_title = f"{stage.title()} confusion matrix"
    if prefix:
        chart_title = f"{prefix.replace('_', ' ').title()} {chart_title}"
    log_confusion_matrix_to_wandb(
        logger,
        stage=stage if prefix is None else f"{prefix}_{stage}",
        confusion_matrix=np.asarray(confusion, dtype=np.int64),
        class_names=class_names,
        step=step,
        title=chart_title,
    )


def log_class_legend(logger, class_names: list[str], *, step: int) -> None:
    wandb = _import_wandb()
    rows = []
    for idx, name in enumerate(class_names[: len(PALETTE)]):
        rgb = tuple(int(round(value * 255.0)) for value in PALETTE[idx].tolist())
        rows.append([int(idx), str(name), *rgb])
    table = wandb.Table(columns=["class_id", "class_name", "r", "g", "b"], data=rows)
    logger.experiment.log({"class_legend": table}, step=step)


def segmented_pointcloud_to_object3d(payload: dict):
    wandb = _import_wandb()
    points = np.asarray(payload["points_xyz"], dtype=np.float32)
    colors = label_colors(
        np.asarray(payload["pred_labels"], dtype=np.int64),
        valid_label=np.asarray(payload["valid_label"], dtype=bool),
    )
    rgb = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)
    data = np.concatenate([points, rgb.astype(np.float32)], axis=1)
    return wandb.Object3D(data)


def _build_point_dataset(
    path: Path,
    *,
    segments: Iterable[str],
    seed: int,
) -> PreprocessedPointCloudDataset:
    return PreprocessedPointCloudDataset(
        path=path,
        segments=list(segments),
        num_points=1,
        deterministic_sampling=True,
        max_cached_segments=1,
        seed=seed,
    )


def build_audit_source(
    datamodule: WaymoLidarDataModule,
    *,
    split: str,
    count: int,
) -> AuditSource | None:
    dataset = {
        "train": datamodule.train_dataset,
        "val": datamodule.val_dataset,
        "test": datamodule.test_dataset,
    }.get(str(split))
    if dataset is None:
        return None

    point_dataset = _build_point_dataset(
        _point_data_dir_for_representation(datamodule.data_dir, datamodule.representation),
        segments=dataset.segment_names,
        seed=datamodule.seed + {"train": 401, "val": 402, "test": 403}[str(split)],
    )
    range_dataset = dataset if isinstance(dataset, PreprocessedRangeImageDataset) else None
    return AuditSource(
        split=str(split),
        point_dataset=point_dataset,
        range_dataset=range_dataset,
        examples=select_audit_examples(dataset, count=int(count)),
    )


def log_audit_pointclouds(
    logger,
    model,
    source: AuditSource | None,
    *,
    step: int,
    prefix: str | None = None,
) -> None:
    if source is None or not source.examples:
        return

    payload: dict[str, object] = {}
    for example_idx, example in enumerate(source.examples):
        point_frame = source.point_dataset.get_dense_frame(example.segment, example.timestamp)
        range_frame = (
            source.range_dataset.get_frame_data(example.segment, example.timestamp)
            if source.range_dataset is not None
            else None
        )
        prediction = model.predict_segmented_pointcloud(point_frame=point_frame, range_frame=range_frame)
        name = f"{source.split}/pointcloud_{example_idx}"
        if prefix:
            name = f"{prefix}_{name}"
        payload[name] = segmented_pointcloud_to_object3d(prediction)
    if payload:
        logger.experiment.log(payload, step=step)


def log_audit_pointcloud_splits(
    logger,
    model,
    datamodule: WaymoLidarDataModule,
    *,
    splits: Iterable[str],
    count: int,
    step: int,
    prefix: str | None = None,
) -> None:
    for split in splits:
        log_audit_pointclouds(
            logger,
            model,
            build_audit_source(datamodule, split=split, count=int(count)),
            step=step,
            prefix=prefix,
        )


class WandbSegmentationCallback(Callback):
    def __init__(self, *, log_pointcloud_count: int = 4) -> None:
        super().__init__()
        self.log_pointcloud_count = int(log_pointcloud_count)
        self._class_legend_logged = False
        self._audit_sources: dict[str, AuditSource | None] = {}

    def _should_skip(self, trainer) -> bool:
        return bool(getattr(trainer, "sanity_checking", False)) or not bool(getattr(trainer, "is_global_zero", True))

    def _ensure_ready(self, trainer, pl_module) -> None:
        if self._should_skip(trainer):
            return
        datamodule = trainer.datamodule
        if datamodule is None:
            return
        pl_module.set_class_names(list(datamodule.class_names))
        if not self._class_legend_logged:
            log_class_legend(trainer.logger, pl_module.class_names, step=trainer.global_step)
            self._class_legend_logged = True
        if not self._audit_sources:
            for split in ("train", "val"):
                self._audit_sources[split] = build_audit_source(
                    datamodule,
                    split=split,
                    count=self.log_pointcloud_count,
                )

    def on_fit_start(self, trainer, pl_module) -> None:
        self._ensure_ready(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._ensure_ready(trainer, pl_module)
        if self._should_skip(trainer):
            return
        log_confusion_matrix_to_wandb(
            trainer.logger,
            stage="train",
            confusion_matrix=pl_module.get_confusion_matrix("train").detach().cpu().numpy(),
            class_names=pl_module.class_names,
            step=trainer.global_step,
            title="Train confusion matrix",
        )
        log_audit_pointclouds(
            trainer.logger,
            pl_module,
            self._audit_sources.get("train"),
            step=trainer.global_step,
        )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._ensure_ready(trainer, pl_module)
        if self._should_skip(trainer):
            return
        log_confusion_matrix_to_wandb(
            trainer.logger,
            stage="val",
            confusion_matrix=pl_module.get_confusion_matrix("val").detach().cpu().numpy(),
            class_names=pl_module.class_names,
            step=trainer.global_step,
            title="Validation confusion matrix",
        )
        log_audit_pointclouds(
            trainer.logger,
            pl_module,
            self._audit_sources.get("val"),
            step=trainer.global_step,
        )
