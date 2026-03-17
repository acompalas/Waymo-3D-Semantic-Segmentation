from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lightning.pytorch.callbacks import Callback
import numpy as np
import torch

from ..data import PreprocessedPointCloudDataset, PreprocessedRangeImageDataset, WaymoLidarDataModule
from .common import (
    confusion_family_section_key,
    confusion_matrix_section_key,
    loss_accuracy_section_key,
    pointcloud_section_key,
    sanitize_metric_name,
)
from ..tools.rendering import PALETTE, label_colors


def _import_wandb():
    import wandb

    return wandb


def _log_wandb_payload(logger, payload: dict[str, object], *, step: int | None = None) -> None:
    if step is not None:
        logger.experiment.log(dict(payload, **{"trainer/global_step": int(step)}))
        return
    logger.experiment.log(payload)


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

    if "loss" in metrics:
        output[loss_accuracy_section_key(stage, "loss", prefix=prefix)] = float(metrics["loss"])
    if "acc" in metrics:
        output[loss_accuracy_section_key(stage, "accuracy", prefix=prefix)] = float(metrics["acc"])
    if "mIoU" in metrics:
        output[confusion_family_section_key(stage, "iou", "mIoU", prefix=prefix)] = float(metrics["mIoU"])
    if "mean_precision" in metrics:
        output[confusion_family_section_key(stage, "precision", "mean_precision", prefix=prefix)] = float(metrics["mean_precision"])
    if "mean_recall" in metrics:
        output[confusion_family_section_key(stage, "recall", "mean_recall", prefix=prefix)] = float(metrics["mean_recall"])

    confusion = np.asarray(stage_payload.get("confusion_matrix", []), dtype=np.int64)
    if confusion.size == 0:
        return output

    for class_idx in range(1, min(len(class_names), confusion.shape[0])):
        metric_key = f"IoU_{sanitize_metric_name(class_names[class_idx])}"
        raw_key = f"IoU_class_{class_idx}"
        if raw_key not in metrics:
            raw_key = None
        if raw_key is not None:
            output[f"{metric_prefix}_{metric_key}"] = float(metrics[raw_key])
            output[confusion_family_section_key(stage, "iou", sanitize_metric_name(class_names[class_idx]), prefix=prefix)] = float(metrics[raw_key])
        precision_key = f"precision_class_{class_idx}"
        if precision_key in metrics:
            output[confusion_family_section_key(stage, "precision", sanitize_metric_name(class_names[class_idx]), prefix=prefix)] = float(metrics[precision_key])
        recall_key = f"recall_class_{class_idx}"
        if recall_key in metrics:
            output[confusion_family_section_key(stage, "recall", sanitize_metric_name(class_names[class_idx]), prefix=prefix)] = float(metrics[recall_key])
    return output


def log_confusion_matrix_to_wandb(
    logger,
    *,
    key: str,
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
    _log_wandb_payload(logger, {key: chart}, step=step)


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
    display_stage = {
        "train": "Train",
        "val": "Validation",
        "test": "Test",
    }.get(str(stage), str(stage).title())
    chart_title = f"{display_stage} confusion matrix"
    if prefix:
        chart_title = f"{prefix.replace('_', ' ').title()} {chart_title}"
    log_confusion_matrix_to_wandb(
        logger,
        key=confusion_matrix_section_key(stage, prefix=prefix),
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
    _log_wandb_payload(logger, {"class_legend": table}, step=step)


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
    num_points: int,
) -> PreprocessedPointCloudDataset:
    return PreprocessedPointCloudDataset(
        path=path,
        segments=list(segments),
        num_points=max(1, int(num_points)),
        deterministic_sampling=True,
        max_cached_segments=1,
        seed=seed,
    )


def _example_key(segment: str, timestamp: int) -> tuple[str, int]:
    return str(segment), int(timestamp)


def _batch_metadata(batch: dict) -> list[tuple[str, int]]:
    segments = batch.get("segment_context_name")
    timestamps = batch.get("frame_timestamp_micros")
    if segments is None or timestamps is None:
        return []

    if isinstance(segments, str):
        segment_values = [segments]
    else:
        segment_values = [str(value) for value in segments]

    if isinstance(timestamps, torch.Tensor):
        timestamp_values = timestamps.detach().cpu().tolist()
    elif isinstance(timestamps, (list, tuple)):
        timestamp_values = list(timestamps)
    else:
        timestamp_values = [timestamps]

    return [_example_key(segment, int(timestamp)) for segment, timestamp in zip(segment_values, timestamp_values)]


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


def build_audit_source(
    datamodule: WaymoLidarDataModule,
    *,
    split: str,
    count: int,
) -> AuditSource | None:
    split_name = str(split)
    if split_name == "val" and hasattr(datamodule, "effective_validation_dataset"):
        dataset = datamodule.effective_validation_dataset()
    else:
        dataset = {
            "train": datamodule.train_dataset,
            "val": datamodule.val_dataset,
            "test": datamodule.test_dataset,
        }.get(split_name)
    if dataset is None:
        return None

    point_dataset = _build_point_dataset(
        Path(datamodule.point_data_dir),
        segments=dataset.segment_names,
        seed=datamodule.seed + {"train": 401, "val": 402, "test": 403}[split_name],
        num_points=datamodule.num_points,
    )
    range_dataset = dataset if isinstance(dataset, PreprocessedRangeImageDataset) else None
    return AuditSource(
        split=split_name,
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
        if source.range_dataset is None:
            point_frame = _sampled_point_frame(
                source.point_dataset,
                segment=example.segment,
                timestamp=example.timestamp,
            )
        else:
            point_frame = source.point_dataset.get_dense_frame(example.segment, example.timestamp)
        range_frame = (
            source.range_dataset.get_frame_data(example.segment, example.timestamp)
            if source.range_dataset is not None
            else None
        )
        with torch.inference_mode():
            prediction = model.predict_segmented_pointcloud(point_frame=point_frame, range_frame=range_frame)
        name = pointcloud_section_key(source.split, f"pointcloud_{example_idx}", prefix=prefix)
        payload[name] = segmented_pointcloud_to_object3d(prediction)
    if payload:
        _log_wandb_payload(logger, payload, step=step)


def log_cached_audit_pointclouds(
    logger,
    source: AuditSource | None,
    cached_predictions: dict[tuple[str, int], dict],
    *,
    step: int,
    prefix: str | None = None,
) -> None:
    if source is None or not source.examples:
        return

    payload: dict[str, object] = {}
    for example_idx, example in enumerate(source.examples):
        prediction = cached_predictions.get(_example_key(example.segment, example.timestamp))
        if prediction is None:
            continue
        name = pointcloud_section_key(source.split, f"pointcloud_{example_idx}", prefix=prefix)
        payload[name] = segmented_pointcloud_to_object3d(prediction)
    if payload:
        _log_wandb_payload(logger, payload, step=step)


def cache_audit_predictions_from_stage_batch(
    batch: dict,
    outputs: dict,
    source: AuditSource | None,
    cached_predictions: dict[tuple[str, int], dict],
) -> None:
    if source is None or "preds" not in outputs:
        return

    target_keys = {_example_key(example.segment, example.timestamp) for example in source.examples}
    if not target_keys:
        return

    metadata = _batch_metadata(batch)
    if not metadata:
        return

    preds = outputs["preds"]
    if not isinstance(preds, torch.Tensor):
        return

    if "points" in batch:
        labels = batch["labels"]
        valid_label = batch["valid_label"]
        points = batch["points"]
        for sample_idx, key in enumerate(metadata):
            if key not in target_keys or key in cached_predictions:
                continue
            cached_predictions[key] = {
                "points_xyz": points[sample_idx].detach().cpu().numpy().astype(np.float32, copy=False),
                "pred_labels": preds[sample_idx].detach().cpu().numpy().astype(np.int64, copy=False),
                "true_labels": labels[sample_idx].detach().cpu().numpy().astype(np.int64, copy=False),
                "valid_label": valid_label[sample_idx].detach().cpu().numpy().astype(bool, copy=False),
            }
        return

    if "range_images" in batch and source.range_dataset is not None:
        labels = batch["semantic"]
        returns = int(batch["range_images"].shape[1])
        for sample_idx, key in enumerate(metadata):
            if key not in target_keys or key in cached_predictions:
                continue
            segment, timestamp = key
            point_frame = source.point_dataset.get_dense_frame(segment, timestamp)
            valid_geometry = point_frame["valid_geometry"].reshape(-1).astype(bool, copy=False)
            point_valid_label = point_frame["valid_label"].reshape(-1).astype(bool, copy=False)[valid_geometry]
            start = sample_idx * returns
            stop = start + returns
            sample_preds = preds[start:stop].reshape(-1).detach().cpu().numpy().astype(np.int64, copy=False)
            sample_labels = labels[sample_idx].reshape(-1).detach().cpu().numpy().astype(np.int64, copy=False)
            cached_predictions[key] = {
                "points_xyz": point_frame["xyz"].reshape(-1, 3)[valid_geometry].astype(np.float32, copy=False),
                "pred_labels": sample_preds[valid_geometry],
                "true_labels": sample_labels[valid_geometry],
                "valid_label": point_valid_label,
            }


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
        self._cached_val_predictions: dict[tuple[str, int], dict] = {}

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

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        _ = trainer
        _ = pl_module
        self._cached_val_predictions = {}

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        _ = trainer
        _ = pl_module
        _ = batch_idx
        _ = dataloader_idx
        if not isinstance(outputs, dict):
            return
        cache_audit_predictions_from_stage_batch(
            batch,
            outputs,
            self._audit_sources.get("val"),
            self._cached_val_predictions,
        )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._ensure_ready(trainer, pl_module)
        if self._should_skip(trainer):
            return
        log_confusion_matrix_to_wandb(
            trainer.logger,
            key=confusion_matrix_section_key("train"),
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
            key=confusion_matrix_section_key("val"),
            confusion_matrix=pl_module.get_confusion_matrix("val").detach().cpu().numpy(),
            class_names=pl_module.class_names,
            step=trainer.global_step,
            title="Validation confusion matrix",
        )
        log_cached_audit_pointclouds(
            trainer.logger,
            self._audit_sources.get("val"),
            self._cached_val_predictions,
            step=trainer.global_step,
        )
