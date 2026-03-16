from typing import Any, Optional

import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_info
import numpy as np
import torch

from ..runtime.metrics import empty_confusion_matrix, iou_metrics_from_confusion_matrix
from ..runtime.stage_eval import consume_stage_output, validate_stage_output


SEGMENTED_POINTCLOUD_KEYS = {
    "points_xyz",
    "pred_labels",
    "true_labels",
    "valid_label",
}


def _format_class_summary(values: torch.Tensor, *, float_values: bool) -> str:
    data = values.detach().cpu()
    if data.numel() == 0:
        return "none"
    if float_values:
        active = [(idx, float(value.item())) for idx, value in enumerate(data) if abs(float(value.item())) > 0.0]
        if not active:
            return "none"
        return ", ".join(f"{idx}:{value:.3f}" for idx, value in active)

    active = [(idx, int(value.item())) for idx, value in enumerate(data) if int(value.item()) > 0]
    if not active:
        return "none"
    total = sum(value for _, value in active)
    items = ", ".join(f"{idx}:{value}" for idx, value in active)
    return f"total={total} | {items}"


class SegmentationLightningModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        *,
        use_balanced_class_weights: bool = True,
        validation_prediction_mode: str = "full",
    ) -> None:
        super().__init__()
        self._metric_num_classes = int(num_classes)
        self._use_balanced_class_weights = bool(use_balanced_class_weights)
        self._sampling_steps_override: int | None = None
        self._validation_prediction_mode = str(validation_prediction_mode)
        self.register_buffer("_class_weights", torch.ones(self._metric_num_classes, dtype=torch.float32), persistent=False)
        self.register_buffer("_train_confmat", empty_confusion_matrix(self._metric_num_classes), persistent=False)
        self.register_buffer("_val_confmat", empty_confusion_matrix(self._metric_num_classes), persistent=False)
        self.register_buffer("_test_confmat", empty_confusion_matrix(self._metric_num_classes), persistent=False)
        self._weights_ready = False

    @property
    def class_weights(self) -> Optional[torch.Tensor]:
        if not self._use_balanced_class_weights or not self._weights_ready:
            return None
        return self._class_weights

    def set_class_weights(self, class_weights: torch.Tensor) -> None:
        class_weights = class_weights.detach().float().to(self.device)
        if class_weights.ndim != 1 or class_weights.shape[0] != self._metric_num_classes:
            raise ValueError(
                f"class_weights shape mismatch: got {tuple(class_weights.shape)}, "
                f"expected ({self._metric_num_classes},)"
            )
        self._class_weights.copy_(class_weights)
        self._weights_ready = True

    def set_sampling_steps(self, steps: int | None) -> None:
        self._sampling_steps_override = None if steps is None else int(steps)

    def resolve_sampling_steps(self, sampling_steps: int | None = None) -> int | None:
        if sampling_steps is not None:
            return int(sampling_steps)
        return self._sampling_steps_override

    def prepare_runtime(self) -> None:
        return None

    def _resolve_prediction_mode(self, stage: str) -> str:
        if stage == "train":
            return "cheap"
        if stage == "val":
            return self._validation_prediction_mode
        return "full"

    def _confmat_for_stage(self, stage: str) -> torch.Tensor:
        if stage == "train":
            return self._train_confmat
        if stage == "val":
            return self._val_confmat
        if stage == "test":
            return self._test_confmat
        raise ValueError(f"Unsupported stage '{stage}'")

    def reset_stage_metrics(self, stage: str) -> None:
        self._confmat_for_stage(stage).zero_()

    def get_confusion_matrix(self, stage: str) -> torch.Tensor:
        return self._confmat_for_stage(stage).detach().clone()

    def _log_iou_metrics(self, stage: str) -> None:
        metrics = iou_metrics_from_confusion_matrix(self._confmat_for_stage(stage), ignore_class_zero=True)
        self.log(f"{stage}_mIoU", metrics["miou"], on_step=False, on_epoch=True, prog_bar=True)
        for cls_idx, cls_iou in enumerate(metrics["per_class_iou"]):
            self.log(f"{stage}_IoU_class_{cls_idx}", cls_iou, on_step=False, on_epoch=True, prog_bar=False)

    def _consume_and_log_stage_output(self, stage: str, output: dict[str, Any]) -> torch.Tensor:
        output = validate_stage_output(output)
        loss = output["loss"]
        acc = consume_stage_output(self._confmat_for_stage(stage), output, num_classes=self._metric_num_classes)
        batch_size = max(1, int(output["batch_size"]))
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def _shared_stage_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        output = self.compute_stage_output(
            batch,
            stage=stage,
            prediction_mode=self._resolve_prediction_mode(stage),
            evaluation=(stage != "train"),
        )
        return self._consume_and_log_stage_output(stage, output)

    def compute_stage_output(
        self,
        batch: dict[str, Any],
        *,
        stage: str,
        prediction_mode: str,
        evaluation: bool,
    ) -> dict[str, Any]:
        output = self._compute_stage_output(
            batch,
            stage=stage,
            prediction_mode=prediction_mode,
            evaluation=evaluation,
        )
        return validate_stage_output(output)

    def _compute_stage_output(
        self,
        batch: dict[str, Any],
        *,
        stage: str,
        prediction_mode: str,
        evaluation: bool,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def on_train_epoch_start(self) -> None:
        self.reset_stage_metrics("train")
        dm = self.trainer.datamodule
        if dm is not None and hasattr(dm, "set_epoch"):
            dm.set_epoch(self.current_epoch)

    def on_train_epoch_end(self) -> None:
        self._log_iou_metrics(stage="train")

    def on_validation_epoch_start(self) -> None:
        self.reset_stage_metrics("val")

    def on_validation_epoch_end(self) -> None:
        self._log_iou_metrics(stage="val")

    def on_test_epoch_start(self) -> None:
        self.reset_stage_metrics("test")

    def on_test_epoch_end(self) -> None:
        self._log_iou_metrics(stage="test")

    def on_fit_start(self) -> None:
        dm = self.trainer.datamodule
        counts = getattr(dm, "class_counts", None) if dm is not None else None
        weights = getattr(dm, "class_weights", None) if dm is not None else None

        if counts is not None:
            rank_zero_info(f"Class counts (train supervised elements): {_format_class_summary(counts, float_values=False)}")

        if not self._use_balanced_class_weights:
            rank_zero_info("Balanced class weights: disabled")
            return

        if weights is not None:
            self.set_class_weights(weights.to(self.device))
            rank_zero_info(f"Class weights: {_format_class_summary(weights, float_values=True)}")
        else:
            rank_zero_info("Class weights: unavailable (not computed by datamodule).")
        self.prepare_runtime()

    def on_validation_start(self) -> None:
        self.prepare_runtime()

    def on_test_start(self) -> None:
        self.prepare_runtime()

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_stage_step(batch, stage="train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_stage_step(batch, stage="val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_stage_step(batch, stage="test")

    def predict_segmented_pointcloud(
        self,
        *,
        point_frame: dict,
        range_frame: dict | None = None,
        sampling_steps: int | None = None,
    ) -> dict:
        raise NotImplementedError

    def _validate_segmented_pointcloud(self, payload: dict) -> dict:
        missing = SEGMENTED_POINTCLOUD_KEYS.difference(payload)
        if missing:
            raise ValueError(f"Segmented point-cloud payload missing keys: {sorted(missing)}")
        return payload


class PointCloudSegmentationModel(SegmentationLightningModule):
    @staticmethod
    def _resolve_point_valid_geometry(payload: dict, labels: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        valid_geometry = payload.get("valid_geometry")
        if valid_geometry is None:
            if isinstance(labels, torch.Tensor):
                return torch.ones_like(labels, dtype=torch.bool)
            return np.ones(labels.shape, dtype=bool)
        if isinstance(valid_geometry, torch.Tensor):
            return valid_geometry.bool()
        return np.asarray(valid_geometry, dtype=bool)

    def _dense_points(
        self,
        point_frame: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        xyz = point_frame["xyz"].reshape(-1, 3)
        point_features = point_frame["point_features"].reshape(-1, 2)
        labels = point_frame["labels"].reshape(-1)
        valid_geometry = self._resolve_point_valid_geometry(point_frame, labels).reshape(-1).astype(bool, copy=False)
        valid_label = point_frame["valid_label"].reshape(-1).astype(bool, copy=False)

        valid_idx = np.flatnonzero(valid_geometry)
        if valid_idx.size == 0:
            raise RuntimeError("Selected frame contains no valid geometry.")

        choice = np.asarray(valid_idx, dtype=np.int64)

        return (
            torch.from_numpy(xyz[choice].astype(np.float32, copy=False)).unsqueeze(0).to(device=self.device),
            torch.from_numpy(point_features[choice].astype(np.float32, copy=False)).unsqueeze(0).to(device=self.device),
            torch.from_numpy(labels[choice].astype(np.int64, copy=False)).unsqueeze(0).to(device=self.device),
            torch.ones((1, choice.shape[0]), dtype=torch.bool, device=self.device),
            valid_label[choice].astype(bool, copy=False),
        )

    def predict_point_labels(
        self,
        points: torch.Tensor,
        point_features: torch.Tensor,
        valid_geometry: torch.Tensor,
        *,
        sampling_steps: int | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def predict_segmented_pointcloud(
        self,
        *,
        point_frame: dict,
        range_frame: dict | None = None,
        sampling_steps: int | None = None,
    ) -> dict:
        _ = range_frame
        self.prepare_runtime()
        points, point_features, labels, valid_geometry, valid_label = self._dense_points(point_frame)
        preds = self.predict_point_labels(
            points,
            point_features,
            valid_geometry,
            sampling_steps=self.resolve_sampling_steps(sampling_steps),
        ).squeeze(0).detach().cpu().numpy().astype(np.int64, copy=False)
        payload = {
            "points_xyz": points.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False),
            "pred_labels": preds,
            "true_labels": labels.squeeze(0).detach().cpu().numpy().astype(np.int64, copy=False),
            "valid_label": valid_label,
        }
        return self._validate_segmented_pointcloud(payload)


class RangeImageSegmentationModel(SegmentationLightningModule):
    @staticmethod
    def _resolve_point_valid_geometry(payload: dict, labels: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        valid_geometry = payload.get("valid_geometry")
        if valid_geometry is None:
            if isinstance(labels, torch.Tensor):
                return torch.ones_like(labels, dtype=torch.bool)
            return np.ones(labels.shape, dtype=bool)
        if isinstance(valid_geometry, torch.Tensor):
            return valid_geometry.bool()
        return np.asarray(valid_geometry, dtype=bool)

    def _build_range_batch(self, range_frame: dict) -> dict:
        return {
            "range_images": torch.from_numpy(range_frame["range_images"]).unsqueeze(0).to(device=self.device),
            "semantic": torch.from_numpy(range_frame["semantic"]).unsqueeze(0).to(device=self.device),
            "valid_label": torch.from_numpy(range_frame["valid_label"]).unsqueeze(0).to(device=self.device),
        }

    def predict_range_labels(self, batch: dict, *, sampling_steps: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def predict_segmented_pointcloud(
        self,
        *,
        point_frame: dict,
        range_frame: dict | None = None,
        sampling_steps: int | None = None,
    ) -> dict:
        if range_frame is None:
            raise ValueError("range_frame is required for range-image models.")

        self.prepare_runtime()
        batch = self._build_range_batch(range_frame)
        preds, labels = self.predict_range_labels(batch, sampling_steps=self.resolve_sampling_steps(sampling_steps))
        valid_geometry = self._resolve_point_valid_geometry(point_frame, point_frame["labels"].reshape(-1)).reshape(-1)
        valid_label = point_frame["valid_label"].reshape(-1).astype(bool, copy=False)
        points = point_frame["xyz"].reshape(-1, 3)[valid_geometry]

        payload = {
            "points_xyz": points.astype(np.float32, copy=False),
            "pred_labels": preds.reshape(-1).detach().cpu().numpy()[valid_geometry].astype(np.int64, copy=False),
            "true_labels": labels.reshape(-1).detach().cpu().numpy()[valid_geometry].astype(np.int64, copy=False),
            "valid_label": valid_label[valid_geometry].astype(bool, copy=False),
        }
        return self._validate_segmented_pointcloud(payload)
