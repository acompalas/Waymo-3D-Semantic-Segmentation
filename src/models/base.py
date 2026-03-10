from typing import Optional

import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_info
import numpy as np
import torch


SEGMENTED_POINTCLOUD_KEYS = {
    "points_xyz",
    "pred_labels",
    "true_labels",
    "valid_label",
}


class SegmentationLightningModule(L.LightningModule):
    def __init__(self, num_classes: int, use_balanced_class_weights: bool = True) -> None:
        super().__init__()
        self._metric_num_classes = int(num_classes)
        self._use_balanced_class_weights = bool(use_balanced_class_weights)
        self._sampling_steps_override: int | None = None
        self.register_buffer("_class_weights", torch.ones(self._metric_num_classes, dtype=torch.float32), persistent=False)
        self.register_buffer(
            "_val_confmat",
            torch.zeros((self._metric_num_classes, self._metric_num_classes), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_test_confmat",
            torch.zeros((self._metric_num_classes, self._metric_num_classes), dtype=torch.long),
            persistent=False,
        )
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

    def update_confmat(self, stage: str, preds: torch.Tensor, labels: torch.Tensor) -> None:
        if preds.numel() == 0 or labels.numel() == 0:
            return
        preds = preds.reshape(-1).long()
        labels = labels.reshape(-1).long()
        keep = (
            (labels > 0)
            & (labels < self._metric_num_classes)
            & (preds >= 0)
            & (preds < self._metric_num_classes)
        )
        if not bool(keep.any()):
            return
        preds = preds[keep]
        labels = labels[keep]
        flat = labels * self._metric_num_classes + preds
        bincount = torch.bincount(
            flat,
            minlength=self._metric_num_classes * self._metric_num_classes,
        ).reshape(self._metric_num_classes, self._metric_num_classes)
        if stage == "val":
            self._val_confmat += bincount
        else:
            self._test_confmat += bincount

    def get_confusion_matrix(self, stage: str) -> torch.Tensor:
        if stage == "val":
            return self._val_confmat.detach().clone()
        if stage == "test":
            return self._test_confmat.detach().clone()
        raise ValueError(f"Unsupported stage '{stage}'")

    def _log_iou_metrics(self, stage: str) -> None:
        conf = self._val_confmat if stage == "val" else self._test_confmat
        conf = conf.to(dtype=torch.float32)
        tp = torch.diag(conf)
        fp = conf.sum(dim=0) - tp
        fn = conf.sum(dim=1) - tp
        union = tp + fp + fn

        iou = torch.full_like(union, fill_value=-1.0, dtype=torch.float32)
        valid = union > 0
        iou[valid] = tp[valid] / union[valid].clamp_min(1e-6)

        metric_valid = valid.clone()
        if metric_valid.numel() > 0:
            metric_valid[0] = False
        miou = iou[metric_valid].mean() if bool(metric_valid.any()) else torch.tensor(0.0, device=conf.device)

        self.log(f"{stage}_mIoU", miou, on_step=False, on_epoch=True, prog_bar=True)
        for cls_idx, cls_iou in enumerate(iou):
            self.log(f"{stage}_IoU_class_{cls_idx}", cls_iou, on_step=False, on_epoch=True, prog_bar=False)

    def on_validation_epoch_start(self) -> None:
        self._val_confmat.zero_()

    def on_validation_epoch_end(self) -> None:
        self._log_iou_metrics(stage="val")

    def on_test_epoch_start(self) -> None:
        self._test_confmat.zero_()

    def on_test_epoch_end(self) -> None:
        self._log_iou_metrics(stage="test")

    def on_fit_start(self) -> None:
        dm = self.trainer.datamodule
        counts = getattr(dm, "class_counts", None) if dm is not None else None
        weights = getattr(dm, "class_weights", None) if dm is not None else None

        if counts is not None:
            rank_zero_info(f"Class counts (train supervised elements): {counts.detach().cpu().tolist()}")

        if not self._use_balanced_class_weights:
            rank_zero_info("Balanced class weights: disabled")
            return

        if weights is not None:
            self.set_class_weights(weights.to(self.device))
            rank_zero_info(f"Class weights: {weights.detach().cpu().tolist()}")
        else:
            rank_zero_info("Class weights: unavailable (not computed by datamodule).")

    def on_train_epoch_start(self) -> None:
        dm = self.trainer.datamodule
        if dm is not None and hasattr(dm, "set_epoch"):
            dm.set_epoch(self.current_epoch)

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
    def _dense_points(
        self,
        point_frame: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        xyz = point_frame["xyz"].reshape(-1, 3)
        point_features = point_frame["point_features"].reshape(-1, 2)
        labels = point_frame["labels"].reshape(-1)
        valid_geometry = point_frame["valid_geometry"].reshape(-1).astype(bool, copy=False)
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

        batch = self._build_range_batch(range_frame)
        preds, labels = self.predict_range_labels(batch, sampling_steps=self.resolve_sampling_steps(sampling_steps))
        valid_geometry = point_frame["valid_geometry"].reshape(-1).astype(bool, copy=False)
        valid_label = point_frame["valid_label"].reshape(-1).astype(bool, copy=False)
        points = point_frame["xyz"].reshape(-1, 3)[valid_geometry]

        payload = {
            "points_xyz": points.astype(np.float32, copy=False),
            "pred_labels": preds.reshape(-1).detach().cpu().numpy()[valid_geometry].astype(np.int64, copy=False),
            "true_labels": labels.reshape(-1).detach().cpu().numpy()[valid_geometry].astype(np.int64, copy=False),
            "valid_label": valid_label[valid_geometry].astype(bool, copy=False),
        }
        return self._validate_segmented_pointcloud(payload)
