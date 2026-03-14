import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PointCloudSegmentationModel
from .inputs import point_geometry, point_input_dim, select_point_model_inputs
from .point_backbones import build_point_backbone


class PointCloudSupervisedSegmenter(PointCloudSegmentationModel):
    def __init__(
        self,
        num_classes: int = 23,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        hidden_dim: int = 256,
        depth: int = 6,
        backbone: str = "edgeconv",
        knn_k: int = 16,
        dropout: float = 0.1,
        use_balanced_class_weights: bool = True,
        geometry_only: bool = False,
    ) -> None:
        super().__init__(num_classes=num_classes, use_balanced_class_weights=use_balanced_class_weights)
        self.save_hyperparameters()
        self.backbone_name = str(backbone).lower()
        self.backbone = build_point_backbone(
            self.backbone_name,
            input_dim=point_input_dim(geometry_only=bool(geometry_only)),
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            dropout=float(dropout),
            knn_k=int(knn_k),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone.output_dim),
            nn.Linear(self.backbone.output_dim, self.backbone.output_dim),
            nn.SiLU(),
            nn.Linear(self.backbone.output_dim, int(num_classes)),
        )

    def _model_inputs(self, points: torch.Tensor, point_features: torch.Tensor) -> torch.Tensor:
        return select_point_model_inputs(
            points,
            point_features,
            geometry_only=bool(self.hparams.geometry_only),
        )

    def _predict_logits(
        self,
        points: torch.Tensor,
        point_features: torch.Tensor,
        valid_geometry: torch.Tensor,
    ) -> torch.Tensor:
        _ = valid_geometry
        model_inputs = self._model_inputs(points, point_features)
        if self.backbone_name == "edgeconv":
            hidden = self.backbone(model_inputs, point_geometry(model_inputs))
        else:
            hidden = self.backbone(model_inputs)
        return self.head(hidden)

    def predict_point_labels(
        self,
        points: torch.Tensor,
        point_features: torch.Tensor,
        valid_geometry: torch.Tensor,
        *,
        sampling_steps: int | None = None,
    ) -> torch.Tensor:
        _ = sampling_steps
        logits = self._predict_logits(points, point_features, valid_geometry)
        preds = logits.argmax(dim=-1)
        return torch.where(valid_geometry, preds, torch.zeros_like(preds))

    def _compute_stage_output(
        self,
        batch: dict,
        *,
        stage: str,
        prediction_mode: str,
        evaluation: bool,
    ) -> dict:
        _ = stage
        _ = prediction_mode
        _ = evaluation
        points = batch["points"].float()
        point_features = batch["point_features"].float()
        labels = batch["labels"].long()
        valid_geometry = batch["valid_geometry"].bool()
        valid_label = batch["valid_label"].bool()
        batch_size = int(points.shape[0])

        logits = self._predict_logits(points, point_features, valid_geometry)
        preds = logits.argmax(dim=-1)
        metric_mask = valid_geometry & valid_label & (labels > 0)
        if not bool(metric_mask.any()):
            loss = logits.sum() * 0.0
            return {
                "loss": loss,
                "preds": torch.zeros_like(labels),
                "labels": labels,
                "metric_mask": metric_mask,
                "batch_size": batch_size,
            }

        target = labels.clone()
        target[~metric_mask] = -100
        loss = F.cross_entropy(logits.transpose(1, 2), target, weight=self.class_weights, ignore_index=-100)
        return {
            "loss": loss,
            "preds": preds,
            "labels": labels,
            "metric_mask": metric_mask,
            "batch_size": batch_size,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(self.trainer.max_epochs)))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
