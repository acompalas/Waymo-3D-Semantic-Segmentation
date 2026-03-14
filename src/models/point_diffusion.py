import torch
import torch.nn as nn

from .base import PointCloudSegmentationModel
from .diffusion import DDPM, labels_to_soft_points
from .inputs import point_geometry, point_input_dim, select_point_model_inputs
from .point_backbones import build_point_backbone


def _resolve_point_geometry_only(geometry_only: bool | None) -> bool:
    return True if geometry_only is None else bool(geometry_only)


class PointDiffusionDenoiser(nn.Module):
    def __init__(
        self,
        *,
        backbone: str = "edgeconv",
        num_classes: int = 23,
        point_input_dim_value: int = 3,
        hidden_dim: int = 256,
        depth: int = 6,
        time_dim: int = 128,
        dropout: float = 0.1,
        knn_k: int = 16,
    ) -> None:
        super().__init__()
        self.backbone_name = str(backbone).lower()
        self.backbone = build_point_backbone(
            self.backbone_name,
            input_dim=int(num_classes) + int(point_input_dim_value),
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            dropout=float(dropout),
            knn_k=int(knn_k),
            time_dim=int(time_dim),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.backbone.output_dim),
            nn.Linear(self.backbone.output_dim, self.backbone.output_dim),
            nn.SiLU(),
            nn.Linear(self.backbone.output_dim, int(num_classes)),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, model_inputs: torch.Tensor) -> torch.Tensor:
        features = torch.cat([x_t, model_inputs], dim=-1)
        xyz = point_geometry(model_inputs)
        if self.backbone_name == "edgeconv":
            hidden = self.backbone(features, xyz, t=t)
        else:
            hidden = self.backbone(features, t=t)
        return self.out_proj(hidden)


class PointCloudDiffusionSegmenter(PointCloudSegmentationModel):
    def __init__(
        self,
        num_classes: int = 23,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        diffusion_steps: int = 1000,
        hidden_dim: int = 256,
        depth: int = 6,
        backbone: str = "edgeconv",
        knn_k: int = 16,
        use_balanced_class_weights: bool = True,
        validation_prediction_mode: str = "cheap",
        geometry_only: bool | None = None,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            use_balanced_class_weights=use_balanced_class_weights,
            validation_prediction_mode=validation_prediction_mode,
        )
        effective_geometry_only = _resolve_point_geometry_only(geometry_only)
        self.save_hyperparameters()
        self.model = PointDiffusionDenoiser(
            backbone=str(backbone),
            num_classes=int(num_classes),
            point_input_dim_value=point_input_dim(geometry_only=effective_geometry_only),
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            knn_k=int(knn_k),
        )
        self.ddpm = DDPM(T=int(diffusion_steps))

    @property
    def geometry_only_enabled(self) -> bool:
        return _resolve_point_geometry_only(self.hparams.geometry_only)

    def prepare_runtime(self) -> None:
        self.ddpm.to(self.device)

    def _model_inputs(self, points: torch.Tensor, point_features: torch.Tensor) -> torch.Tensor:
        return select_point_model_inputs(
            points,
            point_features,
            geometry_only=self.geometry_only_enabled,
        )

    def predict_point_labels(
        self,
        points: torch.Tensor,
        point_features: torch.Tensor,
        valid_geometry: torch.Tensor,
        *,
        sampling_steps: int | None = None,
    ) -> torch.Tensor:
        model_inputs = self._model_inputs(points, point_features)
        x0 = self.ddpm.sample(
            self.model,
            model_inputs,
            sample_shape=(points.shape[0], points.shape[1], int(self.hparams.num_classes)),
            steps=sampling_steps,
        )
        preds = x0.argmax(dim=-1)
        return torch.where(valid_geometry, preds, torch.zeros_like(preds))

    def _evaluation_loss(
        self,
        model_inputs: torch.Tensor,
        labels: torch.Tensor,
        valid_geometry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = labels_to_soft_points(labels, int(self.hparams.num_classes), valid_geometry)
        t = torch.ones(model_inputs.shape[0], device=self.device, dtype=torch.long)
        x_t, eps = self.ddpm.q_sample(x0, t)
        eps_pred = self.model(x_t, t, model_inputs)
        loss = self.ddpm.loss(eps_pred, eps, valid_geometry, labels=labels, class_weights=None, class_dim=2)
        sqrt_ab = self.ddpm._extract(self.ddpm.sqrt_alpha_bars, t, x_t.shape)
        sqrt_1mab = self.ddpm._extract(self.ddpm.sqrt_one_minus_ab, t, x_t.shape)
        x0_est = (x_t - sqrt_1mab * eps_pred) / sqrt_ab.clamp(min=1e-6)
        return loss, x0_est.argmax(dim=-1)

    def _training_loss_and_predictions(
        self,
        model_inputs: torch.Tensor,
        labels: torch.Tensor,
        valid_geometry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = labels_to_soft_points(labels, int(self.hparams.num_classes), valid_geometry)
        t = torch.randint(1, self.ddpm.T + 1, (model_inputs.shape[0],), device=self.device, dtype=torch.long)
        x_t, eps = self.ddpm.q_sample(x0, t)
        eps_pred = self.model(x_t, t, model_inputs)
        loss = self.ddpm.loss(eps_pred, eps, valid_geometry, labels=labels, class_weights=self.class_weights, class_dim=2)
        sqrt_ab = self.ddpm._extract(self.ddpm.sqrt_alpha_bars, t, x_t.shape)
        sqrt_1mab = self.ddpm._extract(self.ddpm.sqrt_one_minus_ab, t, x_t.shape)
        x0_est = (x_t - sqrt_1mab * eps_pred) / sqrt_ab.clamp(min=1e-6)
        return loss, x0_est.argmax(dim=-1)

    def _compute_stage_output(
        self,
        batch: dict,
        *,
        stage: str,
        prediction_mode: str,
        evaluation: bool,
    ) -> dict:
        _ = stage
        points = batch["points"].float()
        point_features = batch["point_features"].float()
        labels = batch["labels"].long()
        valid_geometry = batch["valid_geometry"].bool()
        valid_label = batch["valid_label"].bool()
        batch_size = int(points.shape[0])
        model_inputs = self._model_inputs(points, point_features)
        if evaluation:
            loss, cheap_preds = self._evaluation_loss(model_inputs, labels, valid_geometry)
            if prediction_mode == "full":
                preds = self.predict_point_labels(
                    points,
                    point_features,
                    valid_geometry,
                    sampling_steps=self.resolve_sampling_steps(),
                )
            else:
                preds = torch.where(valid_geometry, cheap_preds, torch.zeros_like(cheap_preds))
        else:
            loss, preds = self._training_loss_and_predictions(model_inputs, labels, valid_geometry)

        metric_mask = valid_geometry & valid_label & (labels > 0)
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
