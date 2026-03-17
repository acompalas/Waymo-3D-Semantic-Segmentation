import torch

from .base import PointCloudSegmentationModel, RangeImageSegmentationModel
from .backbones.point.registry import build_point_backbone
from .backbones.range.registry import build_range_backbone
from .behaviors import DiffusionBehavior, build_behavior
from .diffusion import labels_to_soft_points, labels_to_soft_range
from .heads import PointHead, RangeHead
from .inputs import point_geometry, point_input_dim, range_input_channels, select_point_model_inputs, select_range_model_inputs
from .losses import focal_cross_entropy_loss


def _resolve_class_weight_alpha(class_weight_alpha: float, use_balanced_class_weights: bool | None) -> float:
    if use_balanced_class_weights is None:
        return float(class_weight_alpha)
    return float(class_weight_alpha if use_balanced_class_weights else 0.0)


def _clean_backbone_kwargs(backbone_kwargs: dict) -> dict:
    return {key: value for key, value in backbone_kwargs.items() if value is not None}


class PointCloudTaskModel(PointCloudSegmentationModel):
    def __init__(
        self,
        num_classes: int = 23,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        backbone: str = "edgeconv",
        behavior: str = "direct",
        diffusion_steps: int = 1000,
        class_weight_alpha: float = 1.0,
        focal_loss_gamma: float = 0.0,
        geometry_only: bool = False,
        use_balanced_class_weights: bool | None = None,
        **backbone_kwargs,
    ) -> None:
        super().__init__(num_classes=num_classes)
        class_weight_alpha = _resolve_class_weight_alpha(class_weight_alpha, use_balanced_class_weights)
        backbone_name = str(backbone)
        behavior_name = str(behavior)
        backbone_kwargs = _clean_backbone_kwargs(dict(backbone_kwargs))
        self.save_hyperparameters(
            {
                "num_classes": int(num_classes),
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "backbone": backbone_name,
                "behavior": behavior_name,
                "diffusion_steps": int(diffusion_steps),
                "class_weight_alpha": float(class_weight_alpha),
                "focal_loss_gamma": float(focal_loss_gamma),
                "geometry_only": bool(geometry_only),
                **backbone_kwargs,
            }
        )
        self.behavior_impl = build_behavior(behavior_name, diffusion_steps=int(diffusion_steps))
        model_input_dim = point_input_dim(geometry_only=bool(geometry_only))
        backbone_input_dim = self.behavior_impl.point_backbone_input_dim(
            point_input_dim=model_input_dim,
            num_classes=int(num_classes),
        )
        self.backbone = build_point_backbone(
            backbone_name,
            input_dim=backbone_input_dim,
            time_dim=self.behavior_impl.time_dim,
            point_input_dim=model_input_dim,
            num_classes=int(num_classes),
            **backbone_kwargs,
        )
        self.head = PointHead(input_dim=int(self.backbone.output_dim), output_dim=int(num_classes))

    def prepare_runtime(self) -> None:
        self.behavior_impl.prepare_runtime(self)

    def point_model_inputs(self, points: torch.Tensor, point_features: torch.Tensor) -> torch.Tensor:
        return select_point_model_inputs(
            points,
            point_features,
            geometry_only=bool(self.hparams.geometry_only),
        )

    def predict_logits(self, points: torch.Tensor, point_features: torch.Tensor) -> torch.Tensor:
        model_inputs = self.point_model_inputs(points, point_features)
        hidden = self.backbone(model_inputs, xyz=point_geometry(model_inputs), t=None)
        return self.head(hidden)

    def predict_diffusion_target(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(torch.cat([x_t, cond], dim=-1), xyz=point_geometry(cond), t=t)
        return self.head(hidden)

    def predict_point_labels(
        self,
        points: torch.Tensor,
        point_features: torch.Tensor,
        valid_geometry: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.behavior_impl, DiffusionBehavior):
            model_inputs = self.point_model_inputs(points, point_features)
            x0 = self.behavior_impl.ddpm.sample(
                self.predict_diffusion_target,
                model_inputs,
                sample_shape=(points.shape[0], points.shape[1], int(self.hparams.num_classes)),
            )
            preds = x0.argmax(dim=-1)
        else:
            preds = self.predict_logits(points, point_features).argmax(dim=-1)
        return torch.where(valid_geometry, preds, torch.zeros_like(preds))

    def _point_batch_tensors(
        self,
        batch: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        points = batch["points"].float()
        point_features = batch["point_features"].float()
        labels = batch["labels"].long()
        valid_label = batch["valid_label"].bool()
        valid_geometry = self._resolve_point_valid_geometry(batch, labels)
        batch_size = int(points.shape[0])
        return points, point_features, labels, valid_label, valid_geometry, batch_size

    def compute_direct_stage_output(self, batch: dict) -> dict:
        points, point_features, labels, valid_label, valid_geometry, batch_size = self._point_batch_tensors(batch)
        logits = self.predict_logits(points, point_features)
        preds = logits.argmax(dim=-1)
        if not bool(valid_label.any()):
            loss = logits.sum() * 0.0
            return {
                "loss": loss,
                "preds": torch.zeros_like(labels),
                "labels": labels,
                "metric_mask": valid_label,
                "batch_size": batch_size,
            }

        target = labels.clone()
        target[~valid_label] = -100
        loss = focal_cross_entropy_loss(
            logits.transpose(1, 2),
            target,
            class_weights=self.class_weights,
            gamma=float(self.hparams.focal_loss_gamma),
            ignore_index=-100,
            class_dim=1,
        )
        return {
            "loss": loss,
            "preds": torch.where(valid_geometry, preds, torch.zeros_like(preds)),
            "labels": labels,
            "metric_mask": valid_label,
            "batch_size": batch_size,
        }

    def compute_diffusion_training_stage_output(self, batch: dict, ddpm) -> dict:
        points, point_features, labels, valid_label, _valid_geometry, batch_size = self._point_batch_tensors(batch)
        model_inputs = self.point_model_inputs(points, point_features)
        x0 = labels_to_soft_points(labels, int(self.hparams.num_classes), valid_label)
        t = torch.randint(1, ddpm.T + 1, (model_inputs.shape[0],), device=self.device, dtype=torch.long)
        x_t, eps = ddpm.q_sample(x0, t)
        eps_pred = self.predict_diffusion_target(x_t, t, model_inputs)
        loss = ddpm.loss(eps_pred, eps, valid_label, labels=labels, class_weights=self.class_weights, class_dim=2)
        return {
            "loss": loss,
            "batch_size": batch_size,
        }

    def compute_diffusion_evaluation_stage_output(self, batch: dict, ddpm) -> dict:
        points, point_features, labels, valid_label, valid_geometry, batch_size = self._point_batch_tensors(batch)
        model_inputs = self.point_model_inputs(points, point_features)
        x0 = labels_to_soft_points(labels, int(self.hparams.num_classes), valid_label)
        t = torch.ones(model_inputs.shape[0], device=self.device, dtype=torch.long)
        x_t, eps = ddpm.q_sample(x0, t)
        eps_pred = self.predict_diffusion_target(x_t, t, model_inputs)
        loss = ddpm.loss(eps_pred, eps, valid_label, labels=labels, class_weights=None, class_dim=2)
        preds = self.predict_point_labels(points, point_features, valid_geometry)
        return {
            "loss": loss,
            "preds": preds,
            "labels": labels,
            "metric_mask": valid_label,
            "batch_size": batch_size,
        }

    def _compute_stage_output(
        self,
        batch: dict,
        *,
        evaluation: bool,
    ) -> dict:
        return self.behavior_impl.compute_stage_output(self, batch, evaluation=evaluation)


class RangeImageTaskModel(RangeImageSegmentationModel):
    def __init__(
        self,
        num_classes: int = 23,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        backbone: str = "unet",
        behavior: str = "direct",
        diffusion_steps: int = 1000,
        class_weight_alpha: float = 1.0,
        focal_loss_gamma: float = 0.0,
        geometry_only: bool = False,
        use_balanced_class_weights: bool | None = None,
        **backbone_kwargs,
    ) -> None:
        super().__init__(num_classes=num_classes)
        class_weight_alpha = _resolve_class_weight_alpha(class_weight_alpha, use_balanced_class_weights)
        backbone_name = str(backbone)
        behavior_name = str(behavior)
        backbone_kwargs = _clean_backbone_kwargs(dict(backbone_kwargs))
        input_channels = range_input_channels(geometry_only=bool(geometry_only))
        self.save_hyperparameters(
            {
                "num_classes": int(num_classes),
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "backbone": backbone_name,
                "behavior": behavior_name,
                "diffusion_steps": int(diffusion_steps),
                "class_weight_alpha": float(class_weight_alpha),
                "focal_loss_gamma": float(focal_loss_gamma),
                "geometry_only": bool(geometry_only),
                **backbone_kwargs,
            }
        )
        self.behavior_impl = build_behavior(behavior_name, diffusion_steps=int(diffusion_steps))
        self.backbone = build_range_backbone(
            backbone_name,
            input_channels=int(input_channels),
            num_classes=int(num_classes),
            **backbone_kwargs,
        )
        self.head = RangeHead(
            input_channels=int(self.backbone.output_channels),
            output_channels=int(num_classes),
        )

    def prepare_runtime(self) -> None:
        self.behavior_impl.prepare_runtime(self)

    def prepare_range_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = select_range_model_inputs(batch["range_images"], geometry_only=bool(self.hparams.geometry_only))
        labels = batch["semantic"].long()
        valid = batch["valid_label"].bool()
        if x.ndim != 5:
            raise ValueError(f"Expected selected range input shape [B,R,H,W,C], got {tuple(x.shape)}")
        bsz, returns, height, width, channels = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(bsz * returns, channels, height, width)
        labels = labels.reshape(bsz * returns, height, width)
        valid = valid.reshape(bsz * returns, height, width)
        return x, labels, valid

    def predict_range_logits(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, labels, valid = self.prepare_range_batch(batch)
        hidden = self.backbone(x, cond=None, t=None)
        logits = self.head(hidden)
        return logits, labels, valid & (labels > 0)

    def predict_diffusion_target(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x_t, cond=cond, t=t)
        return self.head(hidden)

    def predict_range_labels(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.behavior_impl, DiffusionBehavior):
            cond, labels, valid = self.prepare_range_batch(batch)
            x0 = self.behavior_impl.ddpm.sample(
                self.predict_diffusion_target,
                cond,
                sample_shape=(cond.shape[0], int(self.hparams.num_classes), cond.shape[2], cond.shape[3]),
            )
            preds = x0.argmax(dim=1)
            preds = torch.where(valid, preds, torch.zeros_like(preds))
            return preds, labels
        logits, labels, valid = self.predict_range_logits(batch)
        preds = logits.argmax(dim=1)
        preds = torch.where(valid, preds, torch.zeros_like(preds))
        return preds, labels

    def compute_direct_stage_output(self, batch: dict) -> dict:
        logits, labels, valid = self.predict_range_logits(batch)
        preds = logits.argmax(dim=1)
        batch_size = int(batch["range_images"].shape[0])

        if not bool(valid.any()):
            loss = logits.sum() * 0.0
            return {
                "loss": loss,
                "preds": torch.zeros_like(labels),
                "labels": labels,
                "metric_mask": valid,
                "batch_size": batch_size,
            }

        target = labels.clone()
        target[~valid] = -100
        loss = focal_cross_entropy_loss(
            logits,
            target,
            class_weights=self.class_weights,
            gamma=float(self.hparams.focal_loss_gamma),
            ignore_index=-100,
            class_dim=1,
        )
        return {
            "loss": loss,
            "preds": preds,
            "labels": labels,
            "metric_mask": valid,
            "batch_size": batch_size,
        }

    def compute_diffusion_training_stage_output(self, batch: dict, ddpm) -> dict:
        cond, labels, valid = self.prepare_range_batch(batch)
        batch_size = int(batch["range_images"].shape[0])
        x0 = labels_to_soft_range(labels, int(self.hparams.num_classes), valid)
        t = torch.randint(1, ddpm.T + 1, (cond.shape[0],), device=self.device, dtype=torch.long)
        x_t, eps = ddpm.q_sample(x0, t)
        eps_pred = self.predict_diffusion_target(x_t, t, cond)
        loss = ddpm.loss(eps_pred, eps, valid, labels=labels, class_weights=self.class_weights, class_dim=1)
        return {
            "loss": loss,
            "batch_size": batch_size,
        }

    def compute_diffusion_evaluation_stage_output(self, batch: dict, ddpm) -> dict:
        cond, labels, valid = self.prepare_range_batch(batch)
        batch_size = int(batch["range_images"].shape[0])
        x0 = labels_to_soft_range(labels, int(self.hparams.num_classes), valid)
        t = torch.ones(cond.shape[0], device=self.device, dtype=torch.long)
        x_t, eps = ddpm.q_sample(x0, t)
        eps_pred = self.predict_diffusion_target(x_t, t, cond)
        loss = ddpm.loss(eps_pred, eps, valid, labels=labels, class_weights=None, class_dim=1)
        preds, labels = self.predict_range_labels(batch)
        return {
            "loss": loss,
            "preds": preds,
            "labels": labels,
            "metric_mask": valid & (labels > 0),
            "batch_size": batch_size,
        }

    def _compute_stage_output(
        self,
        batch: dict,
        *,
        evaluation: bool,
    ) -> dict:
        return self.behavior_impl.compute_stage_output(self, batch, evaluation=evaluation)
