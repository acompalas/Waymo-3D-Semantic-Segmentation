import torch
import torch.nn.functional as F

from .diffusion import DDPM, labels_to_soft_points, labels_to_soft_range


class PointSupervisedBehavior:
    time_dim: int | None = None

    @staticmethod
    def point_backbone_input_dim(*, point_input_dim: int, num_classes: int) -> int:
        _ = num_classes
        return int(point_input_dim)

    def prepare_runtime(self, model) -> None:
        _ = model

    def predict_point_labels(
        self,
        model,
        points: torch.Tensor,
        point_features: torch.Tensor,
        valid_geometry: torch.Tensor,
    ) -> torch.Tensor:
        logits = model.predict_logits(points, point_features)
        preds = logits.argmax(dim=-1)
        return torch.where(valid_geometry, preds, torch.zeros_like(preds))

    def compute_point_stage_output(
        self,
        model,
        batch: dict,
        *,
        stage: str,
        evaluation: bool,
    ) -> dict:
        _ = stage
        _ = evaluation
        points = batch["points"].float()
        point_features = batch["point_features"].float()
        labels = batch["labels"].long()
        valid_label = batch["valid_label"].bool()
        valid_geometry = model._resolve_point_valid_geometry(batch, labels)
        batch_size = int(points.shape[0])

        logits = model.predict_logits(points, point_features)
        preds = logits.argmax(dim=-1)
        metric_mask = valid_label
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
        loss = F.cross_entropy(logits.transpose(1, 2), target, weight=model.class_weights, ignore_index=-100)
        return {
            "loss": loss,
            "preds": torch.where(valid_geometry, preds, torch.zeros_like(preds)),
            "labels": labels,
            "metric_mask": metric_mask,
            "batch_size": batch_size,
        }

    def configure_optimizers(self, model):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(model.hparams.learning_rate),
            weight_decay=float(model.hparams.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(model.trainer.max_epochs)))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class PointDiffusionBehavior:
    time_dim: int | None = 128

    def __init__(self, diffusion_steps: int) -> None:
        self.ddpm = DDPM(T=int(diffusion_steps))

    @staticmethod
    def point_backbone_input_dim(*, point_input_dim: int, num_classes: int) -> int:
        return int(point_input_dim) + int(num_classes)

    def prepare_runtime(self, model) -> None:
        self.ddpm.to(model.device)

    def predict_point_labels(
        self,
        model,
        points: torch.Tensor,
        point_features: torch.Tensor,
        valid_geometry: torch.Tensor,
    ) -> torch.Tensor:
        model_inputs = model.point_model_inputs(points, point_features)
        x0 = self.ddpm.sample(
            model.predict_diffusion_target,
            model_inputs,
            sample_shape=(points.shape[0], points.shape[1], int(model.hparams.num_classes)),
        )
        preds = x0.argmax(dim=-1)
        return torch.where(valid_geometry, preds, torch.zeros_like(preds))

    def _evaluation_loss(self, model, model_inputs: torch.Tensor, labels: torch.Tensor, valid_label: torch.Tensor) -> torch.Tensor:
        x0 = labels_to_soft_points(labels, int(model.hparams.num_classes), valid_label)
        t = torch.ones(model_inputs.shape[0], device=model.device, dtype=torch.long)
        x_t, eps = self.ddpm.q_sample(x0, t)
        eps_pred = model.predict_diffusion_target(x_t, t, model_inputs)
        return self.ddpm.loss(eps_pred, eps, valid_label, labels=labels, class_weights=None, class_dim=2)

    def _training_loss(self, model, model_inputs: torch.Tensor, labels: torch.Tensor, valid_label: torch.Tensor) -> torch.Tensor:
        x0 = labels_to_soft_points(labels, int(model.hparams.num_classes), valid_label)
        t = torch.randint(1, self.ddpm.T + 1, (model_inputs.shape[0],), device=model.device, dtype=torch.long)
        x_t, eps = self.ddpm.q_sample(x0, t)
        eps_pred = model.predict_diffusion_target(x_t, t, model_inputs)
        return self.ddpm.loss(eps_pred, eps, valid_label, labels=labels, class_weights=model.class_weights, class_dim=2)

    def compute_point_stage_output(
        self,
        model,
        batch: dict,
        *,
        stage: str,
        evaluation: bool,
    ) -> dict:
        _ = stage
        points = batch["points"].float()
        point_features = batch["point_features"].float()
        labels = batch["labels"].long()
        valid_label = batch["valid_label"].bool()
        valid_geometry = model._resolve_point_valid_geometry(batch, labels)
        batch_size = int(points.shape[0])
        model_inputs = model.point_model_inputs(points, point_features)
        if evaluation:
            loss = self._evaluation_loss(model, model_inputs, labels, valid_label)
            preds = self.predict_point_labels(
                model,
                points,
                point_features,
                valid_geometry,
            )
            return {
                "loss": loss,
                "preds": preds,
                "labels": labels,
                "metric_mask": valid_label,
                "batch_size": batch_size,
            }
        loss = self._training_loss(model, model_inputs, labels, valid_label)
        return {
            "loss": loss,
            "batch_size": batch_size,
        }

    def configure_optimizers(self, model):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(model.hparams.learning_rate),
            weight_decay=float(model.hparams.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(model.trainer.max_epochs)))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


class RangeSupervisedBehavior:
    def prepare_runtime(self, model) -> None:
        _ = model

    def predict_range_labels(self, model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        logits, labels, valid = model.predict_range_logits(batch)
        preds = logits.argmax(dim=1)
        preds = torch.where(valid, preds, torch.zeros_like(preds))
        return preds, labels

    def compute_range_stage_output(
        self,
        model,
        batch: dict,
        *,
        stage: str,
        evaluation: bool,
    ) -> dict:
        _ = stage
        _ = evaluation
        logits, labels, valid = model.predict_range_logits(batch)
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
        loss = F.cross_entropy(logits, target, weight=model.class_weights, ignore_index=-100)
        return {
            "loss": loss,
            "preds": preds,
            "labels": labels,
            "metric_mask": valid,
            "batch_size": batch_size,
        }

    def configure_optimizers(self, model):
        return torch.optim.Adam(model.parameters(), lr=float(model.hparams.learning_rate))


class RangeDiffusionBehavior:
    def __init__(self, diffusion_steps: int) -> None:
        self.ddpm = DDPM(T=int(diffusion_steps))

    def prepare_runtime(self, model) -> None:
        self.ddpm.to(model.device)

    def _predict_labels(self, model, cond: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x0 = self.ddpm.sample(
            model.predict_diffusion_target,
            cond,
            sample_shape=(cond.shape[0], int(model.hparams.num_classes), cond.shape[2], cond.shape[3]),
        )
        preds = x0.argmax(dim=1)
        return torch.where(valid, preds, torch.zeros_like(preds))

    def predict_range_labels(self, model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        cond, labels, valid = model.prepare_range_batch(batch)
        preds = self._predict_labels(model, cond, valid)
        return preds, labels

    def _evaluation_loss(self, model, cond: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x0 = labels_to_soft_range(labels, int(model.hparams.num_classes), valid)
        t = torch.ones(cond.shape[0], device=model.device, dtype=torch.long)
        x_t, eps = self.ddpm.q_sample(x0, t)
        eps_pred = model.predict_diffusion_target(x_t, t, cond)
        return self.ddpm.loss(eps_pred, eps, valid, labels=labels, class_weights=None, class_dim=1)

    def _training_loss(self, model, cond: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x0 = labels_to_soft_range(labels, int(model.hparams.num_classes), valid)
        t = torch.randint(1, self.ddpm.T + 1, (cond.shape[0],), device=model.device, dtype=torch.long)
        x_t, eps = self.ddpm.q_sample(x0, t)
        eps_pred = model.predict_diffusion_target(x_t, t, cond)
        return self.ddpm.loss(eps_pred, eps, valid, labels=labels, class_weights=model.class_weights, class_dim=1)

    def compute_range_stage_output(
        self,
        model,
        batch: dict,
        *,
        stage: str,
        evaluation: bool,
    ) -> dict:
        _ = stage
        cond, labels, valid = model.prepare_range_batch(batch)
        batch_size = int(batch["range_images"].shape[0])
        if evaluation:
            loss = self._evaluation_loss(model, cond, labels, valid)
            preds = self._predict_labels(model, cond, valid)
            return {
                "loss": loss,
                "preds": preds,
                "labels": labels,
                "metric_mask": valid & (labels > 0),
                "batch_size": batch_size,
            }
        loss = self._training_loss(model, cond, labels, valid)
        return {
            "loss": loss,
            "batch_size": batch_size,
        }

    def configure_optimizers(self, model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(model.hparams.learning_rate))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(model.trainer.max_epochs)))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def build_point_behavior(behavior: str, *, diffusion_steps: int):
    key = str(behavior).lower()
    if key == "supervised":
        return PointSupervisedBehavior()
    if key == "diffusion":
        return PointDiffusionBehavior(diffusion_steps=int(diffusion_steps))
    raise ValueError(f"Unsupported point behavior '{behavior}'.")


def build_range_behavior(behavior: str, *, diffusion_steps: int):
    key = str(behavior).lower()
    if key == "supervised":
        return RangeSupervisedBehavior()
    if key == "diffusion":
        return RangeDiffusionBehavior(diffusion_steps=int(diffusion_steps))
    raise ValueError(f"Unsupported range behavior '{behavior}'.")
