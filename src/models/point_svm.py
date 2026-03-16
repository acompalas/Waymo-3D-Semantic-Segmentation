from typing import Optional

import torch
import torch.nn as nn

from .base import PointCloudSegmentationModel
from .features import MultiScaleKnnEigenFeatureExtractor
from .inputs import select_point_features_for_extractor


def multiclass_hinge_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    true_scores = scores.gather(1, labels.unsqueeze(1))
    losses = (scores - true_scores + margin).clamp_min(0.0)
    losses.scatter_(1, labels.unsqueeze(1), 0.0)
    per_sample = losses.sum(dim=1)

    if class_weights is None:
        return per_sample.mean()

    sample_w = class_weights[labels].clamp_min(0.0)
    denom = sample_w.sum().clamp_min(1e-6)
    return (per_sample * sample_w).sum() / denom


class RunningFeatureStandardizer(nn.Module):
    def __init__(self, feature_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        d = int(feature_dim)
        self.eps = float(eps)
        self.register_buffer("running_mean", torch.zeros(d, dtype=torch.float32), persistent=True)
        self.register_buffer("running_var", torch.ones(d, dtype=torch.float32), persistent=True)
        self.register_buffer("running_count", torch.tensor(0.0, dtype=torch.float32), persistent=True)

    @torch.no_grad()
    def _update_running(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int) -> None:
        n = float(batch_count)
        if n <= 0:
            return
        if float(self.running_count.item()) <= 0:
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_var.clamp_min(self.eps))
            self.running_count.fill_(n)
            return

        old_count = float(self.running_count.item())
        new_count = old_count + n
        delta = batch_mean - self.running_mean
        new_mean = self.running_mean + delta * (n / new_count)

        m_a = self.running_var * old_count
        m_b = batch_var * n
        m2 = m_a + m_b + delta.pow(2) * (old_count * n / new_count)
        self.running_var.copy_((m2 / new_count).clamp_min(self.eps))
        self.running_mean.copy_(new_mean)
        self.running_count.fill_(new_count)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None, *, update_running: bool) -> torch.Tensor:
        if valid_mask is None:
            valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        else:
            valid = valid_mask.bool()
        valid_x = x[valid]

        if bool(valid_x.numel() > 0 and update_running):
            batch_mean = valid_x.mean(dim=0)
            batch_var = valid_x.var(dim=0, unbiased=False).clamp_min(self.eps)
            self._update_running(batch_mean.detach(), batch_var.detach(), int(valid_x.shape[0]))
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean.to(dtype=x.dtype, device=x.device)
            var = self.running_var.to(dtype=x.dtype, device=x.device).clamp_min(self.eps)

        x_norm = (x - mean.view(1, 1, -1)) / torch.sqrt(var.view(1, 1, -1) + self.eps)
        return torch.where(valid.unsqueeze(-1), x_norm, torch.zeros_like(x_norm))


class LinearSVMPointClassifier(PointCloudSegmentationModel):
    def __init__(
        self,
        num_classes: int = 23,
        learning_rate: float = 1e-3,
        svm_reg: float = 1e-4,
        margin: float = 1.0,
        knn_scales: tuple[int, ...] = (16, 32, 64),
        knn_support_size: int = 16384,
        knn_query_chunk: int = 4096,
        use_balanced_class_weights: bool = True,
        geometry_only: bool = False,
    ) -> None:
        super().__init__(num_classes=num_classes, use_balanced_class_weights=use_balanced_class_weights)
        self.save_hyperparameters()

        self.feature_extractor = MultiScaleKnnEigenFeatureExtractor(
            scales=knn_scales,
            knn_support_size=int(knn_support_size),
            knn_query_chunk=int(knn_query_chunk),
        )
        self.feature_standardizer = RunningFeatureStandardizer(self.feature_extractor.out_dim)
        self.classifier = nn.Linear(self.feature_extractor.out_dim, int(num_classes))

    def _predict_scores(
        self,
        points: torch.Tensor,
        point_features: torch.Tensor,
        *,
        update_running: bool = False,
    ) -> torch.Tensor:
        point_features = select_point_features_for_extractor(
            point_features,
            geometry_only=bool(self.hparams.geometry_only),
        )
        features = self.feature_extractor(points, point_features)
        features = self.feature_standardizer(features, update_running=update_running)
        return self.classifier(features)

    def predict_point_labels(
        self,
        points: torch.Tensor,
        point_features: torch.Tensor,
        valid_geometry: torch.Tensor,
        *,
        sampling_steps: int | None = None,
    ) -> torch.Tensor:
        _ = valid_geometry
        _ = sampling_steps
        scores = self._predict_scores(points, point_features, update_running=False)
        return scores.argmax(dim=-1)

    def _compute_stage_output(
        self,
        batch: dict,
        *,
        stage: str,
        prediction_mode: str,
        evaluation: bool,
    ) -> dict:
        _ = prediction_mode
        points = batch["points"].float()
        point_features = batch["point_features"].float()
        labels = batch["labels"].long()
        valid_label = batch["valid_label"].bool()
        batch_size = int(points.shape[0])

        scores = self._predict_scores(
            points,
            point_features,
            update_running=bool(stage == "train" and not evaluation),
        )
        preds = scores.argmax(dim=-1)
        metric_mask = valid_label
        if not bool(metric_mask.any()):
            loss = self.classifier.weight.sum() * 0.0
            return {
                "loss": loss,
                "preds": torch.zeros_like(labels),
                "labels": labels,
                "metric_mask": metric_mask,
                "batch_size": batch_size,
            }

        x = scores[metric_mask]
        y = labels[metric_mask]

        hinge = multiclass_hinge_loss(x, y, margin=float(self.hparams.margin), class_weights=self.class_weights)
        reg = 0.5 * float(self.hparams.svm_reg) * self.classifier.weight.pow(2).sum()
        loss = hinge + reg
        return {
            "loss": loss,
            "preds": preds,
            "labels": labels,
            "metric_mask": metric_mask,
            "batch_size": batch_size,
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=float(self.hparams.learning_rate))
