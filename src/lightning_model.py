from typing import Optional

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from features import MultiScaleKnnEigenFeatureExtractor


def multiclass_hinge_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Crammer-Singer multi-class hinge loss.
    scores: [M, C], labels: [M]
    """
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
        new_var = (m2 / new_count).clamp_min(self.eps)

        self.running_mean.copy_(new_mean)
        self.running_var.copy_(new_var)
        self.running_count.fill_(new_count)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor, *, update_running: bool) -> torch.Tensor:
        # x: [B,N,D], valid_mask: [B,N]
        valid = valid_mask.bool()
        valid_x = x[valid]

        use_batch = bool(valid_x.numel() > 0 and update_running)
        if use_batch:
            batch_mean = valid_x.mean(dim=0)
            batch_var = valid_x.var(dim=0, unbiased=False).clamp_min(self.eps)
            self._update_running(batch_mean.detach(), batch_var.detach(), int(valid_x.shape[0]))
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean.to(dtype=x.dtype, device=x.device)
            var = self.running_var.to(dtype=x.dtype, device=x.device).clamp_min(self.eps)

        x_norm = (x - mean.view(1, 1, -1)) / torch.sqrt(var.view(1, 1, -1) + self.eps)
        # Keep padded/invalid points neutral.
        x_norm = torch.where(valid.unsqueeze(-1), x_norm, torch.zeros_like(x_norm))
        return x_norm


class LinearSVMPointClassifier(L.LightningModule):
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
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.feature_extractor = MultiScaleKnnEigenFeatureExtractor(
            scales=knn_scales,
            knn_support_size=int(knn_support_size),
            knn_query_chunk=int(knn_query_chunk),
        )
        self.feature_standardizer = RunningFeatureStandardizer(
            feature_dim=self.feature_extractor.out_dim,
            eps=1e-6,
        )
        self.classifier = nn.Linear(self.feature_extractor.out_dim, int(num_classes))
        self.register_buffer("_class_weights", torch.ones(int(num_classes), dtype=torch.float32), persistent=False)
        self.register_buffer(
            "_val_confmat",
            torch.zeros((int(num_classes), int(num_classes)), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_test_confmat",
            torch.zeros((int(num_classes), int(num_classes)), dtype=torch.long),
            persistent=False,
        )
        self._weights_ready = False

    @property
    def class_weights(self) -> Optional[torch.Tensor]:
        if not self.hparams.use_balanced_class_weights or not self._weights_ready:
            return None
        return self._class_weights

    def set_class_weights(self, class_weights: torch.Tensor) -> None:
        class_weights = class_weights.detach().float().to(self.device)
        if class_weights.ndim != 1 or class_weights.shape[0] != int(self.hparams.num_classes):
            raise ValueError(
                f"class_weights shape mismatch: got {tuple(class_weights.shape)}, "
                f"expected ({int(self.hparams.num_classes)},)"
            )
        self._class_weights = class_weights
        self._weights_ready = True

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        points = batch["points"].float()
        point_features = batch["point_features"].float()
        labels = batch["labels"].long()
        valid_geometry = batch["valid_geometry"].bool()
        valid_label = batch["valid_label"].bool()
        batch_size = int(points.shape[0])

        features = self.feature_extractor(points, point_features, valid_geometry)
        features = self.feature_standardizer(
            features,
            valid_geometry,
            update_running=bool(self.training),
        )
        valid = valid_geometry & valid_label & (labels >= 0)

        if not bool(valid.any()):
            loss = self.classifier.weight.sum() * 0.0
            self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            return loss

        x = features[valid]
        y = labels[valid]
        scores = self.classifier(x)
        preds = scores.argmax(dim=1)

        if stage in {"val", "test"}:
            self._update_confmat(stage=stage, preds=preds, labels=y)

        hinge = multiclass_hinge_loss(
            scores,
            y,
            margin=float(self.hparams.margin),
            class_weights=self.class_weights,
        )
        reg = 0.5 * float(self.hparams.svm_reg) * self.classifier.weight.pow(2).sum()
        loss = hinge + reg

        acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def _update_confmat(self, stage: str, preds: torch.Tensor, labels: torch.Tensor) -> None:
        num_classes = int(self.hparams.num_classes)
        if preds.numel() == 0:
            return
        preds = preds.reshape(-1).long()
        labels = labels.reshape(-1).long()
        keep = (labels >= 0) & (labels < num_classes) & (preds >= 0) & (preds < num_classes)
        if not bool(keep.any()):
            return
        preds = preds[keep]
        labels = labels[keep]
        flat = labels * num_classes + preds
        bincount = torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        if stage == "val":
            self._val_confmat += bincount
        else:
            self._test_confmat += bincount

    def _log_iou_metrics(self, stage: str) -> None:
        conf = self._val_confmat if stage == "val" else self._test_confmat
        conf = conf.to(dtype=torch.float32)
        tp = torch.diag(conf)
        fp = conf.sum(dim=0) - tp
        fn = conf.sum(dim=1) - tp
        union = tp + fp + fn

        valid = union > 0
        iou = torch.full_like(union, fill_value=-1.0, dtype=torch.float32)
        iou[valid] = tp[valid] / union[valid].clamp_min(1e-6)

        if bool(valid.any()):
            miou = iou[valid].mean()
        else:
            miou = torch.tensor(0.0, dtype=torch.float32, device=conf.device)

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
            counts_cpu = counts.detach().cpu()
            rank_zero_info(f"Class counts (train supervised points): {counts_cpu.tolist()}")

        if not bool(self.hparams.use_balanced_class_weights):
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=float(self.hparams.learning_rate))
