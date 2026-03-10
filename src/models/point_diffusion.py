import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PointCloudSegmentationModel
from .diffusion import DDPM, labels_to_soft_points


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1).float()
        half = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device))
            * torch.arange(0, half, device=t.device).float()
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return F.pad(emb, (0, 1)) if self.dim % 2 == 1 else emb


def knn_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    bsz, npts, _ = xyz.shape
    if npts <= 1:
        return torch.zeros((bsz, npts, 1), dtype=torch.long, device=xyz.device)

    k_eff = max(1, min(k, npts - 1))
    dist = torch.cdist(xyz, xyz)
    eye = torch.eye(npts, dtype=torch.bool, device=xyz.device)[None, :, :]
    dist = dist.masked_fill(eye, float("inf"))
    return dist.topk(k=k_eff, dim=-1, largest=False).indices


class EdgeConvBlock(nn.Module):
    def __init__(self, dim: int, time_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.LayerNorm(dim * 2 + 3),
            nn.Linear(dim * 2 + 3, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
        self.post = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.t_proj = nn.Linear(time_dim, dim)

    def forward(self, x: torch.Tensor, xyz: torch.Tensor, knn_idx: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        bsz, npts, dim = x.shape
        _, _, k = knn_idx.shape
        batch_idx = torch.arange(bsz, device=x.device)[:, None, None]
        x_j = x[batch_idx, knn_idx]
        xyz_j = xyz[batch_idx, knn_idx]
        x_i = x[:, :, None, :].expand(bsz, npts, k, dim)
        xyz_i = xyz[:, :, None, :].expand(bsz, npts, k, 3)
        edge_feat = torch.cat([x_i, x_j - x_i, xyz_j - xyz_i], dim=-1)
        edge_feat = self.edge_mlp(edge_feat)
        agg = edge_feat.max(dim=2).values
        h = self.post(agg) + self.t_proj(t_emb)[:, None, :]
        return x + h


class ResMLPBlock(nn.Module):
    def __init__(self, dim: int, time_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.t_proj = nn.Linear(time_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.fc1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, None, :]
        h = self.fc2(self.drop(F.silu(self.norm2(h))))
        return x + h


class PointDiffusionDenoiserPointNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 23,
        point_dim: int = 3,
        hidden_dim: int = 256,
        depth: int = 6,
        time_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_proj = nn.Linear(num_classes + point_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResMLPBlock(hidden_dim, hidden_dim, dropout=dropout) for _ in range(depth)])
        self.global_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        x = self.in_proj(torch.cat([x_t, xyz], dim=-1))
        for block in self.blocks:
            x = block(x, t_emb)
        x = x + self.global_proj(x.max(dim=1).values)[:, None, :]
        return self.out_proj(x)


class PointDiffusionDenoiserEdgeConv(nn.Module):
    def __init__(
        self,
        num_classes: int = 23,
        point_dim: int = 3,
        hidden_dim: int = 256,
        depth: int = 6,
        time_dim: int = 128,
        dropout: float = 0.1,
        knn_k: int = 16,
    ) -> None:
        super().__init__()
        self.knn_k = int(knn_k)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_proj = nn.Linear(num_classes + point_dim, hidden_dim)
        self.blocks = nn.ModuleList([EdgeConvBlock(hidden_dim, hidden_dim, dropout=dropout) for _ in range(depth)])
        self.global_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        x = self.in_proj(torch.cat([x_t, xyz], dim=-1))
        knn_idx = knn_indices(xyz, self.knn_k)
        for block in self.blocks:
            x = block(x, xyz, knn_idx, t_emb)
        x = x + self.global_proj(x.max(dim=1).values)[:, None, :]
        return self.out_proj(x)


def build_denoiser(
    backbone: str = "edgeconv",
    num_classes: int = 23,
    point_dim: int = 3,
    hidden_dim: int = 256,
    depth: int = 6,
    time_dim: int = 128,
    dropout: float = 0.1,
    knn_k: int = 16,
) -> nn.Module:
    if backbone == "pointnet":
        return PointDiffusionDenoiserPointNet(
            num_classes=num_classes,
            point_dim=point_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            time_dim=time_dim,
            dropout=dropout,
        )
    if backbone == "edgeconv":
        return PointDiffusionDenoiserEdgeConv(
            num_classes=num_classes,
            point_dim=point_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            time_dim=time_dim,
            dropout=dropout,
            knn_k=knn_k,
        )
    raise ValueError(f"Unsupported backbone '{backbone}'. Choose from: pointnet, edgeconv.")


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
    ) -> None:
        super().__init__(num_classes=num_classes, use_balanced_class_weights=use_balanced_class_weights)
        self.save_hyperparameters()
        self.model = build_denoiser(
            backbone=str(backbone),
            num_classes=int(num_classes),
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            knn_k=int(knn_k),
        )
        self.ddpm = DDPM(T=int(diffusion_steps))

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.ddpm.to(self.device)

    def on_validation_start(self) -> None:
        self.ddpm.to(self.device)

    def on_test_start(self) -> None:
        self.ddpm.to(self.device)

    def predict_point_labels(
        self,
        points: torch.Tensor,
        point_features: torch.Tensor,
        valid_geometry: torch.Tensor,
        *,
        sampling_steps: int | None = None,
    ) -> torch.Tensor:
        _ = point_features
        x0 = self.ddpm.sample(
            self.model,
            points,
            sample_shape=(points.shape[0], points.shape[1], int(self.hparams.num_classes)),
            steps=sampling_steps,
        )
        preds = x0.argmax(dim=-1)
        return torch.where(valid_geometry, preds, torch.zeros_like(preds))

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        points = batch["points"].float()
        labels = batch["labels"].long()
        valid_geometry = batch["valid_geometry"].bool()
        x0 = labels_to_soft_points(labels, int(self.hparams.num_classes), valid_geometry)
        t = torch.randint(1, self.ddpm.T + 1, (points.shape[0],), device=self.device, dtype=torch.long)
        x_t, eps = self.ddpm.q_sample(x0, t)
        eps_pred = self.model(x_t, t, points)
        loss = self.ddpm.loss(eps_pred, eps, valid_geometry, labels=labels, class_weights=self.class_weights, class_dim=2)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=int(points.shape[0]))
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        points = batch["points"].float()
        labels = batch["labels"].long()
        valid_geometry = batch["valid_geometry"].bool()
        valid_label = batch["valid_label"].bool()
        x0 = labels_to_soft_points(labels, int(self.hparams.num_classes), valid_geometry)
        t = torch.ones(points.shape[0], device=self.device, dtype=torch.long)
        x_t, eps = self.ddpm.q_sample(x0, t)
        eps_pred = self.model(x_t, t, points)
        loss = self.ddpm.loss(eps_pred, eps, valid_geometry, labels=labels, class_weights=None, class_dim=2)
        sqrt_ab = self.ddpm._extract(self.ddpm.sqrt_alpha_bars, t, x_t.shape)
        sqrt_1mab = self.ddpm._extract(self.ddpm.sqrt_one_minus_ab, t, x_t.shape)
        x0_est = (x_t - sqrt_1mab * eps_pred) / sqrt_ab.clamp(min=1e-6)
        preds = x0_est.argmax(dim=-1)
        metric_mask = valid_geometry & valid_label & (labels > 0)
        self.update_confmat(stage="val", preds=preds[metric_mask], labels=labels[metric_mask])
        acc = (preds[metric_mask] == labels[metric_mask]).float().mean() if bool(metric_mask.any()) else loss.new_tensor(0.0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=int(points.shape[0]))
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=int(points.shape[0]))
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        points = batch["points"].float()
        labels = batch["labels"].long()
        valid_geometry = batch["valid_geometry"].bool()
        valid_label = batch["valid_label"].bool()
        x0 = labels_to_soft_points(labels, int(self.hparams.num_classes), valid_geometry)
        t = torch.ones(points.shape[0], device=self.device, dtype=torch.long)
        x_t, eps = self.ddpm.q_sample(x0, t)
        eps_pred = self.model(x_t, t, points)
        loss = self.ddpm.loss(eps_pred, eps, valid_geometry, labels=labels, class_weights=None, class_dim=2)
        preds = self.predict_point_labels(
            points,
            batch["point_features"].float(),
            valid_geometry,
            sampling_steps=self.resolve_sampling_steps(),
        )
        metric_mask = valid_geometry & valid_label & (labels > 0)
        self.update_confmat(stage="test", preds=preds[metric_mask], labels=labels[metric_mask])
        acc = (preds[metric_mask] == labels[metric_mask]).float().mean() if bool(metric_mask.any()) else loss.new_tensor(0.0)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=int(points.shape[0]))
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=int(points.shape[0]))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(self.trainer.max_epochs)))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
