"""3D point-cloud diffusion denoiser with local EdgeConv-style aggregation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1).float()
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=device))
            * torch.arange(0, half, device=device).float()
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def knn_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """Return KNN indices from xyz with shape (B, N, K)."""
    bsz, npts, _ = xyz.shape
    if npts <= 1:
        return torch.zeros((bsz, npts, 1), dtype=torch.long, device=xyz.device)

    k_eff = max(1, min(k, npts - 1))
    dist = torch.cdist(xyz, xyz)  # (B,N,N)
    eye = torch.eye(npts, dtype=torch.bool, device=xyz.device)[None, :, :]
    dist = dist.masked_fill(eye, float("inf"))
    return dist.topk(k=k_eff, dim=-1, largest=False).indices


class EdgeConvBlock(nn.Module):
    """EdgeConv block with residual connection and timestep conditioning."""

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
        # x: (B,N,D), xyz: (B,N,3), knn_idx: (B,N,K), t_emb: (B,D)
        bsz, npts, dim = x.shape
        _, _, k = knn_idx.shape

        batch_idx = torch.arange(bsz, device=x.device)[:, None, None]

        x_j = x[batch_idx, knn_idx]  # (B,N,K,D)
        xyz_j = xyz[batch_idx, knn_idx]  # (B,N,K,3)

        x_i = x[:, :, None, :].expand(bsz, npts, k, dim)
        xyz_i = xyz[:, :, None, :].expand(bsz, npts, k, 3)

        edge_feat = torch.cat([x_i, x_j - x_i, xyz_j - xyz_i], dim=-1)
        edge_feat = self.edge_mlp(edge_feat)
        agg = edge_feat.max(dim=2).values  # (B,N,D)

        h = self.post(agg)
        h = h + self.t_proj(t_emb)[:, None, :]
        return x + h


class ResMLPBlock(nn.Module):
    """PointNet-style residual pointwise MLP block."""

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
    """PointNet-style diffusion denoiser (no explicit local KNN graph)."""

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
        self.num_classes = int(num_classes)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        in_dim = num_classes + point_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResMLPBlock(hidden_dim, hidden_dim, dropout=dropout) for _ in range(depth)]
        )
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

        x = torch.cat([x_t, xyz], dim=-1)
        x = self.in_proj(x)

        for block in self.blocks:
            x = block(x, t_emb)

        g = x.max(dim=1).values
        g = self.global_proj(g)
        x = x + g[:, None, :]

        return self.out_proj(x)


class PointDiffusionDenoiserEdgeConv(nn.Module):
    """EdgeConv-style diffusion denoiser with local neighborhood aggregation."""

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
        self.num_classes = int(num_classes)
        self.knn_k = int(knn_k)

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        in_dim = num_classes + point_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [EdgeConvBlock(hidden_dim, hidden_dim, dropout=dropout) for _ in range(depth)]
        )

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
        # x_t: (B,N,C), t: (B,), xyz: (B,N,3)
        t_emb = self.time_embed(t)

        x = torch.cat([x_t, xyz], dim=-1)
        x = self.in_proj(x)

        knn_idx = knn_indices(xyz, self.knn_k)

        for block in self.blocks:
            x = block(x, xyz, knn_idx, t_emb)

        g = x.max(dim=1).values
        g = self.global_proj(g)
        x = x + g[:, None, :]

        return self.out_proj(x)


# Backward-compatible default name.
PointDiffusionDenoiser = PointDiffusionDenoiserEdgeConv


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
    name = backbone.lower()
    if name == "pointnet":
        return PointDiffusionDenoiserPointNet(
            num_classes=num_classes,
            point_dim=point_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            time_dim=time_dim,
            dropout=dropout,
        )
    if name == "edgeconv":
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
