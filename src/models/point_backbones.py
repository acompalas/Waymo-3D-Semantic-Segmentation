import torch
import torch.nn as nn
import torch.nn.functional as F


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

    k_eff = max(1, min(int(k), npts - 1))
    dist = torch.cdist(xyz, xyz)
    eye = torch.eye(npts, dtype=torch.bool, device=xyz.device)[None, :, :]
    dist = dist.masked_fill(eye, float("inf"))
    return dist.topk(k=k_eff, dim=-1, largest=False).indices


class ResMLPBlock(nn.Module):
    def __init__(self, dim: int, time_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.t_proj = nn.Linear(int(time_dim), dim) if time_dim is not None else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.fc1(F.silu(self.norm1(x)))
        if self.t_proj is not None:
            if t_emb is None:
                raise ValueError("Time embedding is required for time-conditioned blocks.")
            h = h + self.t_proj(t_emb)[:, None, :]
        h = self.fc2(self.drop(F.silu(self.norm2(h))))
        return x + h


class EdgeConvBlock(nn.Module):
    def __init__(self, dim: int, time_dim: int | None = None, dropout: float = 0.1) -> None:
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
        self.t_proj = nn.Linear(int(time_dim), dim) if time_dim is not None else None

    def forward(
        self,
        x: torch.Tensor,
        xyz: torch.Tensor,
        knn_idx: torch.Tensor,
        t_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, npts, dim = x.shape
        _, _, k = knn_idx.shape
        batch_idx = torch.arange(bsz, device=x.device)[:, None, None]
        x_j = x[batch_idx, knn_idx]
        xyz_j = xyz[batch_idx, knn_idx]
        x_i = x[:, :, None, :].expand(bsz, npts, k, dim)
        xyz_i = xyz[:, :, None, :].expand(bsz, npts, k, 3)
        edge_feat = torch.cat([x_i, x_j - x_i, xyz_j - xyz_i], dim=-1)
        edge_feat = self.edge_mlp(edge_feat)
        h = self.post(edge_feat.max(dim=2).values)
        if self.t_proj is not None:
            if t_emb is None:
                raise ValueError("Time embedding is required for time-conditioned blocks.")
            h = h + self.t_proj(t_emb)[:, None, :]
        return x + h


class PointNetBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        depth: int = 6,
        dropout: float = 0.1,
        time_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.uses_time = time_dim is not None
        if self.uses_time:
            self.time_embed = nn.Sequential(
                SinusoidalTimeEmbedding(int(time_dim)),
                nn.Linear(int(time_dim), int(hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
            )
            block_time_dim = int(hidden_dim)
        else:
            self.time_embed = None
            block_time_dim = None

        self.in_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.blocks = nn.ModuleList(
            [ResMLPBlock(int(hidden_dim), block_time_dim, dropout=dropout) for _ in range(int(depth))]
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.output_dim = int(hidden_dim)

    def forward(self, inputs: torch.Tensor, *, t: torch.Tensor | None = None) -> torch.Tensor:
        if self.uses_time:
            if t is None:
                raise ValueError("PointNetBackbone requires diffusion timesteps when time conditioning is enabled.")
            t_emb = self.time_embed(t)
        else:
            t_emb = None

        x = self.in_proj(inputs)
        for block in self.blocks:
            x = block(x, t_emb)
        return x + self.global_proj(x.max(dim=1).values)[:, None, :]


class EdgeConvBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        depth: int = 6,
        dropout: float = 0.1,
        knn_k: int = 16,
        time_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.knn_k = int(knn_k)
        self.uses_time = time_dim is not None
        if self.uses_time:
            self.time_embed = nn.Sequential(
                SinusoidalTimeEmbedding(int(time_dim)),
                nn.Linear(int(time_dim), int(hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
            )
            block_time_dim = int(hidden_dim)
        else:
            self.time_embed = None
            block_time_dim = None

        self.in_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.blocks = nn.ModuleList(
            [EdgeConvBlock(int(hidden_dim), block_time_dim, dropout=dropout) for _ in range(int(depth))]
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.output_dim = int(hidden_dim)

    def forward(self, inputs: torch.Tensor, xyz: torch.Tensor, *, t: torch.Tensor | None = None) -> torch.Tensor:
        if self.uses_time:
            if t is None:
                raise ValueError("EdgeConvBackbone requires diffusion timesteps when time conditioning is enabled.")
            t_emb = self.time_embed(t)
        else:
            t_emb = None

        x = self.in_proj(inputs)
        knn_idx = knn_indices(xyz, self.knn_k)
        for block in self.blocks:
            x = block(x, xyz, knn_idx, t_emb)
        return x + self.global_proj(x.max(dim=1).values)[:, None, :]


def build_point_backbone(
    backbone: str,
    *,
    input_dim: int,
    hidden_dim: int = 256,
    depth: int = 6,
    dropout: float = 0.1,
    knn_k: int = 16,
    time_dim: int | None = None,
) -> nn.Module:
    key = str(backbone).lower()
    if key == "pointnet":
        return PointNetBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            time_dim=time_dim,
        )
    if key == "edgeconv":
        return EdgeConvBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            knn_k=knn_k,
            time_dim=time_dim,
        )
    raise ValueError(f"Unsupported backbone '{backbone}'. Choose from: pointnet, edgeconv.")
