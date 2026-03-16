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

    def forward(self, x: torch.Tensor, xyz: torch.Tensor, knn_idx: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
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


def _square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return torch.cdist(src, dst)


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    bsz = points.shape[0]
    if idx.ndim == 2:
        batch_idx = torch.arange(bsz, device=points.device)[:, None]
    else:
        batch_idx = torch.arange(bsz, device=points.device)[:, None, None]
    return points[batch_idx, idx]


def _farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    device = xyz.device
    bsz, npts, _ = xyz.shape
    centroids = torch.zeros(bsz, npoint, dtype=torch.long, device=device)
    distance = torch.full((bsz, npts), float("inf"), device=device)
    farthest = torch.randint(0, npts, (bsz,), device=device)
    batch_indices = torch.arange(bsz, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(bsz, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(dim=-1).indices
    return centroids


def _query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    dist = _square_distance(new_xyz, xyz)
    group_idx = dist.argsort(dim=-1)[:, :, :nsample]
    if radius > 0:
        mask = dist.gather(-1, group_idx) > radius
        if mask.any():
            first = group_idx[:, :, :1].expand_as(group_idx)
            group_idx = torch.where(mask, first, group_idx)
    return group_idx


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, mlp: list[int], use_xyz: bool = True, in_channels: int = 0) -> None:
        super().__init__()
        self.npoint = int(npoint)
        self.radius = float(radius)
        self.nsample = int(nsample)
        self.use_xyz = bool(use_xyz)

        last_channel = int(in_channels) + (3 if self.use_xyz else 0)
        layers: list[nn.Module] = []
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, int(out_channel), 1))
            layers.append(nn.SiLU())
            last_channel = int(out_channel)
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, npts, _ = xyz.shape
        npoint = min(self.npoint, npts)
        fps_idx = _farthest_point_sample(xyz, npoint)
        new_xyz = _index_points(xyz, fps_idx)

        group_idx = _query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = _index_points(xyz, group_idx)
        grouped_xyz = grouped_xyz - new_xyz[:, :, None, :]

        if points is not None:
            grouped_points = _index_points(points, group_idx)
            if self.use_xyz:
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz

        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()
        new_points = self.mlp(grouped_points).max(dim=-1).values
        return new_xyz, new_points.transpose(1, 2).contiguous()


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channels: int, mlp: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_channel = int(in_channels)
        for out_channel in mlp:
            layers.append(nn.Conv1d(last_channel, int(out_channel), 1))
            layers.append(nn.SiLU())
            last_channel = int(out_channel)
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor, points1: torch.Tensor | None, points2: torch.Tensor) -> torch.Tensor:
        dist = _square_distance(xyz1, xyz2)
        if xyz2.shape[1] == 1:
            interpolated = points2.repeat(1, xyz1.shape[1], 1)
        else:
            dists, idx = dist.topk(k=3, dim=-1, largest=False)
            dists = torch.clamp(dists, min=1e-10)
            weight = (1.0 / dists)
            weight = weight / weight.sum(dim=-1, keepdim=True)
            interpolated = (_index_points(points2, idx) * weight[..., None]).sum(dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=-1)
        else:
            new_points = interpolated

        return self.mlp(new_points.transpose(1, 2)).transpose(1, 2).contiguous()
