from typing import Sequence

import torch
import torch.nn as nn


class MultiScaleKnnEigenFeatureExtractor(nn.Module):
    def __init__(
        self,
        scales: Sequence[int] = (16, 32, 64),
        eps: float = 1e-6,
        knn_query_chunk: int = 4096,
    ) -> None:
        super().__init__()
        scales = tuple(int(k) for k in scales)
        if not scales or any(k <= 0 for k in scales):
            raise ValueError(f"Invalid scales: {scales}")
        if int(knn_query_chunk) <= 0:
            raise ValueError(f"knn_query_chunk must be > 0, got {knn_query_chunk}")
        self.scales = scales
        self.eps = float(eps)
        self.knn_query_chunk = int(knn_query_chunk)
        self.per_scale_dim = 11
        self.base_dim = 5
        self.out_dim = self.base_dim + len(self.scales) * self.per_scale_dim

    def _chunked_knn(self, points: torch.Tensor, k_max: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = int(points.shape[0])
        if m <= 1:
            return points.new_zeros((m, 0)), torch.zeros((m, 0), dtype=torch.long, device=points.device)

        support = points
        support_size = m
        k_eff = min(int(k_max), support_size - 1)
        if k_eff <= 0:
            return points.new_zeros((m, 0)), torch.zeros((m, 0), dtype=torch.long, device=points.device)

        all_dists: list[torch.Tensor] = []
        all_idx: list[torch.Tensor] = []
        for start in range(0, m, self.knn_query_chunk):
            end = min(start + self.knn_query_chunk, m)
            q = points[start:end]
            d = torch.cdist(q, support, p=2.0)

            q_global = torch.arange(start, end, device=points.device, dtype=torch.long)
            self_rows = q_global < support_size
            if bool(self_rows.any()):
                row_idx = torch.nonzero(self_rows, as_tuple=False).squeeze(1)
                col_idx = q_global[self_rows]
                d[row_idx, col_idx] = float("inf")

            d_chunk, i_chunk = torch.topk(d, k=k_eff, largest=False, dim=1)
            all_dists.append(d_chunk)
            all_idx.append(i_chunk)
        return torch.cat(all_dists, dim=0), torch.cat(all_idx, dim=0)

    def _per_scale_features(
        self,
        points: torch.Tensor,
        knn_dists: torch.Tensor,
        knn_idx: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        k_eff = min(int(k), int(knn_dists.shape[1]))
        if k_eff <= 0:
            return points.new_zeros((points.shape[0], self.per_scale_dim))

        d = knn_dists[:, :k_eff]
        idx = knn_idx[:, :k_eff]
        nbrs = points[idx]

        mean = nbrs.mean(dim=1, keepdim=True)
        centered = nbrs - mean
        cov = centered.transpose(1, 2).matmul(centered) / float(max(k_eff - 1, 1))
        cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        cov = 0.5 * (cov + cov.transpose(1, 2))
        cov = cov + torch.eye(3, dtype=cov.dtype, device=cov.device).unsqueeze(0) * self.eps

        try:
            evals, evecs = torch.linalg.eigh(cov)
        except RuntimeError:
            evals_cpu, evecs_cpu = torch.linalg.eigh(cov.float().cpu())
            evals = evals_cpu.to(device=cov.device, dtype=cov.dtype)
            evecs = evecs_cpu.to(device=cov.device, dtype=cov.dtype)

        evals_desc = torch.flip(evals, dims=[1])
        evecs_desc = torch.flip(evecs, dims=[2])

        l1 = evals_desc[:, 0].clamp_min(self.eps)
        l2 = evals_desc[:, 1].clamp_min(0.0)
        l3 = evals_desc[:, 2].clamp_min(0.0)
        lsum = (l1 + l2 + l3).clamp_min(self.eps)

        normal = evecs_desc[:, :, 2]
        sign = torch.where(normal[:, 2] < 0.0, -1.0, 1.0).unsqueeze(1)
        normal = normal * sign

        return torch.stack(
            [
                (l1 - l2) / l1,
                (l2 - l3) / l1,
                l3 / l1,
                l3 / lsum,
                normal[:, 0],
                normal[:, 1],
                normal[:, 2],
                normal[:, 2].abs(),
                points[:, 2] - nbrs[:, :, 2].mean(dim=1),
                torch.nan_to_num(d.mean(dim=1), nan=0.0, posinf=0.0, neginf=0.0),
                torch.nan_to_num(d[:, -1], nan=0.0, posinf=0.0, neginf=0.0),
            ],
            dim=1,
        )

    def forward(self, points: torch.Tensor, point_features: torch.Tensor) -> torch.Tensor:
        points = points.float()
        point_features = point_features.float()
        batch_size, num_points, _ = points.shape
        out = points.new_zeros((batch_size, num_points, self.out_dim))

        k_max = max(self.scales)
        for batch_idx in range(batch_size):
            finite_points = torch.isfinite(points[batch_idx]).all(dim=1)
            finite_features = torch.isfinite(point_features[batch_idx]).all(dim=1)
            valid = finite_points & finite_features
            if not bool(valid.any()):
                continue

            p = points[batch_idx, valid]
            f = point_features[batch_idx, valid]
            out_valid = p.new_zeros((p.shape[0], self.out_dim))
            out_valid[:, :3] = p
            out_valid[:, 3:5] = f

            if p.shape[0] >= 2:
                knn_dists, knn_idx = self._chunked_knn(p, k_max)
                offset = self.base_dim
                for scale in self.scales:
                    fs = self._per_scale_features(p, knn_dists, knn_idx, int(scale))
                    out_valid[:, offset : offset + self.per_scale_dim] = fs
                    offset += self.per_scale_dim

            out[batch_idx, valid] = out_valid
        return out
