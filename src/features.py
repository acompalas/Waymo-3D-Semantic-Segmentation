from typing import Sequence

import torch
import torch.nn as nn


class MultiScaleKnnEigenFeatureExtractor(nn.Module):
    """
    kNN multi-scale local geometry features for per-point classification.

    Per point output:
    - raw xyz (3) + intensity/elongation (2)
    - for each k in scales:
      linearity, planarity, sphericity, curvature,
      nx, ny, nz, verticality,
      relative_height, mean_knn_distance, kth_distance
    """

    def __init__(
        self,
        scales: Sequence[int] = (16, 32, 64),
        eps: float = 1e-6,
        knn_support_size: int = 16384,
        knn_query_chunk: int = 4096,
    ) -> None:
        super().__init__()
        scales = tuple(int(k) for k in scales)
        if not scales or any(k <= 0 for k in scales):
            raise ValueError(f"Invalid scales: {scales}")
        if int(knn_support_size) <= 1:
            raise ValueError(f"knn_support_size must be > 1, got {knn_support_size}")
        if int(knn_query_chunk) <= 0:
            raise ValueError(f"knn_query_chunk must be > 0, got {knn_query_chunk}")
        self.scales = scales
        self.eps = float(eps)
        self.knn_support_size = int(knn_support_size)
        self.knn_query_chunk = int(knn_query_chunk)
        self.per_scale_dim = 11
        self.base_dim = 5  # xyz + intensity + elongation
        self.out_dim = self.base_dim + len(self.scales) * self.per_scale_dim

    def _chunked_knn(
        self,
        points: torch.Tensor,  # [M,3]
        k_max: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = int(points.shape[0])
        if m <= 1:
            return points.new_zeros((m, 0)), torch.zeros((m, 0), dtype=torch.long, device=points.device)

        support_size = min(m, int(self.knn_support_size))
        support = points[:support_size]  # deterministic, no extra RNG in model forward

        k_eff = min(int(k_max), support_size - 1)
        if k_eff <= 0:
            return points.new_zeros((m, 0)), torch.zeros((m, 0), dtype=torch.long, device=points.device)

        all_dists: list[torch.Tensor] = []
        all_idx: list[torch.Tensor] = []
        chunk = int(self.knn_query_chunk)
        for start in range(0, m, chunk):
            end = min(start + chunk, m)
            q = points[start:end]
            d = torch.cdist(q, support, p=2.0)  # [Q,S]

            # Exclude self-neighbor where query index is included in support.
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
        points: torch.Tensor,  # [M,3]
        knn_dists: torch.Tensor,  # [M,K]
        knn_idx: torch.Tensor,  # [M,K]
        k: int,
    ) -> torch.Tensor:
        k_eff = min(int(k), int(knn_dists.shape[1]))
        if k_eff <= 0:
            return points.new_zeros((points.shape[0], self.per_scale_dim))

        d = knn_dists[:, :k_eff]  # [M,k]
        idx = knn_idx[:, :k_eff]  # [M,k]
        nbrs = points[idx]  # [M,k,3]

        mean = nbrs.mean(dim=1, keepdim=True)  # [M,1,3]
        centered = nbrs - mean  # [M,k,3]
        denom = max(k_eff - 1, 1)
        cov = centered.transpose(1, 2).matmul(centered) / float(denom)  # [M,3,3]
        cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        cov = 0.5 * (cov + cov.transpose(1, 2))
        eye = torch.eye(3, dtype=cov.dtype, device=cov.device).unsqueeze(0)
        cov = cov + eye * self.eps

        try:
            evals, evecs = torch.linalg.eigh(cov)  # ascending λ
        except RuntimeError:
            # Rare cuSOLVER failures can still happen for ill-conditioned batches.
            cov_cpu = cov.float().cpu()
            evals_cpu, evecs_cpu = torch.linalg.eigh(cov_cpu)
            evals = evals_cpu.to(device=cov.device, dtype=cov.dtype)
            evecs = evecs_cpu.to(device=cov.device, dtype=cov.dtype)
        evals_desc = torch.flip(evals, dims=[1])  # [M] -> λ1>=λ2>=λ3
        evecs_desc = torch.flip(evecs, dims=[2])  # cols aligned to desc evals

        l1 = evals_desc[:, 0].clamp_min(self.eps)
        l2 = evals_desc[:, 1].clamp_min(0.0)
        l3 = evals_desc[:, 2].clamp_min(0.0)
        lsum = (l1 + l2 + l3).clamp_min(self.eps)

        linearity = (l1 - l2) / l1
        planarity = (l2 - l3) / l1
        sphericity = l3 / l1
        curvature = l3 / lsum

        normal = evecs_desc[:, :, 2]  # smallest-eigenvalue normal
        sign = torch.where(normal[:, 2] < 0.0, -1.0, 1.0).unsqueeze(1)
        normal = normal * sign
        verticality = normal[:, 2].abs()

        rel_height = points[:, 2] - nbrs[:, :, 2].mean(dim=1)
        mean_knn_dist = torch.nan_to_num(d.mean(dim=1), nan=0.0, posinf=0.0, neginf=0.0)
        kth_dist = torch.nan_to_num(d[:, -1], nan=0.0, posinf=0.0, neginf=0.0)

        return torch.stack(
            [
                linearity,
                planarity,
                sphericity,
                curvature,
                normal[:, 0],
                normal[:, 1],
                normal[:, 2],
                verticality,
                rel_height,
                mean_knn_dist,
                kth_dist,
            ],
            dim=1,
        )

    def forward(
        self,
        points: torch.Tensor,  # [B,N,3]
        point_features: torch.Tensor,  # [B,N,2]
        valid_geometry: torch.Tensor,  # [B,N]
    ) -> torch.Tensor:
        points = points.float()
        point_features = point_features.float()
        valid_geometry = valid_geometry.bool()
        batch_size, num_points, _ = points.shape
        out = points.new_zeros((batch_size, num_points, self.out_dim))

        k_max = max(self.scales)
        for b in range(batch_size):
            finite_points = torch.isfinite(points[b]).all(dim=1)
            finite_features = torch.isfinite(point_features[b]).all(dim=1)
            valid = valid_geometry[b] & finite_points & finite_features
            m = int(valid.sum().item())
            if m <= 0:
                continue

            p = points[b, valid]  # [M,3]
            f = point_features[b, valid]  # [M,2]
            out_valid = p.new_zeros((m, self.out_dim))
            out_valid[:, :3] = p
            out_valid[:, 3:5] = f

            # Need at least one neighbor to compute local stats.
            if m >= 2:
                knn_dists, knn_idx = self._chunked_knn(p, k_max)

                offset = self.base_dim
                for scale in self.scales:
                    fs = self._per_scale_features(p, knn_dists, knn_idx, int(scale))
                    out_valid[:, offset : offset + self.per_scale_dim] = fs
                    offset += self.per_scale_dim

            out[b, valid] = out_valid

        return out
