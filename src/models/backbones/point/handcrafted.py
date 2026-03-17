import argparse

import torch
import torch.nn as nn

from ...features import MultiScaleKnnEigenFeatureExtractor


def add_handcrafted_backbone_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--knn-scales", type=str, default="16,32,64")
    parser.add_argument("--knn-support-size", type=int, default=16384)
    parser.add_argument("--knn-query-chunk", type=int, default=4096)
    parser.add_argument("--proj-dim", type=int, default=0)
    parser.add_argument("--proj-depth", type=int, default=0)
    parser.add_argument("--proj-dropout", type=float, default=0.0)


class HandcraftedPointBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        scales: tuple[int, ...] = (16, 32, 64),
        knn_support_size: int = 16384,
        knn_query_chunk: int = 4096,
        proj_dim: int = 0,
        proj_depth: int = 0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.feature_extractor = MultiScaleKnnEigenFeatureExtractor(
            scales=scales,
            knn_support_size=int(knn_support_size),
            knn_query_chunk=int(knn_query_chunk),
        )
        self.proj_dim = int(proj_dim)
        self.proj_depth = int(proj_depth)
        self.proj_dropout = float(proj_dropout)
        if self.proj_depth < 0:
            raise ValueError(f"proj_depth must be >= 0, got {proj_depth}")
        if self.proj_dim < 0:
            raise ValueError(f"proj_dim must be >= 0, got {proj_dim}")
        if self.proj_depth > 0 and self.proj_dim <= 0:
            raise ValueError("proj_dim must be > 0 when proj_depth is enabled.")

        if self.proj_depth == 0:
            self.post_mlp = nn.Identity()
            self.output_dim = self.feature_extractor.out_dim
        else:
            layers: list[nn.Module] = []
            in_dim = self.feature_extractor.out_dim
            for _ in range(self.proj_depth):
                layers.append(nn.Linear(in_dim, self.proj_dim))
                layers.append(nn.SiLU())
                if self.proj_dropout > 0.0:
                    layers.append(nn.Dropout(self.proj_dropout))
                in_dim = self.proj_dim
            self.post_mlp = nn.Sequential(*layers)
            self.output_dim = self.proj_dim

    def forward(self, inputs: torch.Tensor, *, xyz: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if t is not None:
            raise ValueError("HandcraftedPointBackbone does not support timestep conditioning.")
        if self.input_dim > 3 and inputs.shape[-1] >= 5:
            point_features = inputs[..., 3:5]
        else:
            point_features = xyz.new_zeros((*xyz.shape[:2], 2))
        features = self.feature_extractor(xyz, point_features)
        return self.post_mlp(features)
