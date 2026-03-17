import argparse

import torch
import torch.nn as nn

from ...features import MultiScaleKnnEigenFeatureExtractor


def add_handcrafted_backbone_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--knn-scales", type=str, default="16,32,64")
    parser.add_argument("--knn-support-size", type=int, default=16384)
    parser.add_argument("--knn-query-chunk", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)


class HandcraftedPointBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        scales: tuple[int, ...] = (16, 32, 64),
        knn_support_size: int = 16384,
        knn_query_chunk: int = 4096,
        hidden_dim: int = 0,
        depth: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.feature_extractor = MultiScaleKnnEigenFeatureExtractor(
            scales=scales,
            knn_support_size=int(knn_support_size),
            knn_query_chunk=int(knn_query_chunk),
        )
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)
        if self.depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}")
        if self.hidden_dim < 0:
            raise ValueError(f"hidden_dim must be >= 0, got {hidden_dim}")
        if self.depth > 0 and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0 when depth is enabled.")

        if self.depth == 0:
            self.post_mlp = nn.Identity()
            self.output_dim = self.feature_extractor.out_dim
        else:
            layers: list[nn.Module] = []
            in_dim = self.feature_extractor.out_dim
            for _ in range(self.depth):
                layers.append(nn.Linear(in_dim, self.hidden_dim))
                layers.append(nn.SiLU())
                if self.dropout > 0.0:
                    layers.append(nn.Dropout(self.dropout))
                in_dim = self.hidden_dim
            self.post_mlp = nn.Sequential(*layers)
            self.output_dim = self.hidden_dim

    def forward(self, inputs: torch.Tensor, *, xyz: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if t is not None:
            raise ValueError("HandcraftedPointBackbone does not support timestep conditioning.")
        if self.input_dim > 3 and inputs.shape[-1] >= 5:
            point_features = inputs[..., 3:5]
        else:
            point_features = xyz.new_zeros((*xyz.shape[:2], 2))
        features = self.feature_extractor(xyz, point_features)
        return self.post_mlp(features)
