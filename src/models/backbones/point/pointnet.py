import argparse

import torch
import torch.nn as nn

from .common import ResMLPBlock, SinusoidalTimeEmbedding


def add_pointnet_backbone_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)


class PointNetBackbone(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            depth: int = 6,
            dropout: float = 0.1,
            time_dim: int | None = None
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
        self.blocks = nn.ModuleList([ResMLPBlock(int(hidden_dim), block_time_dim, dropout=dropout) for _ in range(int(depth))])
        self.global_proj = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.output_dim = int(hidden_dim)

    def forward(self, inputs: torch.Tensor, *, xyz: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        _ = xyz
        if self.uses_time:
            if t is None:
                raise ValueError("PointNetBackbone requires diffusion timesteps when time conditioning is enabled.")
            t_emb = self.time_embed(t)
        else:
            if t is not None:
                raise ValueError("PointNetBackbone does not support timestep conditioning.")
            t_emb = None

        x = self.in_proj(inputs)
        for block in self.blocks:
            x = block(x, t_emb)
        return x + self.global_proj(x.max(dim=1).values)[:, None, :]
