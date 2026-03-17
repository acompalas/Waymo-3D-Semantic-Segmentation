import argparse

import torch
import torch.nn as nn

from ..spec import BackboneSpec, make_backbone_spec
from .common import EdgeConvBlock, SinusoidalTimeEmbedding, knn_indices
from .pointnet import add_pointnet_backbone_args


def add_edgeconv_backbone_args(parser: argparse.ArgumentParser) -> tuple[str, ...]:
    names = add_pointnet_backbone_args(parser)
    parser.add_argument("--knn-k", type=int, default=16)
    return names + ("knn_k",)


class EdgeConvBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, depth: int = 6, dropout: float = 0.1, knn_k: int = 16, time_dim: int | None = None) -> None:
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
        self.blocks = nn.ModuleList([EdgeConvBlock(int(hidden_dim), block_time_dim, dropout=dropout) for _ in range(int(depth))])
        self.global_proj = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.output_dim = int(hidden_dim)

    def forward(self, inputs: torch.Tensor, *, xyz: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if self.uses_time:
            if t is None:
                raise ValueError("EdgeConvBackbone requires diffusion timesteps when time conditioning is enabled.")
            t_emb = self.time_embed(t)
        else:
            if t is not None:
                raise ValueError("EdgeConvBackbone does not support timestep conditioning.")
            t_emb = None

        x = self.in_proj(inputs)
        knn_idx = knn_indices(xyz, self.knn_k)
        for block in self.blocks:
            x = block(x, xyz, knn_idx, t_emb)
        return x + self.global_proj(x.max(dim=1).values)[:, None, :]


def _build_edgeconv(**kwargs) -> nn.Module:
    return EdgeConvBackbone(
        input_dim=kwargs["input_dim"],
        hidden_dim=kwargs.get("hidden_dim", 256),
        depth=kwargs.get("depth", 6),
        dropout=kwargs.get("dropout", 0.1),
        knn_k=kwargs.get("knn_k", 16),
        time_dim=kwargs.get("time_dim"),
    )


EDGECONV_BACKBONE_SPEC: BackboneSpec = make_backbone_spec(
    supported_behaviors=("supervised", "diffusion"),
    build=_build_edgeconv,
    add_args_with_names=add_edgeconv_backbone_args,
)
