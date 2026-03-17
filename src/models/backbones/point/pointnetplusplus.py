import argparse

import torch
import torch.nn as nn

from ..spec import BackboneSpec, make_backbone_spec
from .common import PointNetFeaturePropagation, PointNetSetAbstraction


def add_pointnetplusplus_backbone_args(parser: argparse.ArgumentParser) -> tuple[str, ...]:
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    return ("hidden_dim", "dropout")


class PointNetPlusPlusBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        sa_configs: list[tuple[int, float, int, list[int]]] | None = None,
        fp_mlp_configs: list[list[int]] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)

        if sa_configs is None:
            sa_configs = [
                (1024, 0.1, 32, [32, 32, 64]),
                (256, 0.2, 32, [64, 64, 128]),
                (64, 0.4, 32, [128, 128, 256]),
                (16, 0.8, 32, [256, 256, 512]),
            ]
        if fp_mlp_configs is None:
            fp_mlp_configs = [
                [256, 256],
                [256, 256],
                [256, 128],
                [128, 128, 128],
            ]

        self.sa_layers = nn.ModuleList(
            [
                PointNetSetAbstraction(
                    npoint=npoint,
                    radius=radius,
                    nsample=nsample,
                    mlp=mlp,
                    use_xyz=True,
                    in_channels=(self.input_dim - 3 if idx == 0 else sa_configs[idx - 1][3][-1]),
                )
                for idx, (npoint, radius, nsample, mlp) in enumerate(sa_configs)
            ]
        )
        fp_in_channels = [
            sa_configs[2][3][-1] + sa_configs[3][3][-1],
            sa_configs[1][3][-1] + fp_mlp_configs[0][-1],
            sa_configs[0][3][-1] + fp_mlp_configs[1][-1],
            (self.input_dim - 3) + fp_mlp_configs[2][-1],
        ]
        self.fp_layers = nn.ModuleList([PointNetFeaturePropagation(in_channels=in_ch, mlp=mlp) for in_ch, mlp in zip(fp_in_channels, fp_mlp_configs)])
        self.post_mlp = nn.Sequential(
            nn.Conv1d(int(fp_mlp_configs[-1][-1]), int(hidden_dim), 1),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(hidden_dim), int(hidden_dim), 1),
        )
        self.output_dim = int(hidden_dim)

    def forward(self, inputs: torch.Tensor, *, xyz: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if t is not None:
            raise ValueError("PointNetPlusPlusBackbone does not support timestep conditioning.")
        xyz = xyz.contiguous()
        features = inputs[..., 3:].contiguous()
        features = features if features.shape[-1] > 0 else None

        l_xyz = [xyz]
        l_features = [features]
        for sa in self.sa_layers:
            new_xyz, new_features = sa(l_xyz[-1], l_features[-1])
            l_xyz.append(new_xyz)
            l_features.append(new_features)

        for idx in range(len(self.fp_layers)):
            src_idx = -(idx + 1)
            dst_idx = src_idx - 1
            l_features[dst_idx] = self.fp_layers[idx](
                l_xyz[dst_idx],
                l_xyz[src_idx],
                l_features[dst_idx],
                l_features[src_idx],
            )

        x = self.post_mlp(l_features[0].transpose(1, 2)).transpose(1, 2)
        return x


def _build_pointnetplusplus(**kwargs) -> nn.Module:
    return PointNetPlusPlusBackbone(
        input_dim=kwargs["input_dim"],
        hidden_dim=kwargs.get("hidden_dim", 256),
        dropout=kwargs.get("dropout", 0.1),
    )


POINTNETPLUSPLUS_BACKBONE_SPEC: BackboneSpec = make_backbone_spec(
    supported_behaviors=("direct",),
    build=_build_pointnetplusplus,
    add_args_with_names=add_pointnetplusplus_backbone_args,
)
