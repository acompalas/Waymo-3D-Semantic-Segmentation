import argparse

import torch
import torch.nn as nn

from ..spec import BackboneSpec, make_backbone_spec
from .edgeconv import EdgeConvBackbone
from .pointnet import add_pointnet_backbone_args


def add_dp3_backbone_args(parser: argparse.ArgumentParser) -> tuple[str, ...]:
    names = add_pointnet_backbone_args(parser)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--encoder-out-dim", type=int, default=256)
    parser.add_argument("--encoder-use-layernorm", action="store_true")
    parser.add_argument("--encoder-final-norm", type=str, default="none", choices=("none", "layernorm"))
    return names + ("knn_k", "encoder_out_dim", "encoder_use_layernorm", "encoder_final_norm")


class DP3WaymoPointEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = 256,
        use_layernorm: bool = False,
        final_norm: str = "none",
    ) -> None:
        super().__init__()
        hidden = [64, 128, 256, 512]
        layers: list[nn.Module] = []
        prev = int(in_channels)
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.LayerNorm(width) if use_layernorm else nn.Identity())
            layers.append(nn.ReLU())
            prev = width
        self.mlp = nn.Sequential(*layers)

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(hidden[-1], int(out_channels)),
                nn.LayerNorm(int(out_channels)),
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(hidden[-1], int(out_channels))
        else:
            raise ValueError(f"Unsupported final_norm: {final_norm}")

    def forward(self, model_inputs: torch.Tensor) -> torch.Tensor:
        x = self.mlp(model_inputs)
        x = torch.max(x, 1).values
        return self.final_projection(x)


class DP3PointDiffusionBackbone(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        point_input_dim: int,
        num_classes: int,
        encoder_out_dim: int = 256,
        encoder_use_layernorm: bool = False,
        encoder_final_norm: str = "none",
        hidden_dim: int = 256,
        depth: int = 6,
        knn_k: int = 16,
        time_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.point_input_dim = int(point_input_dim)
        self.num_classes = int(num_classes)
        expected_input_dim = self.point_input_dim + self.num_classes
        if int(input_dim) != expected_input_dim:
            raise ValueError(
                f"DP3PointDiffusionBackbone expected input_dim={expected_input_dim}, got {input_dim}."
            )
        self.encoder = DP3WaymoPointEncoder(
            in_channels=int(point_input_dim),
            out_channels=int(encoder_out_dim),
            use_layernorm=bool(encoder_use_layernorm),
            final_norm=str(encoder_final_norm),
        )
        self.backbone = EdgeConvBackbone(
            input_dim=int(point_input_dim) + int(encoder_out_dim) + int(num_classes),
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            dropout=float(dropout),
            knn_k=int(knn_k),
            time_dim=time_dim,
        )
        self.output_dim = int(self.backbone.output_dim)

    def forward(self, inputs: torch.Tensor, *, xyz: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if inputs.shape[-1] != self.point_input_dim + self.num_classes:
            raise ValueError(
                "DP3PointDiffusionBackbone expects concatenated inputs of shape "
                f"[B, N, {self.num_classes} + {self.point_input_dim}]."
            )
        x_t = inputs[..., : self.num_classes]
        model_inputs = inputs[..., self.num_classes :]
        cond_global = self.encoder(model_inputs)
        if cond_global.ndim != 2:
            raise ValueError("DP3 encoder must return shape [B, C].")
        bsz, npts, _ = model_inputs.shape
        cond_expanded = cond_global[:, None, :].expand(bsz, npts, cond_global.shape[-1])
        features = torch.cat([x_t, model_inputs, cond_expanded], dim=-1)
        return self.backbone(features, xyz=xyz, t=t)


def _build_dp3(**kwargs) -> nn.Module:
    return DP3PointDiffusionBackbone(
        input_dim=kwargs["input_dim"],
        point_input_dim=kwargs["point_input_dim"],
        num_classes=kwargs["num_classes"],
        encoder_out_dim=kwargs.get("encoder_out_dim", 256),
        encoder_use_layernorm=kwargs.get("encoder_use_layernorm", False),
        encoder_final_norm=kwargs.get("encoder_final_norm", "none"),
        hidden_dim=kwargs.get("hidden_dim", 256),
        depth=kwargs.get("depth", 6),
        dropout=kwargs.get("dropout", 0.1),
        knn_k=kwargs.get("knn_k", 16),
        time_dim=kwargs.get("time_dim"),
    )


DP3_BACKBONE_SPEC: BackboneSpec = make_backbone_spec(
    supported_behaviors=("diffusion",),
    build=_build_dp3,
    add_args_with_names=add_dp3_backbone_args,
)
