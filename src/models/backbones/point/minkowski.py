import argparse
import importlib

import torch
import torch.nn as nn

from ..spec import BackboneSpec, make_backbone_spec
from .common import SinusoidalTimeEmbedding


def add_minkowski_backbone_args(parser: argparse.ArgumentParser) -> tuple[str, ...]:
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--voxel-size", type=float, default=0.2)
    parser.add_argument("--stem-kernel-size", type=int, default=5)
    parser.add_argument("--kernel-size", type=int, default=3)
    return ("hidden_dim", "depth", "dropout", "voxel_size", "stem_kernel_size", "kernel_size")


def _require_minkowski_engine():
    try:
        return importlib.import_module("MinkowskiEngine")
    except ImportError as exc:
        raise ImportError(
            "The 'minkowski' backbone requires the optional dependency 'MinkowskiEngine'. "
            "Install it in a compatible Linux environment before training or evaluation."
        ) from exc


class _SparseTimeMixin:
    def _coord_batches(self, x) -> torch.Tensor:
        coords = getattr(x, "C", None)
        if coords is None:
            coords = x.coordinates
        return coords[:, 0].long()

    def _add_time_bias(self, x, t_emb: torch.Tensor | None, t_proj: nn.Module | None):
        if t_proj is None:
            return x
        if t_emb is None:
            raise ValueError("Time embedding is required for time-conditioned Minkowski blocks.")
        bias = t_proj(t_emb)[self._coord_batches(x)]
        return self._me.SparseTensor(
            features=x.F + bias,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )


class _SparseUNetBlock(nn.Module, _SparseTimeMixin):
    def __init__(
        self,
        me,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dropout: float,
        dimension: int,
        time_channels: int | None = None,
    ) -> None:
        super().__init__()
        self._me = me
        self.conv1 = me.MinkowskiConvolution(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            stride=1,
            bias=False,
            dimension=int(dimension),
        )
        self.norm1 = me.MinkowskiBatchNorm(int(out_channels))
        self.act1 = me.MinkowskiReLU(inplace=True)
        self.drop = me.MinkowskiDropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.conv2 = me.MinkowskiConvolution(
            int(out_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            stride=1,
            bias=False,
            dimension=int(dimension),
        )
        self.norm2 = me.MinkowskiBatchNorm(int(out_channels))
        self.act2 = me.MinkowskiReLU(inplace=True)
        self.residual = (
            me.MinkowskiConvolution(
                int(in_channels),
                int(out_channels),
                kernel_size=1,
                stride=1,
                bias=False,
                dimension=int(dimension),
            )
            if int(in_channels) != int(out_channels)
            else nn.Identity()
        )
        self.time_proj = nn.Linear(int(time_channels), int(out_channels)) if time_channels is not None else None

    def forward(self, x, t_emb: torch.Tensor | None = None):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self._add_time_bias(x, t_emb, self.time_proj)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.act2(x + residual)


class _SparseDownsample(nn.Module, _SparseTimeMixin):
    def __init__(self, me, in_channels: int, out_channels: int, *, time_channels: int | None = None) -> None:
        super().__init__()
        self._me = me
        self.conv = me.MinkowskiConvolution(
            int(in_channels),
            int(out_channels),
            kernel_size=2,
            stride=2,
            bias=False,
            dimension=3,
        )
        self.norm = me.MinkowskiBatchNorm(int(out_channels))
        self.act = me.MinkowskiReLU(inplace=True)
        self.time_proj = nn.Linear(int(time_channels), int(out_channels)) if time_channels is not None else None

    def forward(self, x, t_emb: torch.Tensor | None = None):
        x = self.conv(x)
        x = self.norm(x)
        x = self._add_time_bias(x, t_emb, self.time_proj)
        return self.act(x)


class _SparseUpsample(nn.Module, _SparseTimeMixin):
    def __init__(self, me, in_channels: int, out_channels: int, *, time_channels: int | None = None) -> None:
        super().__init__()
        self._me = me
        self.conv = me.MinkowskiConvolutionTranspose(
            int(in_channels),
            int(out_channels),
            kernel_size=2,
            stride=2,
            bias=False,
            dimension=3,
        )
        self.norm = me.MinkowskiBatchNorm(int(out_channels))
        self.act = me.MinkowskiReLU(inplace=True)
        self.time_proj = nn.Linear(int(time_channels), int(out_channels)) if time_channels is not None else None

    def forward(self, x, skip, t_emb: torch.Tensor | None = None):
        x = self.conv(x, coordinates=skip)
        x = self.norm(x)
        x = self._add_time_bias(x, t_emb, self.time_proj)
        return self.act(x)


class MinkowskiPointUNetBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        depth: int = 4,
        dropout: float = 0.1,
        voxel_size: float = 0.2,
        stem_kernel_size: int = 5,
        kernel_size: int = 3,
        time_dim: int | None = None,
    ) -> None:
        super().__init__()
        if float(voxel_size) <= 0.0:
            raise ValueError(f"voxel_size must be > 0, got {voxel_size}")
        if int(depth) < 2:
            raise ValueError(f"depth must be >= 2 for a U-Net, got {depth}")

        self._me = _require_minkowski_engine()
        self.voxel_size = float(voxel_size)
        self.uses_time = time_dim is not None

        channels = [int(hidden_dim) * (2 ** level) for level in range(int(depth))]
        time_channels = int(channels[0]) if self.uses_time else None

        self.input_proj = nn.Sequential(
            nn.Linear(int(input_dim), int(channels[0])),
            nn.LayerNorm(int(channels[0])),
            nn.SiLU(),
        )
        if self.uses_time:
            self.time_embed = nn.Sequential(
                SinusoidalTimeEmbedding(int(time_dim)),
                nn.Linear(int(time_dim), int(channels[0])),
                nn.SiLU(),
                nn.Linear(int(channels[0]), int(channels[0])),
            )
        else:
            self.time_embed = None

        self.stem = _SparseUNetBlock(
            self._me,
            int(channels[0]),
            int(channels[0]),
            kernel_size=int(stem_kernel_size),
            dropout=float(dropout),
            dimension=3,
            time_channels=time_channels,
        )
        self.encoder_blocks = nn.ModuleList(
            [
                _SparseUNetBlock(
                    self._me,
                    int(channels[level]),
                    int(channels[level]),
                    kernel_size=int(kernel_size),
                    dropout=float(dropout),
                    dimension=3,
                    time_channels=time_channels,
                )
                for level in range(int(depth))
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                _SparseDownsample(
                    self._me,
                    int(channels[level]),
                    int(channels[level + 1]),
                    time_channels=time_channels,
                )
                for level in range(int(depth) - 1)
            ]
        )
        self.bottleneck = _SparseUNetBlock(
            self._me,
            int(channels[-1]),
            int(channels[-1]),
            kernel_size=int(kernel_size),
            dropout=float(dropout),
            dimension=3,
            time_channels=time_channels,
        )
        self.upsamples = nn.ModuleList(
            [
                _SparseUpsample(
                    self._me,
                    int(channels[level + 1]),
                    int(channels[level]),
                    time_channels=time_channels,
                )
                for level in range(int(depth) - 1)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                _SparseUNetBlock(
                    self._me,
                    int(channels[level]) * 2,
                    int(channels[level]),
                    kernel_size=int(kernel_size),
                    dropout=float(dropout),
                    dimension=3,
                    time_channels=time_channels,
                )
                for level in range(int(depth) - 1)
            ]
        )
        self.out_norm = self._me.MinkowskiBatchNorm(int(channels[0]))
        self.out_act = self._me.MinkowskiReLU(inplace=True)
        self.output_dim = int(channels[0])

    def _point_features(self, inputs: torch.Tensor, t: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.uses_time:
            if t is None:
                raise ValueError("MinkowskiPointUNetBackbone requires diffusion timesteps when time conditioning is enabled.")
            t_emb = self.time_embed(t)
        else:
            if t is not None:
                raise ValueError("MinkowskiPointUNetBackbone does not support timestep conditioning.")
            t_emb = None
        return self.input_proj(inputs), t_emb

    def _tensor_field(self, features: torch.Tensor, xyz: torch.Tensor):
        coords = self._me.utils.batched_coordinates(
            [(sample_xyz / self.voxel_size) for sample_xyz in xyz],
            dtype=torch.float32,
        ).to(device=xyz.device)
        return self._me.TensorField(
            features=features.reshape(-1, features.shape[-1]),
            coordinates=coords,
            quantization_mode=self._me.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=self._me.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=xyz.device,
        )

    def forward(self, inputs: torch.Tensor, *, xyz: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"Expected xyz shape [B,N,3], got {tuple(xyz.shape)}")
        if inputs.ndim != 3:
            raise ValueError(f"Expected inputs shape [B,N,C], got {tuple(inputs.shape)}")
        if inputs.shape[:2] != xyz.shape[:2]:
            raise ValueError(
                f"Point features and coordinates must agree on [B,N], got {tuple(inputs.shape)} and {tuple(xyz.shape)}"
            )

        point_features, t_emb = self._point_features(inputs, t)
        field = self._tensor_field(point_features, xyz)
        x = field.sparse()

        x = self.stem(x, t_emb)
        skips: list = []
        for level, block in enumerate(self.encoder_blocks):
            x = block(x, t_emb)
            skips.append(x)
            if level < len(self.downsamples):
                x = self.downsamples[level](x, t_emb)

        x = self.bottleneck(x, t_emb)
        for level in range(len(self.downsamples) - 1, -1, -1):
            skip = skips[level]
            x = self.upsamples[level](x, skip, t_emb)
            x = self._me.cat(x, skip)
            x = self.decoder_blocks[level](x, t_emb)

        x = self.out_norm(x)
        x = self.out_act(x)
        return x.slice(field).F.reshape(inputs.shape[0], inputs.shape[1], self.output_dim)


def _build_minkowski(**kwargs) -> nn.Module:
    return MinkowskiPointUNetBackbone(
        input_dim=kwargs["input_dim"],
        hidden_dim=kwargs.get("hidden_dim", 64),
        depth=kwargs.get("depth", 4),
        dropout=kwargs.get("dropout", 0.1),
        voxel_size=kwargs.get("voxel_size", 0.2),
        stem_kernel_size=kwargs.get("stem_kernel_size", 5),
        kernel_size=kwargs.get("kernel_size", 3),
        time_dim=kwargs.get("time_dim"),
    )


MINKOWSKI_BACKBONE_SPEC: BackboneSpec = make_backbone_spec(
    supported_behaviors=("direct", "diffusion"),
    build=_build_minkowski,
    add_args_with_names=add_minkowski_backbone_args,
)
