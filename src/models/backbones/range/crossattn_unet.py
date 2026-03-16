import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Downsample, LiDARConditionEncoder, ResBlock, SpatialTransformerBlock, TimeEmbeddingMLP, Upsample, norm_2d


def add_range_crossattn_backbone_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)


class RangeDiffusionBackbone(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cond_channels: int,
        base_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 128,
        context_dim: int = 128,
        n_heads: int = 4,
        head_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        n_levels = len(channel_mults)
        self.time_emb = TimeEmbeddingMLP(time_emb_dim, time_emb_dim)
        self.cond_encoder = LiDARConditionEncoder(cond_channels, context_dim)
        self.in_conv = nn.Conv2d(num_classes, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.skip_channels = [base_channels]
        ch = base_channels
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            use_attn = level == n_levels - 1
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    nn.ModuleDict(
                        {
                            "res": ResBlock(ch, out_ch, time_emb_dim, dropout),
                            "attn": SpatialTransformerBlock(out_ch, context_dim, n_heads, head_dim) if use_attn else nn.Identity(),
                        }
                    )
                )
                ch = out_ch
                self.skip_channels.append(ch)
            if level != n_levels - 1:
                self.down_blocks.append(nn.ModuleDict({"downsample": Downsample(ch)}))
                self.skip_channels.append(ch)

        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = SpatialTransformerBlock(ch, context_dim, n_heads, head_dim)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, dropout)

        self.up_blocks = nn.ModuleList()
        skip_channels = list(self.skip_channels)
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            use_attn = level == n_levels - 1
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                self.up_blocks.append(
                    nn.ModuleDict(
                        {
                            "res": ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout),
                            "attn": SpatialTransformerBlock(out_ch, context_dim, n_heads, head_dim) if use_attn else nn.Identity(),
                        }
                    )
                )
                ch = out_ch
            if level != 0:
                self.up_blocks.append(nn.ModuleDict({"upsample": Upsample(ch)}))

        self.out_norm = norm_2d(ch)
        self.output_channels = ch

    def forward(self, x: torch.Tensor, *, cond: torch.Tensor | None = None, t: torch.Tensor | None = None) -> torch.Tensor:
        if cond is None:
            raise ValueError("RangeDiffusionBackbone requires conditioning inputs.")
        if t is None:
            raise ValueError("RangeDiffusionBackbone requires diffusion timesteps.")
        t_emb = self.time_emb(t)
        context = self.cond_encoder(cond)
        h = self.in_conv(x)
        skips = [h]

        for block in self.down_blocks:
            if "downsample" in block:
                h = block["downsample"](h)
                skips.append(h)
            else:
                h = block["res"](h, t_emb)
                attn = block["attn"]
                h = attn(h, context) if not isinstance(attn, nn.Identity) else h
                skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid_block2(h, t_emb)

        for block in self.up_blocks:
            if "upsample" in block:
                h = block["upsample"](h)
            else:
                skip = skips.pop()
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = block["res"](h, t_emb)
                attn = block["attn"]
                h = attn(h, context) if not isinstance(attn, nn.Identity) else h

        return F.silu(self.out_norm(h))
