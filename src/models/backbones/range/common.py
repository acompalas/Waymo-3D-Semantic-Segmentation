import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad_width_circular_height_replicate(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_w > 0:
        x = F.pad(x, (pad_w, pad_w, 0, 0), mode="circular")
    if pad_h > 0:
        x = F.pad(x, (0, 0, pad_h, pad_h), mode="replicate")
    return x


def norm_2d(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class CircularWidthConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False) -> None:
        super().__init__()
        if int(kernel_size) % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric circular padding.")
        self.pad = int(kernel_size) // 2
        self.conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=0,
            bias=bool(bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(_pad_width_circular_height_replicate(x, pad_h=self.pad, pad_w=self.pad))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = CircularWidthConv2d(in_channels, out_channels, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = CircularWidthConv2d(out_channels, out_channels, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        return self.act(self.bn2(self.conv2(x)))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1).float()
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / max(half - 1, 1))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return F.pad(emb, (0, 1)) if self.dim % 2 == 1 else emb


class TimeEmbeddingMLP(nn.Module):
    def __init__(self, base_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            SinusoidalTimeEmbedding(base_dim),
            nn.Linear(base_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.proj(t)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = norm_2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = norm_2d(out_ch)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, context_dim: int, n_heads: int = 4, head_dim: int = 32) -> None:
        super().__init__()
        inner = n_heads * head_dim
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.scale = head_dim ** -0.5
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, _ = x.shape
        q = self.to_q(x).view(bsz, n_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(bsz, -1, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, n_tokens, self.n_heads * self.head_dim)
        return self.to_out(out)


class GEGLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim * 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class SpatialTransformerBlock(nn.Module):
    def __init__(self, channels: int, context_dim: int, n_heads: int = 4, head_dim: int = 32) -> None:
        super().__init__()
        self.norm = norm_2d(channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.ln2 = nn.LayerNorm(channels)
        self.cross_attn = CrossAttention(channels, context_dim, n_heads, head_dim)
        self.ln3 = nn.LayerNorm(channels)
        self.ff = GEGLU(channels)
        self.ff_out = nn.Linear(channels * 4, channels)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        x_in = x
        x = self.proj_in(self.norm(x))
        x = x.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)
        x = x + self.cross_attn(self.ln2(x), context)
        x = x + self.ff_out(self.ff(self.ln3(x)))
        x = x.view(bsz, height, width, channels).permute(0, 3, 1, 2).contiguous()
        return self.proj_out(x) + x_in


class Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class LiDARConditionEncoder(nn.Module):
    def __init__(self, in_channels: int, context_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, context_dim, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(context_dim, context_dim, 3, stride=2, padding=1),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        feat = self.net(cond)
        bsz, channels, height, width = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(bsz, height * width, channels)
