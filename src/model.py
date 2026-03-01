"""
model.py
========
Conditional DDPM U-Net for LiDAR semantic segmentation in range image space.

Architecture from lecture slides (ECE 285):
  - SinusoidalTimeEmbedding + TimeEmbeddingMLP  (slides pp. 16)
  - ResBlock with time embedding injection       (slides pp. 18)
  - SpatialTransformerBlock (cross-attn + FFN)  (slides pp. 19-20)
  - U-Net encoder / bottleneck / decoder        (slides pp. 21)

Memory adaptations for RTX 2060 (6 GB):
  - Self-attention REMOVED from SpatialTransformerBlock.
    Self-attn is O(N²) in spatial tokens. Our range image is (64,2650)
    so N=169,600 at full res and N=10,608 at the deepest level — both OOM.
    Cross-attention over ~664 downsampled LiDAR tokens is cheap and sufficient.
  - Attention only applied at deepest encoder/decoder level (level 2).
  - LiDAR condition encoder downsamples 16x before producing context tokens.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Normalisation helper ───────────────────────────────────────────────────────
def norm_2d(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


# ── Time embedding (slides pp. 16) ────────────────────────────────────────────
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t      = t.reshape(-1).float()   # ensure (B,)
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeEmbeddingMLP(nn.Module):
    def __init__(self, base_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            SinusoidalTimeEmbedding(base_dim),
            nn.Linear(base_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.proj(t)   # (B, out_dim)


# ── ResBlock (slides pp. 18) ───────────────────────────────────────────────────
class ResBlock(nn.Module):
    """
    ResNet block. Expects a pre-computed time embedding (B, time_dim),
    not raw timesteps — the U-Net calls TimeEmbeddingMLP once and passes
    the result down to every ResBlock.
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1    = norm_2d(in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2    = norm_2d(out_ch)
        self.dropout  = nn.Dropout2d(dropout)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


# ── Cross-attention (slides pp. 19) ───────────────────────────────────────────
class CrossAttention(nn.Module):
    """
    Multi-head cross-attention.
    Q from image tokens, K/V from context (LiDAR features).
    O(N * M) where N = spatial tokens, M = ~664 LiDAR context tokens.
    """
    def __init__(self, dim: int, context_dim: int, n_heads: int = 4, head_dim: int = 32):
        super().__init__()
        inner         = n_heads * head_dim
        self.n_heads  = n_heads
        self.head_dim = head_dim
        self.scale    = head_dim ** -0.5
        self.to_q     = nn.Linear(dim,         inner, bias=False)
        self.to_k     = nn.Linear(context_dim, inner, bias=False)
        self.to_v     = nn.Linear(context_dim, inner, bias=False)
        self.to_out   = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x:       (B, N, dim)
        # context: (B, M, context_dim)
        B, N, _ = x.shape
        H, D    = self.n_heads, self.head_dim

        q = self.to_q(x).view(B, N, H, D).transpose(1, 2)          # (B,H,N,D)
        k = self.to_k(context).view(B, -1, H, D).transpose(1, 2)   # (B,H,M,D)
        v = self.to_v(context).view(B, -1, H, D).transpose(1, 2)   # (B,H,M,D)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale    # (B,H,N,M)
        attn = attn.softmax(dim=-1)
        out  = torch.matmul(attn, v)                                 # (B,H,N,D)
        out  = out.transpose(1, 2).contiguous().view(B, N, H * D)
        return self.to_out(out)


# ── GEGLU feed-forward ────────────────────────────────────────────────────────
class GEGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 4 * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


# ── SpatialTransformerBlock (slides pp. 20, self-attn removed for memory) ─────
class SpatialTransformerBlock(nn.Module):
    """
    Cross-attention + FFN over spatial tokens.

    Self-attention is intentionally removed — it is O(N²) in spatial tokens
    and OOMs on a 6 GB GPU even at the deepest U-Net level (N ≈ 10k).
    Cross-attention over the ~664 downsampled LiDAR context tokens is O(N*664),
    which is fine and still fully conditions the label features on the LiDAR.
    """
    def __init__(self, channels: int, context_dim: int, n_heads: int = 4, head_dim: int = 32):
        super().__init__()
        self.norm       = norm_2d(channels)
        self.proj_in    = nn.Conv2d(channels, channels, 1)
        self.ln2        = nn.LayerNorm(channels)
        self.cross_attn = CrossAttention(channels, context_dim, n_heads, head_dim)
        self.ln3        = nn.LayerNorm(channels)
        self.ff         = GEGLU(channels)
        self.ff_out     = nn.Linear(channels * 4, channels)
        self.proj_out   = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x:       (B, C, H, W)
        # context: (B, ~664, context_dim)
        B, C, H, W = x.shape
        x_in = x

        x = self.proj_in(self.norm(x))
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)   # (B, N, C)

        x = x + self.cross_attn(self.ln2(x), context)     # LiDAR conditioning
        x = x + self.ff_out(self.ff(self.ln3(x)))          # FFN

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.proj_out(x)
        return x + x_in


# ── Down / Upsample ───────────────────────────────────────────────────────────
class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ── LiDAR condition encoder ───────────────────────────────────────────────────
class LiDARConditionEncoder(nn.Module):
    """
    Encodes (B, 4, 64, 2650) → (B, ~664, context_dim).
    Downsamples 16x with strided convolutions so cross-attention is cheap.
    """
    def __init__(self, in_ch: int = 4, context_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,       32,          3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32,          64,          3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64,          context_dim, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(context_dim, context_dim, 3, stride=2, padding=1),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        feat = self.net(cond)                                   # (B, C, H/16, W/16)
        B, C, H, W = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, ~664, C)


# ── Full U-Net ────────────────────────────────────────────────────────────────
class LiDARDiffusionUNet(nn.Module):
    """
    Conditional DDPM U-Net for range image semantic segmentation.

    Input : x_t   (B, num_classes, H, W)  noisy label map at timestep t
            t     (B,)                     diffusion timestep index
            cond  (B, 4, H, W)            LiDAR range image (fixed condition)
    Output: eps   (B, num_classes, H, W)  predicted noise
    """

    def __init__(
        self,
        num_classes:    int   = 23,
        lidar_channels: int   = 4,
        base_channels:  int   = 32,
        channel_mults:  tuple = (1, 2, 4),
        num_res_blocks: int   = 2,
        time_emb_dim:   int   = 128,
        context_dim:    int   = 128,
        n_heads:        int   = 4,
        head_dim:       int   = 32,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        n_levels         = len(channel_mults)

        self.time_emb     = TimeEmbeddingMLP(time_emb_dim, time_emb_dim)
        self.cond_encoder = LiDARConditionEncoder(lidar_channels, context_dim)
        self.in_conv      = nn.Conv2d(num_classes, base_channels, 3, padding=1)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.down_blocks   = nn.ModuleList()
        self.skip_channels = [base_channels]
        ch = base_channels

        for level, mult in enumerate(channel_mults):
            out_ch   = base_channels * mult
            use_attn = (level == n_levels - 1)  # only deepest level
            for _ in range(num_res_blocks):
                self.down_blocks.append(nn.ModuleDict({
                    "res":  ResBlock(ch, out_ch, time_emb_dim, dropout),
                    "attn": SpatialTransformerBlock(out_ch, context_dim, n_heads, head_dim)
                           if use_attn else nn.Identity(),
                }))
                self.skip_channels.append(out_ch)
                ch = out_ch
            if level != n_levels - 1:
                self.down_blocks.append(nn.ModuleDict({"downsample": Downsample(ch)}))
                self.skip_channels.append(ch)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.mid_res1 = ResBlock(ch, ch, time_emb_dim, dropout)
        self.mid_attn = SpatialTransformerBlock(ch, context_dim, n_heads, head_dim)
        self.mid_res2 = ResBlock(ch, ch, time_emb_dim, dropout)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up_blocks = nn.ModuleList()
        skip_stack     = list(self.skip_channels)

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch   = base_channels * mult
            use_attn = (level == n_levels - 1)
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_stack.pop()
                self.up_blocks.append(nn.ModuleDict({
                    "res":  ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout),
                    "attn": SpatialTransformerBlock(out_ch, context_dim, n_heads, head_dim)
                           if use_attn else nn.Identity(),
                }))
                ch = out_ch
            if level != 0:
                self.up_blocks.append(nn.ModuleDict({"upsample": Upsample(ch)}))

        self.out_norm = norm_2d(ch)
        self.out_conv = nn.Conv2d(ch, num_classes, 3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb   = self.time_emb(t)         # (B, time_emb_dim)
        context = self.cond_encoder(cond)  # (B, ~664, context_dim)
        x       = self.in_conv(x_t)        # (B, base_channels, H, W)

        # Encoder
        skips = [x]
        for block in self.down_blocks:
            if "downsample" in block:
                x = block["downsample"](x)
            else:
                x = block["res"](x, t_emb)
                if not isinstance(block["attn"], nn.Identity):
                    x = block["attn"](x, context)
            skips.append(x)

        # Bottleneck
        x = self.mid_res1(x, t_emb)
        x = self.mid_attn(x, context)
        x = self.mid_res2(x, t_emb)

        # Decoder
        for block in self.up_blocks:
            if "upsample" in block:
                x = block["upsample"](x)
            else:
                skip = skips.pop()
                if x.shape != skip.shape:
                    x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                                  0, skip.shape[-2] - x.shape[-2]])
                x = torch.cat([x, skip], dim=1)
                x = block["res"](x, t_emb)
                if not isinstance(block["attn"], nn.Identity):
                    x = block["attn"](x, context)

        return self.out_conv(F.silu(self.out_norm(x)))