import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RangeImageSegmentationModel
from .diffusion import DDPM, labels_to_soft_range
from .inputs import range_input_channels, select_range_model_inputs


def norm_2d(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1).float()
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / max(half - 1, 1))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return F.pad(emb, (0, 1)) if self.dim % 2 == 1 else emb


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
        return self.proj(t)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
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
    def __init__(self, dim: int, context_dim: int, n_heads: int = 4, head_dim: int = 32):
        super().__init__()
        inner = n_heads * head_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
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
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class SpatialTransformerBlock(nn.Module):
    def __init__(self, channels: int, context_dim: int, n_heads: int = 4, head_dim: int = 32):
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
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class LiDARConditionEncoder(nn.Module):
    def __init__(self, in_ch: int = 4, context_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
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


class LiDARDiffusionUNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 23,
        lidar_channels: int = 4,
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
        self.cond_encoder = LiDARConditionEncoder(lidar_channels, context_dim)
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
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            use_attn = level == n_levels - 1
            for _ in range(num_res_blocks + 1):
                skip_ch = self.skip_channels.pop()
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
        self.out_conv = nn.Conv2d(ch, num_classes, 3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        context = self.cond_encoder(cond)
        h = self.in_conv(x_t)
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

        return self.out_conv(F.silu(self.out_norm(h)))


class RangeImageDiffusionSegmenter(RangeImageSegmentationModel):
    def __init__(
        self,
        num_classes: int = 23,
        lidar_channels: int | None = None,
        base_channels: int = 32,
        learning_rate: float = 2e-4,
        diffusion_steps: int = 1000,
        use_balanced_class_weights: bool = True,
        validation_prediction_mode: str = "cheap",
        geometry_only: bool = False,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            use_balanced_class_weights=use_balanced_class_weights,
            validation_prediction_mode=validation_prediction_mode,
        )
        lidar_channels = range_input_channels(geometry_only=bool(geometry_only)) if lidar_channels is None else int(lidar_channels)
        self.save_hyperparameters()
        self.model = LiDARDiffusionUNet(
            num_classes=int(num_classes),
            lidar_channels=int(lidar_channels),
            base_channels=int(base_channels),
        )
        self.ddpm = DDPM(T=int(diffusion_steps))

    def prepare_runtime(self) -> None:
        self.ddpm.to(self.device)

    def prepare_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = select_range_model_inputs(batch["range_images"], geometry_only=bool(self.hparams.geometry_only))
        labels = batch["semantic"].long()
        valid = batch["valid_label"].bool()
        if x.ndim != 5 or x.shape[-1] != int(self.hparams.lidar_channels):
            raise ValueError(f"Expected selected range input shape [B,R,H,W,{int(self.hparams.lidar_channels)}], got {tuple(x.shape)}")
        bsz, returns, height, width, channels = x.shape
        cond = x.permute(0, 1, 4, 2, 3).reshape(bsz * returns, channels, height, width)
        labels = labels.reshape(bsz * returns, height, width)
        valid = valid.reshape(bsz * returns, height, width)
        return cond, labels, valid

    def _predict_labels(self, cond: torch.Tensor, valid: torch.Tensor, sampling_steps: int | None = None) -> torch.Tensor:
        x0 = self.ddpm.sample(
            self.model,
            cond,
            sample_shape=(cond.shape[0], int(self.hparams.num_classes), cond.shape[2], cond.shape[3]),
            steps=sampling_steps,
        )
        preds = x0.argmax(dim=1)
        preds = torch.where(valid, preds, torch.zeros_like(preds))
        return preds

    def predict_range_labels(self, batch: dict, *, sampling_steps: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        cond, labels, valid = self.prepare_batch(batch)
        preds = self._predict_labels(cond, valid, sampling_steps=sampling_steps)
        return preds, labels

    def _evaluation_loss(
        self,
        cond: torch.Tensor,
        labels: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = labels_to_soft_range(labels, int(self.hparams.num_classes), valid)
        t = torch.ones(cond.shape[0], device=self.device, dtype=torch.long)
        x_t, eps = self.ddpm.q_sample(x0, t)
        eps_pred = self.model(x_t, t, cond)
        loss = self.ddpm.loss(eps_pred, eps, valid, labels=labels, class_weights=None, class_dim=1)
        sqrt_ab = self.ddpm._extract(self.ddpm.sqrt_alpha_bars, t, x_t.shape)
        sqrt_1mab = self.ddpm._extract(self.ddpm.sqrt_one_minus_ab, t, x_t.shape)
        x0_est = (x_t - sqrt_1mab * eps_pred) / sqrt_ab.clamp(min=1e-6)
        return loss, x0_est.argmax(dim=1)

    def _training_loss_and_predictions(
        self,
        cond: torch.Tensor,
        labels: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = labels_to_soft_range(labels, int(self.hparams.num_classes), valid)
        t = torch.randint(1, self.ddpm.T + 1, (cond.shape[0],), device=self.device, dtype=torch.long)
        x_t, eps = self.ddpm.q_sample(x0, t)
        eps_pred = self.model(x_t, t, cond)
        loss = self.ddpm.loss(eps_pred, eps, valid, labels=labels, class_weights=self.class_weights, class_dim=1)
        sqrt_ab = self.ddpm._extract(self.ddpm.sqrt_alpha_bars, t, x_t.shape)
        sqrt_1mab = self.ddpm._extract(self.ddpm.sqrt_one_minus_ab, t, x_t.shape)
        x0_est = (x_t - sqrt_1mab * eps_pred) / sqrt_ab.clamp(min=1e-6)
        return loss, x0_est.argmax(dim=1)

    def _compute_stage_output(
        self,
        batch: dict,
        *,
        stage: str,
        prediction_mode: str,
        evaluation: bool,
    ) -> dict:
        _ = stage
        cond, labels, valid = self.prepare_batch(batch)
        batch_size = int(batch["range_images"].shape[0])
        if evaluation:
            loss, cheap_preds = self._evaluation_loss(cond, labels, valid)
            if prediction_mode == "full":
                preds = self._predict_labels(cond, valid, sampling_steps=self.resolve_sampling_steps())
            else:
                preds = torch.where(valid, cheap_preds, torch.zeros_like(cheap_preds))
        else:
            loss, preds = self._training_loss_and_predictions(cond, labels, valid)

        return {
            "loss": loss,
            "preds": preds,
            "labels": labels,
            "metric_mask": valid & (labels > 0),
            "batch_size": batch_size,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=float(self.hparams.learning_rate))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(self.trainer.max_epochs)))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
