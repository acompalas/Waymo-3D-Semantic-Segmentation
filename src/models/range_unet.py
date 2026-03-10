import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RangeImageSegmentationModel


def _pad_width_circular_height_replicate(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_w > 0:
        x = F.pad(x, (pad_w, pad_w, 0, 0), mode="circular")
    if pad_h > 0:
        x = F.pad(x, (0, 0, pad_h, pad_h), mode="replicate")
    return x


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


class RangeUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32, depth: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if int(depth) < 2:
            raise ValueError("depth must be >= 2")
        channels = [int(base_channels) * (2**i) for i in range(int(depth))]
        self.encoders = nn.ModuleList()
        prev = int(in_channels)
        for ch in channels:
            self.encoders.append(ConvBlock(prev, ch, dropout=dropout))
            prev = ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, dropout=dropout)

        self.decoders = nn.ModuleList()
        dec_in = channels[-1] * 2
        for skip_ch in reversed(channels):
            self.decoders.append(ConvBlock(dec_in + skip_ch, skip_ch, dropout=dropout))
            dec_in = skip_ch

        self.head = nn.Conv2d(channels[0], int(num_classes), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for idx, block in enumerate(self.encoders):
            x = block(x)
            skips.append(x)
            if idx != len(self.encoders) - 1:
                x = self.pool(x)

        x = self.pool(x)
        x = self.bottleneck(x)
        for block, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return self.head(x)


class RangeImageUNetSegmenter(RangeImageSegmentationModel):
    def __init__(
        self,
        num_classes: int = 23,
        in_channels: int = 4,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        use_balanced_class_weights: bool = True,
    ) -> None:
        super().__init__(num_classes=num_classes, use_balanced_class_weights=use_balanced_class_weights)
        self.save_hyperparameters()
        self.model = RangeUNet(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            base_channels=int(base_channels),
            depth=int(depth),
            dropout=float(dropout),
        )

    def prepare_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["range_images"].float()
        y = batch["semantic"].long()
        valid = batch["valid_label"].bool()

        if x.ndim != 5 or x.shape[-1] != int(self.hparams.in_channels):
            raise ValueError(
                f"Expected range_images shape [B,R,H,W,{int(self.hparams.in_channels)}], got {tuple(x.shape)}"
            )
        bsz, returns, height, width, channels = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(bsz * returns, channels, height, width)
        y = y.reshape(bsz * returns, height, width)
        valid = valid.reshape(bsz * returns, height, width) & (y > 0)
        return x, y, valid

    def _predict_logits(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, valid = self.prepare_batch(batch)
        return self.model(x), y, valid

    def predict_range_labels(self, batch: dict, *, sampling_steps: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        _ = sampling_steps
        logits, labels, valid = self._predict_logits(batch)
        preds = logits.argmax(dim=1)
        preds = torch.where(valid, preds, torch.zeros_like(preds))
        return preds, labels

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        logits, labels, valid = self._predict_logits(batch)
        preds = logits.argmax(dim=1)
        batch_size = int(batch["range_images"].shape[0])

        if not bool(valid.any()):
            loss = logits.sum() * 0.0
            self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            return loss

        target = labels.clone()
        target[~valid] = -100
        loss = F.cross_entropy(logits, target, weight=self.class_weights, ignore_index=-100)
        if stage in {"val", "test"}:
            self.update_confmat(stage=stage, preds=preds[valid], labels=labels[valid])

        acc = (preds[valid] == labels[valid]).float().mean()
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=float(self.hparams.learning_rate))
