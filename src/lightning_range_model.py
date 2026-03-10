from typing import Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.rank_zero import rank_zero_info


def _pad_width_circular_height_replicate(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_w > 0:
        x = F.pad(x, (pad_w, pad_w, 0, 0), mode="circular")
    if pad_h > 0:
        x = F.pad(x, (0, 0, pad_h, pad_h), mode="replicate")
    return x


class CircularWidthConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = False,
    ) -> None:
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
        x = _pad_width_circular_height_replicate(x, pad_h=self.pad, pad_w=self.pad)
        return self.conv(x)


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
        x = self.act(self.bn2(self.conv2(x)))
        return x


class RangeUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(depth) < 2:
            raise ValueError("depth must be >= 2")
        if int(base_channels) <= 0:
            raise ValueError("base_channels must be > 0")

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

        self.head = nn.Conv2d(channels[0], int(num_classes), kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for i, block in enumerate(self.encoders):
            x = block(x)
            skips.append(x)
            if i != len(self.encoders) - 1:
                x = self.pool(x)

        x = self.pool(x)
        x = self.bottleneck(x)

        for block, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return self.head(x)


class RangeImageUNetSegmenter(L.LightningModule):
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
        super().__init__()
        self.save_hyperparameters()

        self.model = RangeUNet(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            base_channels=int(base_channels),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.register_buffer("_class_weights", torch.ones(int(num_classes), dtype=torch.float32), persistent=False)
        self.register_buffer(
            "_val_confmat",
            torch.zeros((int(num_classes), int(num_classes)), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_test_confmat",
            torch.zeros((int(num_classes), int(num_classes)), dtype=torch.long),
            persistent=False,
        )
        self._weights_ready = False

    @property
    def class_weights(self) -> Optional[torch.Tensor]:
        if not self.hparams.use_balanced_class_weights or not self._weights_ready:
            return None
        return self._class_weights

    def set_class_weights(self, class_weights: torch.Tensor) -> None:
        class_weights = class_weights.detach().float().to(self.device)
        if class_weights.ndim != 1 or class_weights.shape[0] != int(self.hparams.num_classes):
            raise ValueError(
                f"class_weights shape mismatch: got {tuple(class_weights.shape)}, "
                f"expected ({int(self.hparams.num_classes)},)"
            )
        self._class_weights.copy_(class_weights)
        self._weights_ready = True

    def _prepare_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # range_images: [B,R,H,W,4] -> [B*R,4,H,W]
        x = batch["range_images"].float()
        y = batch["semantic"].long()
        valid = batch["valid_label"].bool()

        if x.ndim != 5 or x.shape[-1] != int(self.hparams.in_channels):
            raise ValueError(
                f"Expected range_images shape [B,R,H,W,{int(self.hparams.in_channels)}], got {tuple(x.shape)}"
            )
        b, r, h, w, c = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(b * r, c, h, w)
        y = y.reshape(b * r, h, w)
        valid = valid.reshape(b * r, h, w) & (y >= 0)
        return x, y, valid

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        x, y, valid = self._prepare_batch(batch)
        batch_size = int(batch["range_images"].shape[0])

        logits = self.model(x)
        preds = logits.argmax(dim=1)

        if not bool(valid.any()):
            loss = logits.sum() * 0.0
            self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            return loss

        target = y.clone()
        target[~valid] = -100
        loss = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=-100,
        )

        if stage in {"val", "test"}:
            self._update_confmat(stage=stage, preds=preds[valid], labels=y[valid])

        acc = (preds[valid] == y[valid]).float().mean()
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def _update_confmat(self, stage: str, preds: torch.Tensor, labels: torch.Tensor) -> None:
        num_classes = int(self.hparams.num_classes)
        if preds.numel() == 0:
            return
        preds = preds.reshape(-1).long()
        labels = labels.reshape(-1).long()
        keep = (labels >= 0) & (labels < num_classes) & (preds >= 0) & (preds < num_classes)
        if not bool(keep.any()):
            return
        preds = preds[keep]
        labels = labels[keep]
        flat = labels * num_classes + preds
        bincount = torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        if stage == "val":
            self._val_confmat += bincount
        else:
            self._test_confmat += bincount

    def _log_iou_metrics(self, stage: str) -> None:
        conf = self._val_confmat if stage == "val" else self._test_confmat
        conf = conf.to(dtype=torch.float32)
        tp = torch.diag(conf)
        fp = conf.sum(dim=0) - tp
        fn = conf.sum(dim=1) - tp
        union = tp + fp + fn

        valid = union > 0
        iou = torch.full_like(union, fill_value=-1.0, dtype=torch.float32)
        iou[valid] = tp[valid] / union[valid].clamp_min(1e-6)

        if bool(valid.any()):
            miou = iou[valid].mean()
        else:
            miou = torch.tensor(0.0, dtype=torch.float32, device=conf.device)

        self.log(f"{stage}_mIoU", miou, on_step=False, on_epoch=True, prog_bar=True)
        for cls_idx, cls_iou in enumerate(iou):
            self.log(f"{stage}_IoU_class_{cls_idx}", cls_iou, on_step=False, on_epoch=True, prog_bar=False)

    def on_validation_epoch_start(self) -> None:
        self._val_confmat.zero_()

    def on_validation_epoch_end(self) -> None:
        self._log_iou_metrics(stage="val")

    def on_test_epoch_start(self) -> None:
        self._test_confmat.zero_()

    def on_test_epoch_end(self) -> None:
        self._log_iou_metrics(stage="test")

    def on_fit_start(self) -> None:
        dm = self.trainer.datamodule
        counts = getattr(dm, "class_counts", None) if dm is not None else None
        weights = getattr(dm, "class_weights", None) if dm is not None else None

        if counts is not None:
            counts_cpu = counts.detach().cpu()
            rank_zero_info(f"Class counts (train supervised pixels): {counts_cpu.tolist()}")

        if not bool(self.hparams.use_balanced_class_weights):
            rank_zero_info("Balanced class weights: disabled")
            return

        if weights is not None:
            self.set_class_weights(weights.to(self.device))
            rank_zero_info(f"Class weights: {weights.detach().cpu().tolist()}")
        else:
            rank_zero_info("Class weights: unavailable (not computed by datamodule).")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=float(self.hparams.learning_rate))
