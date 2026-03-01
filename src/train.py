"""
train.py
========
PyTorch Lightning training script for LiDAR semantic segmentation diffusion model.

Usage:
    cd "D:/ECE 271B SemSeg Project"

    # Train on 10 segments (default), evaluate on last 10
    python src/train.py

    # Train on 20 segments
    python src/train.py --num-segments 20

    # Train on all 40 with more epochs
    python src/train.py --num-segments 40 --epochs 100

    # Quick smoke test
    python src/train.py --num-segments 2 --epochs 2 --batch-size 1

    # Half precision for faster training (RTX cards support this well)
    python src/train.py --num-segments 40 --epochs 50 --precision 16-mixed

Full options:
    --num-segments  N   Segments to train on, front of sorted list (default: 10, max: 40)
    --epochs        N   Training epochs (default: 50)
    --batch-size    N   Batch size (default: 2)
    --lr            F   Learning rate (default: 2e-4)
    --T             N   Diffusion timesteps (default: 1000)
    --base-channels N   U-Net base channel width (default: 32)
    --data-root     P   Path to training/ directory
    --out-dir       P   Where to save checkpoints and outputs (default: outputs/)
    --seed          N   Random seed (default: 0)
    --workers       N   DataLoader workers (default: 0, safe on Windows)
    --precision     S   32, 16-mixed, bf16-mixed (default: 32)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

sys.path.insert(0, str(Path(__file__).parent))
from dataset   import get_train_dataset, get_test_dataset, NUM_CLASSES, CLASS_NAMES
from model     import LiDARDiffusionUNet
from diffusion import DDPM

DEFAULT_DATA_ROOT = Path("Waymo Data/training")
DEFAULT_OUT_DIR   = Path("outputs")


# ── Soft label encoding ────────────────────────────────────────────────────────
def labels_to_soft(labels: torch.Tensor, num_classes: int, valid: torch.Tensor) -> torch.Tensor:
    """
    Integer label map → soft one-hot in [-1, 1], invalid pixels zeroed out.

    labels : (B, H, W)  int64
    valid  : (B, H, W)  bool
    returns: (B, C, H, W) float32
    """
    one_hot = F.one_hot(labels.clamp(0, num_classes - 1), num_classes)  # (B,H,W,C)
    one_hot = one_hot.permute(0, 3, 1, 2).float()                        # (B,C,H,W)
    x0 = one_hot * 2.0 - 1.0                                             # scale to [-1,1]
    x0 = x0 * valid.unsqueeze(1).float()                                 # zero invalid pixels
    return x0


# ── LightningModule ────────────────────────────────────────────────────────────
class LiDARDiffusionModule(L.LightningModule):
    """
    Lightning wrapper around the U-Net + DDPM.

    Lightning handles: device placement, gradient clipping, checkpointing,
    logging, mixed precision, and multi-GPU — zero extra code needed.

    training_step   : one batch, compute noise prediction loss
    validation_step : fast proxy mIoU (single low-noise step, not full sampling)
    configure_optimizers: AdamW + cosine LR decay
    """

    def __init__(self, hparams: dict):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = LiDARDiffusionUNet(
            num_classes    = NUM_CLASSES,
            lidar_channels = 4,
            base_channels  = hparams["base_channels"],
            channel_mults  = (1, 2, 4),
            num_res_blocks = 2,
            time_emb_dim   = 128,
            context_dim    = 128,
            n_heads        = 4,
            head_dim       = 32,
            dropout        = 0.1,
        )

        self.ddpm = DDPM(T=hparams["T"])

        # IoU accumulators reset each validation epoch
        self.register_buffer("_val_intersection", torch.zeros(NUM_CLASSES, dtype=torch.long))
        self.register_buffer("_val_union",        torch.zeros(NUM_CLASSES, dtype=torch.long))

    def on_validation_start(self):
        self.ddpm.to(self.device)

    # ── Training step ─────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        lidar  = batch["lidar"]    # (B, 4, H, W)
        labels = batch["labels"]   # (B, H, W)
        valid  = batch["valid"]    # (B, H, W)
        B      = lidar.shape[0]

        x0     = labels_to_soft(labels, NUM_CLASSES, valid)
        t      = torch.randint(1, self.hparams["T"] + 1, (B,), device=self.device, dtype=torch.long)
        x_t, eps = self.ddpm.q_sample(x0, t)
        eps_pred = self.model(x_t, t, lidar)
        loss     = self.ddpm.loss(eps_pred, eps, valid)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ── Validation step ───────────────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        """
        Fast validation proxy — not full reverse diffusion (that's too slow per epoch).
        We corrupt labels to t=1 (barely noisy) and check how well the model denoises.
        Full mIoU from complete sampling runs in evaluate.py after training is done.
        """
        lidar  = batch["lidar"]
        labels = batch["labels"]
        valid  = batch["valid"]
        B      = lidar.shape[0]

        x0       = labels_to_soft(labels, NUM_CLASSES, valid)
        t_low    = torch.ones(B, device=self.device, dtype=torch.long)
        x_t, eps = self.ddpm.q_sample(x0, t_low)

        with torch.no_grad():
            eps_pred = self.model(x_t, t_low, lidar)

        val_loss = self.ddpm.loss(eps_pred, eps, valid)
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Reconstruct x_0 estimate to get class predictions
        sqrt_ab   = self.ddpm._extract(self.ddpm.sqrt_alpha_bars,   t_low, x_t.shape)
        sqrt_1mab = self.ddpm._extract(self.ddpm.sqrt_one_minus_ab, t_low, x_t.shape)
        x0_est    = (x_t - sqrt_1mab * eps_pred) / sqrt_ab.clamp(min=1e-6)
        pred      = x0_est.argmax(dim=1).cpu()   # (B, H, W)

        v      = valid.bool().cpu()
        pred_v = pred[v]
        true_v = labels.cpu()[v]

        for c in range(NUM_CLASSES):
            self._val_intersection[c] += ((pred_v == c) & (true_v == c)).sum()
            self._val_union[c]        += ((pred_v == c) | (true_v == c)).sum()

        return val_loss

    def on_validation_epoch_end(self):
        ious = []
        for c in range(NUM_CLASSES):
            u = self._val_union[c].item()
            ious.append(self._val_intersection[c].item() / u if u > 0 else float("nan"))

        valid_ious = [v for v in ious if not np.isnan(v)]
        miou = float(np.mean(valid_ious)) if valid_ious else 0.0
        self.log("val/miou", miou, prog_bar=True)

        self._val_intersection.zero_()
        self._val_union.zero_()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams["epochs"], eta_min=self.hparams["lr"] / 10
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    L.seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n── Loading dataset ──")
    train_ds = get_train_dataset(args.data_root, num_segments=args.num_segments)
    test_ds  = get_test_dataset(args.data_root,  num_segments=args.num_val_segs)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False,
    )
    val_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False,
    )

    # ── Module ────────────────────────────────────────────────────────────────
    print("\n── Building model ──")
    hparams = vars(args)
    module  = LiDARDiffusionModule(hparams)

    n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath   = out_dir,
        filename  = "best_model",
        monitor   = "val/loss",
        mode      = "min",
        save_last = True,
        verbose   = True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ── Trainer ───────────────────────────────────────────────────────────────
    # Lightning auto-detects GPU, handles device placement, mixed precision,
    # gradient clipping — nothing to configure manually.
    trainer = L.Trainer(
        max_epochs          = args.epochs,
        accelerator         = "auto",
        devices             = "auto",
        precision           = args.precision,
        callbacks           = [checkpoint_cb, lr_monitor],
        logger              = CSVLogger(save_dir=str(out_dir), name="logs"),
        log_every_n_steps   = max(1, len(train_loader) // 5),
        gradient_clip_val   = 1.0,
        enable_progress_bar = True,
    )

    print(f"\n── Training for {args.epochs} epochs ──")
    trainer.fit(module, train_loader, val_loader)

    # Save hparams for evaluate.py to read
    with open(out_dir / "hparams.json", "w") as f:
        json.dump(hparams, f, indent=2, default=str)

    print(f"\nDone. Best checkpoint: {checkpoint_cb.best_model_path}")
    print("Run evaluation:  python src/evaluate.py")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LiDAR diffusion segmentation model (Lightning).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--num-segments",  type=int,   default=40)
    parser.add_argument("--num-val-segs",  type=int,   default=3,
                        help="Test segments to use for validation during training (default 3)")
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch-size",    type=int,   default=1)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--T",             type=int,   default=1000)
    parser.add_argument("--base-channels", type=int,   default=32)
    parser.add_argument("--data-root",     type=str,   default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir",       type=str,   default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--workers",       type=int,   default=0)
    parser.add_argument("--precision",     type=str,   default="32",
                        choices=["32", "16-mixed", "bf16-mixed"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.num_segments = max(1, min(40, args.num_segments))
    print(f"Training on {args.num_segments} segments, validating on {args.num_val_segs} test segment(s).")
    main(args)