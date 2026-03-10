"""Training script for 3D point-cloud diffusion semantic segmentation."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from diffusion import DDPM, labels_to_soft
from model_3d import build_denoiser
from waymo_lidar_dataset_loader import WaymoLidarDataset

NUM_CLASSES = 23


def resolve_device(device_arg: str = "auto") -> torch.device:
    req = str(device_arg).lower()
    if req == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return torch.device("mps" if has_mps else "cpu")
    if req == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_indices_by_segment(
    dataset: WaymoLidarDataset,
    num_val_segments: int,
    num_train_segments: Optional[int] = None,
) -> tuple[list[int], list[int]]:
    segments = sorted({segment for segment, _ in dataset.records})
    n_val = max(0, min(len(segments), int(num_val_segments)))
    val_set = set(segments[-n_val:]) if n_val > 0 else set()
    train_segments = [s for s in segments if s not in val_set]
    if num_train_segments is not None:
        n_train = max(1, min(len(train_segments), int(num_train_segments)))
        train_segments = train_segments[:n_train]
    train_set = set(train_segments)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for idx, (segment, _) in enumerate(dataset.records):
        if segment in val_set:
            val_idx.append(idx)
        elif segment in train_set:
            train_idx.append(idx)

    return train_idx, val_idx


def compute_class_weights(dataset, indices: list[int], num_classes: int = NUM_CLASSES) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.int64)

    for idx in indices:
        sample = dataset[idx]
        valid = sample["mask"].numpy().astype(bool)
        labels = sample["labels"].numpy()
        labels = labels[valid]
        labels = labels[labels != 0]
        if labels.size > 0:
            binc = np.bincount(labels, minlength=num_classes)
            counts[: len(binc)] += binc[:num_classes]

    total = int(counts.sum())
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    if total > 0 and present.any():
        weights[present] = total / (num_classes * counts[present])
        weights[present] /= weights[present].mean()
    weights[0] = 0.0

    return torch.from_numpy(weights)


@torch.no_grad()
def compute_iou(pred: torch.Tensor, true: torch.Tensor, valid: torch.Tensor, num_classes: int = NUM_CLASSES) -> Dict[str, float]:
    # pred/true/valid: (B,N)
    pred_v = pred[valid]
    true_v = true[valid]

    defined = true_v != 0
    pred_v = pred_v[defined]
    true_v = true_v[defined]

    ious = []
    per_class: Dict[str, float] = {}

    for c in range(num_classes):
        inter = ((pred_v == c) & (true_v == c)).sum().item()
        union = ((pred_v == c) | (true_v == c)).sum().item()
        if union > 0:
            iou = float(inter / union)
            per_class[str(c)] = iou
            if c != 0:
                ious.append(iou)
        else:
            per_class[str(c)] = float("nan")

    per_class["mIoU"] = float(np.mean(ious)) if ious else 0.0
    return per_class


def run_epoch(
    model: torch.nn.Module,
    ddpm: DDPM,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    class_weights: Optional[torch.Tensor] = None,
    log_every: int = 1,
    phase: str = "train",
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_batches = 0

    all_pred = []
    all_true = []
    all_valid = []

    num_batches = len(loader)
    for step, batch in enumerate(loader, start=1):
        points = batch["points"].to(device)  # (B,N,3)
        labels = batch["labels"].to(device)  # (B,N)
        valid = batch["mask"].to(device)     # (B,N)
        bsz = points.shape[0]

        x0 = labels_to_soft(labels, NUM_CLASSES, valid)
        t = torch.randint(1, ddpm.T + 1, (bsz,), device=device, dtype=torch.long)
        x_t, eps = ddpm.q_sample(x0, t)
        eps_pred = model(x_t, t, points)

        loss = ddpm.loss(eps_pred, eps, valid, labels=labels, class_weights=class_weights)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

        if log_every > 0 and (step == 1 or step % log_every == 0 or step == num_batches):
            print(
                f"[{phase}] step {step:04d}/{num_batches:04d} loss={loss.item():.4f}",
                flush=True,
            )

        # one-step proxy segmentation estimate
        with torch.no_grad():
            sqrt_ab = ddpm._extract(ddpm.sqrt_alpha_bars, t, x_t.shape)
            sqrt_1mab = ddpm._extract(ddpm.sqrt_one_minus_ab, t, x_t.shape)
            x0_est = (x_t - sqrt_1mab * eps_pred) / sqrt_ab.clamp(min=1e-6)
            pred = x0_est.argmax(dim=-1)

        all_pred.append(pred.cpu())
        all_true.append(labels.cpu())
        all_valid.append(valid.cpu())

    if total_batches == 0:
        return 0.0, 0.0

    pred_cat = torch.cat(all_pred, dim=0)
    true_cat = torch.cat(all_true, dim=0)
    valid_cat = torch.cat(all_valid, dim=0)
    metrics = compute_iou(pred_cat, true_cat, valid_cat)

    return total_loss / total_batches, metrics["mIoU"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D point-cloud diffusion segmentation model.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-train-segments", type=int, default=None)
    parser.add_argument("--num-val-segments", type=int, default=5)
    parser.add_argument("--max-cached-segments", type=int, default=2)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--backbone", type=str, default="edgeconv", choices=["edgeconv", "pointnet"])
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    print("Initializing dataset...", flush=True)
    dataset = WaymoLidarDataset(
        path=args.data_root,
        num_points=args.num_points,
        max_cached_segments=args.max_cached_segments,
        seed=args.seed,
    )
    print(f"Dataset ready: {len(dataset)} frames", flush=True)
    train_idx, val_idx = split_indices_by_segment(
        dataset,
        args.num_val_segments,
        num_train_segments=args.num_train_segments,
    )

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx) if len(val_idx) > 0 else None

    print("Building dataloaders...", flush=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
    print(f"Train batches per epoch: {len(train_loader)}", flush=True)
    if val_loader is not None:
        print(f"Val batches per epoch: {len(val_loader)}", flush=True)

    print("Computing class weights...", flush=True)
    class_weights = compute_class_weights(dataset, train_idx).to(device)
    print("Class weights ready", flush=True)

    model = build_denoiser(
        backbone=args.backbone,
        num_classes=NUM_CLASSES,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        knn_k=args.knn_k,
    ).to(device)

    ddpm = DDPM(T=args.T).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_miou", "val_loss", "val_miou", "lr"],
        )
        writer.writeheader()

        best_val = float("inf")
        best_ckpt = out_dir / "best_model.pt"

        for epoch in range(1, args.epochs + 1):
            print(f"Starting epoch {epoch}/{args.epochs}", flush=True)
            train_loss, train_miou = run_epoch(
                model=model,
                ddpm=ddpm,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                class_weights=class_weights,
                log_every=args.log_every,
                phase="train",
            )

            if val_loader is not None:
                val_loss, val_miou = run_epoch(
                    model=model,
                    ddpm=ddpm,
                    loader=val_loader,
                    device=device,
                    optimizer=None,
                    class_weights=None,
                    log_every=args.log_every,
                    phase="val",
                )
            else:
                val_loss, val_miou = float("nan"), float("nan")

            scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_miou": train_miou,
                "val_loss": val_loss,
                "val_miou": val_miou,
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            f.flush()

            print(
                f"epoch {epoch:03d} | train_loss={train_loss:.4f} train_mIoU={train_miou:.4f} "
                f"| val_loss={val_loss:.4f} val_mIoU={val_miou:.4f}"
            )

            metric_for_best = val_loss if val_loader is not None else train_loss
            if metric_for_best < best_val:
                best_val = metric_for_best
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                        "class_weights": class_weights.detach().cpu(),
                    },
                    best_ckpt,
                )

    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "epoch": args.epochs,
            "class_weights": class_weights.detach().cpu(),
        },
        out_dir / "last_model.pt",
    )

    with (out_dir / "hparams.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved metrics to {metrics_path}")
    print(f"Best checkpoint: {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
