"""Evaluate trained 3D point-cloud diffusion segmentation model."""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from diffusion import DDPM
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


def split_indices_by_segment(dataset: WaymoLidarDataset, num_test_segments: int) -> list[int]:
    segments = sorted({segment for segment, _ in dataset.records})
    n_test = max(1, min(len(segments), int(num_test_segments)))
    test_set = set(segments[-n_test:])
    return [i for i, (segment, _) in enumerate(dataset.records) if segment in test_set]


@torch.no_grad()
def compute_iou_and_confusion(pred: torch.Tensor, true: torch.Tensor, valid: torch.Tensor) -> tuple[Dict[str, float], np.ndarray]:
    pred_v = pred[valid]
    true_v = true[valid]

    defined = true_v != 0
    pred_v = pred_v[defined]
    true_v = true_v[defined]

    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(conf, (true_v.numpy(), pred_v.numpy()), 1)

    per_class: Dict[str, float] = {}
    ious = []

    for c in range(NUM_CLASSES):
        inter = int(((pred_v == c) & (true_v == c)).sum().item())
        union = int(((pred_v == c) | (true_v == c)).sum().item())
        if union > 0:
            iou = inter / union
            per_class[str(c)] = float(iou)
            if c != 0:
                ious.append(iou)
        else:
            per_class[str(c)] = float("nan")

    per_class["mIoU"] = float(np.mean(ious)) if ious else 0.0
    return per_class, conf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 3D point-cloud diffusion segmentation model.")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pt")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-test-segments", type=int, default=10)
    parser.add_argument("--max-cached-segments", type=int, default=2)
    parser.add_argument("--num-ddpm-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt.get("args", {})

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model = build_denoiser(
        backbone=str(train_args.get("backbone", "edgeconv")),
        num_classes=NUM_CLASSES,
        hidden_dim=int(train_args.get("hidden_dim", 256)),
        depth=int(train_args.get("depth", 6)),
        knn_k=int(train_args.get("knn_k", 16)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    T = int(train_args.get("T", 1000))
    ddpm = DDPM(T=T).to(device)

    dataset = WaymoLidarDataset(
        path=args.data_root,
        num_points=args.num_points,
        max_cached_segments=args.max_cached_segments,
        seed=0,
    )
    test_idx = split_indices_by_segment(dataset, args.num_test_segments)
    test_ds = Subset(dataset, test_idx)

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_pred = []
    all_true = []
    all_valid = []

    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            labels = batch["labels"].to(device)
            valid = batch["mask"].to(device)

            bsz, npts, _ = points.shape
            x_t = torch.randn(bsz, npts, NUM_CLASSES, device=device)

            if args.num_ddpm_steps and args.num_ddpm_steps < T:
                step_indices = np.linspace(T, 1, args.num_ddpm_steps, dtype=int)
            else:
                step_indices = np.arange(T, 0, -1, dtype=int)

            for t_int in step_indices:
                t = torch.full((bsz,), int(t_int), device=device, dtype=torch.long)
                x_t = ddpm.p_sample(model, x_t, t, points)

            pred = x_t.argmax(dim=-1)

            all_pred.append(pred.cpu())
            all_true.append(labels.cpu())
            all_valid.append(valid.cpu())

    pred_cat = torch.cat(all_pred, dim=0)
    true_cat = torch.cat(all_true, dim=0)
    valid_cat = torch.cat(all_valid, dim=0)

    per_class_iou, conf = compute_iou_and_confusion(pred_cat, true_cat, valid_cat)

    print("Per-class IoU:")
    for k, v in per_class_iou.items():
        if k == "mIoU":
            continue
        print(f"  class {k:>2}: {v:.4f}" if not np.isnan(v) else f"  class {k:>2}: N/A")
    print(f"mIoU: {per_class_iou['mIoU']:.4f}")

    with (out_dir / "miou_table.json").open("w") as f:
        json.dump({"per_class_iou": per_class_iou}, f, indent=2)

    with (out_dir / "confusion_matrix.json").open("w") as f:
        json.dump(conf.tolist(), f)

    print(f"Saved {out_dir / 'miou_table.json'}")
    print(f"Saved {out_dir / 'confusion_matrix.json'}")


if __name__ == "__main__":
    main()
