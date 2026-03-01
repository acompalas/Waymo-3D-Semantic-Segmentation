# Waymo 3D Semantic Segmentation — Diffusion Model

LiDAR semantic segmentation on the Waymo Open Dataset using a DDPM diffusion model with a U-Net conditioned on range images.

---

## Project Structure

```
Waymo-3D-Semantic-Segmentation/
├── README.md
├── .gitignore
├── src/
│   ├── dataset.py        — Waymo parquet dataset loader
│   ├── model.py          — U-Net diffusion model
│   ├── diffusion.py      — DDPM noise schedule and sampling
│   ├── train.py          — PyTorch Lightning training script
│   ├── evaluate.py       — Inference, mIoU table, error visualization GIFs
│   ├── plot_metrics.py   — Loss and mIoU curves from training logs
│   └── visualize.py      — Interactive ground truth point cloud viewer
├── outputs/              — Generated automatically on first run
│   ├── best_model.ckpt   — Best checkpoint by validation loss
│   ├── last.ckpt         — Most recent checkpoint
│   ├── miou_table.json   — Per-class IoU and overall mIoU
│   ├── loss_curve.png    — Train vs val loss over epochs
│   ├── miou_curve.png    — Validation mIoU over epochs
│   ├── error_viz_seg1.gif — Error visualization GIF for last test segment
│   └── logs/             — Lightning CSVLogger metrics (read by plot_metrics.py)
└── Waymo Data/           — Not committed (in .gitignore)
    └── training/
        ├── lidar/
        ├── lidar_calibration/
        └── lidar_segmentation/
```

---

## Data Setup

Download the Waymo Open Dataset v2 parquet files and arrange them as follows:

```
Waymo Data/
└── training/
    ├── lidar/                  ← LiDAR range image parquet files
    ├── lidar_calibration/      ← Beam inclination + extrinsic calibration
    └── lidar_segmentation/     ← Semantic label parquet files
```

Each subfolder should contain `.parquet` files, one per segment. The code sorts segments alphabetically — training always uses segments from the **front** of the sorted list, evaluation always uses segments from the **back**.

> The `.txt` URI files are not required. Only the parquet files matter.

---

## Installation

```bash
pip install torch torchvision lightning polars numpy open3d pillow matplotlib pandas
```

---

## Full Training Run (Copy & Paste)

```bash
# Train on 40 segments, validate on last 10, 50 epochs
python src/train.py --num-segments 40 --epochs 50 --batch-size 1

# Evaluate on all 10 test segments, produce 1 error GIF
python src/evaluate.py

# Plot loss and mIoU curves
python src/plot_metrics.py
```

Outputs saved to `outputs/`:
- `best_model.ckpt` — best checkpoint by validation loss
- `miou_table.json` — per-class IoU and mIoU
- `error_viz_seg1.gif` — error visualization of last test segment
- `loss_curve.png` — train vs val loss over epochs
- `miou_curve.png` — validation mIoU over epochs

---

## CLI Reference

### `train.py`

| Argument | Default | Description |
|---|---|---|
| `--num-segments` | 40 | How many segments to train on (from front of sorted list) |
| `--num-val-segs` | 10 | How many segments to validate on during training (from back) |
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 1 | Batch size (1 recommended for 6GB VRAM) |
| `--data-root` | `Waymo Data/training` | Path to training directory |
| `--out-dir` | `outputs/` | Where to save checkpoints and logs |

### `evaluate.py`

| Argument | Default | Description |
|---|---|---|
| `--num-test-segs` | 10 | How many segments to evaluate (from back of sorted list) |
| `--gif-segs` | 1 | How many segments to render as error GIFs |
| `--gif-fps` | 5.0 | GIF frame rate |
| `--no-viz` | off | Skip GIF rendering entirely |
| `--num-ddpm-steps` | 1000 | Denoising steps (fewer = faster but lower quality, e.g. 50) |
| `--checkpoint` | `outputs/best_model.ckpt` | Path to model checkpoint |

### `plot_metrics.py`

No arguments needed. Reads from `outputs/logs/` automatically.

---

## Smoke Test (Quick Pipeline Check)

```bash
# Train: 2 segments, 2 epochs, 1 val segment
python src/train.py --num-segments 2 --epochs 2 --batch-size 1 --num-val-segs 1

# Evaluate: 1 segment, 10 denoising steps, 1 GIF
python src/evaluate.py --num-test-segs 1 --num-ddpm-steps 10 --gif-segs 1

# Plot
python src/plot_metrics.py
```

---

## Visualize Ground Truth

```bash
# View 1 random segment interactively
python src/visualize.py

# View specific segment by index
python src/visualize.py --segment-indices 5

# Export first segment to GIF
python src/visualize.py --export-gif outputs/viz.gif
```

---

## Notes

- Training and evaluation segments are **independent** — you can train on 10 segments and still evaluate on the last 10 regardless of overlap
- If you specify more segments than you have downloaded, the code will warn you and clamp automatically
- `val/miou` shown during training is a proxy metric (1 denoising step) — use `evaluate.py` for the real mIoU
- Mixed precision training (`--precision 16-mixed`) is supported automatically on compatible GPUs