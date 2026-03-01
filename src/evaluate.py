"""
evaluate.py
===========
Load a trained model checkpoint and evaluate on the test set.
Produces:
  1. Per-class mIoU table (printed + saved as JSON)
  2. 3D error visualization GIF — gray=correct, red intensity=confidence of wrong prediction

Usage:
    cd "D:/ECE 271B SemSeg Project"

    # Evaluate best checkpoint, visualize 3 test frames as GIF
    python src/evaluate.py

    # Evaluate specific checkpoint, more viz frames
    python src/evaluate.py --checkpoint outputs/final_model.pt --viz-frames 10

    # Only compute mIoU, skip visualization
    python src/evaluate.py --no-viz

Full options:
    --checkpoint    PATH   Model checkpoint to load (default: outputs/best_model.pt)
    --data-root     PATH   Path to training/ directory
    --out-dir       PATH   Where to save outputs (default: outputs/)
    --viz-frames    N      How many test frames to include in error GIF (default: 5)
    --gif-fps       F      GIF frame rate (default: 5.0)
    --no-viz               Skip 3D error visualization
    --num-ddpm-steps N     Use fewer reverse steps for faster inference (default: full T)
                           e.g. 50 gives ~20x speedup with minor quality loss
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import open3d as o3d
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from dataset   import get_test_dataset, NUM_CLASSES, CLASS_NAMES, WaymoRangeImageDataset
from model     import LiDARDiffusionUNet
from diffusion import DDPM

DEFAULT_DATA_ROOT = Path("Waymo Data/training")
DEFAULT_OUT_DIR   = Path("outputs")
DEFAULT_CKPT      = Path("outputs/best_model.ckpt")


# ── Calibration helpers (needed to project range image → point cloud) ──────────
import polars as pl

LASER_COL   = "key.laser_name"
SEGMENT_COL = "key.segment_context_name"
PRIMARY_LASER = 1


def _load_calibration(data_root: Path, segment: str) -> dict:
    calib_df = (
        pl.scan_parquet(str(data_root / "lidar_calibration" / "*.parquet"))
        .filter(
            (pl.col(SEGMENT_COL) == segment) &
            (pl.col(LASER_COL) == PRIMARY_LASER)
        )
        .collect()
    )
    if calib_df.height == 0:
        raise KeyError(f"No calibration for segment {segment}")
    return calib_df.row(0, named=True)


def _beam_inclinations(calib_row: dict, H: int) -> np.ndarray:
    vals = calib_row["[LiDARCalibrationComponent].beam_inclination.values"]
    if vals is not None and len(vals) == H:
        return np.asarray(vals, dtype=np.float32)[::-1].copy()
    mn = float(calib_row["[LiDARCalibrationComponent].beam_inclination.min"])
    mx = float(calib_row["[LiDARCalibrationComponent].beam_inclination.max"])
    return np.linspace(mn, mx, H, dtype=np.float32)[::-1].copy()


def _extrinsic(calib_row: dict) -> np.ndarray:
    T = np.asarray(calib_row["[LiDARCalibrationComponent].extrinsic.transform"], dtype=np.float64)
    return T.reshape(4, 4)


def range_image_to_xyz(
    range_image: np.ndarray,   # (4, H, W) float32
    inclinations: np.ndarray,  # (H,)
    extrinsic: np.ndarray,     # (4, 4)
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (xyz_vehicle, valid_mask). Matches visualize.py projection exactly."""
    if range_image.shape[0] == 4:
        range_image = range_image.transpose(1, 2, 0)  # (H, W, 4)

    H, W = range_image.shape[:2]
    r     = range_image[:, :, 0]
    valid = r > 0

    # Apply azimuth correction from extrinsic (same as visualize.py)
    az_correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0]).astype(np.float32)
    az   = np.linspace(np.pi, -np.pi, W, endpoint=False, dtype=np.float32) - az_correction
    inc2d = inclinations[:, None]
    az2d  = az[None, :]

    xs = (r * np.cos(inc2d) * np.cos(az2d))[valid]
    ys = (r * np.cos(inc2d) * np.sin(az2d))[valid]
    zs = (r * np.sin(inc2d))[valid]

    pts   = np.stack([xs, ys, zs], axis=1)
    ones  = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts.astype(np.float64), ones], axis=1)
    xyz   = (pts_h @ extrinsic.T)[:, :3].astype(np.float32)
    return xyz, valid


# ── Error color mapping ────────────────────────────────────────────────────────
def error_colors(
    pred:       np.ndarray,   # (N,) int  predicted class per valid point
    true:       np.ndarray,   # (N,) int  ground truth class per valid point
    confidence: np.ndarray,   # (N,) float  softmax probability of predicted class
) -> np.ndarray:
    """
    Light gray = correct prediction
    Magenta, scaled by confidence = wrong prediction
      low confidence wrong  → dim magenta
      high confidence wrong → bright magenta
    Returns (N, 3) float32 RGB in [0, 1].
    """
    correct = (pred == true)
    colors  = np.full((len(pred), 3), 0.85, dtype=np.float32)  # light gray for correct

    wrong_mask = ~correct
    if wrong_mask.any():
        t = confidence[wrong_mask].clip(0.0, 1.0)   # 0 = uncertain, 1 = confident
        # Cyan (0, 1, 1) → Pink (1.0, 0.4, 0.8)
        colors[wrong_mask, 0] = t              # R: 0 → 1.0
        colors[wrong_mask, 1] = 1.0 - t * 0.6 # G: 1 → 0.4
        colors[wrong_mask, 2] = 1.0 - t * 0.2 # B: 1 → 0.8

    return colors


# ── GIF saving ────────────────────────────────────────────────────────────────
def save_gif(frames, path, fps=5.0):
    if not frames:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dur = max(1, int(round(1000.0 / fps)))
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:], duration=dur, loop=0)
    print(f"Saved error GIF → {path} ({len(frames)} frames @ {fps:.1f} fps)")


# ── Open3D frame capture ───────────────────────────────────────────────────────
# Same car_pov preset as visualize.py
CAMERA_PRESET = np.array([
    [-0.026570101741086316, -0.9996106358387303,  0.008520939593593985, -0.6305623596589306],
    [-0.02978545966939474,  -0.007728510895758548,-0.9995264361244364,   3.275704827536906 ],
    [ 0.9992031105264592,   -0.02681131920334203, -0.02956851495806514,  13.644113467242853],
    [ 0.0,                   0.0,                  0.0,                   1.0              ],
], dtype=np.float64)


def render_frame_to_image(xyz: np.ndarray, colors: np.ndarray, w=1280, h=720) -> np.ndarray:
    """Render a point cloud to an RGB numpy array using Open3D offscreen."""
    pcd        = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=w, height=h)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.12, 0.14, 0.16])
    opt.point_size       = 2.5
    vis.add_geometry(pcd)
    vis.reset_view_point(True)

    ctr    = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    params.extrinsic = CAMERA_PRESET
    try:
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    except TypeError:
        ctr.convert_from_pinhole_camera_parameters(params)

    vis.poll_events()
    vis.update_renderer()
    rgb  = np.asarray(vis.capture_screen_float_buffer(do_render=True), dtype=np.float32)
    vis.destroy_window()
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


# ── Main evaluation ────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        # Try .ckpt extension if .pt was given
        alt = ckpt_path.with_suffix(".ckpt")
        if alt.exists():
            ckpt_path = alt
        else:
            print(f"ERROR: checkpoint not found: {ckpt_path}")
            sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)

    # Lightning checkpoints store everything under specific keys
    if "state_dict" in ckpt:
        # Lightning format: state_dict has keys like "model.encoder.weight"
        # Strip the "model." prefix to get plain model state dict
        raw_sd     = ckpt["state_dict"]
        model_sd   = {k[len("model."):]: v for k, v in raw_sd.items() if k.startswith("model.")}
        train_args = ckpt.get("hyper_parameters", {})
        epoch      = ckpt.get("epoch", "?")
        print(f"Loaded Lightning checkpoint (epoch {epoch})")
    else:
        # Plain PyTorch format (legacy)
        model_sd   = ckpt["model_state"]
        train_args = ckpt.get("args", {})
        epoch      = ckpt.get("epoch", "?")
        loss       = ckpt.get("loss", float("nan"))
        print(f"Loaded checkpoint from epoch {epoch}, loss={loss:.4f}")

    model = LiDARDiffusionUNet(
        num_classes    = NUM_CLASSES,
        lidar_channels = 4,
        base_channels  = train_args.get("base_channels", 32),
        channel_mults  = (1, 2, 4),
        num_res_blocks = 2,
        time_emb_dim   = 128,
        context_dim    = 128,
        n_heads        = 4,
        head_dim       = 32,
    ).to(device)
    model.load_state_dict(model_sd)
    model.eval()

    T      = train_args.get("T", 1000)
    n_steps = args.num_ddpm_steps if args.num_ddpm_steps else T
    ddpm    = DDPM(T=T, device=str(device)).to(device)

    # ── Test dataset ───────────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    test_ds   = get_test_dataset(data_root, num_segments=args.num_test_segs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Evaluation loop ────────────────────────────────────────────────────────
    intersection = torch.zeros(NUM_CLASSES, dtype=torch.long)
    union        = torch.zeros(NUM_CLASSES, dtype=torch.long)

    n_frames = len(test_ds)
    print(f"\nEvaluating {args.num_test_segs} segment(s), {n_frames} frames total...")

    # Track current segment for progress display
    current_seg      = None
    seg_frame_idx    = 0
    seg_total_frames = 0

    # Figure out which segments we'll produce GIFs for (last N segments evaluated)
    # Group frame indices by segment first
    all_segments_in_order = []
    seen = {}
    for idx in range(n_frames):
        seg = test_ds.records[idx][0]
        if seg not in seen:
            seen[seg] = len(all_segments_in_order)
            all_segments_in_order.append(seg)

    n_gif_segs   = args.gif_segs if not args.no_viz else 0
    gif_segments  = set(all_segments_in_order[-n_gif_segs:]) if n_gif_segs > 0 else set()

    # seg_name → list of rendered frames
    gif_buffers: dict[str, list] = {}

    with torch.no_grad():
        for i in range(n_frames):
            sample  = test_ds[i]
            lidar   = sample["lidar"].unsqueeze(0).to(device)
            labels  = sample["labels"]
            valid   = sample["valid"]
            segment = sample["segment_context_name"]

            # ── Inference: reverse diffusion ──────────────────────────────────
            if n_steps < T:
                step_indices = np.linspace(T, 1, n_steps, dtype=int)
            else:
                step_indices = np.arange(T, 0, -1, dtype=int)

            B, _, H, W = lidar.shape
            x_t = torch.randn(1, NUM_CLASSES, H, W, device=device)
            for t_int in step_indices:
                t_tensor = torch.full((1,), int(t_int), device=device, dtype=torch.long)
                x_t = ddpm.p_sample(model, x_t, t_tensor, lidar)
            x0_pred = x_t

            probs    = torch.softmax(x0_pred.squeeze(0), dim=0).cpu()  # (C, H, W)
            pred_cls = probs.argmax(dim=0)                              # (H, W)

            # ── mIoU accumulation ─────────────────────────────────────────────
            v      = valid.bool()
            pred_v = pred_cls[v]
            true_v = labels[v]

            for c in range(NUM_CLASSES):
                intersection[c] += ((pred_v == c) & (true_v == c)).sum().item()
                union[c]        += ((pred_v == c) | (true_v == c)).sum().item()

            # Track segment transitions for progress display
            if segment != current_seg:
                if current_seg is not None:
                    print()  # newline after previous segment's progress
                current_seg   = segment
                seg_frame_idx = 0
                seg_total_frames = sum(1 for r in test_ds.records if r[0] == segment)
                print(f"\nSegment: {segment[:50]}")
            seg_frame_idx += 1

            # ── 3D error visualization ─────────────────────────────────────────
            if segment in gif_segments:
                try:
                    calib = _load_calibration(data_root, segment)
                    ri_np = sample["lidar"].numpy()
                    inc   = _beam_inclinations(calib, ri_np.shape[1])
                    ext   = _extrinsic(calib)
                    xyz, _ = range_image_to_xyz(ri_np, inc, ext)

                    pred_np = pred_cls[v].numpy()
                    true_np = labels[v].numpy()
                    conf_np = probs.max(dim=0).values[v].numpy()

                    colors = error_colors(pred_np, true_np, conf_np)
                    frame  = render_frame_to_image(xyz, colors)

                    if segment not in gif_buffers:
                        gif_buffers[segment] = []
                    gif_buffers[segment].append(frame)
                    print(f"  Rendering frame {seg_frame_idx}/{seg_total_frames}", end="\r")
                except Exception as e:
                    print(f"  Frame {seg_frame_idx} viz skipped: {e}")
            else:
                print(f"  Frame {seg_frame_idx}/{seg_total_frames}", end="\r")

    print()

    # ── mIoU table ─────────────────────────────────────────────────────────────
    iou_per_class = {}
    for c in range(NUM_CLASSES):
        u   = union[c].item()
        iou = (intersection[c].item() / u) if u > 0 else float("nan")
        iou_per_class[CLASS_NAMES[c]] = iou

    valid_ious = [v for v in iou_per_class.values() if not np.isnan(v)]
    miou       = float(np.mean(valid_ious)) if valid_ious else 0.0

    print(f"\n{'Class':<30} {'IoU':>8}")
    print("─" * 40)
    for name, iou in iou_per_class.items():
        print(f"{name:<30} {f'{iou:.4f}' if not np.isnan(iou) else 'N/A':>8}")
    print("─" * 40)
    print(f"{'mIoU':<30} {miou:>8.4f}")

    result = {"per_class_iou": iou_per_class, "mIoU": miou}
    miou_path = out_dir / "miou_table.json"
    with open(miou_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {miou_path}")

    # ── Save error GIFs (one per segment) ─────────────────────────────────────
    for seg_idx, (seg_name, frames) in enumerate(gif_buffers.items()):
        gif_path = out_dir / f"error_viz_seg{seg_idx+1}.gif"
        save_gif(frames, gif_path, fps=args.gif_fps)
        print(f"  Segment: {seg_name[:40]}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained diffusion segmentation model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint",      type=str,   default=str(DEFAULT_CKPT))
    parser.add_argument("--data-root",       type=str,   default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir",         type=str,   default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--num-test-segs",   type=int,   default=10,
                        help="How many test segments to evaluate (default 10, taken from back of sorted list)")
    parser.add_argument("--gif-segs",       type=int,   default=1,
                        help="Produce a GIF for the last N evaluated segments (default 1, 0 = none)")
    parser.add_argument("--gif-fps",         type=float, default=5.0)
    parser.add_argument("--no-viz",          action="store_true",
                        help="Skip 3D error visualization entirely")
    parser.add_argument("--num-ddpm-steps",  type=int,   default=None,
                        help="Use fewer reverse steps for faster inference (e.g. 50)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)