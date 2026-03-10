"""
evaluate.py
===========
Load a trained model checkpoint and evaluate on the test set.
Produces:
  1. Per-class mIoU table (printed + saved as JSON)
  2. 3D class-colored GIF — points colored by PREDICTED class,
     correct predictions bright, wrong predictions darkened to 25%
  3. Confusion matrix PNG — 23x23 row-normalized heatmap

Usage:
    cd "D:/ECE 271B SemSeg Project"

    python src/evaluate.py
    python src/evaluate.py --num-ddpm-steps 50   # faster inference
    python src/evaluate.py --no-viz               # mIoU only

Full options:
    --checkpoint      PATH   Model checkpoint (default: outputs/best_model.ckpt)
    --data-root       PATH   Path to training/ directory
    --out-dir         PATH   Where to save outputs (default: outputs/)
    --num-test-segs   N      Test segments to evaluate (default: 10)
    --gif-segs        N      Produce GIF for last N evaluated segments (default: 1)
    --gif-fps         F      GIF frame rate (default: 5.0)
    --no-viz                 Skip GIF visualization entirely
    --num-ddpm-steps  N      Fewer reverse steps for faster inference (e.g. 50)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import open3d as o3d
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from dataset   import get_test_dataset, NUM_CLASSES, CLASS_NAMES
from model     import LiDARDiffusionUNet
from diffusion import DDPM

DEFAULT_DATA_ROOT = Path("Waymo Data/training")
DEFAULT_OUT_DIR   = Path("outputs")
DEFAULT_CKPT      = Path("outputs/best_model.ckpt")

LASER_COL     = "key.laser_name"
SEGMENT_COL   = "key.segment_context_name"
PRIMARY_LASER = 1

# Same class colors as visualize.py
CLASS_COLORS = np.array([
    [0.50, 0.50, 0.50],  #  0 TYPE_UNDEFINED
    [0.95, 0.25, 0.22],  #  1 TYPE_CAR
    [0.80, 0.20, 0.00],  #  2 TYPE_TRUCK
    [1.00, 0.40, 0.00],  #  3 TYPE_BUS
    [1.00, 0.60, 0.20],  #  4 TYPE_OTHER_VEHICLE
    [0.60, 0.00, 0.80],  #  5 TYPE_MOTORCYCLIST
    [0.80, 0.20, 0.80],  #  6 TYPE_BICYCLIST
    [1.00, 0.80, 0.00],  #  7 TYPE_PEDESTRIAN
    [0.00, 0.80, 0.80],  #  8 TYPE_SIGN
    [0.00, 1.00, 1.00],  #  9 TYPE_TRAFFIC_LIGHT
    [0.40, 0.40, 0.80],  # 10 TYPE_POLE
    [1.00, 0.60, 0.60],  # 11 TYPE_CONSTRUCTION_CONE
    [0.60, 0.00, 0.40],  # 12 TYPE_BICYCLE
    [0.40, 0.00, 0.60],  # 13 TYPE_MOTORCYCLE
    [0.60, 0.40, 0.20],  # 14 TYPE_BUILDING
    [0.18, 0.80, 0.44],  # 15 TYPE_VEGETATION
    [0.30, 0.50, 0.10],  # 16 TYPE_TREE_TRUNK
    [0.70, 0.60, 0.40],  # 17 TYPE_CURB
    [0.40, 0.40, 0.40],  # 18 TYPE_ROAD
    [0.80, 0.80, 0.00],  # 19 TYPE_LANE_MARKER
    [0.60, 0.60, 0.60],  # 20 TYPE_OTHER_GROUND
    [0.20, 0.80, 0.60],  # 21 TYPE_WALKABLE
    [0.20, 0.60, 0.80],  # 22 TYPE_SIDEWALK
], dtype=np.float32)


# ── Calibration helpers ────────────────────────────────────────────────────────
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
    range_image: np.ndarray,
    inclinations: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if range_image.shape[0] == 4:
        range_image = range_image.transpose(1, 2, 0)
    H, W = range_image.shape[:2]
    r     = range_image[:, :, 0]
    valid = r > 0

    az_correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0]).astype(np.float32)
    az    = np.linspace(np.pi, -np.pi, W, endpoint=False, dtype=np.float32) - az_correction
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


# ── Color functions ─────────────────────────────────────────────────────────────────────────
def true_colors(true: np.ndarray) -> np.ndarray:
    """Ground truth class colors."""
    idx = np.clip(true.astype(np.int64), 0, NUM_CLASSES - 1)
    return CLASS_COLORS[idx].copy()


def pred_colors(pred: np.ndarray) -> np.ndarray:
    """Predicted class colors."""
    idx = np.clip(pred.astype(np.int64), 0, NUM_CLASSES - 1)
    return CLASS_COLORS[idx].copy()


def error_colors(pred: np.ndarray, true: np.ndarray, prob_pred: np.ndarray, prob_true: np.ndarray) -> np.ndarray:
    """
    Color all points by margin = p(pred_class) - p(true_class).
    Correct predictions always have margin = 0 -> grey.
    Wrong predictions scaled by how confidently wrong -> vivid pink.

    margin = p(pred) - p(true):
      correct (pred == true): margin = 0          -> grey
      wrong, confident:       margin near 1.0     -> vivid pink
      wrong, uncertain:       margin near 0.0     -> barely pink
    """
    GREY = np.array([0.6, 0.6, 0.6], dtype=np.float32)
    PINK = np.array([1.0, 0.2, 0.5], dtype=np.float32)

    margin = (prob_pred - prob_true).clip(0.0, 1.0)[:, None]  # (N, 1), always 0 when correct
    colors = GREY + margin * (PINK - GREY)                      # lerp grey -> pink
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
    print(f"Saved GIF → {path} ({len(frames)} frames @ {fps:.1f} fps)")


# ── Open3D render ──────────────────────────────────────────────────────────────
CAMERA_PRESET = np.array([
    [-0.026570101741086316, -0.9996106358387303,  0.008520939593593985, -0.6305623596589306],
    [-0.02978545966939474,  -0.007728510895758548,-0.9995264361244364,   3.275704827536906 ],
    [ 0.9992031105264592,   -0.02681131920334203, -0.02956851495806514,  13.644113467242853],
    [ 0.0,                   0.0,                  0.0,                   1.0              ],
], dtype=np.float64)


def render_frame_to_image(xyz: np.ndarray, colors: np.ndarray, w=1280, h=720) -> np.ndarray:
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


# ── Confusion matrix PNG ───────────────────────────────────────────────────────
def save_confusion_matrix(conf_matrix: np.ndarray, out_path: Path):
    """Row-normalized confusion matrix. Rows = true class, Cols = predicted class."""
    row_sums = conf_matrix.sum(axis=1, keepdims=True).clip(min=1)
    norm     = conf_matrix / row_sums

    short_names = [n.replace("TYPE_", "") for n in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    ax.set_title(
        "Confusion Matrix (row-normalized)\n"
        "Diagonal = correct predictions | Off-diagonal = class confusions",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix → {out_path}")


# ── Main evaluation ────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        alt = ckpt_path.with_suffix(".ckpt")
        if alt.exists():
            ckpt_path = alt
        else:
            print(f"ERROR: checkpoint not found: {ckpt_path}")
            sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt:
        raw_sd     = ckpt["state_dict"]
        model_sd   = {k[len("model."):]: v for k, v in raw_sd.items() if k.startswith("model.")}
        train_args = ckpt.get("hyper_parameters", {})
        epoch      = ckpt.get("epoch", "?")
        print(f"Loaded Lightning checkpoint (epoch {epoch})")
    else:
        model_sd   = ckpt["model_state"]
        train_args = ckpt.get("args", {})
        epoch      = ckpt.get("epoch", "?")
        print(f"Loaded checkpoint (epoch {epoch})")

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

    T       = train_args.get("T", 1000)
    n_steps = args.num_ddpm_steps if args.num_ddpm_steps else T
    ddpm    = DDPM(T=T, device=str(device)).to(device)

    data_root = Path(args.data_root)
    test_ds   = get_test_dataset(data_root, num_segments=args.num_test_segs)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intersection = torch.zeros(NUM_CLASSES, dtype=torch.long)
    union        = torch.zeros(NUM_CLASSES, dtype=torch.long)
    conf_matrix  = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    n_frames = len(test_ds)
    print(f"\nEvaluating {args.num_test_segs} segment(s), {n_frames} frames total...")

    all_segments_in_order = []
    seen = {}
    for idx in range(n_frames):
        seg = test_ds.records[idx][0]
        if seg not in seen:
            seen[seg] = len(all_segments_in_order)
            all_segments_in_order.append(seg)

    n_gif_segs   = args.gif_segs if not args.no_viz else 0
    gif_segments  = set(all_segments_in_order[-n_gif_segs:]) if n_gif_segs > 0 else set()
    # Three buffers per segment: true, pred, error
    true_buffers:  dict[str, list] = {}
    pred_buffers:  dict[str, list] = {}
    error_buffers: dict[str, list] = {}

    current_seg      = None
    seg_frame_idx    = 0
    seg_total_frames = 0

    with torch.no_grad():
        for i in range(n_frames):
            sample  = test_ds[i]
            lidar   = sample["lidar"].unsqueeze(0).to(device)
            labels  = sample["labels"]
            valid   = sample["valid"]
            segment = sample["segment_context_name"]

            if n_steps < T:
                step_indices = np.linspace(T, 1, n_steps, dtype=int)
            else:
                step_indices = np.arange(T, 0, -1, dtype=int)

            B, _, H, W = lidar.shape
            x_t = torch.randn(1, NUM_CLASSES, H, W, device=device)
            for t_int in step_indices:
                t_tensor = torch.full((1,), int(t_int), device=device, dtype=torch.long)
                x_t = ddpm.p_sample(model, x_t, t_tensor, lidar)

            probs    = torch.softmax(x_t.squeeze(0), dim=0).cpu()
            pred_cls = probs.argmax(dim=0)

            v      = valid.bool()
            pred_v = pred_cls[v].numpy()
            true_v = labels[v].numpy()

            # Exclude undefined pixels (label 0) from metrics
            defined = (true_v != 0)
            pred_d  = pred_v[defined]
            true_d  = true_v[defined]

            for c in range(NUM_CLASSES):
                intersection[c] += int(((pred_d == c) & (true_d == c)).sum())
                union[c]        += int(((pred_d == c) | (true_d == c)).sum())

            np.add.at(conf_matrix, (true_d, pred_d), 1)

            if segment != current_seg:
                if current_seg is not None:
                    print()
                current_seg      = segment
                seg_frame_idx    = 0
                seg_total_frames = sum(1 for r in test_ds.records if r[0] == segment)
                print(f"\nSegment: {segment[:50]}")
            seg_frame_idx += 1

            if segment in gif_segments:
                try:
                    calib  = _load_calibration(data_root, segment)
                    ri_np  = sample["lidar"].numpy()
                    inc    = _beam_inclinations(calib, ri_np.shape[1])
                    ext    = _extrinsic(calib)
                    xyz, _ = range_image_to_xyz(ri_np, inc, ext)
                    # Extract p(pred_class) and p(true_class) per valid pixel
                    # probs: (C, H, W) -> flatten to (C, N_valid)
                    probs_flat  = probs.reshape(NUM_CLASSES, -1)[:, v.reshape(-1)]
                    n_valid     = len(true_v)
                    arange      = np.arange(n_valid)
                    prob_pred_v = probs_flat[pred_v, arange].numpy()
                    prob_true_v = probs_flat[true_v, arange].numpy()

                    true_buffers.setdefault(segment, []).append(
                        render_frame_to_image(xyz, true_colors(true_v)))
                    pred_buffers.setdefault(segment, []).append(
                        render_frame_to_image(xyz, pred_colors(pred_v)))
                    error_buffers.setdefault(segment, []).append(
                        render_frame_to_image(xyz, error_colors(pred_v, true_v, prob_pred_v, prob_true_v)))

                    print(f"  Rendering frame {seg_frame_idx}/{seg_total_frames}", end="\r")
                except Exception as e:
                    print(f"  Frame {seg_frame_idx} viz skipped: {e}")
            else:
                print(f"  Frame {seg_frame_idx}/{seg_total_frames}", end="\r")

    print()

    # ── mIoU ──────────────────────────────────────────────────────────────────
    iou_per_class = {}
    for c in range(NUM_CLASSES):
        u   = union[c].item()
        iou = (intersection[c].item() / u) if u > 0 else float("nan")
        iou_per_class[CLASS_NAMES[c]] = iou

    # Exclude TYPE_UNDEFINED (class 0) from mIoU — it is not a real semantic class
    valid_ious = [iou for name, iou in iou_per_class.items()
                  if not np.isnan(iou) and name != "TYPE_UNDEFINED"]
    miou       = float(np.mean(valid_ious)) if valid_ious else 0.0

    print(f"\n{'Class':<30} {'IoU':>8}")
    print("─" * 40)
    for name, iou in iou_per_class.items():
        print(f"{name:<30} {f'{iou:.4f}' if not np.isnan(iou) else 'N/A':>8}")
    print("─" * 40)
    print(f"{'mIoU':<30} {miou:>8.4f}")

    miou_path = out_dir / "miou_table.json"
    with open(miou_path, "w") as f:
        json.dump({"per_class_iou": iou_per_class, "mIoU": miou}, f, indent=2)
    print(f"\nSaved mIoU → {miou_path}")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    save_confusion_matrix(conf_matrix, out_dir / "confusion_matrix.png")
    with open(out_dir / "confusion_matrix.json", "w") as f:
        json.dump(conf_matrix.tolist(), f)

    # ── GIFs ──────────────────────────────────────────────────────────────────
    all_segs = list(true_buffers.keys())
    for seg_idx, seg_name in enumerate(all_segs):
        i = seg_idx + 1
        save_gif(true_buffers[seg_name],  out_dir / f"true_viz_seg{i}.gif",  fps=args.gif_fps)
        save_gif(pred_buffers[seg_name],  out_dir / f"pred_viz_seg{i}.gif",  fps=args.gif_fps)
        save_gif(error_buffers[seg_name], out_dir / f"error_viz_seg{i}.gif", fps=args.gif_fps)
        print(f"  Segment {i}: {seg_name[:40]}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained diffusion segmentation model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint",     type=str,   default=str(DEFAULT_CKPT))
    parser.add_argument("--data-root",      type=str,   default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out-dir",        type=str,   default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--num-test-segs",  type=int,   default=10)
    parser.add_argument("--gif-segs",       type=int,   default=1)
    parser.add_argument("--gif-fps",        type=float, default=5.0)
    parser.add_argument("--no-viz",         action="store_true")
    parser.add_argument("--num-ddpm-steps", type=int,   default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)