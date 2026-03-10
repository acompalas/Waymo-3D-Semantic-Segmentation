import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from PIL import Image
import torch

from lightning_model import LinearSVMPointClassifier
from preprocessed_datasets import PreprocessedPointCloudDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Render model predictions for a point-cloud segment and export an animated GIF. "
            "By default, uses the latest lightning_logs version and its best checkpoint."
        )
    )
    p.add_argument("--data-dir", type=Path, default=Path("preprocessed/point_clouds"))
    p.add_argument("--logs-root", type=Path, default=Path("lightning_logs"))
    p.add_argument("--ckpt-path", type=Path, default=None)
    p.add_argument("--version", type=int, default=None, help="Optional explicit lightning_logs version number.")
    p.add_argument("--source-subdirs", type=str, default="validation")
    p.add_argument("--segment", type=str, default=None, help="Optional explicit segment context name.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-points", type=int, default=16384)
    p.add_argument("--output-gif", type=Path, default=Path("predictions.gif"))
    p.add_argument(
        "--output-correctness-gif",
        type=Path,
        default=None,
        help="Optional path for correctness/error GIF. Default: <output-gif stem>_correctness.gif",
    )
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--max-frame-delay-s", type=float, default=0.2)
    p.add_argument(
        "--error-scale",
        type=float,
        default=2.0,
        help="Score-difference scale for incorrect-point redness in correctness GIF.",
    )
    p.add_argument("--camera-preset", type=str, default="car_pov", choices=["car_pov", "third_person", "top", "topdown"])
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--point-size", type=float, default=2.5)
    p.add_argument("--max-frames", type=int, default=0, help="0 means all frames in the selected segment.")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def parse_csv(values: str) -> list[str]:
    return [x.strip() for x in str(values).split(",") if x.strip()]


def resolve_device(raw: str) -> torch.device:
    key = str(raw).strip().lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _version_key(path: Path) -> int:
    match = re.fullmatch(r"version_(\d+)", path.name)
    if not match:
        return -1
    return int(match.group(1))


def resolve_checkpoint(logs_root: Path, ckpt_path: Optional[Path], version: Optional[int]) -> Path:
    if ckpt_path is not None:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    if not logs_root.exists():
        raise FileNotFoundError(f"logs_root not found: {logs_root}")

    if version is not None:
        version_dir = logs_root / f"version_{int(version)}"
        if not version_dir.exists():
            raise FileNotFoundError(f"Version directory not found: {version_dir}")
    else:
        candidates = [p for p in logs_root.iterdir() if p.is_dir() and _version_key(p) >= 0]
        if not candidates:
            raise FileNotFoundError(f"No version_* directories found under {logs_root}")
        version_dir = sorted(candidates, key=_version_key)[-1]

    ckpt_dir = version_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    best_candidates = sorted(ckpt_dir.glob("best-*.ckpt"))
    if best_candidates:
        return best_candidates[-1]

    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt

    raise FileNotFoundError(f"No best-*.ckpt or last.ckpt found under {ckpt_dir}")


def choose_segment(dataset: PreprocessedPointCloudDataset, segment: Optional[str], seed: int) -> str:
    if segment is not None:
        if segment not in set(dataset.segment_names):
            raise ValueError(f"Requested segment not found in selection: {segment}")
        return segment
    if not dataset.segment_names:
        raise RuntimeError("No segments available for rendering.")
    rng = np.random.default_rng(int(seed))
    return str(rng.choice(np.array(dataset.segment_names, dtype=object)))


def label_colors(labels: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [0.95, 0.25, 0.22],
            [0.18, 0.80, 0.44],
            [0.20, 0.60, 0.98],
            [0.95, 0.77, 0.06],
            [0.61, 0.35, 0.71],
            [0.10, 0.74, 0.61],
            [0.90, 0.49, 0.13],
            [0.91, 0.30, 0.24],
            [0.18, 0.80, 0.80],
            [0.55, 0.34, 0.29],
            [0.85, 0.37, 0.01],
            [0.50, 0.50, 0.50],
        ],
        dtype=np.float32,
    )
    idx = np.mod(labels.astype(np.int64), len(palette))
    return palette[idx]


def apply_camera_preset(
    vis: o3d.visualization.VisualizerWithKeyCallback,
    frame_pcd: o3d.geometry.PointCloud,
    preset: str | None,
) -> None:
    if preset is None:
        return

    preset_key = str(preset).lower()
    if preset_key in {"top", "topdown"}:
        extrinsic = np.array(
            [
                [-0.3856370173389184, 0.9225134468464795, 0.015906955880067963, 6.141559756838577],
                [0.47505989338065996, 0.21330916748662962, -0.8537079692537238, -6.064283949433857],
                [-0.790950180832585, -0.32166463817707514, -0.5205090508217053, 109.91047324199376],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    elif preset_key in {"car_pov", "third_person"}:
        extrinsic = np.array(
            [
                [-0.026570101741086316, -0.9996106358387303, 0.008520939593593985, -0.6305623596589306],
                [-0.02978545966939474, -0.007728510895758548, -0.9995264361244364, 3.275704827536906],
                [0.9992031105264592, -0.02681131920334203, -0.02956851495806514, 13.644113467242853],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(
            f"Unknown camera preset: {preset}. Use top (or topdown) or car_pov (or third_person)."
        )

    _ = frame_pcd  # kept for parity with test_script.py signature
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    params.extrinsic = extrinsic
    try:
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    except TypeError:
        ctr.convert_from_pinhole_camera_parameters(params)


def save_gif(frames_rgb: list[np.ndarray], output_path: Path, durations_ms: list[int]) -> None:
    if not frames_rgb:
        raise RuntimeError("No frames captured for GIF export.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(frame) for frame in frames_rgb]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=[max(1, int(v)) for v in durations_ms],
        loop=0,
    )


def correctness_colors(
    preds: np.ndarray,
    labels: np.ndarray,
    valid_label: np.ndarray,
    error_mag: np.ndarray,
    error_scale: float,
) -> np.ndarray:
    colors = np.full((preds.shape[0], 3), 0.45, dtype=np.float32)  # unlabeled/ignored -> gray
    labeled = valid_label.astype(bool)
    if not np.any(labeled):
        return colors

    correct = labeled & (preds == labels)
    wrong = labeled & (~correct)

    # Correct labeled points -> green.
    colors[correct] = np.array([0.14, 0.84, 0.25], dtype=np.float32)

    # Incorrect labeled points -> red intensity by score gap (pred_score - true_score).
    if np.any(wrong):
        scale = max(float(error_scale), 1e-6)
        sev = np.clip(error_mag[wrong] / scale, 0.0, 1.0)
        # light red -> deep red as severity increases
        colors[wrong, 0] = 1.0
        colors[wrong, 1] = 0.55 * (1.0 - sev)
        colors[wrong, 2] = 0.55 * (1.0 - sev)
    return colors


@torch.no_grad()
def predict_frame(
    model: LinearSVMPointClassifier,
    sample: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = sample["points"].unsqueeze(0).to(device=device, dtype=torch.float32)
    point_features = sample["point_features"].unsqueeze(0).to(device=device, dtype=torch.float32)
    valid_geometry = sample["valid_geometry"].unsqueeze(0).to(device=device).bool()
    valid_label = sample["valid_label"].unsqueeze(0).to(device=device).bool()
    labels = sample["labels"].unsqueeze(0).to(device=device).long()

    features = model.feature_extractor(points, point_features, valid_geometry)
    features = model.feature_standardizer(features, valid_geometry, update_running=False)
    scores = model.classifier(features)
    preds = scores.argmax(dim=-1)

    keep = valid_geometry[0]
    points_np = points[0][keep].detach().cpu().numpy().astype(np.float32, copy=False)
    preds_np = preds[0][keep].detach().cpu().numpy().astype(np.int64, copy=False)
    labels_np = labels[0][keep].detach().cpu().numpy().astype(np.int64, copy=False)
    vlabel_t = valid_label[0][keep]
    vlabel_np = vlabel_t.detach().cpu().numpy().astype(bool, copy=False)

    # For labeled points: error magnitude = pred_score - true_score (>=0 usually when wrong).
    scores_keep = scores[0][keep]
    pred_scores = scores_keep.gather(1, preds[0][keep].unsqueeze(1)).squeeze(1)
    safe_labels = labels[0][keep].clamp(min=0, max=scores_keep.shape[1] - 1)
    true_scores = scores_keep.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    err_mag = (pred_scores - true_scores).clamp_min(0.0)
    err_mag = torch.where(vlabel_t, err_mag, torch.zeros_like(err_mag))
    err_mag_np = err_mag.detach().cpu().numpy().astype(np.float32, copy=False)

    return points_np, preds_np, labels_np, vlabel_np, err_mag_np


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ckpt_path = resolve_checkpoint(
        logs_root=args.logs_root,
        ckpt_path=args.ckpt_path,
        version=args.version,
    )
    print(f"Using checkpoint: {ckpt_path}")
    print(f"Using device: {device}")

    model = LinearSVMPointClassifier.load_from_checkpoint(str(ckpt_path), map_location=device)
    model = model.to(device)
    model.eval()

    dataset = PreprocessedPointCloudDataset(
        path=args.data_dir,
        source_subdirs=None if args.segment is not None else parse_csv(args.source_subdirs),
        segments=[args.segment] if args.segment is not None else None,
        num_points=int(args.num_points),
        deterministic_sampling=True,
        max_cached_segments=2,
        seed=int(args.seed),
    )
    segment = choose_segment(dataset, args.segment, seed=args.seed)
    print(f"Rendering segment: {segment}")

    frame_indices = [i for i, (seg, _) in enumerate(dataset.records) if seg == segment]
    if not frame_indices:
        raise RuntimeError(f"No frames found for segment '{segment}'")
    if args.max_frames > 0:
        frame_indices = frame_indices[: int(args.max_frames)]

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"Predictions: {segment}",
        width=int(args.width),
        height=int(args.height),
    )
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.12, 0.14, 0.16], dtype=np.float64)
    render_opt.point_size = float(args.point_size)
    render_opt.point_color_option = o3d.visualization.PointColorOption.Color

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    frames_rgb: list[np.ndarray] = []
    frames_correctness_rgb: list[np.ndarray] = []
    durations_ms: list[int] = []
    did_set_camera = False
    last_ts: Optional[int] = None
    labeled_points_total = 0
    wrong_points_total = 0

    if args.output_correctness_gif is None:
        output_correctness_gif = args.output_gif.with_name(f"{args.output_gif.stem}_correctness{args.output_gif.suffix}")
    else:
        output_correctness_gif = args.output_correctness_gif

    try:
        for i, dataset_idx in enumerate(frame_indices, start=1):
            sample = dataset[dataset_idx]
            points_np, preds_np, labels_np, valid_label_np, err_mag_np = predict_frame(
                model=model,
                sample=sample,
                device=device,
            )
            if points_np.shape[0] == 0:
                continue

            pcd.points = o3d.utility.Vector3dVector(points_np.astype(np.float64, copy=False))
            pcd.colors = o3d.utility.Vector3dVector(label_colors(preds_np).astype(np.float64, copy=False))
            vis.update_geometry(pcd)
            if not did_set_camera:
                vis.reset_view_point(True)
                apply_camera_preset(vis, pcd, args.camera_preset)
                did_set_camera = True

            if not vis.poll_events():
                break
            vis.update_renderer()

            ts = int(sample["frame_timestamp_micros"])
            if last_ts is None:
                dt = 1.0 / max(float(args.fps), 1e-6)
            else:
                dt = (ts - last_ts) / 1e6
                dt = max(0.0, min(float(dt), float(args.max_frame_delay_s)))
                if dt <= 0:
                    dt = 1.0 / max(float(args.fps), 1e-6)
            last_ts = ts

            rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True), dtype=np.float32)
            rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            if i == 1 and np.mean(rgb8) < 1.0:
                vis.poll_events()
                vis.update_renderer()
                rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True), dtype=np.float32)
                rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
            frames_rgb.append(rgb8)

            corr_colors = correctness_colors(
                preds=preds_np,
                labels=labels_np,
                valid_label=valid_label_np,
                error_mag=err_mag_np,
                error_scale=float(args.error_scale),
            )
            pcd.colors = o3d.utility.Vector3dVector(corr_colors.astype(np.float64, copy=False))
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            rgb_corr = np.asarray(vis.capture_screen_float_buffer(do_render=True), dtype=np.float32)
            rgb_corr8 = np.clip(rgb_corr * 255.0, 0, 255).astype(np.uint8)
            frames_correctness_rgb.append(rgb_corr8)

            labeled_points_total += int(valid_label_np.sum())
            wrong_points_total += int((valid_label_np & (preds_np != labels_np)).sum())
            durations_ms.append(max(1, int(round(dt * 1000.0))))
            print(f"Rendered frame {i}/{len(frame_indices)}", end="\r")
    finally:
        vis.destroy_window()

    if not frames_rgb:
        raise RuntimeError("No frames were rendered. Nothing to save.")

    save_gif(frames_rgb, args.output_gif, durations_ms)
    save_gif(frames_correctness_rgb, output_correctness_gif, durations_ms)
    print()
    print(f"Saved GIF: {args.output_gif}")
    print(f"Saved correctness GIF: {output_correctness_gif}")
    print(f"Frames: {len(frames_rgb)}")
    print(f"Segment: {segment}")
    if labeled_points_total > 0:
        wrong_rate = 100.0 * float(wrong_points_total) / float(labeled_points_total)
        print(
            f"Labeled points (rendered): {labeled_points_total} | "
            f"Wrong: {wrong_points_total} ({wrong_rate:.2f}%)"
        )
    else:
        print("No labeled points in rendered frames; correctness GIF will appear mostly gray.")


if __name__ == "__main__":
    main()
