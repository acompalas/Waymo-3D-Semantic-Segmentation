"""
visualize.py
============
Visualize Waymo LiDAR segments as animated colored point clouds.

Usage examples:

    # Visualize 1 random segment (default)
    python src/visualize.py

    # Visualize 3 random segments in sequence
    python src/visualize.py --num-segments 3

    # Visualize all 50 segments
    python src/visualize.py --num-segments 50

    # Pick specific segments by index (0-based) from the sorted segment list
    python src/visualize.py --segment-indices 0 5 12

    # Change camera, speed, export GIF
    python src/visualize.py --num-segments 2 --camera top --speed 2.0 --export-gif outputs/viz.gif

    # No loop (play once and move to next segment)
    python src/visualize.py --num-segments 3 --no-loop

Full options:
    --num-segments      N       How many segments to visualize (default: 1)
    --segment-indices   i j k   Specific segment indices instead of random
    --seed              N       Random seed for segment selection (default: 42)
    --camera            PRESET  Camera preset: car_pov or top (default: car_pov)
    --speed             F       Playback speed multiplier (default: 1.0)
    --fps               F       Playback FPS when not using realtime (default: 10.0)
    --no-realtime               Use fixed FPS instead of real timestamps
    --no-loop                   Play each segment once then move to next
    --use-return2               Also show second LiDAR return
    --export-gif        PATH    Export animation to GIF (only first segment)
    --gif-fps           F       GIF frame rate (default: same as playback)
    --width             N       Render window width (default: 1280)
    --height            N       Render window height (default: 720)
    --data-root         PATH    Path to training/ directory
                                (default: Waymo Data/training)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import polars as pl
from PIL import Image


# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_DATA_ROOT = Path("Waymo Data/training")


# ── Calibration helpers ────────────────────────────────────────────────────────
def reshape_range_image(values_list, shape_arr):
    shape = tuple(int(x) for x in shape_arr)
    return np.asarray(values_list, dtype=np.float32).reshape(shape)


def beam_inclinations(calib_row, H):
    vals = calib_row["[LiDARCalibrationComponent].beam_inclination.values"]
    if vals is not None and len(vals) == H:
        return np.asarray(vals, dtype=np.float32)[::-1]
    mn = float(calib_row["[LiDARCalibrationComponent].beam_inclination.min"])
    mx = float(calib_row["[LiDARCalibrationComponent].beam_inclination.max"])
    return np.linspace(mn, mx, H, dtype=np.float32)[::-1]


def extrinsic_matrix(calib_row):
    T = np.asarray(
        calib_row["[LiDARCalibrationComponent].extrinsic.transform"], dtype=np.float64
    )
    return T.reshape(4, 4)


# ── Range image → point cloud ──────────────────────────────────────────────────
def range_image_to_points_vehicle(range_image, inc, T, range_channel=0, feature_channel_for_color=None):
    H, W, C = range_image.shape
    r = range_image[:, :, range_channel]
    valid = r > 0

    azimuth_correction = np.arctan2(T[1, 0], T[0, 0]).astype(np.float32)
    az = np.linspace(np.pi, -np.pi, W, endpoint=False, dtype=np.float32) - azimuth_correction

    inc2d = inc[:, None]
    az2d  = az[None, :]

    xs = (r * np.cos(inc2d) * np.cos(az2d))[valid]
    ys = (r * np.cos(inc2d) * np.sin(az2d))[valid]
    zs = (r * np.sin(inc2d))[valid]

    pts   = np.stack([xs, ys, zs], axis=1)
    ones  = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts.astype(np.float64), ones], axis=1)
    pts_v = (pts_h @ T.T)[:, :3].astype(np.float32)

    scalars = None
    if feature_channel_for_color is not None:
        scalars = range_image[:, :, feature_channel_for_color][valid].astype(np.float32)

    return pts_v, scalars, valid


# ── Segmentation labels → colors ───────────────────────────────────────────────
# NOTE: channel 0 = instance ID, channel 1 = semantic label
SEMANTIC_CHANNEL = 1

CLASS_COLORS = np.array([
    [0.50, 0.50, 0.50],  #  0 TYPE_UNDEFINED       gray
    [0.95, 0.25, 0.22],  #  1 TYPE_CAR              red
    [0.80, 0.20, 0.00],  #  2 TYPE_TRUCK            dark orange
    [1.00, 0.40, 0.00],  #  3 TYPE_BUS              orange
    [1.00, 0.60, 0.20],  #  4 TYPE_OTHER_VEHICLE    light orange
    [0.60, 0.00, 0.80],  #  5 TYPE_MOTORCYCLIST     purple
    [0.80, 0.20, 0.80],  #  6 TYPE_BICYCLIST        violet
    [1.00, 0.80, 0.00],  #  7 TYPE_PEDESTRIAN       yellow
    [0.00, 0.80, 0.80],  #  8 TYPE_SIGN             cyan
    [0.00, 1.00, 1.00],  #  9 TYPE_TRAFFIC_LIGHT    bright cyan
    [0.40, 0.40, 0.80],  # 10 TYPE_POLE             slate blue
    [1.00, 0.60, 0.60],  # 11 TYPE_CONSTRUCTION_CONE pink
    [0.60, 0.00, 0.40],  # 12 TYPE_BICYCLE          dark magenta
    [0.40, 0.00, 0.60],  # 13 TYPE_MOTORCYCLE       dark purple
    [0.60, 0.40, 0.20],  # 14 TYPE_BUILDING         brown
    [0.18, 0.80, 0.44],  # 15 TYPE_VEGETATION       green
    [0.30, 0.50, 0.10],  # 16 TYPE_TREE_TRUNK       dark green
    [0.70, 0.60, 0.40],  # 17 TYPE_CURB             tan
    [0.40, 0.40, 0.40],  # 18 TYPE_ROAD             dark gray
    [0.80, 0.80, 0.00],  # 19 TYPE_LANE_MARKER      olive
    [0.60, 0.60, 0.60],  # 20 TYPE_OTHER_GROUND     medium gray
    [0.20, 0.80, 0.60],  # 21 TYPE_WALKABLE         teal
    [0.20, 0.60, 0.80],  # 22 TYPE_SIDEWALK         steel blue
], dtype=np.float32)


def label_colors(labels):
    """Map integer semantic labels to RGB colors."""
    idx = np.clip(labels.astype(np.int64), 0, len(CLASS_COLORS) - 1)
    return CLASS_COLORS[idx]


# ── Point cloud builder ────────────────────────────────────────────────────────
def build_combined_point_cloud(frame_lidar, frame_seg, frame_calib, use_return2=False):
    calib_by_laser = {int(r["key.laser_name"]): r for r in frame_calib.iter_rows(named=True)}
    seg_by_laser   = {int(r["key.laser_name"]): r for r in frame_seg.iter_rows(named=True)}
    has_any_seg    = len(seg_by_laser) > 0

    all_pts, all_colors = [], []

    for row in frame_lidar.iter_rows(named=True):
        laser = int(row["key.laser_name"])
        calib = calib_by_laser[laser]
        T     = extrinsic_matrix(calib)

        ri1 = reshape_range_image(
            row["[LiDARComponent].range_image_return1.values"],
            row["[LiDARComponent].range_image_return1.shape"],
        )
        H, W, C = ri1.shape
        inc = beam_inclinations(calib, H)

        pts_v, intensity, valid_mask = range_image_to_points_vehicle(
            ri1, inc, T,
            range_channel=0,
            feature_channel_for_color=1 if C > 1 else None,
        )

        if laser in seg_by_laser:
            seg_row = seg_by_laser[laser]
            seg1    = reshape_range_image(
                seg_row["[LiDARSegmentationLabelComponent].range_image_return1.values"],
                seg_row["[LiDARSegmentationLabelComponent].range_image_return1.shape"],
            ).astype(np.int32)
            # channel 1 = semantic label (channel 0 = instance id)
            labels = seg1[:, :, SEMANTIC_CHANNEL][valid_mask]
            colors = label_colors(labels)
        elif intensity is not None:
            s = intensity
            s = (s - s.min()) / (s.max() - s.min() + 1e-6)
            s = 0.25 + 0.35 * s if has_any_seg else s
            colors = np.stack([s, s, s], axis=1)
        else:
            colors = np.full((pts_v.shape[0], 3), 0.45, dtype=np.float32)

        all_pts.append(pts_v)
        all_colors.append(colors.astype(np.float32))

        if use_return2:
            ri2_vals = row["[LiDARComponent].range_image_return2.values"]
            if ri2_vals is not None and len(ri2_vals) > 0:
                ri2 = reshape_range_image(ri2_vals, row["[LiDARComponent].range_image_return2.shape"])
                pts_v2, intensity2, _ = range_image_to_points_vehicle(
                    ri2, inc, T, range_channel=0,
                    feature_channel_for_color=1 if ri2.shape[2] > 1 else None,
                )
                all_pts.append(pts_v2)
                if intensity2 is not None:
                    s2 = (intensity2 - intensity2.min()) / (intensity2.max() - intensity2.min() + 1e-6)
                    all_colors.append(np.stack([s2, s2, s2], axis=1).astype(np.float32))
                else:
                    all_colors.append(np.full((pts_v2.shape[0], 3), 0.45, dtype=np.float32))

    pts  = np.concatenate(all_pts,   axis=0) if all_pts   else np.zeros((0, 3), np.float32)
    cols = np.concatenate(all_colors, axis=0) if all_colors else np.zeros((0, 3), np.float32)

    pcd        = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    return pcd


# ── Camera helpers ─────────────────────────────────────────────────────────────
CAMERA_PRESETS = {
    "top": np.array([
        [-0.3856370173389184,  0.9225134468464795,  0.015906955880067963,  6.141559756838577],
        [ 0.47505989338065996, 0.21330916748662962, -0.8537079692537238,  -6.064283949433857],
        [-0.790950180832585,  -0.32166463817707514, -0.5205090508217053,  109.91047324199376],
        [ 0.0,                 0.0,                  0.0,                   1.0             ],
    ], dtype=np.float64),
    "car_pov": np.array([
        [-0.026570101741086316, -0.9996106358387303,  0.008520939593593985, -0.6305623596589306],
        [-0.02978545966939474,  -0.007728510895758548,-0.9995264361244364,   3.275704827536906 ],
        [ 0.9992031105264592,   -0.02681131920334203, -0.02956851495806514,  13.644113467242853],
        [ 0.0,                   0.0,                  0.0,                   1.0              ],
    ], dtype=np.float64),
}


def apply_camera_preset(vis, preset_name):
    extrinsic = CAMERA_PRESETS.get(preset_name)
    if extrinsic is None:
        return
    ctr    = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    params.extrinsic = extrinsic
    try:
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    except TypeError:
        ctr.convert_from_pinhole_camera_parameters(params)


def print_camera_pose(vis):
    ctr    = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    pose   = {"extrinsic": np.asarray(params.extrinsic).tolist()}
    print(f"\nCAMERA_POSE={json.dumps(pose)}")


# ── GIF export ─────────────────────────────────────────────────────────────────
def save_gif(frames_rgb, output_path, fps=10.0):
    if not frames_rgb:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = max(1, int(round(1000.0 / max(fps, 1e-6))))
    pil_frames = [Image.fromarray(f) for f in frames_rgb]
    pil_frames[0].save(
        out, save_all=True, append_images=pil_frames[1:],
        duration=duration, loop=0,
    )
    print(f"\nSaved GIF → {out} ({len(frames_rgb)} frames @ {fps:.1f} fps)")


# ── Segment loading ────────────────────────────────────────────────────────────
def get_all_segments(data_root: Path) -> list[str]:
    seg_files = sorted((data_root / "lidar_segmentation").glob("*.parquet"))
    return [f.stem for f in seg_files]


def load_segment(data_root: Path, segment: str):
    lidar_df = pl.scan_parquet(str(data_root / "lidar" / "*.parquet"))
    seg_df   = pl.scan_parquet(str(data_root / "lidar_segmentation" / "*.parquet"))
    calib_df = pl.scan_parquet(str(data_root / "lidar_calibration" / "*.parquet"))

    seg_lidar = lidar_df.filter(pl.col("key.segment_context_name") == segment).collect()
    seg_seg   = seg_df.filter(pl.col("key.segment_context_name") == segment).collect()
    seg_calib = calib_df.filter(pl.col("key.segment_context_name") == segment).collect()
    return seg_lidar, seg_seg, seg_calib


def labeled_timestamps(seg_lidar, seg_seg) -> list[int]:
    lidar_ts = seg_lidar.select("key.frame_timestamp_micros").unique()
    seg_ts   = seg_seg.select("key.frame_timestamp_micros").unique()
    ts = (
        lidar_ts
        .join(seg_ts, on="key.frame_timestamp_micros", how="inner")
        .sort("key.frame_timestamp_micros")
        ["key.frame_timestamp_micros"]
        .to_list()
    )
    return [int(t) for t in ts]


# ── Animation ──────────────────────────────────────────────────────────────────
def animate_segment(
    segment, seg_lidar, seg_seg, seg_calib,
    camera_preset="car_pov",
    fps=10.0,
    speed=1.0,
    realtime=True,
    loop=True,
    use_return2=False,
    export_gif_path=None,
    gif_fps=None,
    render_width=1280,
    render_height=720,
):
    timestamps = labeled_timestamps(seg_lidar, seg_seg)
    if not timestamps:
        print(f"  No labeled timestamps for segment {segment}, skipping.")
        return

    # Pre-build all point clouds
    lidar_by_ts = {
        int(df["key.frame_timestamp_micros"][0]): df
        for df in seg_lidar.partition_by("key.frame_timestamp_micros", maintain_order=True)
    }
    seg_by_ts = {
        int(df["key.frame_timestamp_micros"][0]): df
        for df in seg_seg.partition_by("key.frame_timestamp_micros", maintain_order=True)
    }

    print(f"  Building {len(timestamps)} point clouds...", end="", flush=True)
    frame_sequence = []
    for ts in timestamps:
        fl = lidar_by_ts.get(ts)
        fs = seg_by_ts.get(ts, seg_seg.head(0))
        if fl is None:
            continue
        pcd = build_combined_point_cloud(fl, fs, seg_calib, use_return2=use_return2)
        if len(pcd.points) > 0:
            frame_sequence.append((ts, pcd))
    print(f" done ({len(frame_sequence)} frames)")

    if not frame_sequence:
        print(f"  No renderable frames, skipping.")
        return

    # Compute playback fps
    if realtime and len(frame_sequence) > 1:
        ts_arr = np.array([ts for ts, _ in frame_sequence], dtype=np.float64)
        dts    = np.diff(ts_arr) / 1e6
        dts    = dts[dts > 0]
        playback_fps = (speed / np.median(dts)) if len(dts) else fps
    else:
        playback_fps = fps
    frame_dt = 1.0 / max(playback_fps, 1e-6)
    gif_fps  = float(gif_fps) if gif_fps else playback_fps

    # Open3D window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"Waymo — {segment[:40]}",
        width=render_width,
        height=render_height,
    )
    opt = vis.get_render_option()
    opt.background_color = np.array([0.12, 0.14, 0.16])
    opt.point_size       = 2.5

    pcd_render = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_render)
    vis.register_key_callback(ord("P"), lambda v: print_camera_pose(v) or False)

    print(f"  Playing segment. Press P to print camera pose. Close window to continue.")

    capturing   = export_gif_path is not None
    gif_frames  = []
    did_reset   = False
    keep_going  = True
    last_ts     = None

    try:
        while keep_going:
            for i, (ts, frame_pcd) in enumerate(frame_sequence):
                t0 = time.perf_counter()

                pcd_render.points = frame_pcd.points
                pcd_render.colors = frame_pcd.colors
                vis.update_geometry(pcd_render)

                if not did_reset:
                    vis.reset_view_point(True)
                    apply_camera_preset(vis, camera_preset)
                    did_reset = True

                if not vis.poll_events():
                    keep_going = False
                    break
                vis.update_renderer()

                print(f"  Frame {i+1}/{len(frame_sequence)}", end="\r")

                if realtime and last_ts is not None:
                    target_dt = min((ts - last_ts) / 1e6 / max(speed, 1e-6), 0.2)
                else:
                    target_dt = frame_dt

                if capturing:
                    rgb  = np.asarray(vis.capture_screen_float_buffer(do_render=True), dtype=np.float32)
                    rgb8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                    gif_frames.append(rgb8)

                elapsed = time.perf_counter() - t0
                sleep   = target_dt - elapsed
                if sleep > 0:
                    time.sleep(sleep)
                last_ts = ts

            if capturing and gif_frames:
                save_gif(gif_frames, export_gif_path, fps=gif_fps)
                capturing  = False
                gif_frames = []

            if not loop:
                keep_going = False

    finally:
        if capturing and gif_frames:
            save_gif(gif_frames, export_gif_path, fps=gif_fps)
        vis.destroy_window()


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Waymo LiDAR segments as colored point clouds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--num-segments",     type=int,   default=1,
                        help="Number of random segments to visualize (default: 1)")
    parser.add_argument("--segment-indices",  type=int,   nargs="+",
                        help="Visualize specific segments by 0-based index instead of random")
    parser.add_argument("--seed",             type=int,   default=42,
                        help="Random seed for segment selection (default: 42)")
    parser.add_argument("--camera",           type=str,   default="car_pov",
                        choices=["car_pov", "top"],
                        help="Camera preset (default: car_pov)")
    parser.add_argument("--speed",            type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--fps",              type=float, default=10.0,
                        help="FPS when not using realtime mode (default: 10.0)")
    parser.add_argument("--no-realtime",      action="store_true",
                        help="Use fixed FPS instead of real sensor timestamps")
    parser.add_argument("--no-loop",         action="store_true",
                        help="Play each segment once then move to next")
    parser.add_argument("--use-return2",      action="store_true",
                        help="Also render the second LiDAR return")
    parser.add_argument("--export-gif",       type=str,   default=None,
                        help="Export first segment to GIF at this path")
    parser.add_argument("--gif-fps",          type=float, default=None,
                        help="GIF frame rate (default: same as playback)")
    parser.add_argument("--width",            type=int,   default=1280,
                        help="Render window width (default: 1280)")
    parser.add_argument("--height",           type=int,   default=720,
                        help="Render window height (default: 720)")
    parser.add_argument("--data-root",        type=str,   default=str(DEFAULT_DATA_ROOT),
                        help=f"Path to training/ directory (default: {DEFAULT_DATA_ROOT})")
    return parser.parse_args()


def main():
    args      = parse_args()
    data_root = Path(args.data_root)

    if not data_root.exists():
        print(f"ERROR: data root not found: {data_root}")
        sys.exit(1)

    all_segments = get_all_segments(data_root)
    total        = len(all_segments)
    print(f"Found {total} segments with segmentation labels.\n")

    # Select which segments to visualize
    if args.segment_indices is not None:
        bad = [i for i in args.segment_indices if i < 0 or i >= total]
        if bad:
            print(f"ERROR: indices out of range [0, {total-1}]: {bad}")
            sys.exit(1)
        chosen = [all_segments[i] for i in args.segment_indices]
    else:
        n   = min(args.num_segments, total)
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(total, size=n, replace=False)
        idx.sort()
        chosen = [all_segments[i] for i in idx]

    print(f"Visualizing {len(chosen)} segment(s):\n")
    for i, seg in enumerate(chosen):
        print(f"  [{i+1}/{len(chosen)}]  {seg}")
    print()

    for i, segment in enumerate(chosen):
        print(f"─── Segment {i+1}/{len(chosen)}: {segment[:60]} ───")

        seg_lidar, seg_seg, seg_calib = load_segment(data_root, segment)

        export = args.export_gif if i == 0 else None  # only export first segment

        animate_segment(
            segment        = segment,
            seg_lidar      = seg_lidar,
            seg_seg        = seg_seg,
            seg_calib      = seg_calib,
            camera_preset  = args.camera,
            fps            = args.fps,
            speed          = args.speed,
            realtime       = not args.no_realtime,
            loop           = not args.no_loop,
            use_return2    = args.use_return2,
            export_gif_path= export,
            gif_fps        = args.gif_fps,
            render_width   = args.width,
            render_height  = args.height,
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()