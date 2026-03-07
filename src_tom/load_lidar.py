# %%
import polars as pl
import numpy as np
import time
import json
from pathlib import Path
from PIL import Image

try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None


def _require_open3d():
    if o3d is None:
        raise RuntimeError(
            "open3d is not installed. Install it for load_lidar.py visualization, "
            "or use train.py/evaluate.py which do not require open3d."
        )

# %%
lidar_df = pl.scan_parquet("data/training/lidar/*.parquet")
lidar_segmentation_df = pl.scan_parquet("data/training/lidar_segmentation/*.parquet")
lidar_calibration_df = pl.scan_parquet("data/training/lidar_calibration/*.parquet")


def reshape_range_image(values_list, shape_arr):
    shape = tuple(int(x) for x in shape_arr)  # (H,W,C)
    return np.asarray(values_list, dtype=np.float32).reshape(shape)

def beam_inclinations(calib_row, H):
    vals = calib_row["[LiDARCalibrationComponent].beam_inclination.values"]
    if vals is not None and len(vals) == H:
        return np.asarray(vals, dtype=np.float32)[::-1]
    mn = float(calib_row["[LiDARCalibrationComponent].beam_inclination.min"])
    mx = float(calib_row["[LiDARCalibrationComponent].beam_inclination.max"])
    return np.linspace(mn, mx, H, dtype=np.float32)[::-1]

def extrinsic_matrix(calib_row):
    T = np.asarray(calib_row["[LiDARCalibrationComponent].extrinsic.transform"], dtype=np.float64)
    return T.reshape(4, 4)

def range_image_to_points_vehicle(
    range_image,
    inc,
    T,
    range_channel=0,
    feature_channel_for_color=None,
):
    H, W, C = range_image.shape
    r = range_image[:, :, range_channel]
    valid = r > 0

    azimuth_correction = np.arctan2(T[1, 0], T[0, 0]).astype(np.float32)
    az = np.linspace(np.pi, -np.pi, W, endpoint=False, dtype=np.float32) - azimuth_correction
    inc2d = inc[:, None]  # (H,1)
    az2d = az[None, :]    # (1,W)

    cos_inc = np.cos(inc2d)
    sin_inc = np.sin(inc2d)
    cos_az = np.cos(az2d)
    sin_az = np.sin(az2d)

    xs = (r * cos_inc * cos_az)[valid]
    ys = (r * cos_inc * sin_az)[valid]
    zs = (r * sin_inc)[valid]

    pts = np.stack([xs, ys, zs], axis=1)  # Nx3

    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts.astype(np.float64), ones], axis=1)  # Nx4
    pts_v = (pts_h @ T.T)[:, :3].astype(np.float32)

    scalars = None
    if feature_channel_for_color is not None:
        scalars = range_image[:, :, feature_channel_for_color][valid].astype(np.float32)

    return pts_v, scalars, valid

def labels_from_segmentation(seg_image, valid_mask):
    labels = seg_image[:, :, 0][valid_mask].astype(np.int32)
    return labels

def label_colors(labels):
    palette = np.array(
        [
            [0.95, 0.25, 0.22],  # red
            [0.18, 0.80, 0.44],  # green
            [0.20, 0.60, 0.98],  # blue
            [0.95, 0.77, 0.06],  # yellow
            [0.61, 0.35, 0.71],  # purple
            [0.10, 0.74, 0.61],  # teal
            [0.90, 0.49, 0.13],  # orange
            [0.91, 0.30, 0.24],  # crimson
            [0.18, 0.80, 0.80],  # cyan
            [0.55, 0.34, 0.29],  # brown
        ],
        dtype=np.float32,
    )
    idx = np.mod(labels.astype(np.int64), len(palette))
    return palette[idx]

def _normalize(vec):
    n = np.linalg.norm(vec)
    if n < 1e-8:
        return vec
    return vec / n

def _camera_from_eye(eye, lookat, world_up):
    front = _normalize(lookat - eye)
    right = np.cross(front, world_up)
    if np.linalg.norm(right) < 1e-8:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(front, world_up)
    right = _normalize(right)
    up = _normalize(np.cross(right, front))
    return front, up

def apply_camera_preset(vis, frame_pcd, preset):
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

    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    params.extrinsic = extrinsic
    try:
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    except TypeError:
        ctr.convert_from_pinhole_camera_parameters(params)

def save_gif_from_frames(frames_rgb, output_path, fps=None, durations_ms=None):
    if len(frames_rgb) == 0:
        return

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if durations_ms is not None and len(durations_ms) == len(frames_rgb):
        duration = [max(1, int(round(v))) for v in durations_ms]
        fps_text = "variable"
    else:
        fps = 10.0 if fps is None else float(fps)
        duration = max(1, int(round(1000.0 / max(fps, 1e-6))))
        fps_text = f"{fps:.2f}"

    pil_frames = [Image.fromarray(frame) for frame in frames_rgb]
    pil_frames[0].save(
        out,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"\nSaved GIF to {out} ({len(frames_rgb)} frames @ {fps_text} fps)")

def _to_float_list(arr):
    return [float(x) for x in np.asarray(arr, dtype=np.float64).reshape(-1)]

def camera_pose_from_view_control(ctr):
    params = ctr.convert_to_pinhole_camera_parameters()
    extrinsic = np.asarray(params.extrinsic, dtype=np.float64)
    intrinsic = np.asarray(params.intrinsic.intrinsic_matrix, dtype=np.float64)

    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    camera_center = -R.T @ t

    pose = {
        "camera_center": _to_float_list(camera_center),
        "extrinsic": extrinsic.tolist(),
        "intrinsic": intrinsic.tolist(),
    }
    if hasattr(ctr, "get_lookat"):
        pose["lookat"] = _to_float_list(ctr.get_lookat())
    if hasattr(ctr, "get_front"):
        pose["front"] = _to_float_list(ctr.get_front())
    if hasattr(ctr, "get_up"):
        pose["up"] = _to_float_list(ctr.get_up())
    if hasattr(ctr, "get_zoom"):
        pose["zoom"] = float(ctr.get_zoom())
    return pose

def print_camera_pose(ctr, prefix="CAMERA_POSE"):
    pose = camera_pose_from_view_control(ctr)
    print(f"\n{prefix}={json.dumps(pose)}")

def load_frame_polars(lidar_df, lidar_seg_df, lidar_calib_df, segment, ts_micros):
    frame_lidar = (
        lidar_df
        .filter((pl.col("key.segment_context_name") == segment) &
                (pl.col("key.frame_timestamp_micros") == ts_micros))
        .collect()
    )
    frame_seg = (
        lidar_seg_df
        .filter((pl.col("key.segment_context_name") == segment) &
                (pl.col("key.frame_timestamp_micros") == ts_micros))
        .collect()
    )
    frame_calib = (
        lidar_calib_df
        .filter(pl.col("key.segment_context_name") == segment)
        .collect()
    )
    return frame_lidar, frame_seg, frame_calib

def load_segment_polars(lidar_df, lidar_seg_df, lidar_calib_df, segment):
    segment_lidar = (
        lidar_df
        .filter(pl.col("key.segment_context_name") == segment)
        .collect()
    )
    segment_seg = (
        lidar_seg_df
        .filter(pl.col("key.segment_context_name") == segment)
        .collect()
    )
    segment_calib = (
        lidar_calib_df
        .filter(pl.col("key.segment_context_name") == segment)
        .collect()
    )
    return segment_lidar, segment_seg, segment_calib

def build_combined_point_cloud(frame_lidar, frame_seg, frame_calib, use_return2=False):
    _require_open3d()
    # index calibration rows by laser_name
    calib_by_laser = {}
    for row in frame_calib.iter_rows(named=True):
        calib_by_laser[int(row["key.laser_name"])] = row

    # index segmentation rows by laser_name
    seg_by_laser = {}
    for row in frame_seg.iter_rows(named=True):
        seg_by_laser[int(row["key.laser_name"])] = row

    has_any_seg = len(seg_by_laser) > 0
    all_pts = []
    all_colors = []

    for row in frame_lidar.iter_rows(named=True):
        laser = int(row["key.laser_name"])
        calib = calib_by_laser[laser]
        T = extrinsic_matrix(calib)

        # Return 1
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
            seg1 = reshape_range_image(
                seg_row["[LiDARSegmentationLabelComponent].range_image_return1.values"],
                seg_row["[LiDARSegmentationLabelComponent].range_image_return1.shape"],
            )
            labels = labels_from_segmentation(seg1, valid_mask)
            colors = label_colors(labels)
        elif intensity is not None:
            s = intensity
            if has_any_seg:
                s = (s - s.min()) / (s.max() - s.min() + 1e-6)
                s = 0.25 + 0.35 * s
            else:
                s = (s - s.min()) / (s.max() - s.min() + 1e-6)
            colors = np.stack([s, s, s], axis=1)
        else:
            colors = np.full((pts_v.shape[0], 3), 0.45, dtype=np.float32)

        all_pts.append(pts_v)
        all_colors.append(colors.astype(np.float32))

        if use_return2:
            ri2_vals = row["[LiDARComponent].range_image_return2.values"]
            if ri2_vals is not None and len(ri2_vals) > 0:
                ri2 = reshape_range_image(
                    ri2_vals,
                    row["[LiDARComponent].range_image_return2.shape"],
                )
                pts_v2, intensity2, valid_mask2 = range_image_to_points_vehicle(
                    ri2, inc, T,
                    range_channel=0,
                    feature_channel_for_color=1 if ri2.shape[2] > 1 else None,
                )
                all_pts.append(pts_v2)
                if intensity2 is not None:
                    s2 = (intensity2 - intensity2.min()) / (intensity2.max() - intensity2.min() + 1e-6)
                    if has_any_seg:
                        s2 = 0.25 + 0.35 * s2
                    all_colors.append(np.stack([s2, s2, s2], axis=1).astype(np.float32))
                else:
                    all_colors.append(np.full((pts_v2.shape[0], 3), 0.45, dtype=np.float32))

    pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3), dtype=np.float32)
    cols = np.concatenate(all_colors, axis=0) if all_colors else np.zeros((0, 3), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))
    return pcd

def choose_random_segment_with_seg(lidar_seg_df, seed=None):
    segments = (
        lidar_seg_df
        .select("key.segment_context_name")
        .unique()
        .sort("key.segment_context_name")
        .collect()["key.segment_context_name"]
        .to_list()
    )
    if not segments:
        raise RuntimeError("No segmentation frames found in lidar_segmentation dataset.")

    rng = np.random.default_rng(seed)
    return str(rng.choice(segments))

def labeled_timestamps_for_segment(segment_lidar, segment_seg):
    lidar_ts = segment_lidar.select("key.frame_timestamp_micros").unique()
    seg_ts = segment_seg.select("key.frame_timestamp_micros").unique()
    ts = (
        lidar_ts
        .join(seg_ts, on="key.frame_timestamp_micros", how="inner")
        .sort("key.frame_timestamp_micros")["key.frame_timestamp_micros"]
        .to_list()
    )
    return [int(v) for v in ts]

def animate_segment(
    segment,
    timestamps,
    segment_lidar,
    segment_seg,
    segment_calib,
    fps=8.0,
    use_return2=False,
    loop=True,
    realtime=True,
    speed=1.0,
    max_frame_delay_s=0.2,
    camera_preset="angled",
    export_gif_path=None,
    export_gif_loops=1,
    export_gif_fps=None,
    print_camera_on_move=False,
    camera_print_interval_s=0.25,
    camera_change_threshold=1e-3,
    render_width=1920,
    render_height=1080,
):
    _require_open3d()
    if len(timestamps) == 0:
        raise RuntimeError(f"No labeled timestamps to animate for segment {segment}.")

    lidar_by_ts = {
        int(df["key.frame_timestamp_micros"][0]): df
        for df in segment_lidar.partition_by("key.frame_timestamp_micros", maintain_order=True)
    }
    seg_by_ts = {
        int(df["key.frame_timestamp_micros"][0]): df
        for df in segment_seg.partition_by("key.frame_timestamp_micros", maintain_order=True)
    }
    frame_sequence = []
    skipped_empty_frames = 0
    for ts_micros in timestamps:
        frame_lidar = lidar_by_ts.get(ts_micros)
        frame_seg = seg_by_ts.get(ts_micros)
        if frame_lidar is None:
            continue
        if frame_seg is None:
            frame_seg = segment_seg.head(0)

        frame_pcd = build_combined_point_cloud(
            frame_lidar,
            frame_seg,
            segment_calib,
            use_return2=use_return2,
        )
        if len(frame_pcd.points) == 0:
            skipped_empty_frames += 1
            continue
        frame_sequence.append((ts_micros, frame_pcd))

    if len(frame_sequence) == 0:
        raise RuntimeError(f"No point clouds could be built for segment {segment}.")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"Waymo segment {segment}",
        width=int(render_width),
        height=int(render_height),
    )
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.12, 0.14, 0.16], dtype=np.float64)
    render_opt.point_size = 2.5
    render_opt.point_color_option = o3d.visualization.PointColorOption.Color
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    frame_dt = 1.0 / max(float(fps), 1e-6)
    if realtime and len(frame_sequence) > 1:
        ts_arr = np.array([ts for ts, _ in frame_sequence], dtype=np.float64)
        dt = np.diff(ts_arr) / 1e6
        dt = dt[dt > 0]
        playback_fps = (max(float(speed), 1e-6) / np.median(dt)) if len(dt) else fps
    else:
        playback_fps = fps
    gif_fps = float(export_gif_fps) if export_gif_fps is not None else float(playback_fps)
    should_capture = export_gif_path is not None and export_gif_loops > 0
    gif_frames = []
    gif_durations_ms = []
    gif_saved = False
    loops_captured = 0
    last_camera_print_t = 0.0
    last_camera_extrinsic = None

    def _print_camera_hotkey(v):
        print_camera_pose(v.get_view_control(), prefix="CAMERA_POSE_MANUAL")
        return False

    vis.register_key_callback(ord("P"), _print_camera_hotkey)

    print(f"Animating segment {segment}")
    print(f"Frames with segmentation: {len(timestamps)}")
    print(f"Frames used for playback/export: {len(frame_sequence)}")
    if skipped_empty_frames > 0:
        print(f"Skipped empty point-cloud frames: {skipped_empty_frames}")
    print(f"Camera preset: {camera_preset}")
    if should_capture:
        print(f"GIF export: {export_gif_path} (capture loops={export_gif_loops}, fps={gif_fps:.2f})")
    if print_camera_on_move:
        print(f"Camera trace: enabled (interval={camera_print_interval_s:.2f}s)")
    print("Press 'P' to print current camera pose.")
    print("Close the Open3D window to stop.")

    try:
        did_reset_view = False
        keep_running = True
        while keep_running:
            last_ts_micros = None
            for i, (ts_micros, frame_pcd) in enumerate(frame_sequence, start=1):
                t0 = time.perf_counter()
                pcd.points = frame_pcd.points
                pcd.colors = frame_pcd.colors
                vis.update_geometry(pcd)
                if not did_reset_view and len(frame_pcd.points) > 0:
                    vis.reset_view_point(True)
                    apply_camera_preset(vis, frame_pcd, camera_preset)
                    did_reset_view = True

                if not vis.poll_events():
                    return
                vis.update_renderer()
                now = time.perf_counter()
                if print_camera_on_move:
                    ctr = vis.get_view_control()
                    pose = camera_pose_from_view_control(ctr)
                    extrinsic = np.asarray(pose["extrinsic"], dtype=np.float64)
                    changed = (
                        last_camera_extrinsic is None
                        or np.linalg.norm(extrinsic - last_camera_extrinsic) > camera_change_threshold
                    )
                    if changed and (now - last_camera_print_t) >= camera_print_interval_s:
                        print(f"\nCAMERA_POSE_AUTO={json.dumps(pose)}")
                        last_camera_print_t = now
                        last_camera_extrinsic = extrinsic

                print(f"Frame {i}/{len(frame_sequence)} ts={ts_micros}", end="\r")

                if realtime and last_ts_micros is not None:
                    target_dt = (ts_micros - last_ts_micros) / 1e6
                    target_dt = max(0.0, target_dt / max(float(speed), 1e-6))
                    target_dt = min(target_dt, max_frame_delay_s)
                else:
                    target_dt = frame_dt

                if should_capture and loops_captured < export_gif_loops:
                    # Force render during capture to avoid initial black frame artifacts.
                    rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True), dtype=np.float32)
                    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
                    # Retry once if we still got an all-black frame.
                    if i == 1 and np.mean(rgb8) < 1.0:
                        vis.poll_events()
                        vis.update_renderer()
                        rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True), dtype=np.float32)
                        rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
                    gif_frames.append(rgb8)
                    gif_durations_ms.append(max(1, int(round(target_dt * 1000.0))))

                elapsed = time.perf_counter() - t0
                sleep_dt = target_dt - elapsed
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
                last_ts_micros = ts_micros

            if should_capture and loops_captured < export_gif_loops:
                loops_captured += 1
                if loops_captured >= export_gif_loops and not gif_saved:
                    save_gif_from_frames(
                        gif_frames,
                        export_gif_path,
                        fps=gif_fps,
                        durations_ms=gif_durations_ms,
                    )
                    gif_saved = True
                    gif_frames = []
                    gif_durations_ms = []

            if not loop:
                keep_running = False
        print()
    finally:
        if should_capture and not gif_saved and len(gif_frames) > 0:
            save_gif_from_frames(
                gif_frames,
                export_gif_path,
                fps=gif_fps,
                durations_ms=gif_durations_ms,
            )
        vis.destroy_window()


# %%
if __name__ == "__main__":
    SCENE_SEED = 438756
    CAMERA_PRESET = "car_pov"
    EXPORT_GIF_PATH = "pointcloud.gif"
    EXPORT_GIF_LOOPS = 1
    EXPORT_GIF_FPS = None
    EXPORT_WIDTH = 1920
    EXPORT_HEIGHT = 1080
    PRINT_CAMERA_ON_MOVE = True
    CAMERA_PRINT_INTERVAL_S = 0.25

    segment = choose_random_segment_with_seg(lidar_segmentation_df, seed=SCENE_SEED)
    segment_lidar, segment_seg, segment_calib = load_segment_polars(
        lidar_df,
        lidar_segmentation_df,
        lidar_calibration_df,
        segment,
    )
    timestamps = labeled_timestamps_for_segment(segment_lidar, segment_seg)
    animate_segment(
        segment,
        timestamps,
        segment_lidar,
        segment_seg,
        segment_calib,
        fps=10.0,
        use_return2=False,
        loop=True,
        realtime=True,
        speed=1.0,
        camera_preset=CAMERA_PRESET,
        export_gif_path=EXPORT_GIF_PATH,
        export_gif_loops=EXPORT_GIF_LOOPS,
        export_gif_fps=EXPORT_GIF_FPS,
        print_camera_on_move=PRINT_CAMERA_ON_MOVE,
        camera_print_interval_s=CAMERA_PRINT_INTERVAL_S,
        render_width=EXPORT_WIDTH,
        render_height=EXPORT_HEIGHT,
    )
