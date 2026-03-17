import os
from pathlib import Path
import sys

import numpy as np


PALETTE = np.array(
    [
        [0.50, 0.50, 0.50],
        [0.95, 0.25, 0.22],
        [0.80, 0.20, 0.00],
        [1.00, 0.40, 0.00],
        [1.00, 0.60, 0.20],
        [0.60, 0.00, 0.80],
        [0.80, 0.20, 0.80],
        [1.00, 0.80, 0.00],
        [0.00, 0.80, 0.80],
        [0.00, 1.00, 1.00],
        [0.40, 0.40, 0.80],
        [1.00, 0.60, 0.60],
        [0.60, 0.00, 0.40],
        [0.40, 0.00, 0.60],
        [0.60, 0.40, 0.20],
        [0.18, 0.80, 0.44],
        [0.30, 0.50, 0.10],
        [0.70, 0.60, 0.40],
        [0.40, 0.40, 0.40],
        [0.80, 0.80, 0.00],
        [0.60, 0.60, 0.60],
        [0.20, 0.80, 0.60],
        [0.20, 0.60, 0.80],
    ],
    dtype=np.float32,
)

CAMERA_PRESETS: dict[str, np.ndarray] = {
    "top": np.array(
        [
            [-0.3856370173389184, 0.9225134468464795, 0.015906955880067963, 6.141559756838577],
            [0.47505989338065996, 0.21330916748662962, -0.8537079692537238, -6.064283949433857],
            [-0.790950180832585, -0.32166463817707514, -0.5205090508217053, 109.91047324199376],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    "car_pov": np.array(
        [
            [-0.026570101741086316, -0.9996106358387303, 0.008520939593593985, -0.6305623596589306],
            [-0.02978545966939474, -0.007728510895758548, -0.9995264361244364, 3.275704827536906],
            [0.9992031105264592, -0.02681131920334203, -0.02956851495806514, 13.644113467242853],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
}


def label_colors(labels: np.ndarray, valid_label: np.ndarray | None = None) -> np.ndarray:
    idx = np.clip(labels.astype(np.int64), 0, len(PALETTE) - 1)
    colors = PALETTE[idx].copy()
    if valid_label is not None:
        colors[~valid_label.astype(bool)] = np.array([0.45, 0.45, 0.45], dtype=np.float32)
    return colors


def save_gif(frames_rgb: list[np.ndarray], output_path: Path, fps: float) -> None:
    if not frames_rgb:
        raise RuntimeError("No frames captured for GIF export.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(1, int(round(1000.0 / max(float(fps), 1e-6))))
    from PIL import Image

    pil_frames = [Image.fromarray(frame) for frame in frames_rgb]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
    )


def ensure_open3d_linux_env() -> None:
    if not sys.platform.startswith("linux"):
        return
    os.environ["GDK_BACKEND"] = "x11"
    os.environ["XDG_SESSION_TYPE"] = "x11"
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ.pop("WAYLAND_DISPLAY", None)


def render_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    width: int,
    height: int,
    point_size: float,
    camera_preset: str,
) -> np.ndarray:
    ensure_open3d_linux_env()
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    vis = o3d.visualization.Visualizer()
    created = vis.create_window(visible=False, width=int(width), height=int(height))
    if not bool(created):
        vis.destroy_window()
        raise RuntimeError(
            "Open3D failed to create a rendering window. On Linux this usually means no usable X11/OpenGL "
            "context is available for offscreen rendering."
        )
    opt = vis.get_render_option()
    if opt is None:
        vis.destroy_window()
        raise RuntimeError(
            "Open3D created a window but did not provide render options. This usually indicates the OpenGL "
            "context failed to initialize."
        )
    opt.background_color = np.array([0.12, 0.14, 0.16], dtype=np.float64)
    opt.point_size = float(point_size)
    vis.add_geometry(pcd)
    vis.reset_view_point(True)

    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    key = "top" if camera_preset in {"top", "topdown"} else "car_pov"
    params.extrinsic = CAMERA_PRESETS[key]
    try:
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    except TypeError:
        ctr.convert_from_pinhole_camera_parameters(params)

    vis.poll_events()
    vis.update_renderer()
    rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True), dtype=np.float32)
    vis.destroy_window()
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
