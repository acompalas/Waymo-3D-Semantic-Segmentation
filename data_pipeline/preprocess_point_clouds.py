import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert preprocessed range-image artifacts into dense point-cloud tensors "
            "with explicit geometry/supervision masks."
        )
    )
    p.add_argument("--range-dir", type=Path, default=Path("preprocessed/range_images"))
    p.add_argument("--output-dir", type=Path, default=Path("preprocessed/point_clouds"))
    p.add_argument(
        "--segments",
        nargs="*",
        default=None,
        help="Optional explicit segment whitelist. If omitted, process all segments in segment_source.json.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for segment conversion.",
    )
    p.add_argument(
        "--progress-style",
        type=str,
        choices=["tqdm", "print", "none"],
        default="tqdm",
        help="Progress reporting style.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output. Deprecated in favor of --progress-style none.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


class _ProgressReporter:
    def __init__(self, *, total: int, style: str, desc: str) -> None:
        self.total = int(total)
        self.style = str(style)
        self.desc = str(desc)
        self.count = 0
        self._bar = None
        if self.style == "tqdm":
            self._bar = tqdm(total=self.total, desc=self.desc, unit="segment")

    def update(self, segment: str, frames: int) -> None:
        self.count += 1
        if self.style == "tqdm":
            assert self._bar is not None
            self._bar.update(1)
            return
        if self.style == "print":
            print(f"{self.desc}: {self.count}/{self.total} segments done - {segment} ({int(frames)} frames)", flush=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def _resolve_progress_style(args: argparse.Namespace) -> str:
    if bool(args.no_progress):
        return "none"
    return str(args.progress_style)


def _log_print_progress(progress_style: str, message: str) -> None:
    if str(progress_style) == "print":
        print(message, flush=True)


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n", encoding="utf-8")


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output path exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def _validate_range_meta(meta: dict[str, Any], range_dir: Path) -> None:
    representation = str(meta.get("representation", ""))
    if representation != "range_images":
        raise ValueError(
            f"Expected representation='range_images' in {range_dir / 'meta.json'}, got '{representation}'."
        )


def _build_output_meta(input_meta: dict[str, Any], range_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "representation": "point_clouds_dense",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_range_root": str(range_dir),
        "source_representation": str(input_meta.get("representation", "range_images")),
        "laser_id": int(input_meta.get("laser_id", 1)),
        "validity_rules": {
            "valid_geometry": "range > 0",
            "valid_label": "(range > 0) & (is_in_nlz <= 0) & (semantic > 0) & has_labels",
        },
        "segment_arrays": {
            "points_xyz": {
                "dtype": "float32",
                "shape": ["num_frames", "num_returns", "height", "width", 3],
                "coords": ["x", "y", "z"],
                "frame": "vehicle",
            },
            "features": {
                "dtype": "float32",
                "shape": ["num_frames", "num_returns", "height", "width", 2],
                "channels": ["lidar_intensity", "lidar_elongation"],
            },
            "semantic": {
                "dtype": "int16",
                "shape": ["num_frames", "num_returns", "height", "width"],
                "notes": "Labeled segments retain source semantic ids (0..22). Unlabeled segments are filled with -1.",
            },
            "valid_geometry": {
                "dtype": "bool",
                "shape": ["num_frames", "num_returns", "height", "width"],
            },
            "valid_label": {
                "dtype": "bool",
                "shape": ["num_frames", "num_returns", "height", "width"],
            },
            "class_counts": {
                "dtype": "int64",
                "shape": ["num_frames", "num_returns", "num_classes"],
                "notes": "Per-frame semantic counts copied from the source range-image artifacts using the valid_label mask.",
            },
            "timestamps": {"dtype": "int64", "shape": ["num_frames"]},
        },
    }


def _points_vehicle_dense(
    range_image: np.ndarray,  # [H,W,4], float32
    inclinations: np.ndarray,  # [H], float32
    azimuths: np.ndarray,  # [W], float32
    extrinsic: np.ndarray,  # [4,4], float32/float64
) -> np.ndarray:
    if range_image.ndim != 3 or range_image.shape[2] != 4:
        raise ValueError(f"Expected range image shape [H,W,4], got {range_image.shape}")

    ranges = range_image[:, :, 0].astype(np.float32, copy=False)  # [H,W]
    in2d = inclinations[:, None]  # [H,1]
    az2d = azimuths[None, :]  # [1,W]

    cos_in = np.cos(in2d)
    sin_in = np.sin(in2d)
    cos_az = np.cos(az2d)
    sin_az = np.sin(az2d)

    points_sensor = np.empty((range_image.shape[0], range_image.shape[1], 3), dtype=np.float32)
    points_sensor[:, :, 0] = ranges * cos_in * cos_az
    points_sensor[:, :, 1] = ranges * cos_in * sin_az
    points_sensor[:, :, 2] = ranges * sin_in

    rotation = extrinsic[:3, :3].astype(np.float32, copy=False)
    translation = extrinsic[:3, 3].astype(np.float32, copy=False)
    # Row-vector transform: p_vehicle = p_sensor @ R^T + t
    return points_sensor @ rotation.T + translation


def _process_segment_worker(
    in_segments_root: Path,
    out_segments_root: Path,
    segment: str,
    is_labeled: bool,
) -> tuple[str, int]:
    in_dir = in_segments_root / segment
    if not in_dir.exists():
        raise FileNotFoundError(f"Missing segment directory: {in_dir}")

    ri1 = np.load(in_dir / "ri1.npy", mmap_mode="r")
    ri2 = np.load(in_dir / "ri2.npy", mmap_mode="r")
    timestamps = np.load(in_dir / "timestamps.npy")
    inclinations = np.load(in_dir / "inclinations.npy")
    azimuths = np.load(in_dir / "azimuths.npy")
    extrinsic = np.load(in_dir / "extrinsic.npy")
    class_counts = np.load(in_dir / "class_counts.npy")

    if ri1.ndim != 4 or ri1.shape[-1] != 4:
        raise ValueError(f"Unexpected ri1 shape for segment '{segment}': {ri1.shape}")
    if ri2.shape != ri1.shape:
        raise ValueError(f"RI2 shape mismatch for segment '{segment}': {ri2.shape} vs {ri1.shape}")

    num_frames, height, width, _ = ri1.shape
    if timestamps.shape[0] != num_frames:
        raise ValueError(
            f"timestamps length mismatch for segment '{segment}': {timestamps.shape[0]} vs num_frames={num_frames}"
        )
    if inclinations.shape[0] != height:
        raise ValueError(
            f"inclinations length mismatch for segment '{segment}': {inclinations.shape[0]} vs height={height}"
        )
    if azimuths.shape[0] != width:
        raise ValueError(f"azimuths length mismatch for segment '{segment}': {azimuths.shape[0]} vs width={width}")
    if extrinsic.shape != (4, 4):
        raise ValueError(f"Unexpected extrinsic shape for segment '{segment}': {extrinsic.shape}")

    seg1: Optional[np.ndarray] = None
    seg2: Optional[np.ndarray] = None
    if is_labeled:
        seg1_path = in_dir / "seg1.npy"
        seg2_path = in_dir / "seg2.npy"
        if not seg1_path.exists() or not seg2_path.exists():
            # Fall back to unlabeled handling if segmentation arrays are absent.
            is_labeled = False
        else:
            seg1 = np.load(seg1_path, mmap_mode="r")
            seg2 = np.load(seg2_path, mmap_mode="r")
            if seg1.ndim != 4 or seg1.shape[-1] != 2:
                raise ValueError(f"Unexpected seg1 shape for segment '{segment}': {seg1.shape}")
            if seg2.shape != seg1.shape:
                raise ValueError(f"SEG2 shape mismatch for segment '{segment}': {seg2.shape} vs {seg1.shape}")
            if seg1.shape[:3] != ri1.shape[:3]:
                raise ValueError(
                    f"Segmentation spatial/temporal shape mismatch for segment '{segment}': "
                    f"{seg1.shape[:3]} vs {ri1.shape[:3]}"
                )
    if class_counts.shape[:2] != (num_frames, 2):
        raise ValueError(
            f"class_counts shape mismatch for segment '{segment}': expected leading shape {(num_frames, 2)}, got {class_counts.shape}"
        )

    out_dir = out_segments_root / segment
    out_dir.mkdir(parents=True, exist_ok=False)

    xyz = np.empty((num_frames, 2, height, width, 3), dtype=np.float32)
    feat = np.empty((num_frames, 2, height, width, 2), dtype=np.float32)
    semantic = np.empty((num_frames, 2, height, width), dtype=np.int16)
    valid_geometry = np.empty((num_frames, 2, height, width), dtype=np.bool_)
    valid_label = np.empty((num_frames, 2, height, width), dtype=np.bool_)

    for frame_idx in range(num_frames):
        for ret_idx, ri in enumerate((ri1, ri2)):
            ri_frame = ri[frame_idx]
            if ri_frame.shape != (height, width, 4):
                raise ValueError(
                    f"Unexpected frame shape for segment '{segment}', frame={frame_idx}, return={ret_idx + 1}: "
                    f"{ri_frame.shape}"
                )

            points = _points_vehicle_dense(
                range_image=ri_frame,
                inclinations=inclinations,
                azimuths=azimuths,
                extrinsic=extrinsic,
            )
            xyz[frame_idx, ret_idx] = points
            feat[frame_idx, ret_idx, :, :, 0] = ri_frame[:, :, 1]  # intensity
            feat[frame_idx, ret_idx, :, :, 1] = ri_frame[:, :, 2]  # elongation

            geom_mask = ri_frame[:, :, 0] > 0.0
            valid_geometry[frame_idx, ret_idx] = geom_mask

            if is_labeled and seg1 is not None and seg2 is not None:
                seg_frame = seg1[frame_idx] if ret_idx == 0 else seg2[frame_idx]
                sem = seg_frame[:, :, 1].astype(np.int16, copy=False)  # semantic_class
                semantic[frame_idx, ret_idx] = sem
                valid_label[frame_idx, ret_idx] = geom_mask & (ri_frame[:, :, 3] <= 0.0) & (sem > 0)
            else:
                semantic[frame_idx, ret_idx] = -1
                valid_label[frame_idx, ret_idx] = False

    np.save(out_dir / "xyz.npy", xyz)
    np.save(out_dir / "feat.npy", feat)
    np.save(out_dir / "semantic.npy", semantic)
    np.save(out_dir / "valid_geometry.npy", valid_geometry)
    np.save(out_dir / "valid_label.npy", valid_label)
    np.save(out_dir / "timestamps.npy", timestamps.astype(np.int64, copy=False))
    np.save(out_dir / "class_counts.npy", class_counts.astype(np.int64, copy=False))
    return segment, int(num_frames)


def _filtered_segment_source(
    records: list[dict[str, Any]],
    allowed_segments: Optional[set[str]],
) -> tuple[list[dict[str, Any]], list[tuple[str, bool]]]:
    out_records: list[dict[str, Any]] = []
    tasks: list[tuple[str, bool]] = []
    seen: set[str] = set()

    for record in records:
        source_subdir = str(record["source_subdir"])
        is_labeled = bool(record["is_labeled"])
        original_segments = [str(s) for s in record.get("segments", [])]

        if allowed_segments is None:
            keep_segments = original_segments
        else:
            keep_segments = [s for s in original_segments if s in allowed_segments]

        for segment in keep_segments:
            if segment in seen:
                raise ValueError(f"Segment '{segment}' appears multiple times in segment_source.json")
            seen.add(segment)
            tasks.append((segment, is_labeled))

        out_records.append(
            {
                "source_subdir": source_subdir,
                "is_labeled": is_labeled,
                "segments": keep_segments,
            }
        )

    if allowed_segments is not None:
        missing = sorted(allowed_segments.difference(seen))
        if missing:
            preview = missing[:10]
            raise ValueError(f"Requested segments not found in segment_source.json: {preview}")

    return out_records, tasks


def _process_segments(
    in_segments_root: Path,
    out_segments_root: Path,
    tasks: list[tuple[str, bool]],
    num_workers: int,
    progress_style: str,
) -> None:
    for segment, _ in tasks:
        if not (in_segments_root / segment).exists():
            raise FileNotFoundError(f"Segment listed in source file but missing in range-dir/segments: {segment}")

    _log_print_progress(progress_style, f"Prepared {len(tasks)} segment jobs.")

    if num_workers <= 1:
        progress = _ProgressReporter(total=len(tasks), style=progress_style, desc="segments")
        try:
            for segment, is_labeled in tasks:
                _log_print_progress(progress_style, f"Starting segment: {segment}")
                done_segment, num_frames = _process_segment_worker(
                    in_segments_root=in_segments_root,
                    out_segments_root=out_segments_root,
                    segment=segment,
                    is_labeled=is_labeled,
                )
                progress.update(done_segment, num_frames)
        finally:
            progress.close()
        return

    _log_print_progress(progress_style, f"Launching {max(1, int(num_workers))} worker processes.")
    progress = _ProgressReporter(total=len(tasks), style=progress_style, desc="segments")
    try:
        with ProcessPoolExecutor(max_workers=max(1, int(num_workers)), mp_context=mp.get_context("spawn")) as pool:
            futures = [
                pool.submit(
                    _process_segment_worker,
                    in_segments_root,
                    out_segments_root,
                    segment,
                    is_labeled,
                )
                for segment, is_labeled in tasks
            ]
            for future in as_completed(futures):
                done_segment, num_frames = future.result()
                progress.update(done_segment, num_frames)
    finally:
        progress.close()


def main() -> None:
    args = parse_args()
    range_dir = args.range_dir
    output_dir = args.output_dir
    progress_style = _resolve_progress_style(args)

    _log_print_progress(progress_style, f"Reading range-image metadata from {range_dir} ...")
    input_meta = _read_json(range_dir / "meta.json")
    _validate_range_meta(input_meta, range_dir)
    input_segment_source = _read_json(range_dir / "segment_source.json")
    if not isinstance(input_segment_source, list):
        raise ValueError(f"Expected list in {range_dir / 'segment_source.json'}")

    allowed_segments = None if args.segments is None else {str(s) for s in args.segments}
    out_segment_source, tasks = _filtered_segment_source(input_segment_source, allowed_segments)
    if not tasks:
        raise ValueError("No segments selected for processing.")

    _prepare_output_dir(output_dir=output_dir, overwrite=bool(args.overwrite))
    out_segments_root = output_dir / "segments"
    out_segments_root.mkdir(parents=True, exist_ok=False)

    _write_json(output_dir / "meta.json", _build_output_meta(input_meta=input_meta, range_dir=range_dir))
    _write_json(output_dir / "segment_source.json", out_segment_source, sort_keys=False)
    _log_print_progress(progress_style, f"Wrote metadata to {output_dir}.")

    classes_path = range_dir / "classes.json"
    if classes_path.exists():
        shutil.copy2(classes_path, output_dir / "classes.json")
        _log_print_progress(progress_style, f"Copied classes.json to {output_dir}.")

    _process_segments(
        in_segments_root=range_dir / "segments",
        out_segments_root=out_segments_root,
        tasks=tasks,
        num_workers=int(args.num_workers),
        progress_style=progress_style,
    )
    print(f"Wrote dense point-cloud artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
