import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import polars as pl
from tqdm.auto import tqdm

# Keep Polars worker thread usage low during multiprocessing unless caller overrides it.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

SEGMENT_COL = "key.segment_context_name"
TIMESTAMP_COL = "key.frame_timestamp_micros"
LASER_COL = "key.laser_name"

LIDAR_RI1_VALUES_COL = "[LiDARComponent].range_image_return1.values"
LIDAR_RI1_SHAPE_COL = "[LiDARComponent].range_image_return1.shape"
LIDAR_RI2_VALUES_COL = "[LiDARComponent].range_image_return2.values"
LIDAR_RI2_SHAPE_COL = "[LiDARComponent].range_image_return2.shape"

SEG_RI1_VALUES_COL = "[LiDARSegmentationLabelComponent].range_image_return1.values"
SEG_RI1_SHAPE_COL = "[LiDARSegmentationLabelComponent].range_image_return1.shape"
SEG_RI2_VALUES_COL = "[LiDARSegmentationLabelComponent].range_image_return2.values"
SEG_RI2_SHAPE_COL = "[LiDARSegmentationLabelComponent].range_image_return2.shape"

CALIB_EXTRINSIC_COL = "[LiDARCalibrationComponent].extrinsic.transform"
CALIB_INCL_VALUES_COL = "[LiDARCalibrationComponent].beam_inclination.values"
CALIB_INCL_MIN_COL = "[LiDARCalibrationComponent].beam_inclination.min"
CALIB_INCL_MAX_COL = "[LiDARCalibrationComponent].beam_inclination.max"

LIDAR_COLUMNS = [
    TIMESTAMP_COL,
    LIDAR_RI1_VALUES_COL,
    LIDAR_RI1_SHAPE_COL,
    LIDAR_RI2_VALUES_COL,
    LIDAR_RI2_SHAPE_COL,
]
SEG_COLUMNS = [
    TIMESTAMP_COL,
    SEG_RI1_VALUES_COL,
    SEG_RI1_SHAPE_COL,
    SEG_RI2_VALUES_COL,
    SEG_RI2_SHAPE_COL,
]
CALIB_COLUMNS = [
    CALIB_EXTRINSIC_COL,
    CALIB_INCL_VALUES_COL,
    CALIB_INCL_MIN_COL,
    CALIB_INCL_MAX_COL,
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess Waymo LiDAR range-image artifacts.")
    p.add_argument("--data-dir", type=Path, default=Path("waymo_open_dataset_v_2_0_1"))
    p.add_argument("--output-dir", type=Path, default=Path("preprocessed/range_images"))
    p.add_argument(
        "--proto-path",
        type=Path,
        default=None,
        help="Optional segmentation proto path. Defaults to <data-dir>/segmentation.proto.",
    )
    p.add_argument("--laser-id", type=int, default=1)
    p.add_argument(
        "--labeled-subdirs",
        nargs="*",
        default=["training", "validation"],
        help=(
            "Source subdirs to treat as labeled. These must contain lidar, lidar_segmentation, "
            "and lidar_calibration."
        ),
    )
    p.add_argument(
        "--unlabeled-subdirs",
        nargs="*",
        default=[],
        help=(
            "Source subdirs to treat as unlabeled. These are preprocessed from lidar+calibration "
            "only, and seg1/seg2 are omitted."
        ),
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes for segment-level preprocessing.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _to_name(enum_key: str) -> str:
    return enum_key.removeprefix("TYPE_").lower()


def parse_classes(proto_path: Path) -> dict[str, str]:
    if not proto_path.exists():
        raise FileNotFoundError(f"Missing segmentation proto: {proto_path}")

    classes: dict[int, str] = {}
    pattern = re.compile(r"^\s*(TYPE_[A-Z0-9_]+)\s*=\s*(\d+)\s*;")
    for line in proto_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        enum_key = match.group(1)
        class_id = int(match.group(2))
        classes[class_id] = _to_name(enum_key)

    if not classes:
        raise ValueError(f"No class definitions parsed from {proto_path}")
    # Keep insertion order sorted numerically by class id.
    return {str(k): classes[k] for k in sorted(classes)}


def _resolve_proto_path(data_dir: Path, proto_path: Optional[Path]) -> Path:
    return Path(proto_path) if proto_path is not None else data_dir / "segmentation.proto"


def _class_count_vector(semantic: np.ndarray, valid_label: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.zeros(int(num_classes), dtype=np.int64)
    classes = semantic[valid_label]
    if classes.size == 0:
        return counts
    classes = classes[(classes >= 0) & (classes < int(num_classes))]
    if classes.size == 0:
        return counts
    counts[:] = np.bincount(classes.astype(np.int64, copy=False), minlength=int(num_classes))[: int(num_classes)]
    return counts


def _map_segment_files(directory: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(directory.glob("*.parquet"))}


def _normalize_subdir_list(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        subdir = str(value).strip()
        if not subdir or subdir in seen:
            continue
        seen.add(subdir)
        out.append(subdir)
    return out


def _resolve_source_roots(
    data_dir: Path,
    labeled_subdirs: Sequence[str],
    unlabeled_subdirs: Sequence[str],
) -> dict[str, Path]:
    labeled = _normalize_subdir_list(labeled_subdirs)
    unlabeled = _normalize_subdir_list(unlabeled_subdirs)

    overlap = sorted(set(labeled).intersection(unlabeled))
    if overlap:
        raise ValueError(f"Source subdirs cannot be both labeled and unlabeled: {overlap}")

    requested = labeled + unlabeled
    if not requested:
        raise ValueError("No source subdirs requested. Pass --labeled-subdirs and/or --unlabeled-subdirs.")

    roots: dict[str, Path] = {}
    for subdir in requested:
        root = data_dir / subdir
        if not root.is_dir():
            raise FileNotFoundError(f"Missing source subdir: {root}")

        required = ["lidar", "lidar_calibration"]
        if subdir in labeled:
            required.append("lidar_segmentation")

        missing = [name for name in required if not (root / name).is_dir()]
        if missing:
            raise FileNotFoundError(f"Source subdir '{subdir}' is missing required directories: {missing}")
        roots[subdir] = root

    return roots


def _build_labeled_frame_index(split_root: Path, laser_id: int) -> pl.DataFrame:
    seg_keys = (
        pl.scan_parquet(str(split_root / "lidar_segmentation" / "*.parquet"))
        .filter(pl.col(LASER_COL) == int(laser_id))
        .select(SEGMENT_COL, TIMESTAMP_COL)
        .unique()
    )
    lidar_keys = (
        pl.scan_parquet(str(split_root / "lidar" / "*.parquet"))
        .filter(pl.col(LASER_COL) == int(laser_id))
        .select(SEGMENT_COL, TIMESTAMP_COL)
        .unique()
    )
    return (
        lidar_keys.join(seg_keys, on=[SEGMENT_COL, TIMESTAMP_COL], how="inner")
        .sort([SEGMENT_COL, TIMESTAMP_COL])
        .collect()
    )


def _read_segment_table(parquet_path: Path, columns: Sequence[str], laser_id: int) -> pl.DataFrame:
    return (
        pl.scan_parquet(str(parquet_path))
        .filter(pl.col(LASER_COL) == int(laser_id))
        .select(*columns)
        .collect()
    )


def _reshape_array(values_list: Sequence[Any], shape_arr: Sequence[int], dtype: Any) -> np.ndarray:
    shape = tuple(int(x) for x in shape_arr)
    return np.asarray(values_list, dtype=dtype).reshape(shape)


def _reshape_optional(
    values_list: Optional[Sequence[Any]],
    shape_arr: Optional[Sequence[int]],
    dtype: Any,
    fallback_shape: Sequence[int],
) -> np.ndarray:
    if values_list is None or shape_arr is None:
        return np.zeros(tuple(int(x) for x in fallback_shape), dtype=dtype)
    if len(values_list) == 0 or len(shape_arr) == 0:
        return np.zeros(tuple(int(x) for x in fallback_shape), dtype=dtype)
    return _reshape_array(values_list, shape_arr, dtype)


def _extrinsic_matrix(calib_row: dict[str, Any]) -> np.ndarray:
    transform = np.asarray(calib_row[CALIB_EXTRINSIC_COL], dtype=np.float32)
    if transform.size != 16:
        raise ValueError("Calibration extrinsic transform must have 16 elements.")
    return transform.reshape(4, 4)


def _beam_inclinations(calib_row: dict[str, Any], height: int) -> np.ndarray:
    vals = calib_row[CALIB_INCL_VALUES_COL]
    if vals is not None and len(vals) == height:
        incl = np.asarray(vals, dtype=np.float32)
    else:
        mn = float(calib_row[CALIB_INCL_MIN_COL])
        mx = float(calib_row[CALIB_INCL_MAX_COL])
        incl = np.linspace(mn, mx, height, dtype=np.float32)
    return incl[::-1]


def _beam_azimuths(width: int) -> np.ndarray:
    return np.linspace(np.pi, -np.pi, width, endpoint=False, dtype=np.float32)


def _write_json(path: Path, payload: Any, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n", encoding="utf-8")


def _save_segment(
    segments_root: Path,
    segment: str,
    ri1: np.ndarray,
    ri2: np.ndarray,
    seg1: Optional[np.ndarray],
    seg2: Optional[np.ndarray],
    extrinsic: np.ndarray,
    inclinations: np.ndarray,
    azimuths: np.ndarray,
    timestamps: np.ndarray,
    class_counts: np.ndarray,
) -> None:
    segment_dir = segments_root / segment
    segment_dir.mkdir(parents=True, exist_ok=False)

    np.save(segment_dir / "ri1.npy", ri1)
    np.save(segment_dir / "ri2.npy", ri2)
    if (seg1 is None) != (seg2 is None):
        raise ValueError("seg1/seg2 must either both be present or both be omitted.")
    if seg1 is not None and seg2 is not None:
        np.save(segment_dir / "seg1.npy", seg1)
        np.save(segment_dir / "seg2.npy", seg2)
    np.save(segment_dir / "extrinsic.npy", extrinsic)
    np.save(segment_dir / "inclinations.npy", inclinations)
    np.save(segment_dir / "azimuths.npy", azimuths)
    np.save(segment_dir / "timestamps.npy", timestamps)
    np.save(segment_dir / "class_counts.npy", class_counts)


def _build_root_meta(data_dir: Path, laser_id: int) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "representation": "range_images",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_data_root": str(data_dir),
        "laser_id": int(laser_id),
        "segment_arrays": {
            "ri1": {
                "dtype": "float32",
                "shape": ["num_frames", "height", "width", "channel"],
                "channels": ["range", "lidar_intensity", "lidar_elongation", "is_in_nlz"],
            },
            "ri2": {
                "dtype": "float32",
                "shape": ["num_frames", "height", "width", "channel"],
                "channels": ["range", "lidar_intensity", "lidar_elongation", "is_in_nlz"],
            },
            "seg1": {
                "dtype": "int32",
                "shape": ["num_frames", "height", "width", "channel"],
                "channels": ["instance_id", "semantic_class"],
            },
            "seg2": {
                "dtype": "int32",
                "shape": ["num_frames", "height", "width", "channel"],
                "channels": ["instance_id", "semantic_class"],
            },
            "class_counts": {
                "dtype": "int64",
                "shape": ["num_frames", "num_returns", "num_classes"],
                "notes": "Per-frame semantic counts using valid_label = (range > 0) & (is_in_nlz <= 0) & (semantic_class > 0).",
            },
            "extrinsic": {"dtype": "float32", "shape": [4, 4]},
            "inclinations": {"dtype": "float32", "shape": ["height"]},
            "azimuths": {"dtype": "float32", "shape": ["width"]},
            "timestamps": {"dtype": "int64", "shape": ["num_frames"]},
        }
    }


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output path exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def _load_joined_segment_frames(
    lidar_path: Path,
    seg_path: Path,
    calib_path: Path,
    segment: str,
    laser_id: int,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    lidar_df = _read_segment_table(lidar_path, LIDAR_COLUMNS, laser_id)
    seg_df = _read_segment_table(seg_path, SEG_COLUMNS, laser_id)
    calib_df = _read_segment_table(calib_path, CALIB_COLUMNS, laser_id)
    if calib_df.height == 0:
        raise ValueError(f"Missing calibration for segment '{segment}' and laser_id={laser_id}")
    calib_row = calib_df.row(0, named=True)

    joined = lidar_df.join(seg_df, on=TIMESTAMP_COL, how="inner").sort(TIMESTAMP_COL)
    if joined.height == 0:
        raise ValueError(f"No labeled frames after join for segment '{segment}'")
    return joined, calib_row


def _load_unlabeled_segment_frames(
    lidar_path: Path,
    calib_path: Path,
    segment: str,
    laser_id: int,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    lidar_df = _read_segment_table(lidar_path, LIDAR_COLUMNS, laser_id).sort(TIMESTAMP_COL)
    calib_df = _read_segment_table(calib_path, CALIB_COLUMNS, laser_id)
    if calib_df.height == 0:
        raise ValueError(f"Missing calibration for segment '{segment}' and laser_id={laser_id}")
    if lidar_df.height == 0:
        raise ValueError(f"No lidar frames available for segment '{segment}'")
    calib_row = calib_df.row(0, named=True)
    return lidar_df, calib_row


def _process_segment_worker(
    segments_root: Path,
    segment: str,
    lidar_path: Path,
    seg_path: Optional[Path],
    calib_path: Path,
    laser_id: int,
    is_labeled: bool,
    num_classes: int,
) -> tuple[str, int]:
    if is_labeled:
        if seg_path is None:
            raise ValueError(f"Missing segmentation parquet for labeled segment '{segment}'")
        frame_df, calib_row = _load_joined_segment_frames(
            lidar_path=lidar_path,
            seg_path=seg_path,
            calib_path=calib_path,
            segment=segment,
            laser_id=laser_id,
        )
    else:
        frame_df, calib_row = _load_unlabeled_segment_frames(
            lidar_path=lidar_path,
            calib_path=calib_path,
            segment=segment,
            laser_id=laser_id,
        )

    extrinsic = _extrinsic_matrix(calib_row)
    ri1_frames: list[np.ndarray] = []
    ri2_frames: list[np.ndarray] = []
    seg1_frames: list[np.ndarray] = []
    seg2_frames: list[np.ndarray] = []
    timestamps: list[int] = []
    frame_hw: Optional[tuple[int, int]] = None

    for row in frame_df.iter_rows(named=True):
        ri1 = _reshape_array(row[LIDAR_RI1_VALUES_COL], row[LIDAR_RI1_SHAPE_COL], np.float32)

        if ri1.ndim != 3 or ri1.shape[2] != 4:
            raise ValueError(f"Unexpected RI1 shape {ri1.shape} for segment '{segment}'")

        if frame_hw is None:
            frame_hw = (int(ri1.shape[0]), int(ri1.shape[1]))
        elif frame_hw != (int(ri1.shape[0]), int(ri1.shape[1])):
            raise ValueError(
                f"Frame shape changed within segment '{segment}': expected {frame_hw}, got {ri1.shape[:2]}"
            )

        ri2 = _reshape_optional(
            row[LIDAR_RI2_VALUES_COL],
            row[LIDAR_RI2_SHAPE_COL],
            np.float32,
            fallback_shape=ri1.shape,
        )
        if ri2.shape != ri1.shape:
            raise ValueError(f"RI2 shape mismatch in segment '{segment}': {ri2.shape} vs {ri1.shape}")

        if is_labeled:
            seg1 = _reshape_array(row[SEG_RI1_VALUES_COL], row[SEG_RI1_SHAPE_COL], np.int32)
            if seg1.ndim != 3 or seg1.shape[2] != 2:
                raise ValueError(f"Unexpected SEG1 shape {seg1.shape} for segment '{segment}'")
            if ri1.shape[:2] != seg1.shape[:2]:
                raise ValueError(f"RI1/SEG1 spatial mismatch in segment '{segment}'")

            seg2 = _reshape_optional(
                row[SEG_RI2_VALUES_COL],
                row[SEG_RI2_SHAPE_COL],
                np.int32,
                fallback_shape=seg1.shape,
            )
            if seg2.shape != seg1.shape:
                raise ValueError(f"SEG2 shape mismatch in segment '{segment}': {seg2.shape} vs {seg1.shape}")
            seg1_frames.append(seg1)
            seg2_frames.append(seg2)

        ri1_frames.append(ri1)
        ri2_frames.append(ri2)
        timestamps.append(int(row[TIMESTAMP_COL]))

    if frame_hw is None:
        raise ValueError(f"No frames available for segment '{segment}'")
    h, w = frame_hw
    ri1_arr = np.stack(ri1_frames, axis=0)
    ri2_arr = np.stack(ri2_frames, axis=0)
    seg1_arr = np.stack(seg1_frames, axis=0) if is_labeled else None
    seg2_arr = np.stack(seg2_frames, axis=0) if is_labeled else None
    class_counts = np.zeros((len(timestamps), 2, int(num_classes)), dtype=np.int64)
    if seg1_arr is not None and seg2_arr is not None:
        sem1 = seg1_arr[:, :, :, 1].astype(np.int64, copy=False)
        sem2 = seg2_arr[:, :, :, 1].astype(np.int64, copy=False)
        valid1 = (ri1_arr[:, :, :, 0] > 0.0) & (ri1_arr[:, :, :, 3] <= 0.0) & (sem1 > 0)
        valid2 = (ri2_arr[:, :, :, 0] > 0.0) & (ri2_arr[:, :, :, 3] <= 0.0) & (sem2 > 0)
        for frame_idx in range(len(timestamps)):
            class_counts[frame_idx, 0] = _class_count_vector(sem1[frame_idx], valid1[frame_idx], int(num_classes))
            class_counts[frame_idx, 1] = _class_count_vector(sem2[frame_idx], valid2[frame_idx], int(num_classes))
    _save_segment(
        segments_root=segments_root,
        segment=segment,
        ri1=ri1_arr,
        ri2=ri2_arr,
        seg1=seg1_arr,
        seg2=seg2_arr,
        extrinsic=extrinsic,
        inclinations=_beam_inclinations(calib_row, h),
        azimuths=_beam_azimuths(w),
        timestamps=np.asarray(timestamps, dtype=np.int64),
        class_counts=class_counts,
    )
    return segment, len(timestamps)


def _process_segments(
    segments_root: Path,
    source_roots: dict[str, Path],
    labeled_source_segments: dict[str, list[str]],
    unlabeled_source_segments: dict[str, list[str]],
    laser_id: int,
    num_classes: int,
    num_workers: int,
    show_progress: bool,
) -> None:
    tasks: list[tuple[str, Path, Optional[Path], Path, bool]] = []
    for source_subdir, segment_names in labeled_source_segments.items():
        root = source_roots[source_subdir]
        lidar_files = _map_segment_files(root / "lidar")
        seg_files = _map_segment_files(root / "lidar_segmentation")
        calib_files = _map_segment_files(root / "lidar_calibration")

        for segment in segment_names:
            lidar_path = lidar_files.get(segment)
            seg_path = seg_files.get(segment)
            calib_path = calib_files.get(segment)
            if lidar_path is None or seg_path is None or calib_path is None:
                raise KeyError(f"Missing lidar/seg/calibration parquet for segment '{segment}' under {root}")
            tasks.append((segment, lidar_path, seg_path, calib_path, True))

    for source_subdir, segment_names in unlabeled_source_segments.items():
        root = source_roots[source_subdir]
        lidar_files = _map_segment_files(root / "lidar")
        calib_files = _map_segment_files(root / "lidar_calibration")

        for segment in segment_names:
            lidar_path = lidar_files.get(segment)
            calib_path = calib_files.get(segment)
            if lidar_path is None or calib_path is None:
                raise KeyError(f"Missing lidar/calibration parquet for segment '{segment}' under {root}")
            tasks.append((segment, lidar_path, None, calib_path, False))

    if num_workers <= 1:
        progress = tqdm(total=len(tasks), desc="segments", unit="segment", disable=not show_progress)
        try:
            for segment, lidar_path, seg_path, calib_path, is_labeled in tasks:
                _process_segment_worker(
                    segments_root=segments_root,
                    segment=segment,
                    lidar_path=lidar_path,
                    seg_path=seg_path,
                    calib_path=calib_path,
                    laser_id=laser_id,
                    is_labeled=is_labeled,
                    num_classes=num_classes,
                )
                progress.update(1)
        finally:
            progress.close()
        return

    max_workers = max(1, int(num_workers))
    progress = tqdm(total=len(tasks), desc="segments", unit="segment", disable=not show_progress)
    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as pool:
            futures = [
                pool.submit(
                    _process_segment_worker,
                    segments_root,
                    segment,
                    lidar_path,
                    seg_path,
                    calib_path,
                    laser_id,
                    is_labeled,
                    num_classes,
                )
                for segment, lidar_path, seg_path, calib_path, is_labeled in tasks
            ]
            for future in as_completed(futures):
                future.result()
                progress.update(1)
    finally:
        progress.close()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    laser_id = int(args.laser_id)
    labeled_subdirs = _normalize_subdir_list(args.labeled_subdirs)
    unlabeled_subdirs = _normalize_subdir_list(args.unlabeled_subdirs)
    source_roots = _resolve_source_roots(
        data_dir=data_dir,
        labeled_subdirs=labeled_subdirs,
        unlabeled_subdirs=unlabeled_subdirs,
    )
    proto_path = _resolve_proto_path(data_dir, args.proto_path)
    unlabeled_subdir_set = set(unlabeled_subdirs)

    classes = parse_classes(proto_path)
    num_classes = max(int(key) for key in classes) + 1
    segment_source_records: list[dict[str, Any]] = []
    labeled_source_segments: dict[str, list[str]] = {}
    unlabeled_source_segments: dict[str, list[str]] = {}
    for source_subdir, root in source_roots.items():
        is_labeled = source_subdir not in unlabeled_subdir_set
        if is_labeled:
            labeled_index = _build_labeled_frame_index(root, laser_id)
            segments = sorted(str(x) for x in labeled_index.get_column(SEGMENT_COL).unique().to_list())
            labeled_source_segments[source_subdir] = segments
        else:
            segments = sorted(_map_segment_files(root / "lidar").keys())
            unlabeled_source_segments[source_subdir] = segments

        segment_source_records.append(
            {
                "source_subdir": source_subdir,
                "is_labeled": bool(is_labeled),
                "segments": segments,
            }
        )

    segment_to_sources: dict[str, list[str]] = {}
    for record in segment_source_records:
        source_subdir = str(record["source_subdir"])
        for segment in record["segments"]:
            segment_to_sources.setdefault(str(segment), []).append(source_subdir)
    overlaps = sorted((segment, sorted(sources)) for segment, sources in segment_to_sources.items() if len(sources) > 1)
    if overlaps:
        preview = [seg for seg, _ in overlaps[:5]]
        raise ValueError(f"Segments appear in multiple source subdirs: {preview}")

    _prepare_output_dir(output_dir=output_dir, overwrite=bool(args.overwrite))
    segments_root = output_dir / "segments"
    segments_root.mkdir(parents=True, exist_ok=False)

    # Keep numeric class index ordering in output.
    _write_json(output_dir / "classes.json", classes, sort_keys=False)
    _write_json(output_dir / "meta.json", _build_root_meta(data_dir=data_dir, laser_id=laser_id))
    _write_json(output_dir / "segment_source.json", segment_source_records, sort_keys=False)

    _process_segments(
        segments_root=segments_root,
        source_roots=source_roots,
        labeled_source_segments=labeled_source_segments,
        unlabeled_source_segments=unlabeled_source_segments,
        laser_id=laser_id,
        num_classes=num_classes,
        num_workers=int(args.num_workers),
        show_progress=not bool(args.no_progress),
    )
    print(f"Wrote range-image artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
