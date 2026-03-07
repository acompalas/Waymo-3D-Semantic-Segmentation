from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

SEGMENT_COL = "key.segment_context_name"
TIMESTAMP_COL = "key.frame_timestamp_micros"
LASER_COL = "key.laser_name"
PRIMARY_LASER_ID = 1
SEMANTIC_CHANNEL = 1

# ----------------------------
# Array reshaping / calibration
# ----------------------------
def reshape_array(values_list: Sequence[Any], shape_arr: Sequence[int], dtype: Any) -> np.ndarray:
    shape = tuple(int(x) for x in shape_arr)
    return np.asarray(values_list, dtype=dtype).reshape(shape)


def extrinsic_matrix(calib_row: dict[str, Any]) -> np.ndarray:
    transform = np.asarray(calib_row["[LiDARCalibrationComponent].extrinsic.transform"], dtype=np.float64)
    mat = transform.reshape(4, 4)
    if (not np.isfinite(mat).all()) or np.abs(mat).max() > 1e3:
        return np.eye(4, dtype=np.float64)
    return mat


def beam_inclinations(calib_row: dict[str, Any], height: int) -> np.ndarray:
    vals = calib_row["[LiDARCalibrationComponent].beam_inclination.values"]
    if vals is not None and len(vals) == height:
        inc = np.asarray(vals, dtype=np.float32)
    else:
        mn = float(calib_row["[LiDARCalibrationComponent].beam_inclination.min"])
        mx = float(calib_row["[LiDARCalibrationComponent].beam_inclination.max"])
        inc = np.linspace(mn, mx, height, dtype=np.float32)

    return inc[::-1]

def beam_azimuths(calib_row: dict[str, Any], width: int) -> np.ndarray:
    return np.linspace(np.pi, -np.pi, width, endpoint=False, dtype=np.float32)

# ----------------------------
# Range image -> points + labels
# ----------------------------
def range_image_to_points(
    range_image: np.ndarray,  # [H,W,4] float32
    inclinations: np.ndarray, # [H] float32
    azimuths: np.ndarray,     # [W] float32
    extrinsic: np.ndarray,    # [4,4] float64
    range_channel: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    ranges = range_image[:, :, range_channel]
    valid = (ranges > 0) & np.isfinite(ranges) & (ranges < 1e4)

    in2d = inclinations[:, None] # [H,1]
    az2d = azimuths[None, :]     # [1,W]

    cos_in = np.cos(in2d)
    sin_in = np.sin(in2d)
    cos_az = np.cos(az2d)
    sin_az = np.sin(az2d)

    xs = (ranges * cos_in * cos_az)[valid]
    ys = (ranges * cos_in * sin_az)[valid]
    zs = (ranges * sin_in)[valid]
    points_sensor = np.stack([xs, ys, zs], axis=1) # [N,3] in sensor frame

    ones = np.ones((points_sensor.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points_sensor.astype(np.float64), ones], axis=1) # [N,4]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        points_vehicle = (points_h @ extrinsic.T)[:, :3].astype(np.float32)     # [N,3]
    return points_vehicle, valid


def labels_from_segmentation(
    seg_image: np.ndarray,  # [H,W,2] int32
    valid_mask: np.ndarray, # [H,W] bool
) -> np.ndarray:
    return seg_image[:, :, SEMANTIC_CHANNEL][valid_mask].astype(np.int64)


def reconstruct_points_and_labels(
    lidar_row: dict[str, Any],
    seg_row: dict[str, Any],
    calib_row: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    extrinsic = extrinsic_matrix(calib_row)

    def process_one_return(
        ri_values_key: str,
        ri_shape_key: str,
        seg_values_key: str,
        seg_shape_key: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        ri_values = lidar_row.get(ri_values_key)
        ri_shape = lidar_row.get(ri_shape_key)
        seg_values = seg_row.get(seg_values_key)
        seg_shape = seg_row.get(seg_shape_key)

        if not ri_values or not seg_values:
            return np.zeros((0, 3), np.float32), np.zeros((0,), np.int64)

        range_image = reshape_array(ri_values, ri_shape, np.float32)  # [H,W,4]
        seg_image = reshape_array(seg_values, seg_shape, np.int32)    # [H,W,2]
        inclinations = beam_inclinations(calib_row, range_image.shape[0])
        azimuths = beam_azimuths(calib_row, range_image.shape[1])

        points, valid_mask = range_image_to_points(range_image, inclinations, azimuths, extrinsic, range_channel=0)
        if points.shape[0] == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0,), np.int64)

        labels = labels_from_segmentation(seg_image, valid_mask)
        return points, labels

    p1, y1 = process_one_return(
        "[LiDARComponent].range_image_return1.values",
        "[LiDARComponent].range_image_return1.shape",
        "[LiDARSegmentationLabelComponent].range_image_return1.values",
        "[LiDARSegmentationLabelComponent].range_image_return1.shape",
    )
    p2, y2 = process_one_return(
        "[LiDARComponent].range_image_return2.values",
        "[LiDARComponent].range_image_return2.shape",
        "[LiDARSegmentationLabelComponent].range_image_return2.values",
        "[LiDARSegmentationLabelComponent].range_image_return2.shape",
    )

    points = np.concatenate([p1, p2], axis=0) if (p1.size or p2.size) else np.zeros((0, 3), np.float32)
    labels = np.concatenate([y1, y2], axis=0) if (y1.size or y2.size) else np.zeros((0,), np.int64)
    return points, labels


# ----------------------------
# Parquet indexing / caching
# ----------------------------
def _map_segment_files(directory: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(directory.glob("*.parquet"))}


def _build_labeled_frame_index(split_root: Path) -> pl.DataFrame:
    seg_keys = (
        pl.scan_parquet(str(split_root / "lidar_segmentation" / "*.parquet"))
        .filter(pl.col(LASER_COL) == PRIMARY_LASER_ID)
        .select(SEGMENT_COL, TIMESTAMP_COL)
        .unique()
    )
    lidar_keys = (
        pl.scan_parquet(str(split_root / "lidar" / "*.parquet"))
        .filter(pl.col(LASER_COL) == PRIMARY_LASER_ID)
        .select(SEGMENT_COL, TIMESTAMP_COL)
        .unique()
    )
    return (
        lidar_keys.join(seg_keys, on=[SEGMENT_COL, TIMESTAMP_COL], how="inner")
        .sort([SEGMENT_COL, TIMESTAMP_COL])
        .collect()
    )


@dataclass
class _SegmentCacheEntry:
    lidar_by_timestamp: dict[int, dict[str, Any]]
    seg_by_timestamp: dict[int, dict[str, Any]]
    calib_row: dict[str, Any]


class WaymoLidarDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Path],
        segments: Optional[Sequence[str]] = None,
        frame_index: Optional[pl.DataFrame] = None,
        num_points: int = 16384,
        max_cached_segments: int = 4,
        seed: int = 0,
    ) -> None:
        self.path = Path(path)
        self.num_points = int(num_points)
        self.rng = np.random.default_rng(seed)

        self.lidar_files = _map_segment_files(self.path / "lidar")
        self.seg_files = _map_segment_files(self.path / "lidar_segmentation")
        self.calib_files = _map_segment_files(self.path / "lidar_calibration")

        if frame_index is None:
            frame_index = _build_labeled_frame_index(self.path)
        else:
            frame_index = frame_index.select(SEGMENT_COL, TIMESTAMP_COL).sort([SEGMENT_COL, TIMESTAMP_COL])

        if segments is not None:
            segset = set(str(s) for s in segments)
            frame_index = frame_index.filter(pl.col(SEGMENT_COL).is_in(segset))

        self.records: list[tuple[str, int]] = [
            (str(seg), int(ts)) for seg, ts in frame_index.select(SEGMENT_COL, TIMESTAMP_COL).iter_rows()
        ]

        self._cache: OrderedDict[str, _SegmentCacheEntry] = OrderedDict()
        self.max_cached_segments = max(1, int(max_cached_segments))

    def __len__(self) -> int:
        return len(self.records)

    def _load_segment(self, segment: str) -> _SegmentCacheEntry:
        if segment in self._cache:
            entry = self._cache.pop(segment)
            self._cache[segment] = entry
            return entry

        lidar_path = self.lidar_files.get(segment)
        seg_path = self.seg_files.get(segment)
        calib_path = self.calib_files.get(segment)
        if lidar_path is None or seg_path is None or calib_path is None:
            raise KeyError(f"Missing parquet for segment {segment}")

        lidar_df = (
            pl.read_parquet(lidar_path)
            .filter(pl.col(LASER_COL) == PRIMARY_LASER_ID)
        )
        seg_df = (
            pl.read_parquet(seg_path)
            .filter(pl.col(LASER_COL) == PRIMARY_LASER_ID)
        )
        calib_df = (
            pl.read_parquet(calib_path)
            .filter(pl.col(LASER_COL) == PRIMARY_LASER_ID)
        )

        if calib_df.height == 0:
            raise KeyError(f"Missing calibration for LiDAR {PRIMARY_LASER_ID} in segment {segment}")
        calib_row = calib_df.row(0, named=True)

        # Build timestamp -> row mapping
        lidar_by_timestamp = {int(r[TIMESTAMP_COL]): r for r in lidar_df.iter_rows(named=True)}
        seg_by_timestamp = {int(r[TIMESTAMP_COL]): r for r in seg_df.iter_rows(named=True)}

        entry = _SegmentCacheEntry(
            lidar_by_timestamp=lidar_by_timestamp,
            seg_by_timestamp=seg_by_timestamp,
            calib_row=calib_row,
        )
        self._cache[segment] = entry
        while len(self._cache) > self.max_cached_segments:
            self._cache.popitem(last=False)
        return entry

    def _sample_or_pad(self, points: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target = self.num_points
        count = points.shape[0]

        if count == 0:
            return (
                np.zeros((target, 3), np.float32),
                np.full((target,), -1, np.int64),
                np.zeros((target,), bool),
            )

        if count >= target:
            idx = self.rng.choice(count, size=target, replace=False)
            return points[idx], labels[idx], np.ones((target,), bool)

        order = self.rng.permutation(count)
        out_p = np.zeros((target, 3), np.float32)
        out_y = np.full((target,), -1, np.int64)
        mask = np.zeros((target,), bool)

        out_p[:count] = points[order]
        out_y[:count] = labels[order]
        mask[:count] = True
        return out_p, out_y, mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        segment, timestamp = self.records[index]
        entry = self._load_segment(segment)

        lidar_row = entry.lidar_by_timestamp.get(timestamp)
        seg_row = entry.seg_by_timestamp.get(timestamp)
        if lidar_row is None or seg_row is None:
            raise IndexError(f"Missing labeled frame for segment={segment} ts={timestamp}")

        points, labels = reconstruct_points_and_labels(
            lidar_row=lidar_row,
            seg_row=seg_row,
            calib_row=entry.calib_row
        )

        finite = np.isfinite(points).all(axis=1)
        keep = (labels >= 0) & finite
        points = points[keep]
        labels = labels[keep]

        points, labels, mask = self._sample_or_pad(points, labels)

        return {
            "points": torch.from_numpy(points).float(),   # [N,3]
            "labels": torch.from_numpy(labels).long(),    # [N]
            "mask": torch.from_numpy(mask),               # [N] bool
            "segment_context_name": segment,
            "frame_timestamp_micros": timestamp,
        }
