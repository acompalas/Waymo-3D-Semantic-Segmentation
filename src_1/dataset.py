"""
dataset.py
==========
Range-image dataset for the LiDAR semantic segmentation diffusion model.

Memory strategy:
  - Frame index built lazily (no array data)
  - Optional per-segment LRU cache (default 1 segment) for fast frame lookup
  - Cache can be disabled (segment_cache_size=0) for minimum RAM usage
  - Segment-local sampling improves cache hit rate and disk locality

Split convention:
  Training : first N segments sorted alphabetically (up to 40)
  Test     : last 10 segments sorted alphabetically
"""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, Sampler

# ── Column names ───────────────────────────────────────────────────────────────
SEGMENT_COL   = "key.segment_context_name"
TIMESTAMP_COL = "key.frame_timestamp_micros"
LASER_COL     = "key.laser_name"
PRIMARY_LASER = 1
SEMANTIC_CH   = 1
NUM_CLASSES   = 23

CLASS_NAMES = [
    "TYPE_UNDEFINED", "TYPE_CAR", "TYPE_TRUCK", "TYPE_BUS",
    "TYPE_OTHER_VEHICLE", "TYPE_MOTORCYCLIST", "TYPE_BICYCLIST",
    "TYPE_PEDESTRIAN", "TYPE_SIGN", "TYPE_TRAFFIC_LIGHT", "TYPE_POLE",
    "TYPE_CONSTRUCTION_CONE", "TYPE_BICYCLE", "TYPE_MOTORCYCLE",
    "TYPE_BUILDING", "TYPE_VEGETATION", "TYPE_TREE_TRUNK", "TYPE_CURB",
    "TYPE_ROAD", "TYPE_LANE_MARKER", "TYPE_OTHER_GROUND",
    "TYPE_WALKABLE", "TYPE_SIDEWALK",
]

LIDAR_VALUES_COL = "[LiDARComponent].range_image_return1.values"
LIDAR_SHAPE_COL  = "[LiDARComponent].range_image_return1.shape"
SEG_VALUES_COL   = "[LiDARSegmentationLabelComponent].range_image_return1.values"
SEG_SHAPE_COL    = "[LiDARSegmentationLabelComponent].range_image_return1.shape"


# ── Helpers ────────────────────────────────────────────────────────────────────
def _reshape(values, shape, dtype):
    return np.asarray(values, dtype=dtype).reshape(tuple(int(x) for x in shape))


def _as_list(values):
    if hasattr(values, "to_list"):
        return values.to_list()
    return values


def _map_stems(directory: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(directory.glob("*.parquet"))}


def _build_frame_index(split_root: Path) -> pl.DataFrame:
    """Lazy join of lidar and segmentation key columns only — no array data."""
    lidar_keys = (
        pl.scan_parquet(str(split_root / "lidar" / "*.parquet"))
        .filter(pl.col(LASER_COL) == PRIMARY_LASER)
        .select(SEGMENT_COL, TIMESTAMP_COL)
        .unique()
    )
    seg_keys = (
        pl.scan_parquet(str(split_root / "lidar_segmentation" / "*.parquet"))
        .filter(pl.col(LASER_COL) == PRIMARY_LASER)
        .select(SEGMENT_COL, TIMESTAMP_COL)
        .unique()
    )
    return (
        lidar_keys
        .join(seg_keys, on=[SEGMENT_COL, TIMESTAMP_COL], how="inner")
        .sort([SEGMENT_COL, TIMESTAMP_COL])
        .collect()
    )


# ── Dataset ────────────────────────────────────────────────────────────────────
class WaymoRangeImageDataset(Dataset):
    """
    PyTorch Dataset returning range images + segmentation labels.

    Each __getitem__ reads exactly one row from the lidar parquet and one row
    from the segmentation parquet for the requested frame. No segment-level
    DataFrames are kept in memory between calls.

    Each __getitem__ returns:
      lidar  : (4, 64, 2650)  float32
      labels : (64, 2650)     int64
      valid  : (64, 2650)     bool
    """

    def __init__(
        self,
        path: str | Path,
        segments: Optional[Sequence[str]] = None,
        segment_cache_size: int = 1,
    ) -> None:
        self.root = Path(path)
        self.segment_cache_size = max(0, int(segment_cache_size))

        self.lidar_dir = self.root / "lidar"
        self.seg_dir   = self.root / "lidar_segmentation"

        self.lidar_files = _map_stems(self.lidar_dir)
        self.seg_files   = _map_stems(self.seg_dir)

        frame_index = _build_frame_index(self.root)

        if segments is not None:
            seg_set = set(str(s) for s in segments)
            frame_index = frame_index.filter(pl.col(SEGMENT_COL).is_in(seg_set))

        self.records: list[tuple[str, int]] = [
            (str(seg), int(ts))
            for seg, ts in frame_index.select(SEGMENT_COL, TIMESTAMP_COL).iter_rows()
        ]

        self.segment_to_indices: dict[str, list[int]] = {}
        for idx, (segment, _) in enumerate(self.records):
            self.segment_to_indices.setdefault(segment, []).append(idx)

        self._segment_cache: OrderedDict[str, _SegmentTables] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def _load_segment_tables(self, segment: str) -> "_SegmentTables":
        lidar_path = self.lidar_files.get(segment)
        seg_path   = self.seg_files.get(segment)
        if lidar_path is None or seg_path is None:
            raise KeyError(f"Missing parquet for segment: {segment}")

        lidar_df = (
            pl.read_parquet(
                str(lidar_path),
                columns=[TIMESTAMP_COL, LASER_COL, LIDAR_VALUES_COL, LIDAR_SHAPE_COL],
            )
            .filter(pl.col(LASER_COL) == PRIMARY_LASER)
            .select(TIMESTAMP_COL, LIDAR_VALUES_COL, LIDAR_SHAPE_COL)
        )
        seg_df = (
            pl.read_parquet(
                str(seg_path),
                columns=[TIMESTAMP_COL, LASER_COL, SEG_VALUES_COL, SEG_SHAPE_COL],
            )
            .filter(pl.col(LASER_COL) == PRIMARY_LASER)
            .select(TIMESTAMP_COL, SEG_VALUES_COL, SEG_SHAPE_COL)
        )

        lidar_index = {int(ts): i for i, ts in enumerate(lidar_df[TIMESTAMP_COL].to_list())}
        seg_index   = {int(ts): i for i, ts in enumerate(seg_df[TIMESTAMP_COL].to_list())}
        return _SegmentTables(
            lidar_df=lidar_df,
            seg_df=seg_df,
            lidar_index=lidar_index,
            seg_index=seg_index,
        )

    def _get_segment_tables(self, segment: str) -> "_SegmentTables":
        cached = self._segment_cache.pop(segment, None)
        if cached is not None:
            self._segment_cache[segment] = cached
            return cached

        loaded = self._load_segment_tables(segment)
        self._segment_cache[segment] = loaded

        while len(self._segment_cache) > self.segment_cache_size:
            self._segment_cache.popitem(last=False)

        return loaded

    def __getitem__(self, index: int) -> dict[str, Any]:
        segment, timestamp = self.records[index]

        if self.segment_cache_size > 0:
            tables = self._get_segment_tables(segment)
            lidar_i = tables.lidar_index.get(timestamp)
            seg_i   = tables.seg_index.get(timestamp)
            if lidar_i is None or seg_i is None:
                raise IndexError(f"Missing frame: segment={segment} ts={timestamp}")

            lidar_values = tables.lidar_df[LIDAR_VALUES_COL][lidar_i]
            lidar_shape  = tables.lidar_df[LIDAR_SHAPE_COL][lidar_i]
            seg_values   = tables.seg_df[SEG_VALUES_COL][seg_i]
            seg_shape    = tables.seg_df[SEG_SHAPE_COL][seg_i]
        else:
            lidar_path = self.lidar_files.get(segment)
            seg_path   = self.seg_files.get(segment)
            if lidar_path is None or seg_path is None:
                raise KeyError(f"Missing parquet for segment: {segment}")

            lidar_row = (
                pl.scan_parquet(str(lidar_path))
                .filter(
                    (pl.col(LASER_COL) == PRIMARY_LASER) &
                    (pl.col(TIMESTAMP_COL) == timestamp)
                )
                .select(LIDAR_VALUES_COL, LIDAR_SHAPE_COL)
                .collect()
            )
            seg_row = (
                pl.scan_parquet(str(seg_path))
                .filter(
                    (pl.col(LASER_COL) == PRIMARY_LASER) &
                    (pl.col(TIMESTAMP_COL) == timestamp)
                )
                .select(SEG_VALUES_COL, SEG_SHAPE_COL)
                .collect()
            )

            if lidar_row.height == 0 or seg_row.height == 0:
                raise IndexError(f"Missing frame: segment={segment} ts={timestamp}")

            lidar_values = lidar_row[LIDAR_VALUES_COL][0]
            lidar_shape  = lidar_row[LIDAR_SHAPE_COL][0]
            seg_values   = seg_row[SEG_VALUES_COL][0]
            seg_shape    = seg_row[SEG_SHAPE_COL][0]

        # ── Range image (H, W, 4) → (4, H, W) ────────────────────────────────
        ri = _reshape(
            _as_list(lidar_values),
            _as_list(lidar_shape),
            np.float32,
        )
        lidar = torch.from_numpy(ri.transpose(2, 0, 1).copy())   # (4, H, W)

        # ── Segmentation (H, W, 2) → semantic channel ─────────────────────────
        seg = _reshape(
            _as_list(seg_values),
            _as_list(seg_shape),
            np.int32,
        )
        labels = torch.from_numpy(seg[:, :, SEMANTIC_CH].astype(np.int64))  # (H, W)

        valid = lidar[0] > 0   # (H, W) bool

        return {
            "lidar":                  lidar,
            "labels":                 labels,
            "valid":                  valid,
            "segment_context_name":   segment,
            "frame_timestamp_micros": timestamp,
        }


@dataclass
class _SegmentTables:
    lidar_df: pl.DataFrame
    seg_df: pl.DataFrame
    lidar_index: dict[int, int]
    seg_index: dict[int, int]


class SegmentShuffleSampler(Sampler[int]):
    def __init__(self, dataset: WaymoRangeImageDataset, seed: int = 0):
        self.dataset = dataset
        self.segments = list(dataset.segment_to_indices.keys())
        self.indices_by_segment = dataset.segment_to_indices
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self):
        seg_perm = torch.randperm(len(self.segments), generator=self.generator).tolist()
        for seg_idx in seg_perm:
            seg = self.segments[seg_idx]
            segment_indices = self.indices_by_segment[seg]
            frame_perm = torch.randperm(len(segment_indices), generator=self.generator).tolist()
            for frame_idx in frame_perm:
                yield segment_indices[frame_idx]

    def __len__(self) -> int:
        return len(self.dataset)


# ── Split helper ───────────────────────────────────────────────────────────────
def get_train_dataset(
    data_root: str | Path,
    num_segments: int = 40,
    segment_cache_size: int = 1,
) -> "WaymoRangeImageDataset":
    """
    Return training dataset using the first num_segments segments
    (sorted alphabetically). Always reserves the last 10 for testing.
    """
    data_root = Path(data_root)
    all_stems = sorted(p.stem for p in (data_root / "lidar_segmentation").glob("*.parquet"))

    available  = len(all_stems)
    max_train  = max(0, available - 10)  # always reserve last 10 for test

    if num_segments > max_train:
        print(f"  WARNING: requested {num_segments} train segments but only {max_train} available without overlapping test set. Using {max_train}.")
        num_segments = max_train

    train_segs = all_stems[:num_segments]
    print(f"Train segments: {len(train_segs)} (of {max_train} available for training)")

    ds = WaymoRangeImageDataset(
        data_root,
        segments=train_segs,
        segment_cache_size=segment_cache_size,
    )
    print(f"  Train frames: {len(ds)}")
    return ds


def get_test_dataset(
    data_root: str | Path,
    num_segments: int = 10,
    segment_cache_size: int = 1,
) -> "WaymoRangeImageDataset":
    """
    Return test dataset always from the last 10 segments.
    num_segments controls how many of those 10 to use (max 10).
    """
    data_root = Path(data_root)
    all_stems = sorted(p.stem for p in (data_root / "lidar_segmentation").glob("*.parquet"))

    available    = len(all_stems)
    num_segments = min(num_segments, 10, available)

    test_segs = all_stems[-10:][-num_segments:] if num_segments < 10 else all_stems[-10:]
    print(f"Test segments: {len(test_segs)} (from last 10 of {available} total)")

    ds = WaymoRangeImageDataset(
        data_root,
        segments=test_segs,
        segment_cache_size=segment_cache_size,
    )
    print(f"  Test frames: {len(ds)}")
    return ds


def get_splits(
    data_root: str | Path,
    num_train: int = 40,
    num_test:  int = 10,
    segment_cache_size: int = 1,
) -> tuple["WaymoRangeImageDataset", "WaymoRangeImageDataset"]:
    """Convenience wrapper — returns (train_ds, test_ds)."""
    print("\n── Loading dataset ──")
    train_ds = get_train_dataset(data_root, num_train, segment_cache_size=segment_cache_size)
    test_ds  = get_test_dataset(data_root, num_test, segment_cache_size=segment_cache_size)
    return train_ds, test_ds
