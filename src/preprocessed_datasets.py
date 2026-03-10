from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_segment_source_records(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    records = _read_json(root / "segment_source.json")
    if not isinstance(records, list):
        raise ValueError(f"Expected list in {root / 'segment_source.json'}")
    return records


def source_segments_map(path: str | Path) -> dict[str, list[str]]:
    records = load_segment_source_records(path)
    by_source: dict[str, list[str]] = {}
    for rec in records:
        source = str(rec.get("source_subdir"))
        by_source[source] = [str(s) for s in rec.get("segments", [])]
    return by_source


def _normalize_str_list(values: Optional[Sequence[str] | str]) -> Optional[list[str]]:
    if values is None:
        return None
    if isinstance(values, str):
        items = [x.strip() for x in values.split(",")]
        return [x for x in items if x]
    return [str(x) for x in values]


def _select_segments(
    root: Path,
    records: list[dict[str, Any]],
    source_subdirs: Optional[Sequence[str] | str],
    segments: Optional[Sequence[str]],
) -> list[str]:
    source_filter = _normalize_str_list(source_subdirs)
    source_filter_set = None if source_filter is None else set(source_filter)

    selected: list[str] = []
    seen: set[str] = set()
    for rec in records:
        source = str(rec.get("source_subdir"))
        if source_filter_set is not None and source not in source_filter_set:
            continue
        for seg in rec.get("segments", []):
            seg_name = str(seg)
            if seg_name in seen:
                raise ValueError(f"Segment '{seg_name}' appears more than once in segment_source.json")
            seen.add(seg_name)
            selected.append(seg_name)

    if segments is not None:
        requested = {str(s) for s in segments}
        selected = [s for s in selected if s in requested]
        missing = sorted(requested.difference(selected))
        if missing:
            preview = missing[:10]
            raise ValueError(f"Requested segments not found for selection under {root}: {preview}")

    if not selected:
        raise ValueError(f"No segments selected under {root}")
    return selected


def _build_records(segments_root: Path, segment_names: Sequence[str]) -> tuple[list[tuple[str, int]], list[tuple[str, int, int]]]:
    records: list[tuple[str, int]] = []
    frames: list[tuple[str, int, int]] = []
    missing: list[str] = []
    for seg in segment_names:
        ts_path = segments_root / seg / "timestamps.npy"
        if not ts_path.exists():
            missing.append(seg)
            continue
        timestamps = np.load(ts_path)
        for frame_idx, ts in enumerate(timestamps.tolist()):
            ts_int = int(ts)
            records.append((seg, ts_int))
            frames.append((seg, int(frame_idx), ts_int))
    if missing:
        warnings.warn(
            f"Skipping {len(missing)} segments missing from {segments_root}. "
            f"First few: {missing[:5]}",
            RuntimeWarning,
            stacklevel=2,
        )
    if not records:
        raise ValueError(f"No frame records found under {segments_root} for selected segments.")
    return records, frames


class _BasePreprocessedDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        path: str | Path,
        source_subdirs: Optional[Sequence[str] | str] = None,
        segments: Optional[Sequence[str]] = None,
        max_cached_segments: int = 4,
        seed: int = 0,
    ) -> None:
        self.path = Path(path)
        self.meta = _read_json(self.path / "meta.json")
        self.segments_root = self.path / "segments"
        if not self.segments_root.is_dir():
            raise FileNotFoundError(f"Missing segments directory: {self.segments_root}")

        records = load_segment_source_records(self.path)
        self.segment_names = _select_segments(
            root=self.path,
            records=records,
            source_subdirs=source_subdirs,
            segments=segments,
        )
        self.records, self._frames = _build_records(self.segments_root, self.segment_names)
        # Keep only segments that actually produced frame records.
        self.segment_names = sorted({seg for seg, _, _ in self._frames})

        self.max_cached_segments = max(1, int(max_cached_segments))
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.records)


@dataclass
class _RangeSegmentCacheEntry:
    ri1: np.ndarray
    ri2: np.ndarray
    seg1: Optional[np.ndarray]
    seg2: Optional[np.ndarray]


class PreprocessedRangeImageDataset(_BasePreprocessedDataset):
    def __init__(
        self,
        path: str | Path,
        source_subdirs: Optional[Sequence[str] | str] = None,
        segments: Optional[Sequence[str]] = None,
        max_cached_segments: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__(
            path=path,
            source_subdirs=source_subdirs,
            segments=segments,
            max_cached_segments=max_cached_segments,
            seed=seed,
        )
        self._cache: OrderedDict[str, _RangeSegmentCacheEntry] = OrderedDict()

    def _load_segment(self, segment: str) -> _RangeSegmentCacheEntry:
        if segment in self._cache:
            entry = self._cache.pop(segment)
            self._cache[segment] = entry
            return entry

        segment_dir = self.segments_root / segment
        ri1 = np.load(segment_dir / "ri1.npy", mmap_mode="r")
        ri2 = np.load(segment_dir / "ri2.npy", mmap_mode="r")
        seg1_path = segment_dir / "seg1.npy"
        seg2_path = segment_dir / "seg2.npy"
        seg1 = np.load(seg1_path, mmap_mode="r") if seg1_path.exists() else None
        seg2 = np.load(seg2_path, mmap_mode="r") if seg2_path.exists() else None

        entry = _RangeSegmentCacheEntry(
            ri1=ri1,
            ri2=ri2,
            seg1=seg1,
            seg2=seg2,
        )
        self._cache[segment] = entry
        while len(self._cache) > self.max_cached_segments:
            self._cache.popitem(last=False)
        return entry

    def __getitem__(self, index: int) -> dict[str, Any]:
        segment, frame_idx, timestamp = self._frames[index]
        entry = self._load_segment(segment)

        ri_frame = np.stack([entry.ri1[frame_idx], entry.ri2[frame_idx]], axis=0).astype(np.float32, copy=False)
        # [R,H,W]
        valid_geometry = (ri_frame[:, :, :, 0] > 0.0)
        is_in_nlz = (ri_frame[:, :, :, 3] > 0.0)

        if entry.seg1 is not None and entry.seg2 is not None:
            semantic = np.stack([entry.seg1[frame_idx, :, :, 1], entry.seg2[frame_idx, :, :, 1]], axis=0).astype(
                np.int64, copy=False
            )
            valid_label = valid_geometry & (~is_in_nlz) & (semantic > 0)
        else:
            semantic = np.full(valid_geometry.shape, -1, dtype=np.int64)
            valid_label = np.zeros(valid_geometry.shape, dtype=bool)

        return {
            "range_images": torch.from_numpy(ri_frame).float(),  # [2,H,W,4]
            "semantic": torch.from_numpy(semantic).long(),  # [2,H,W]
            "valid_geometry": torch.from_numpy(valid_geometry),  # [2,H,W]
            "valid_label": torch.from_numpy(valid_label),  # [2,H,W]
            "is_in_nlz": torch.from_numpy(is_in_nlz),  # [2,H,W]
            "segment_context_name": segment,
            "frame_timestamp_micros": int(timestamp),
        }

    def compute_class_counts(self, num_classes: int) -> np.ndarray:
        counts = np.zeros(int(num_classes), dtype=np.int64)
        for segment in self.segment_names:
            entry = self._load_segment(segment)
            if entry.seg1 is None or entry.seg2 is None:
                continue

            sem1 = entry.seg1[:, :, :, 1].astype(np.int64, copy=False)
            sem2 = entry.seg2[:, :, :, 1].astype(np.int64, copy=False)
            valid1 = (entry.ri1[:, :, :, 0] > 0.0) & (entry.ri1[:, :, :, 3] <= 0.0) & (sem1 > 0)
            valid2 = (entry.ri2[:, :, :, 0] > 0.0) & (entry.ri2[:, :, :, 3] <= 0.0) & (sem2 > 0)

            cls1 = sem1[valid1]
            cls2 = sem2[valid2]
            for cls in (cls1, cls2):
                if cls.size == 0:
                    continue
                cls = cls[(cls >= 0) & (cls < int(num_classes))]
                if cls.size == 0:
                    continue
                counts += np.bincount(cls, minlength=int(num_classes))[: int(num_classes)]
        return counts


@dataclass
class _PointSegmentCacheEntry:
    xyz: np.ndarray
    feat: np.ndarray
    semantic: np.ndarray
    valid_geometry: np.ndarray
    valid_label: np.ndarray


class PreprocessedPointCloudDataset(_BasePreprocessedDataset):
    def __init__(
        self,
        path: str | Path,
        source_subdirs: Optional[Sequence[str] | str] = None,
        segments: Optional[Sequence[str]] = None,
        num_points: int = 16384,
        deterministic_sampling: bool = False,
        max_cached_segments: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__(
            path=path,
            source_subdirs=source_subdirs,
            segments=segments,
            max_cached_segments=max_cached_segments,
            seed=seed,
        )
        self.num_points = int(num_points)
        self.seed = int(seed)
        self.deterministic_sampling = bool(deterministic_sampling)
        self._cache: OrderedDict[str, _PointSegmentCacheEntry] = OrderedDict()

    def _load_segment(self, segment: str) -> _PointSegmentCacheEntry:
        if segment in self._cache:
            entry = self._cache.pop(segment)
            self._cache[segment] = entry
            return entry

        segment_dir = self.segments_root / segment
        entry = _PointSegmentCacheEntry(
            xyz=np.load(segment_dir / "xyz.npy", mmap_mode="r"),
            feat=np.load(segment_dir / "feat.npy", mmap_mode="r"),
            semantic=np.load(segment_dir / "semantic.npy", mmap_mode="r"),
            valid_geometry=np.load(segment_dir / "valid_geometry.npy", mmap_mode="r"),
            valid_label=np.load(segment_dir / "valid_label.npy", mmap_mode="r"),
        )
        self._cache[segment] = entry
        while len(self._cache) > self.max_cached_segments:
            self._cache.popitem(last=False)
        return entry

    def _sample_or_pad(
        self,
        points: np.ndarray,
        point_features: np.ndarray,
        labels: np.ndarray,
        valid_label: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        target = self.num_points
        count = points.shape[0]

        out_p = np.zeros((target, 3), dtype=np.float32)
        out_f = np.zeros((target, 2), dtype=np.float32)
        out_y = np.full((target,), -1, dtype=np.int64)
        out_geom = np.zeros((target,), dtype=bool)
        out_vlabel = np.zeros((target,), dtype=bool)

        if count == 0:
            return out_p, out_f, out_y, out_geom, out_vlabel

        if count >= target:
            idx = rng.choice(count, size=target, replace=False)
            out_p[:] = points[idx]
            out_f[:] = point_features[idx]
            out_y[:] = labels[idx]
            out_geom[:] = True
            out_vlabel[:] = valid_label[idx]
            return out_p, out_f, out_y, out_geom, out_vlabel

        order = rng.permutation(count)
        out_p[:count] = points[order]
        out_f[:count] = point_features[order]
        out_y[:count] = labels[order]
        out_geom[:count] = True
        out_vlabel[:count] = valid_label[order]
        return out_p, out_f, out_y, out_geom, out_vlabel

    def _frame_seed(self, segment: str, timestamp: int) -> int:
        payload = f"{segment}:{int(timestamp)}:{self.seed}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def __getitem__(self, index: int) -> dict[str, Any]:
        segment, frame_idx, timestamp = self._frames[index]
        entry = self._load_segment(segment)

        xyz = entry.xyz[frame_idx].reshape(-1, 3).astype(np.float32, copy=False)
        feat = entry.feat[frame_idx].reshape(-1, 2).astype(np.float32, copy=False)
        semantic = entry.semantic[frame_idx].reshape(-1).astype(np.int64, copy=False)
        valid_geometry = entry.valid_geometry[frame_idx].reshape(-1).astype(bool, copy=False)
        valid_label = entry.valid_label[frame_idx].reshape(-1).astype(bool, copy=False)

        keep = valid_geometry
        points = xyz[keep]
        point_features = feat[keep]
        labels = semantic[keep]
        vlabel = valid_label[keep]

        if self.deterministic_sampling:
            rng = np.random.default_rng(self._frame_seed(segment=segment, timestamp=timestamp))
        else:
            rng = self.rng

        points, point_features, labels, geom_mask, label_mask = self._sample_or_pad(
            points=points,
            point_features=point_features,
            labels=labels,
            valid_label=vlabel,
            rng=rng,
        )

        return {
            "points": torch.from_numpy(points).float(),  # [N,3]
            "point_features": torch.from_numpy(point_features).float(),  # [N,2] intensity, elongation
            "labels": torch.from_numpy(labels).long(),  # [N]
            "valid_geometry": torch.from_numpy(geom_mask),  # [N]
            "valid_label": torch.from_numpy(label_mask),  # [N]
            "segment_context_name": segment,
            "frame_timestamp_micros": int(timestamp),
        }

    def compute_class_counts(self, num_classes: int) -> np.ndarray:
        counts = np.zeros(int(num_classes), dtype=np.int64)
        for segment in self.segment_names:
            entry = self._load_segment(segment)
            sem = entry.semantic.reshape(-1).astype(np.int64, copy=False)
            vlabel = entry.valid_label.reshape(-1).astype(bool, copy=False)
            classes = sem[vlabel]
            if classes.size == 0:
                continue
            classes = classes[(classes >= 0) & (classes < int(num_classes))]
            if classes.size == 0:
                continue
            counts += np.bincount(classes, minlength=int(num_classes))[: int(num_classes)]
        return counts
