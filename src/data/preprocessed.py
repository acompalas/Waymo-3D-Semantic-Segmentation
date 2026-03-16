from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Optional, Sequence
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


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
            f"Skipping {len(missing)} segments missing from {segments_root}. First few: {missing[:5]}",
            RuntimeWarning,
            stacklevel=2,
        )
    if not records:
        raise ValueError(f"No frame records found under {segments_root} for selected segments.")
    return records, frames


def _progress_segments(segments: Sequence[str], *, desc: str):
    values = list(segments)
    if not sys.stderr.isatty() or len(values) <= 1:
        return values
    return tqdm(values, desc=desc, leave=False, unit="segment")


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
        _, frames = _build_records(self.segments_root, self.segment_names)
        self._set_frames(frames)

        self.max_cached_segments = max(1, int(max_cached_segments))
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.records)

    def resolve_frame_index(self, segment: str, timestamp: int) -> int:
        key = (str(segment), int(timestamp))
        if key not in self._frame_lookup:
            raise KeyError(f"Unknown frame key: {key}")
        return int(self._frame_lookup[key])

    def frames_for_segment(self, segment: str) -> list[tuple[int, int]]:
        return list(self._segment_frames.get(str(segment), []))

    def resolve_dataset_index(self, segment: str, timestamp: int) -> int:
        key = (str(segment), int(timestamp))
        if key not in self._dataset_index_lookup:
            raise KeyError(f"Unknown frame key: {key}")
        return int(self._dataset_index_lookup[key])

    def _set_frames(self, frames: Sequence[tuple[str, int, int]]) -> None:
        self._frames = [(str(seg), int(frame_idx), int(ts)) for seg, frame_idx, ts in frames]
        self.records = [(seg, ts) for seg, _, ts in self._frames]
        self.segment_names = sorted({seg for seg, _, _ in self._frames})
        self._frame_lookup = {(seg, ts): frame_idx for seg, frame_idx, ts in self._frames}
        self._dataset_index_lookup = {(seg, ts): dataset_idx for dataset_idx, (seg, _, ts) in enumerate(self._frames)}
        self._segment_frames: dict[str, list[tuple[int, int]]] = {}
        for seg, frame_idx, ts in self._frames:
            self._segment_frames.setdefault(seg, []).append((frame_idx, ts))


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
        entry = _RangeSegmentCacheEntry(
            ri1=np.load(segment_dir / "ri1.npy", mmap_mode="r"),
            ri2=np.load(segment_dir / "ri2.npy", mmap_mode="r"),
            seg1=np.load(segment_dir / "seg1.npy", mmap_mode="r") if (segment_dir / "seg1.npy").exists() else None,
            seg2=np.load(segment_dir / "seg2.npy", mmap_mode="r") if (segment_dir / "seg2.npy").exists() else None,
        )
        self._cache[segment] = entry
        while len(self._cache) > self.max_cached_segments:
            self._cache.popitem(last=False)
        return entry

    def get_frame_data(self, segment: str, timestamp: int) -> dict[str, Any]:
        frame_idx = self.resolve_frame_index(segment, timestamp)
        entry = self._load_segment(segment)

        ri_frame = np.stack([entry.ri1[frame_idx], entry.ri2[frame_idx]], axis=0).astype(np.float32, copy=False)
        valid_geometry = ri_frame[:, :, :, 0] > 0.0
        is_in_nlz = ri_frame[:, :, :, 3] > 0.0

        if entry.seg1 is not None and entry.seg2 is not None:
            semantic = np.stack([entry.seg1[frame_idx, :, :, 1], entry.seg2[frame_idx, :, :, 1]], axis=0).astype(
                np.int64,
                copy=False,
            )
            valid_label = valid_geometry & (~is_in_nlz) & (semantic > 0)
        else:
            semantic = np.full(valid_geometry.shape, -1, dtype=np.int64)
            valid_label = np.zeros(valid_geometry.shape, dtype=bool)

        return {
            "range_images": ri_frame,
            "semantic": semantic,
            "valid_geometry": valid_geometry,
            "valid_label": valid_label,
            "is_in_nlz": is_in_nlz,
            "segment_context_name": str(segment),
            "frame_timestamp_micros": int(timestamp),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        segment, frame_idx, timestamp = self._frames[index]
        _ = frame_idx
        frame = self.get_frame_data(segment, timestamp)
        return {
            "range_images": torch.from_numpy(frame["range_images"]).float(),
            "semantic": torch.from_numpy(frame["semantic"]).long(),
            "valid_geometry": torch.from_numpy(frame["valid_geometry"]),
            "valid_label": torch.from_numpy(frame["valid_label"]),
            "is_in_nlz": torch.from_numpy(frame["is_in_nlz"]),
            "segment_context_name": frame["segment_context_name"],
            "frame_timestamp_micros": frame["frame_timestamp_micros"],
        }

    def compute_class_counts(self, num_classes: int) -> np.ndarray:
        counts = np.zeros(int(num_classes), dtype=np.int64)
        for segment in _progress_segments(self.segment_names, desc="Counting range-image class labels"):
            entry = self._load_segment(segment)
            if entry.seg1 is None or entry.seg2 is None:
                continue

            sem1 = entry.seg1[:, :, :, 1].astype(np.int64, copy=False)
            sem2 = entry.seg2[:, :, :, 1].astype(np.int64, copy=False)
            valid1 = (entry.ri1[:, :, :, 0] > 0.0) & (entry.ri1[:, :, :, 3] <= 0.0) & (sem1 > 0)
            valid2 = (entry.ri2[:, :, :, 0] > 0.0) & (entry.ri2[:, :, :, 3] <= 0.0) & (sem2 > 0)

            for classes in (sem1[valid1], sem2[valid2]):
                if classes.size == 0:
                    continue
                classes = classes[(classes >= 0) & (classes < int(num_classes))]
                if classes.size == 0:
                    continue
                counts += np.bincount(classes, minlength=int(num_classes))[: int(num_classes)]
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
        self.num_points = int(num_points)
        if self.num_points <= 0:
            raise ValueError(f"num_points must be > 0, got {self.num_points}")
        self.seed = int(seed)
        self.deterministic_sampling = bool(deterministic_sampling)
        self._cache: OrderedDict[str, _PointSegmentCacheEntry] = OrderedDict()
        super().__init__(
            path=path,
            source_subdirs=source_subdirs,
            segments=segments,
            max_cached_segments=max_cached_segments,
            seed=seed,
        )
        self._filter_frames_with_min_geometry()

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

    def get_dense_frame(self, segment: str, timestamp: int) -> dict[str, Any]:
        frame_idx = self.resolve_frame_index(segment, timestamp)
        entry = self._load_segment(segment)
        return {
            "xyz": entry.xyz[frame_idx].astype(np.float32, copy=False),
            "point_features": entry.feat[frame_idx].astype(np.float32, copy=False),
            "labels": entry.semantic[frame_idx].astype(np.int64, copy=False),
            "valid_geometry": entry.valid_geometry[frame_idx].astype(bool, copy=False),
            "valid_label": entry.valid_label[frame_idx].astype(bool, copy=False),
            "segment_context_name": str(segment),
            "frame_timestamp_micros": int(timestamp),
        }

    def _filter_frames_with_min_geometry(self) -> None:
        kept_frames: list[tuple[str, int, int]] = []
        dropped = 0
        for segment in self.segment_names:
            entry = self._load_segment(segment)
            valid_counts = entry.valid_geometry.reshape(entry.valid_geometry.shape[0], -1).sum(axis=1)
            for frame_idx, timestamp in self.frames_for_segment(segment):
                if int(valid_counts[frame_idx]) >= self.num_points:
                    kept_frames.append((segment, frame_idx, timestamp))
                else:
                    dropped += 1

        if not kept_frames:
            raise ValueError(
                f"No point-cloud frames under {self.path} contain at least {self.num_points} valid geometry points."
            )

        if dropped > 0:
            warnings.warn(
                f"Dropping {dropped} point-cloud frames with fewer than {self.num_points} valid geometry points.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._set_frames(kept_frames)

    def _sample_points(
        self,
        points: np.ndarray,
        point_features: np.ndarray,
        labels: np.ndarray,
        valid_label: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        target = self.num_points
        count = points.shape[0]
        if count < target:
            raise RuntimeError(
                f"Frame has only {count} valid geometry points, fewer than required num_points={target}."
            )

        preferred = np.flatnonzero(valid_label)
        if preferred.size >= target:
            idx = rng.choice(preferred, size=target, replace=False)
        elif preferred.size == count:
            idx = rng.choice(count, size=target, replace=False)
        else:
            other = np.flatnonzero(~valid_label)
            need = target - preferred.size
            supplement = rng.choice(other, size=need, replace=False)
            idx = np.concatenate([preferred.astype(np.int64, copy=False), supplement.astype(np.int64, copy=False)])
            idx = rng.permutation(idx)

        return (
            points[idx].astype(np.float32, copy=False),
            point_features[idx].astype(np.float32, copy=False),
            labels[idx].astype(np.int64, copy=False),
            valid_label[idx].astype(bool, copy=False),
        )

    def _frame_seed(self, segment: str, timestamp: int) -> int:
        payload = f"{segment}:{int(timestamp)}:{self.seed}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def __getitem__(self, index: int) -> dict[str, Any]:
        segment, frame_idx, timestamp = self._frames[index]
        _ = frame_idx
        frame = self.get_dense_frame(segment, timestamp)
        xyz = frame["xyz"].reshape(-1, 3)
        feat = frame["point_features"].reshape(-1, 2)
        semantic = frame["labels"].reshape(-1)
        valid_geometry = frame["valid_geometry"].reshape(-1)
        valid_label = frame["valid_label"].reshape(-1)

        keep = valid_geometry
        points = xyz[keep]
        point_features = feat[keep]
        labels = semantic[keep]
        label_mask = valid_label[keep]

        if self.deterministic_sampling:
            rng = np.random.default_rng(self._frame_seed(segment=segment, timestamp=timestamp))
        else:
            rng = self.rng

        points, point_features, labels, sampled_label_mask = self._sample_points(
            points=points,
            point_features=point_features,
            labels=labels,
            valid_label=label_mask,
            rng=rng,
        )

        return {
            "points": torch.from_numpy(points).float(),
            "point_features": torch.from_numpy(point_features).float(),
            "labels": torch.from_numpy(labels).long(),
            "valid_label": torch.from_numpy(sampled_label_mask),
            "segment_context_name": frame["segment_context_name"],
            "frame_timestamp_micros": frame["frame_timestamp_micros"],
        }

    def compute_class_counts(self, num_classes: int) -> np.ndarray:
        counts = np.zeros(int(num_classes), dtype=np.int64)
        for segment in _progress_segments(self.segment_names, desc="Counting point-cloud class labels"):
            entry = self._load_segment(segment)
            frame_indices = [frame_idx for frame_idx, _ in self.frames_for_segment(segment)]
            if not frame_indices:
                continue
            frame_sel = np.asarray(frame_indices, dtype=np.int64)
            sem = entry.semantic[frame_sel].reshape(-1).astype(np.int64, copy=False)
            valid_label = entry.valid_label[frame_sel].reshape(-1).astype(bool, copy=False)
            classes = sem[valid_label]
            if classes.size == 0:
                continue
            classes = classes[(classes >= 0) & (classes < int(num_classes))]
            if classes.size == 0:
                continue
            counts += np.bincount(classes, minlength=int(num_classes))[: int(num_classes)]
        return counts
