from pathlib import Path
from typing import Optional, Sequence

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from preprocessed_datasets import (
    PreprocessedPointCloudDataset,
    PreprocessedRangeImageDataset,
    source_segments_map,
)
from sampler import SceneShuffleBatchSampler


def _normalize_subdirs(values: Optional[Sequence[str] | str]) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        parts = [x.strip() for x in values.split(",")]
        return [x for x in parts if x]
    return [str(x) for x in values]


def _balanced_class_weights(counts: np.ndarray) -> torch.Tensor:
    counts = np.asarray(counts, dtype=np.float64)
    num_classes = int(counts.shape[0])
    total = float(counts.sum())
    weights = np.zeros_like(counts, dtype=np.float64)
    present = counts > 0
    if total > 0 and np.any(present):
        weights[present] = total / (num_classes * counts[present])
        # Normalize present-class mean to 1 for stable loss scale.
        weights[present] = weights[present] / np.mean(weights[present])
    return torch.tensor(weights.astype(np.float32))


class WaymoLidarDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        dataset_type: str = "point_clouds",
        batch_size: int = 8,
        num_points: int = 16384,
        num_classes: int = 23,
        train_subdirs: Sequence[str] | str = ("training",),
        val_subdirs: Sequence[str] | str = (),
        test_subdirs: Sequence[str] | str = ("validation",),
        val_fraction: float = 0.1,
        num_workers: int = 4,
        max_cached_segments: int = 4,
        seed: int = 0,
        drop_last_train: bool = True,
        worker_start_method: str = "spawn",
        balanced_class_weights: bool = True,
        train_segment_fraction: float = 1.0,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.dataset_type = str(dataset_type).strip().lower()
        self.batch_size = int(batch_size)
        self.num_points = int(num_points)
        self.num_classes = int(num_classes)
        self.train_subdirs = _normalize_subdirs(train_subdirs)
        self.val_subdirs = _normalize_subdirs(val_subdirs)
        self.test_subdirs = _normalize_subdirs(test_subdirs)
        self.val_fraction = float(val_fraction)
        self.num_workers = int(num_workers)
        self.max_cached_segments = int(max_cached_segments)
        self.seed = int(seed)
        self.drop_last_train = bool(drop_last_train)
        self.worker_start_method = str(worker_start_method)
        self.balanced_class_weights = bool(balanced_class_weights)
        self.train_segment_fraction = float(train_segment_fraction)
        if not (0.0 < self.train_segment_fraction <= 1.0):
            raise ValueError(
                f"train_segment_fraction must be in (0, 1], got {self.train_segment_fraction}"
            )

        self.train_dataset: Optional[PreprocessedPointCloudDataset | PreprocessedRangeImageDataset] = None
        self.val_dataset: Optional[PreprocessedPointCloudDataset | PreprocessedRangeImageDataset] = None
        self.test_dataset: Optional[PreprocessedPointCloudDataset | PreprocessedRangeImageDataset] = None

        self._train_sampler: Optional[SceneShuffleBatchSampler] = None
        self._val_sampler: Optional[SceneShuffleBatchSampler] = None
        self._test_sampler: Optional[SceneShuffleBatchSampler] = None

        self.class_weights: Optional[torch.Tensor] = None
        self.class_counts: Optional[torch.Tensor] = None

    def _dataset_cls(self):
        if self.dataset_type == "point_clouds":
            return PreprocessedPointCloudDataset
        if self.dataset_type == "range_images":
            return PreprocessedRangeImageDataset
        raise ValueError(
            f"Unsupported dataset_type='{self.dataset_type}'. Use one of: "
            "point_clouds, range_images."
        )

    def _segments_for_subdirs(self, subdirs: Sequence[str]) -> list[str]:
        by_source = source_segments_map(self.data_dir)
        segments: list[str] = []
        for subdir in subdirs:
            segments.extend(by_source.get(subdir, []))
        segments = sorted(set(segments))
        return segments

    def _subsample_train_segments(self, segments: Sequence[str]) -> list[str]:
        values = [str(s) for s in segments]
        if self.train_segment_fraction >= 1.0:
            return values
        if not values:
            return values
        keep = max(1, int(np.ceil(len(values) * self.train_segment_fraction)))
        rng = np.random.default_rng(self.seed + 17)
        order = rng.permutation(np.array(values, dtype=object))
        picked = sorted(str(x) for x in order[:keep].tolist())
        return picked

    def _make_dataset(
        self,
        *,
        subdirs: Optional[Sequence[str]] = None,
        segments: Optional[Sequence[str]] = None,
        seed: int,
        deterministic_sampling: bool = False,
    ) -> PreprocessedPointCloudDataset | PreprocessedRangeImageDataset:
        cls = self._dataset_cls()
        if cls is PreprocessedPointCloudDataset:
            return PreprocessedPointCloudDataset(
                path=self.data_dir,
                source_subdirs=subdirs,
                segments=segments,
                num_points=self.num_points,
                deterministic_sampling=deterministic_sampling,
                max_cached_segments=self.max_cached_segments,
                seed=seed,
            )
        return PreprocessedRangeImageDataset(
            path=self.data_dir,
            source_subdirs=subdirs,
            segments=segments,
            max_cached_segments=self.max_cached_segments,
            seed=seed,
        )

    def _split_train_val_segments(self) -> tuple[list[str], list[str]]:
        train_segments = self._segments_for_subdirs(self.train_subdirs)
        if not train_segments:
            raise ValueError(f"No train segments found for subdirs={self.train_subdirs}")

        if len(train_segments) <= 1:
            raise ValueError("Need at least 2 train segments to create a train/val split.")
        if self.val_fraction <= 0.0:
            raise ValueError(
                "val_fraction must be > 0 when val_subdirs is empty. "
                "Set val_subdirs explicitly or use a positive val_fraction."
            )

        rng = np.random.default_rng(self.seed)
        perm = np.array(train_segments, dtype=object)
        rng.shuffle(perm)
        val_count = int(round(len(perm) * self.val_fraction))
        val_count = max(1, min(val_count, len(perm) - 1))
        val_segments = sorted(str(x) for x in perm[:val_count].tolist())
        train_only = sorted(str(x) for x in perm[val_count:].tolist())
        return train_only, val_segments

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            if self.val_subdirs:
                train_segments = self._segments_for_subdirs(self.train_subdirs)
                train_segments = self._subsample_train_segments(train_segments)
                self.train_dataset = self._make_dataset(
                    subdirs=None,
                    segments=train_segments,
                    seed=self.seed,
                    deterministic_sampling=False,
                )
                self.val_dataset = self._make_dataset(
                    subdirs=self.val_subdirs,
                    segments=None,
                    seed=self.seed + 1,
                    deterministic_sampling=True,
                )
            else:
                train_segments, val_segments = self._split_train_val_segments()
                train_segments = self._subsample_train_segments(train_segments)
                self.train_dataset = self._make_dataset(
                    subdirs=None,
                    segments=train_segments,
                    seed=self.seed,
                    deterministic_sampling=False,
                )
                self.val_dataset = self._make_dataset(
                    subdirs=None,
                    segments=val_segments,
                    seed=self.seed + 1,
                    deterministic_sampling=True,
                )

            if self.train_dataset is not None and self.num_classes > 0 and hasattr(self.train_dataset, "compute_class_counts"):
                counts = self.train_dataset.compute_class_counts(self.num_classes)
                self.class_counts = torch.tensor(counts.astype(np.int64), dtype=torch.long)
                if self.balanced_class_weights:
                    self.class_weights = _balanced_class_weights(counts)
                else:
                    self.class_weights = None
            else:
                self.class_counts = None
                self.class_weights = None

        if stage in (None, "test"):
            if self.test_subdirs:
                self.test_dataset = self._make_dataset(
                    subdirs=self.test_subdirs,
                    segments=None,
                    seed=self.seed + 2,
                    deterministic_sampling=True,
                )
            else:
                self.test_dataset = None

    def set_epoch(self, epoch: int) -> None:
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)
        if self._val_sampler is not None:
            self._val_sampler.set_epoch(epoch)
        if self._test_sampler is not None:
            self._test_sampler.set_epoch(epoch)

    def _loader(
        self,
        dataset: PreprocessedPointCloudDataset | PreprocessedRangeImageDataset,
        sampler: SceneShuffleBatchSampler,
    ) -> DataLoader:
        loader_kwargs = {
            "dataset": dataset,
            "batch_sampler": sampler,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            loader_kwargs["multiprocessing_context"] = self.worker_start_method
        return DataLoader(**loader_kwargs)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting train_dataloader().")
        self._train_sampler = SceneShuffleBatchSampler(
            records=self.train_dataset.records,
            batch_size=self.batch_size,
            shuffle_scenes=True,
            shuffle_within_scene=True,
            drop_last=self.drop_last_train,
            seed=self.seed,
        )
        return self._loader(self.train_dataset, self._train_sampler)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting val_dataloader().")
        self._val_sampler = SceneShuffleBatchSampler(
            records=self.val_dataset.records,
            batch_size=self.batch_size,
            shuffle_scenes=False,
            shuffle_within_scene=False,
            drop_last=False,
            seed=self.seed + 101,
        )
        return self._loader(self.val_dataset, self._val_sampler)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before requesting test_dataloader().")
        self._test_sampler = SceneShuffleBatchSampler(
            records=self.test_dataset.records,
            batch_size=self.batch_size,
            shuffle_scenes=False,
            shuffle_within_scene=False,
            drop_last=False,
            seed=self.seed + 202,
        )
        return self._loader(self.test_dataset, self._test_sampler)
