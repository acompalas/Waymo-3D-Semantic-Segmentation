from collections import OrderedDict
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class SceneShuffleBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        records: Sequence[tuple[str, int]],
        batch_size: int,
        shuffle_scenes: bool = True,
        shuffle_within_scene: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.shuffle_scenes = bool(shuffle_scenes)
        self.shuffle_within_scene = bool(shuffle_within_scene)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        scene_to_indices: OrderedDict[str, list[int]] = OrderedDict()
        for idx, (segment, _) in enumerate(records):
            scene_to_indices.setdefault(str(segment), []).append(int(idx))
        self._scene_indices: list[list[int]] = list(scene_to_indices.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        total_batches = 0
        for idxs in self._scene_indices:
            count = len(idxs)
            if self.drop_last:
                total_batches += count // self.batch_size
            else:
                total_batches += (count + self.batch_size - 1) // self.batch_size
        return total_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)

        scene_order = np.arange(len(self._scene_indices), dtype=np.int64)
        if self.shuffle_scenes:
            rng.shuffle(scene_order)

        for scene_idx in scene_order:
            scene_frame_indices = np.asarray(self._scene_indices[int(scene_idx)], dtype=np.int64)
            if self.shuffle_within_scene:
                scene_frame_indices = rng.permutation(scene_frame_indices)

            scene_count = int(scene_frame_indices.shape[0])
            limit = scene_count - (scene_count % self.batch_size) if self.drop_last else scene_count
            for start in range(0, limit, self.batch_size):
                batch = scene_frame_indices[start : start + self.batch_size]
                if batch.shape[0] < self.batch_size and self.drop_last:
                    continue
                yield batch.tolist()
