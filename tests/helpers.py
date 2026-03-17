import json
from pathlib import Path

from lightning.pytorch.loggers import CSVLogger
import numpy as np


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class FakeWandbTable:
    def __init__(self, columns=None, data=None, rows=None, **_kwargs) -> None:
        self.columns = list(columns or [])
        self.data = list(data or rows or [])


class FakeWandbObject3D:
    def __init__(self, data_or_path, caption=None, **_kwargs) -> None:
        self.data = data_or_path
        self.caption = caption


class FakeWandbRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict, int | None]] = []
        self.finished = False
        self.hparams: list[dict] = []

    def log(self, payload: dict, step: int | None = None) -> None:
        self.logged.append((dict(payload), step))

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self.log(metrics, step=step)

    def log_hparams(self, params: dict) -> None:
        self.hparams.append(dict(params))

    def save(self) -> None:
        return None

    def finish(self) -> None:
        self.finished = True


class FakeWandbLogger(CSVLogger):
    instances: list["FakeWandbLogger"] = []

    def __init__(
        self,
        *,
        project: str,
        entity=None,
        name: str | None = None,
        save_dir: str | None = None,
        id: str | None = None,
        tags=None,
        job_type: str | None = None,
        log_model: bool = False,
    ) -> None:
        super().__init__(save_dir=str(save_dir or "."), name=str(name or "wandb"))
        self.project = project
        self.entity = entity
        self.id = id
        self.tags = list(tags or [])
        self.job_type = job_type
        self.log_model = log_model
        self._experiment = FakeWandbRun()
        FakeWandbLogger.instances.append(self)

    @property
    def experiment(self) -> FakeWandbRun:
        return self._experiment

    def log_hyperparams(self, params) -> None:
        self._experiment.log_hparams(dict(params))


class FakeWandbModule:
    Table = FakeWandbTable
    Object3D = FakeWandbObject3D

    @staticmethod
    def plot_table(*, vega_spec_name: str, data_table, fields: dict, string_fields=None, split_table: bool = False):
        return {
            "vega_spec_name": vega_spec_name,
            "table": data_table,
            "fields": dict(fields),
            "string_fields": dict(string_fields or {}),
            "split_table": bool(split_table),
        }


def build_synthetic_preprocessed_roots(base_dir: Path) -> tuple[Path, Path]:
    range_root = base_dir / "range_images"
    point_root = base_dir / "point_clouds"
    _write_json(range_root / "meta.json", {"representation": "range_images"})
    _write_json(point_root / "meta.json", {"representation": "point_clouds_dense"})

    records = [
        {"source_subdir": "training", "segments": ["segment_train"]},
        {"source_subdir": "validation", "segments": ["segment_val"]},
    ]
    _write_json(range_root / "segment_source.json", records)
    _write_json(point_root / "segment_source.json", records)

    for root in (range_root, point_root):
        _write_json(root / "classes.json", {"0": "undefined", "1": "car", "2": "pedestrian", "3": "cyclist"})

    for idx, segment in enumerate(("segment_train", "segment_val")):
        label_offset = idx + 1
        _build_range_segment(range_root / "segments" / segment, timestamp=100 + idx, label_offset=label_offset)
        _build_point_segment(point_root / "segments" / segment, timestamp=100 + idx, label_offset=label_offset)

    return point_root, range_root


def _build_range_segment(segment_dir: Path, *, timestamp: int, label_offset: int) -> None:
    segment_dir.mkdir(parents=True, exist_ok=True)
    h, w = 4, 4
    base = np.zeros((1, h, w, 4), dtype=np.float32)
    base[0, :, :, 0] = np.array(
        [
            [1.0, 1.5, 2.0, 0.0],
            [0.0, 2.0, 2.5, 1.0],
            [3.0, 3.5, 0.0, 2.0],
            [4.0, 4.5, 5.0, 5.5],
        ],
        dtype=np.float32,
    )
    base[0, :, :, 1] = 0.5 + label_offset
    base[0, :, :, 2] = 1.5 + label_offset
    base[0, :, :, 3] = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    ri2 = base.copy()
    ri2[0, :, :, 0] += 0.25

    seg = np.zeros((1, h, w, 2), dtype=np.int16)
    seg[0, :, :, 1] = np.array(
        [
            [label_offset, label_offset, 0, label_offset],
            [label_offset, label_offset + 1, label_offset + 1, label_offset],
            [label_offset, label_offset, label_offset, label_offset + 1],
            [label_offset + 1, label_offset + 1, label_offset, label_offset],
        ],
        dtype=np.int16,
    )
    class_counts = np.zeros((1, 2, 4), dtype=np.int64)
    valid1 = (base[0, :, :, 0] > 0.0) & (base[0, :, :, 3] <= 0.0) & (seg[0, :, :, 1] > 0)
    valid2 = (ri2[0, :, :, 0] > 0.0) & (ri2[0, :, :, 3] <= 0.0) & (seg[0, :, :, 1] > 0)
    class_counts[0, 0] = np.bincount(seg[0, :, :, 1][valid1], minlength=4)[:4]
    class_counts[0, 1] = np.bincount(seg[0, :, :, 1][valid2], minlength=4)[:4]

    np.save(segment_dir / "ri1.npy", base)
    np.save(segment_dir / "ri2.npy", ri2)
    np.save(segment_dir / "seg1.npy", seg)
    np.save(segment_dir / "seg2.npy", seg)
    np.save(segment_dir / "timestamps.npy", np.array([timestamp], dtype=np.int64))
    np.save(segment_dir / "class_counts.npy", class_counts)


def _build_point_segment(segment_dir: Path, *, timestamp: int, label_offset: int) -> None:
    segment_dir.mkdir(parents=True, exist_ok=True)
    h, w = 4, 4
    xyz = np.zeros((1, 2, h, w, 3), dtype=np.float32)
    feat = np.zeros((1, 2, h, w, 2), dtype=np.float32)
    semantic = np.full((1, 2, h, w), -1, dtype=np.int16)
    valid_geometry = np.zeros((1, 2, h, w), dtype=bool)
    valid_label = np.zeros((1, 2, h, w), dtype=bool)

    for ret in range(2):
        for y in range(h):
            for x in range(w):
                xyz[0, ret, y, x] = np.array([x, y, ret], dtype=np.float32) + 0.1 * label_offset
                feat[0, ret, y, x] = np.array([0.5 + ret, 1.0 + x], dtype=np.float32)
                valid_geometry[0, ret, y, x] = not (y == 1 and x == 0)
                semantic[0, ret, y, x] = label_offset if (x + y + ret) % 3 else label_offset + 1
                valid_label[0, ret, y, x] = valid_geometry[0, ret, y, x] and semantic[0, ret, y, x] > 0 and not (y == 1 and x == 3)
    class_counts = np.zeros((1, 2, 4), dtype=np.int64)
    for ret in range(2):
        class_counts[0, ret] = np.bincount(semantic[0, ret][valid_label[0, ret]], minlength=4)[:4]

    np.save(segment_dir / "xyz.npy", xyz)
    np.save(segment_dir / "feat.npy", feat)
    np.save(segment_dir / "semantic.npy", semantic)
    np.save(segment_dir / "valid_geometry.npy", valid_geometry)
    np.save(segment_dir / "valid_label.npy", valid_label)
    np.save(segment_dir / "timestamps.npy", np.array([timestamp], dtype=np.int64))
    np.save(segment_dir / "class_counts.npy", class_counts)
