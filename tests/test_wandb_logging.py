import tempfile
import unittest
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import torch

from src.data import PreprocessedPointCloudDataset, WaymoLidarDataModule
from src.runtime.wandb_logging import WandbSegmentationCallback, build_audit_source, confusion_table_rows, log_audit_pointclouds, select_audit_examples, stage_metric_dict
from tests.helpers import FakeWandbLogger, FakeWandbModule, build_synthetic_preprocessed_roots


class WandbLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmpdir.name)
        self.point_root, _ = build_synthetic_preprocessed_roots(self.base_dir)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _duplicate_point_segment_frames(self, segment: str, timestamps: list[int]) -> None:
        segment_dir = self.point_root / "segments" / segment
        for name in ("xyz", "feat", "semantic", "valid_geometry", "valid_label", "class_counts"):
            path = segment_dir / f"{name}.npy"
            data = np.load(path)
            tiled = np.repeat(data, repeats=len(timestamps), axis=0)
            np.save(path, tiled)
        np.save(segment_dir / "timestamps.npy", np.asarray(timestamps, dtype=np.int64))

    def test_confusion_rows_exclude_class_zero(self) -> None:
        confusion = np.array(
            [
                [5, 1, 0],
                [2, 3, 4],
                [0, 6, 7],
            ],
            dtype=np.int64,
        )
        rows, names = confusion_table_rows(confusion, ["undefined", "car", "pedestrian"])

        self.assertEqual(names, ["car", "pedestrian"])
        self.assertEqual(rows[0], ["car", "car", 3])
        self.assertEqual(rows[1], ["car", "pedestrian", 4])

    def test_stage_metric_dict_uses_class_names(self) -> None:
        payload = {
            "metrics": {
                "loss": 1.0,
                "acc": 0.5,
                "mIoU": 0.25,
                "IoU_class_1": 0.3,
                "IoU_class_2": 0.2,
            },
            "confusion_matrix": [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
        }
        metrics = stage_metric_dict("val", payload, ["undefined", "car", "pedestrian"], prefix="final")

        self.assertIn("final_val_mIoU", metrics)
        self.assertIn("final_val_IoU_car", metrics)
        self.assertIn("final_val_IoU_pedestrian", metrics)

    def test_select_audit_examples_returns_exact_requested_count(self) -> None:
        dataset = PreprocessedPointCloudDataset(self.point_root, source_subdirs="validation", num_points=8, seed=0)
        examples = select_audit_examples(dataset, count=4)

        self.assertEqual(len(examples), 4)
        self.assertTrue(all(example.segment == "segment_val" for example in examples))
        self.assertTrue(all(example.timestamp == 101 for example in examples))

    def test_build_audit_source_uses_explicit_point_data_dir_for_range_data(self) -> None:
        relocated_root = self.base_dir / "custom_layout" / "range_images"
        relocated_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.point_root.parent / "range_images", relocated_root)

        datamodule = WaymoLidarDataModule(
            data_dir=relocated_root,
            point_data_dir=self.point_root,
            range_data_dir=relocated_root,
            representation="range_images",
            batch_size=1,
            train_subdirs="training",
            val_subdirs="validation",
            test_subdirs="validation",
            num_workers=0,
        )
        datamodule.setup("fit")

        source = build_audit_source(datamodule, split="val", count=1)
        self.assertIsNotNone(source)
        self.assertEqual(source.point_dataset.path, self.point_root)

    def test_build_audit_source_matches_datamodule_num_points(self) -> None:
        datamodule = WaymoLidarDataModule(
            data_dir=self.point_root,
            representation="point_clouds",
            batch_size=1,
            num_points=5,
            train_subdirs="training",
            val_subdirs="validation",
            test_subdirs="validation",
            num_workers=0,
        )
        datamodule.setup("fit")

        source = build_audit_source(datamodule, split="val", count=1)

        self.assertIsNotNone(source)
        self.assertEqual(source.point_dataset.num_points, 5)

    def test_log_audit_pointclouds_uses_sampled_point_frames_for_point_models(self) -> None:
        datamodule = WaymoLidarDataModule(
            data_dir=self.point_root,
            representation="point_clouds",
            batch_size=1,
            num_points=5,
            train_subdirs="training",
            val_subdirs="validation",
            test_subdirs="validation",
            num_workers=0,
        )
        datamodule.setup("fit")
        source = build_audit_source(datamodule, split="val", count=1)
        logger = FakeWandbLogger(project="test", save_dir=str(self.base_dir))
        observed_counts: list[int] = []

        class DummyModel:
            def predict_segmented_pointcloud(self, *, point_frame, range_frame=None):
                _ = range_frame
                count = int(point_frame["xyz"].reshape(-1, 3).shape[0])
                observed_counts.append(count)
                labels = point_frame["labels"].reshape(-1).astype(np.int64, copy=False)
                return {
                    "points_xyz": point_frame["xyz"].reshape(-1, 3).astype(np.float32, copy=False),
                    "pred_labels": labels,
                    "true_labels": labels,
                    "valid_label": point_frame["valid_label"].reshape(-1).astype(bool, copy=False),
                }

        with patch("src.runtime.wandb_logging._import_wandb", return_value=FakeWandbModule):
            log_audit_pointclouds(
                logger,
                DummyModel(),
                source,
                step=7,
            )

        self.assertEqual(observed_counts, [5])
        self.assertEqual(len(logger.experiment.logged), 1)

    def test_build_audit_source_uses_effective_validation_subset(self) -> None:
        self._duplicate_point_segment_frames("segment_val", [101, 102, 103])
        datamodule = WaymoLidarDataModule(
            data_dir=self.point_root,
            representation="point_clouds",
            batch_size=1,
            num_points=5,
            train_subdirs="training",
            val_subdirs="validation",
            test_subdirs="validation",
            num_workers=0,
            val_samples_per_segment=1,
        )
        datamodule.setup("fit")

        source = build_audit_source(datamodule, split="val", count=3)

        self.assertIsNotNone(source)
        self.assertEqual({example.timestamp for example in source.examples}, {101})

    def test_validation_epoch_end_logs_cached_predictions_without_reinference(self) -> None:
        datamodule = WaymoLidarDataModule(
            data_dir=self.point_root,
            representation="point_clouds",
            batch_size=1,
            num_points=5,
            train_subdirs="training",
            val_subdirs="validation",
            test_subdirs="validation",
            num_workers=0,
        )
        datamodule.setup("fit")
        logger = FakeWandbLogger(project="test", save_dir=str(self.base_dir))
        callback = WandbSegmentationCallback(log_pointcloud_count=1)

        class DummyTrainer:
            def __init__(self, datamodule, logger) -> None:
                self.datamodule = datamodule
                self.logger = logger
                self.global_step = 9
                self.sanity_checking = False
                self.is_global_zero = True

        class DummyModel:
            def __init__(self) -> None:
                self._class_names = []

            @property
            def class_names(self) -> list[str]:
                return list(self._class_names)

            def set_class_names(self, class_names: list[str]) -> None:
                self._class_names = list(class_names)

            def get_confusion_matrix(self, stage: str) -> torch.Tensor:
                _ = stage
                return torch.zeros((23, 23), dtype=torch.int64)

            def predict_segmented_pointcloud(self, *, point_frame, range_frame=None):
                _ = point_frame
                _ = range_frame
                raise AssertionError("validation audit should reuse cached predictions")

        trainer = DummyTrainer(datamodule, logger)
        model = DummyModel()
        batch = next(iter(datamodule.val_dataloader()))
        outputs = {
            "loss": torch.tensor(0.0),
            "preds": batch["labels"].clone(),
            "labels": batch["labels"].clone(),
            "metric_mask": batch["valid_label"].clone(),
            "batch_size": int(batch["labels"].shape[0]),
        }

        with patch("src.runtime.wandb_logging._import_wandb", return_value=FakeWandbModule):
            callback.on_fit_start(trainer, model)
            callback.on_validation_epoch_start(trainer, model)
            callback.on_validation_batch_end(trainer, model, outputs, batch, batch_idx=0)
            callback.on_validation_epoch_end(trainer, model)

        logged_keys = {key for payload, _ in logger.experiment.logged for key in payload}
        self.assertIn("val/confusion_matrix", logged_keys)
        self.assertTrue(any(key.startswith("val/pointcloud_") for key in logged_keys))


if __name__ == "__main__":
    unittest.main()
