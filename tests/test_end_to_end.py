import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np

from src.data import WaymoLidarDataModule
from src.main import run_render
from src.models import (
    LinearSVMPointClassifier,
    PointCloudDiffusionSegmenter,
    RangeImageDiffusionSegmenter,
    RangeImageUNetSegmenter,
)
from tests.helpers import build_synthetic_preprocessed_roots


def _fit_and_save(model, datamodule, root_dir: Path) -> Path:
    logger = CSVLogger(save_dir=str(root_dir), name="logs")
    checkpoint = ModelCheckpoint(monitor="val_mIoU", mode="max", save_top_k=1, save_last=True)
    trainer = L.Trainer(
        default_root_dir=str(root_dir),
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=logger,
        callbacks=[checkpoint],
        enable_progress_bar=False,
        log_every_n_steps=1,
    )
    trainer.fit(model=model, datamodule=datamodule)
    path = Path(checkpoint.last_model_path)
    if not path.exists():
        raise AssertionError("Expected Lightning to save a checkpoint.")
    return path


class EndToEndSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmpdir.name)
        self.point_root, self.range_root = build_synthetic_preprocessed_roots(self.base_dir)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _point_dm(self) -> WaymoLidarDataModule:
        return WaymoLidarDataModule(
            data_dir=self.point_root,
            representation="point_clouds",
            batch_size=1,
            num_points=8,
            num_classes=23,
            train_subdirs="training",
            val_subdirs="validation",
            test_subdirs="validation",
            num_workers=0,
            max_cached_segments=1,
        )

    def _range_dm(self) -> WaymoLidarDataModule:
        return WaymoLidarDataModule(
            data_dir=self.range_root,
            representation="range_images",
            batch_size=1,
            num_points=8,
            num_classes=23,
            train_subdirs="training",
            val_subdirs="validation",
            test_subdirs="validation",
            num_workers=0,
            max_cached_segments=1,
        )

    def test_checkpoint_round_trip_for_all_models(self) -> None:
        configs = [
            (LinearSVMPointClassifier(num_classes=23, knn_scales=(2,), knn_support_size=8, knn_query_chunk=8), self._point_dm()),
            (RangeImageUNetSegmenter(num_classes=23, base_channels=4, depth=2), self._range_dm()),
            (RangeImageDiffusionSegmenter(num_classes=23, base_channels=4, diffusion_steps=4), self._range_dm()),
            (
                PointCloudDiffusionSegmenter(
                    num_classes=23,
                    hidden_dim=32,
                    depth=2,
                    diffusion_steps=4,
                    backbone="pointnet",
                ),
                self._point_dm(),
            ),
        ]
        for idx, (model, datamodule) in enumerate(configs):
            ckpt_path = _fit_and_save(model, datamodule, self.base_dir / f"fit_{idx}")
            loaded = model.__class__.load_from_checkpoint(str(ckpt_path))
            self.assertIsInstance(loaded, model.__class__)

    def test_render_smoke_for_point_and_range_models(self) -> None:
        point_ckpt = _fit_and_save(
            LinearSVMPointClassifier(num_classes=23, knn_scales=(2,), knn_support_size=8, knn_query_chunk=8),
            self._point_dm(),
            self.base_dir / "render_point",
        )
        range_ckpt = _fit_and_save(
            RangeImageUNetSegmenter(num_classes=23, base_channels=4, depth=2),
            self._range_dm(),
            self.base_dir / "render_range",
        )

        fake_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch("src.main.render_point_cloud", return_value=fake_frame):
            run_render(
                argparse.Namespace(
                    command="render",
                    model="point_svm",
                    checkpoint=point_ckpt,
                    data_dir=self.point_root,
                    point_data_dir=self.point_root,
                    source_subdirs="validation",
                    segment="segment_val",
                    seed=0,
                    sampling_steps=None,
                    render_num_points=8,
                    output_gif=self.base_dir / "point.gif",
                    fps=2.0,
                    max_frames=1,
                    camera_preset="car_pov",
                    width=32,
                    height=32,
                    point_size=1.0,
                    device="cpu",
                )
            )
            run_render(
                argparse.Namespace(
                    command="render",
                    model="range_unet",
                    checkpoint=range_ckpt,
                    data_dir=self.range_root,
                    point_data_dir=self.point_root,
                    source_subdirs="validation",
                    segment="segment_val",
                    seed=0,
                    sampling_steps=None,
                    render_num_points=8,
                    output_gif=self.base_dir / "range.gif",
                    fps=2.0,
                    max_frames=1,
                    camera_preset="car_pov",
                    width=32,
                    height=32,
                    point_size=1.0,
                    device="cpu",
                )
            )

        self.assertTrue((self.base_dir / "point.gif").exists())
        self.assertTrue((self.base_dir / "range.gif").exists())

    def test_run_render_uses_shared_model_interface(self) -> None:
        point_ckpt = _fit_and_save(
            LinearSVMPointClassifier(num_classes=23, knn_scales=(2,), knn_support_size=8, knn_query_chunk=8),
            self._point_dm(),
            self.base_dir / "render_contract",
        )

        calls: list[tuple[bool, bool]] = []

        def fake_predict_segmented_pointcloud(*, point_frame, range_frame=None, sampling_steps=None):
            _ = sampling_steps
            calls.append((point_frame is not None, range_frame is not None))
            valid_geometry = point_frame["valid_geometry"].reshape(-1).astype(bool, copy=False)
            return {
                "points_xyz": point_frame["xyz"].reshape(-1, 3)[valid_geometry].astype(np.float32, copy=False),
                "pred_labels": point_frame["labels"].reshape(-1)[valid_geometry].astype(np.int64, copy=False),
                "true_labels": point_frame["labels"].reshape(-1)[valid_geometry].astype(np.int64, copy=False),
                "valid_label": point_frame["valid_label"].reshape(-1)[valid_geometry].astype(bool, copy=False),
            }

        fake_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch("src.main.render_point_cloud", return_value=fake_frame), patch.object(
            LinearSVMPointClassifier,
            "predict_segmented_pointcloud",
            side_effect=fake_predict_segmented_pointcloud,
        ):
            run_render(
                argparse.Namespace(
                    command="render",
                    model="point_svm",
                    checkpoint=point_ckpt,
                    data_dir=self.point_root,
                    point_data_dir=self.point_root,
                    source_subdirs="validation",
                    segment="segment_val",
                    seed=0,
                    sampling_steps=None,
                    render_num_points=8,
                    output_gif=self.base_dir / "contract.gif",
                    fps=2.0,
                    max_frames=1,
                    camera_preset="car_pov",
                    width=32,
                    height=32,
                    point_size=1.0,
                    device="cpu",
                )
            )

        self.assertEqual(calls, [(True, False)])


if __name__ == "__main__":
    unittest.main()
