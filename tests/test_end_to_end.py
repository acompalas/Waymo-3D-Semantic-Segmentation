import argparse
import tempfile
import unittest
from pathlib import Path
import json
import os
from unittest.mock import patch

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np

from src.data import WaymoLidarDataModule
from src.main import run_evaluate, run_render, run_train
from src.models import (
    LinearSVMPointClassifier,
    PointCloudDiffusionSegmenter,
    RangeImageDiffusionSegmenter,
    RangeImageUNetSegmenter,
)
from src.runtime import get_model_spec
from src.tools.rendering import ensure_open3d_linux_env
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

    def test_train_writes_final_report_artifacts_and_uses_checkpoint_reload(self) -> None:
        output_dir = self.base_dir / "train_outputs"
        spec = get_model_spec("point_svm")
        with patch.object(spec.module_cls, "load_from_checkpoint", wraps=spec.module_cls.load_from_checkpoint) as load_mock:
            run_train(
                argparse.Namespace(
                    command="train",
                    model="point_svm",
                    data_dir=self.point_root,
                    train_subdirs="training",
                    val_subdirs="validation",
                    test_subdirs="validation",
                    batch_size=1,
                    num_points=8,
                    num_classes=23,
                    val_fraction=0.1,
                    num_workers=0,
                    worker_start_method="spawn",
                    max_epochs=1,
                    lr=1e-3,
                    no_balanced_class_weights=False,
                    early_stopping_patience=2,
                    early_stopping_min_delta=0.0,
                    train_segment_fraction=1.0,
                    max_cached_segments=1,
                    accelerator="cpu",
                    devices=1,
                    precision="32",
                    seed=0,
                    output_dir=output_dir,
                    auto_evaluate=True,
                    svm_reg=1e-4,
                    margin=1.0,
                    knn_scales="2",
                    knn_support_size=8,
                    knn_query_chunk=8,
                )
            )

        self.assertGreaterEqual(load_mock.call_count, 1)
        run_dir = next((output_dir / "point_svm").glob("version_*"))
        report_path = run_dir / "final_report.json"
        self.assertTrue(report_path.exists())
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(sorted(payload["stages"]), ["test", "train", "val"])
        for name in ("train_confusion_raw.png", "train_confusion_normalized.png", "val_confusion_raw.png", "test_confusion_raw.png"):
            self.assertTrue((run_dir / name).exists())

    def test_train_can_skip_auto_evaluation(self) -> None:
        output_dir = self.base_dir / "train_no_eval"
        run_train(
            argparse.Namespace(
                command="train",
                model="point_svm",
                data_dir=self.point_root,
                train_subdirs="training",
                val_subdirs="validation",
                test_subdirs="validation",
                batch_size=1,
                num_points=8,
                num_classes=23,
                val_fraction=0.1,
                num_workers=0,
                worker_start_method="spawn",
                max_epochs=1,
                lr=1e-3,
                no_balanced_class_weights=False,
                early_stopping_patience=2,
                early_stopping_min_delta=0.0,
                train_segment_fraction=1.0,
                max_cached_segments=1,
                accelerator="cpu",
                devices=1,
                precision="32",
                seed=0,
                output_dir=output_dir,
                auto_evaluate=False,
                svm_reg=1e-4,
                margin=1.0,
                knn_scales="2",
                knn_support_size=8,
                knn_query_chunk=8,
            )
        )
        run_dir = next((output_dir / "point_svm").glob("version_*"))
        self.assertFalse((run_dir / "final_report.json").exists())

    def test_evaluate_writes_requested_split_report(self) -> None:
        ckpt_path = _fit_and_save(
            LinearSVMPointClassifier(num_classes=23, knn_scales=(2,), knn_support_size=8, knn_query_chunk=8),
            self._point_dm(),
            self.base_dir / "eval_ckpt",
        )
        output_dir = self.base_dir / "eval_report"
        run_evaluate(
            argparse.Namespace(
                command="evaluate",
                model="point_svm",
                checkpoint=ckpt_path,
                data_dir=self.point_root,
                test_subdirs="validation",
                splits="test",
                batch_size=1,
                num_points=8,
                num_classes=23,
                num_workers=0,
                worker_start_method="spawn",
                max_cached_segments=1,
                sampling_steps=None,
                accelerator="cpu",
                devices=1,
                precision="32",
                seed=0,
                output_dir=output_dir,
            )
        )
        report_path = output_dir / "point_svm_report.json"
        self.assertTrue(report_path.exists())
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(sorted(payload["stages"]), ["test"])
        self.assertTrue((output_dir / "test_confusion_raw.png").exists())
        self.assertTrue((output_dir / "test_confusion_normalized.png").exists())

    def test_linux_open3d_environment_defaults_are_applied(self) -> None:
        previous_gdk = os.environ.pop("GDK_BACKEND", None)
        previous_session = os.environ.pop("XDG_SESSION_TYPE", None)
        try:
            with patch("src.tools.rendering.sys.platform", "linux"):
                ensure_open3d_linux_env()
            self.assertEqual(os.environ["GDK_BACKEND"], "x11")
            self.assertEqual(os.environ["XDG_SESSION_TYPE"], "x11")
        finally:
            if previous_gdk is None:
                os.environ.pop("GDK_BACKEND", None)
            else:
                os.environ["GDK_BACKEND"] = previous_gdk
            if previous_session is None:
                os.environ.pop("XDG_SESSION_TYPE", None)
            else:
                os.environ["XDG_SESSION_TYPE"] = previous_session


if __name__ == "__main__":
    unittest.main()
