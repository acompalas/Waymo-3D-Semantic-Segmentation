import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import torch

from src.data import WaymoLidarDataModule
from src.main import run_evaluate, run_render, run_train
from src.models import PointCloudTaskModel, RangeImageTaskModel
from src.runtime import get_model_selection
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

    def test_checkpoint_round_trip_for_representative_models(self) -> None:
        configs = [
            (
                PointCloudTaskModel(num_classes=23, backbone="handcrafted", head="mlp", behavior="supervised"),
                self._point_dm(),
            ),
            (
                PointCloudTaskModel(num_classes=23, hidden_dim=32, depth=2, backbone="pointnet", head="mlp", behavior="supervised"),
                self._point_dm(),
            ),
            (
                PointCloudTaskModel(num_classes=23, hidden_dim=32, depth=2, diffusion_steps=4, backbone="pointnet", head="mlp", behavior="diffusion"),
                self._point_dm(),
            ),
            (
                RangeImageTaskModel(num_classes=23, base_channels=4, depth=2, backbone="unet", head="segmentation", behavior="supervised"),
                self._range_dm(),
            ),
            (
                RangeImageTaskModel(num_classes=23, base_channels=4, diffusion_steps=4, backbone="crossattn_unet", head="denoising", behavior="diffusion"),
                self._range_dm(),
            ),
        ]
        for idx, (model, datamodule) in enumerate(configs):
            ckpt_path = _fit_and_save(model, datamodule, self.base_dir / f"fit_{idx}")
            loaded = model.__class__.load_from_checkpoint(str(ckpt_path))
            self.assertIsInstance(loaded, model.__class__)

    def test_render_smoke_for_point_and_range_models(self) -> None:
        point_model = PointCloudTaskModel(num_classes=23, backbone="handcrafted", head="mlp", behavior="supervised")
        point_ckpt = _fit_and_save(point_model, self._point_dm(), self.base_dir / "render_point")
        range_model = RangeImageTaskModel(num_classes=23, base_channels=4, depth=2, backbone="unet", head="segmentation", behavior="supervised")
        range_ckpt = _fit_and_save(range_model, self._range_dm(), self.base_dir / "render_range")

        fake_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch("src.main.render_point_cloud", return_value=fake_frame):
            run_render(
                argparse.Namespace(
                    command="render",
                    representation="point_clouds",
                    behavior="supervised",
                    backbone="handcrafted",
                    head="mlp",
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
                    representation="range_images",
                    behavior="supervised",
                    backbone="unet",
                    head="segmentation",
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
        point_model = PointCloudTaskModel(num_classes=23, backbone="handcrafted", head="mlp", behavior="supervised")
        point_ckpt = _fit_and_save(point_model, self._point_dm(), self.base_dir / "render_contract")

        calls: list[tuple[bool, bool]] = []

        def fake_predict_segmented_pointcloud(*, point_frame, range_frame=None, sampling_steps=None):
            _ = sampling_steps
            calls.append((point_frame is not None, range_frame is not None))
            return {
                "points_xyz": point_frame["xyz"].reshape(-1, 3).astype(np.float32, copy=False),
                "pred_labels": point_frame["labels"].reshape(-1).astype(np.int64, copy=False),
                "true_labels": point_frame["labels"].reshape(-1).astype(np.int64, copy=False),
                "valid_label": point_frame["valid_label"].reshape(-1).astype(bool, copy=False),
            }

        fake_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch("src.main.render_point_cloud", return_value=fake_frame), patch.object(
            PointCloudTaskModel,
            "predict_segmented_pointcloud",
            side_effect=fake_predict_segmented_pointcloud,
        ):
            run_render(
                argparse.Namespace(
                    command="render",
                    representation="point_clouds",
                    behavior="supervised",
                    backbone="handcrafted",
                    head="mlp",
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

    def test_run_render_limits_point_model_inputs_to_render_num_points(self) -> None:
        point_model = PointCloudTaskModel(num_classes=23, backbone="handcrafted", head="mlp", behavior="supervised")
        point_ckpt = _fit_and_save(point_model, self._point_dm(), self.base_dir / "render_point_limit")

        observed_point_count: list[int] = []

        def fake_predict_segmented_pointcloud(*, point_frame, range_frame=None, sampling_steps=None):
            _ = range_frame
            _ = sampling_steps
            observed_point_count.append(int(point_frame["xyz"].reshape(-1, 3).shape[0]))
            return {
                "points_xyz": point_frame["xyz"].reshape(-1, 3).astype(np.float32, copy=False),
                "pred_labels": point_frame["labels"].reshape(-1).astype(np.int64, copy=False),
                "true_labels": point_frame["labels"].reshape(-1).astype(np.int64, copy=False),
                "valid_label": point_frame["valid_label"].reshape(-1).astype(bool, copy=False),
            }

        fake_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch("src.main.render_point_cloud", return_value=fake_frame), patch.object(
            PointCloudTaskModel,
            "predict_segmented_pointcloud",
            side_effect=fake_predict_segmented_pointcloud,
        ):
            run_render(
                argparse.Namespace(
                    command="render",
                    representation="point_clouds",
                    behavior="supervised",
                    backbone="handcrafted",
                    head="mlp",
                    checkpoint=point_ckpt,
                    data_dir=self.point_root,
                    point_data_dir=self.point_root,
                    source_subdirs="validation",
                    segment="segment_val",
                    seed=0,
                    sampling_steps=None,
                    render_num_points=5,
                    output_gif=self.base_dir / "point_limit.gif",
                    fps=2.0,
                    max_frames=1,
                    camera_preset="car_pov",
                    width=32,
                    height=32,
                    point_size=1.0,
                    device="cpu",
                )
            )

        self.assertEqual(observed_point_count, [5])

    def test_train_writes_final_report_artifacts_and_uses_checkpoint_reload(self) -> None:
        output_dir = self.base_dir / "train_outputs"
        selection = get_model_selection("point_clouds", "handcrafted", "mlp", "supervised")
        with patch.object(selection.spec.module_cls, "load_from_checkpoint", wraps=selection.spec.module_cls.load_from_checkpoint) as load_mock:
            run_train(
                argparse.Namespace(
                    command="train",
                    representation="point_clouds",
                    behavior="supervised",
                    backbone="handcrafted",
                    head="mlp",
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
                    weight_decay=1e-4,
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
                    geometry_only=False,
                    knn_scales="2",
                    knn_support_size=8,
                    knn_query_chunk=8,
                )
            )

        self.assertGreaterEqual(load_mock.call_count, 1)
        run_dir = next((output_dir / selection.model_id).glob("version_*"))
        report_path = run_dir / "final_report.json"
        self.assertTrue(report_path.exists())
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(sorted(payload["stages"]), ["test", "train", "val"])
        for name in ("train_confusion_raw.png", "train_confusion_normalized.png", "val_confusion_raw.png", "test_confusion_raw.png"):
            self.assertTrue((run_dir / name).exists())

    def test_train_can_skip_auto_evaluation(self) -> None:
        output_dir = self.base_dir / "train_no_eval"
        selection = get_model_selection("point_clouds", "handcrafted", "mlp", "supervised")
        run_train(
            argparse.Namespace(
                command="train",
                representation="point_clouds",
                behavior="supervised",
                backbone="handcrafted",
                head="mlp",
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
                weight_decay=1e-4,
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
                geometry_only=False,
                knn_scales="2",
                knn_support_size=8,
                knn_query_chunk=8,
            )
        )
        run_dir = next((output_dir / selection.model_id).glob("version_*"))
        self.assertFalse((run_dir / "final_report.json").exists())

    def test_evaluate_writes_requested_split_report(self) -> None:
        model = PointCloudTaskModel(num_classes=23, backbone="handcrafted", head="mlp", behavior="supervised")
        ckpt_path = _fit_and_save(model, self._point_dm(), self.base_dir / "eval_ckpt")
        output_dir = self.base_dir / "eval_report"
        selection = get_model_selection("point_clouds", "handcrafted", "mlp", "supervised")
        run_evaluate(
            argparse.Namespace(
                command="evaluate",
                representation="point_clouds",
                behavior="supervised",
                backbone="handcrafted",
                head="mlp",
                checkpoint=ckpt_path,
                data_dir=self.point_root,
                test_subdirs="validation",
                splits="test",
                max_batches=1,
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
        report_path = output_dir / f"{selection.model_id}_report.json"
        self.assertTrue(report_path.exists())
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(sorted(payload["stages"]), ["test"])
        self.assertTrue((output_dir / "test_confusion_raw.png").exists())
        self.assertTrue((output_dir / "test_confusion_normalized.png").exists())

    def test_linux_open3d_environment_forces_x11(self) -> None:
        previous_gdk = os.environ.get("GDK_BACKEND")
        previous_session = os.environ.get("XDG_SESSION_TYPE")
        previous_qt = os.environ.get("QT_QPA_PLATFORM")
        previous_wayland = os.environ.get("WAYLAND_DISPLAY")
        os.environ["GDK_BACKEND"] = "wayland"
        os.environ["XDG_SESSION_TYPE"] = "wayland"
        os.environ["QT_QPA_PLATFORM"] = "wayland"
        os.environ["WAYLAND_DISPLAY"] = "wayland-0"
        try:
            with patch("src.tools.rendering.sys.platform", "linux"):
                ensure_open3d_linux_env()
            self.assertEqual(os.environ["GDK_BACKEND"], "x11")
            self.assertEqual(os.environ["XDG_SESSION_TYPE"], "x11")
            self.assertEqual(os.environ["QT_QPA_PLATFORM"], "xcb")
            self.assertNotIn("WAYLAND_DISPLAY", os.environ)
        finally:
            if previous_gdk is None:
                os.environ.pop("GDK_BACKEND", None)
            else:
                os.environ["GDK_BACKEND"] = previous_gdk
            if previous_session is None:
                os.environ.pop("XDG_SESSION_TYPE", None)
            else:
                os.environ["XDG_SESSION_TYPE"] = previous_session
            if previous_qt is None:
                os.environ.pop("QT_QPA_PLATFORM", None)
            else:
                os.environ["QT_QPA_PLATFORM"] = previous_qt
            if previous_wayland is None:
                os.environ.pop("WAYLAND_DISPLAY", None)
            else:
                os.environ["WAYLAND_DISPLAY"] = previous_wayland


if __name__ == "__main__":
    unittest.main()
