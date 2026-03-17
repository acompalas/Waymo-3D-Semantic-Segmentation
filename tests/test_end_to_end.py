import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import re

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
from tests.helpers import FakeWandbLogger, FakeWandbModule, build_synthetic_preprocessed_roots


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
        FakeWandbLogger.instances.clear()

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

    def _latest_hparams(self) -> dict:
        logger = FakeWandbLogger.instances[-1]
        self.assertTrue(logger.experiment.hparams)
        return logger.experiment.hparams[-1]

    def test_checkpoint_round_trip_for_representative_models(self) -> None:
        configs = [
            (
                PointCloudTaskModel(num_classes=23, backbone="handcrafted", behavior="direct"),
                self._point_dm(),
            ),
            (
                PointCloudTaskModel(num_classes=23, hidden_dim=32, depth=2, backbone="pointnet", behavior="direct"),
                self._point_dm(),
            ),
            (
                PointCloudTaskModel(num_classes=23, hidden_dim=32, depth=2, diffusion_steps=4, backbone="pointnet", behavior="diffusion"),
                self._point_dm(),
            ),
            (
                RangeImageTaskModel(num_classes=23, base_channels=4, depth=2, backbone="unet", behavior="direct"),
                self._range_dm(),
            ),
            (
                RangeImageTaskModel(num_classes=23, base_channels=4, diffusion_steps=4, backbone="crossattn_unet", behavior="diffusion"),
                self._range_dm(),
            ),
        ]
        for idx, (model, datamodule) in enumerate(configs):
            ckpt_path = _fit_and_save(model, datamodule, self.base_dir / f"fit_{idx}")
            loaded = model.__class__.load_from_checkpoint(str(ckpt_path))
            self.assertIsInstance(loaded, model.__class__)

    def test_render_smoke_for_point_and_range_models(self) -> None:
        point_model = PointCloudTaskModel(num_classes=23, backbone="handcrafted", behavior="direct")
        point_ckpt = _fit_and_save(point_model, self._point_dm(), self.base_dir / "render_point")
        range_model = RangeImageTaskModel(num_classes=23, base_channels=4, depth=2, backbone="unet", behavior="direct")
        range_ckpt = _fit_and_save(range_model, self._range_dm(), self.base_dir / "render_range")

        fake_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch("src.main.render_point_cloud", return_value=fake_frame), patch("src.main.save_gif") as save_gif_mock:
            run_render(
                argparse.Namespace(
                    command="render",
                    representation="point_clouds",
                    behavior="direct",
                    backbone="handcrafted",
                    checkpoint=point_ckpt,
                    point_data_dir=self.point_root,
                    range_data_dir=self.range_root,
                    source_subdirs="validation",
                    segment="segment_val",
                    seed=0,
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
                    behavior="direct",
                    backbone="unet",
                    checkpoint=range_ckpt,
                    point_data_dir=self.point_root,
                    range_data_dir=self.range_root,
                    source_subdirs="validation",
                    segment="segment_val",
                    seed=0,
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
        self.assertEqual(save_gif_mock.call_count, 2)

    def test_run_render_uses_shared_model_interface(self) -> None:
        point_model = PointCloudTaskModel(num_classes=23, backbone="handcrafted", behavior="direct")
        point_ckpt = _fit_and_save(point_model, self._point_dm(), self.base_dir / "render_contract")

        calls: list[tuple[bool, bool]] = []

        def fake_predict_segmented_pointcloud(*, point_frame, range_frame=None):
            calls.append((point_frame is not None, range_frame is not None))
            return {
                "points_xyz": point_frame["xyz"].reshape(-1, 3).astype(np.float32, copy=False),
                "pred_labels": point_frame["labels"].reshape(-1).astype(np.int64, copy=False),
                "true_labels": point_frame["labels"].reshape(-1).astype(np.int64, copy=False),
                "valid_label": point_frame["valid_label"].reshape(-1).astype(bool, copy=False),
            }

        fake_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch("src.main.render_point_cloud", return_value=fake_frame), patch("src.main.save_gif"), patch.object(
            PointCloudTaskModel,
            "predict_segmented_pointcloud",
            side_effect=fake_predict_segmented_pointcloud,
        ):
            run_render(
                argparse.Namespace(
                    command="render",
                    representation="point_clouds",
                    behavior="direct",
                    backbone="handcrafted",
                    checkpoint=point_ckpt,
                    point_data_dir=self.point_root,
                    range_data_dir=self.range_root,
                    source_subdirs="validation",
                    segment="segment_val",
                    seed=0,
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
        point_model = PointCloudTaskModel(num_classes=23, backbone="handcrafted", behavior="direct")
        point_ckpt = _fit_and_save(point_model, self._point_dm(), self.base_dir / "render_point_limit")

        observed_point_count: list[int] = []

        def fake_predict_segmented_pointcloud(*, point_frame, range_frame=None):
            _ = range_frame
            observed_point_count.append(int(point_frame["xyz"].reshape(-1, 3).shape[0]))
            return {
                "points_xyz": point_frame["xyz"].reshape(-1, 3).astype(np.float32, copy=False),
                "pred_labels": point_frame["labels"].reshape(-1).astype(np.int64, copy=False),
                "true_labels": point_frame["labels"].reshape(-1).astype(np.int64, copy=False),
                "valid_label": point_frame["valid_label"].reshape(-1).astype(bool, copy=False),
            }

        fake_frame = np.zeros((32, 32, 3), dtype=np.uint8)
        with patch("src.main.render_point_cloud", return_value=fake_frame), patch("src.main.save_gif"), patch.object(
            PointCloudTaskModel,
            "predict_segmented_pointcloud",
            side_effect=fake_predict_segmented_pointcloud,
        ):
            run_render(
                argparse.Namespace(
                    command="render",
                    representation="point_clouds",
                    behavior="direct",
                    backbone="handcrafted",
                    checkpoint=point_ckpt,
                    point_data_dir=self.point_root,
                    range_data_dir=self.range_root,
                    source_subdirs="validation",
                    segment="segment_val",
                    seed=0,
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

    def test_train_logs_final_wandb_artifacts_and_uses_checkpoint_reload(self) -> None:
        output_dir = self.base_dir / "train_outputs"
        selection = get_model_selection("point_clouds", "handcrafted", "direct")
        with (
            patch.object(selection.spec.module_cls, "load_from_checkpoint", wraps=selection.spec.module_cls.load_from_checkpoint) as load_mock,
            patch("src.main.WandbLogger", FakeWandbLogger),
            patch("src.runtime.wandb_logging._import_wandb", return_value=FakeWandbModule),
        ):
            run_train(
                argparse.Namespace(
                    command="train",
                    representation="point_clouds",
                    behavior="direct",
                    backbone="handcrafted",
                    point_data_dir=self.point_root,
                    range_data_dir=self.range_root,
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
                    class_weight_alpha=1.0,
                    focal_loss_gamma=0.0,
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
                    knn_query_chunk=8,
                    wandb_project="test-project",
                    wandb_entity=None,
                    wandb_run_name=None,
                    wandb_tags="smoke",
                    log_pointcloud_count=4,
                )
            )

        self.assertGreaterEqual(load_mock.call_count, 1)
        logger = FakeWandbLogger.instances[-1]
        self.assertRegex(logger.name, r"^point_clouds-handcrafted-\d{8}-\d{6}$")
        hparams = self._latest_hparams()
        self.assertEqual(hparams["representation"], "point_clouds")
        self.assertEqual(hparams["behavior"], "direct")
        self.assertEqual(hparams["backbone"], "handcrafted")
        self.assertEqual(hparams["point_data_dir"], str(self.point_root))
        self.assertEqual(hparams["knn_scales"], "2")
        self.assertEqual(hparams["knn_query_chunk"], 8)
        logged_keys = {key for payload, _ in logger.experiment.logged for key in payload}
        self.assertIn("train/confusion_matrix", logged_keys)
        self.assertIn("val/confusion_matrix", logged_keys)
        self.assertIn("final_test/confusion_matrix", logged_keys)
        self.assertIn("class_legend", logged_keys)
        self.assertTrue(any(key.startswith("final_train/pointcloud_") for key in logged_keys))
        self.assertTrue(logger.experiment.finished)

    def test_train_can_skip_auto_evaluation(self) -> None:
        output_dir = self.base_dir / "train_no_eval"
        with (
            patch("src.main.WandbLogger", FakeWandbLogger),
            patch("src.runtime.wandb_logging._import_wandb", return_value=FakeWandbModule),
        ):
            run_train(
                argparse.Namespace(
                    command="train",
                    representation="point_clouds",
                    behavior="direct",
                    backbone="handcrafted",
                    point_data_dir=self.point_root,
                    range_data_dir=self.range_root,
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
                    class_weight_alpha=1.0,
                    focal_loss_gamma=0.0,
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
                    knn_query_chunk=8,
                    wandb_project="test-project",
                    wandb_entity=None,
                    wandb_run_name=None,
                    wandb_tags="smoke",
                    log_pointcloud_count=4,
                )
            )
        logger = FakeWandbLogger.instances[-1]
        self.assertRegex(logger.name, r"^point_clouds-handcrafted-\d{8}-\d{6}$")
        hparams = self._latest_hparams()
        self.assertEqual(hparams["representation"], "point_clouds")
        self.assertEqual(hparams["auto_evaluate"], False)
        self.assertEqual(hparams["behavior"], "direct")
        logged_keys = {key for payload, _ in logger.experiment.logged for key in payload}
        self.assertNotIn("final_test/confusion_matrix", logged_keys)
        self.assertTrue(logger.experiment.finished)

    def test_evaluate_logs_requested_split_to_wandb(self) -> None:
        model = PointCloudTaskModel(num_classes=23, backbone="handcrafted", behavior="direct")
        ckpt_path = _fit_and_save(model, self._point_dm(), self.base_dir / "eval_ckpt")
        output_dir = self.base_dir / "eval_report"
        with (
            patch("src.main.WandbLogger", FakeWandbLogger),
            patch("src.runtime.wandb_logging._import_wandb", return_value=FakeWandbModule),
        ):
            run_evaluate(
                argparse.Namespace(
                    command="evaluate",
                    representation="point_clouds",
                    behavior="direct",
                    backbone="handcrafted",
                    checkpoint=ckpt_path,
                    point_data_dir=self.point_root,
                    range_data_dir=self.range_root,
                    test_subdirs="validation",
                    splits="test",
                    max_batches=1,
                    batch_size=1,
                    num_points=8,
                    num_classes=23,
                    num_workers=0,
                    worker_start_method="spawn",
                    max_cached_segments=1,
                    accelerator="cpu",
                    devices=1,
                    precision="32",
                    seed=0,
                    output_dir=output_dir,
                    wandb_project="test-project",
                    wandb_entity=None,
                    wandb_run_name=None,
                    wandb_tags="eval",
                    log_pointcloud_count=4,
                )
            )
        logger = FakeWandbLogger.instances[-1]
        self.assertRegex(logger.name, r"^point_clouds-handcrafted-\d{8}-\d{6}$")
        hparams = self._latest_hparams()
        self.assertEqual(hparams["representation"], "point_clouds")
        self.assertEqual(hparams["splits"], "test")
        self.assertEqual(hparams["max_batches"], 1)
        self.assertEqual(hparams["backbone"], "handcrafted")
        logged_keys = {key for payload, _ in logger.experiment.logged for key in payload}
        self.assertIn("eval_test/confusion_matrix", logged_keys)
        self.assertTrue(any(key.startswith("eval_test/pointcloud_") for key in logged_keys))
        self.assertTrue(logger.experiment.finished)

    def test_default_wandb_run_name_uses_timestamp_suffix(self) -> None:
        output_dir = self.base_dir / "wandb_name_increment"
        common_args = dict(
            command="train",
            representation="point_clouds",
            behavior="direct",
            backbone="handcrafted",
            point_data_dir=self.point_root,
            range_data_dir=self.range_root,
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
            class_weight_alpha=1.0,
            focal_loss_gamma=0.0,
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
            knn_query_chunk=8,
            wandb_project="test-project",
            wandb_entity=None,
            wandb_run_name=None,
            wandb_tags="smoke",
            log_pointcloud_count=4,
        )
        with (
            patch("src.main.WandbLogger", FakeWandbLogger),
            patch("src.runtime.wandb_logging._import_wandb", return_value=FakeWandbModule),
            patch("src.main._wandb_name_timestamp", side_effect=["20260316-154500", "20260316-154501"]),
        ):
            run_train(argparse.Namespace(**common_args))
            run_train(argparse.Namespace(**common_args))
        self.assertEqual(FakeWandbLogger.instances[-2].name, "point_clouds-handcrafted-20260316-154500")
        self.assertEqual(FakeWandbLogger.instances[-1].name, "point_clouds-handcrafted-20260316-154501")

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
