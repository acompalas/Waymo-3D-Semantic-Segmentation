import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.data import PreprocessedPointCloudDataset, PreprocessedRangeImageDataset, WaymoLidarDataModule
from src.models import (
    LinearSVMPointClassifier,
    PointCloudDiffusionSegmenter,
    PointCloudSupervisedSegmenter,
    RangeImageDiffusionSegmenter,
    RangeImageUNetSegmenter,
)
from tests.helpers import build_synthetic_preprocessed_roots


class DataAndModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmpdir.name)
        self.point_root, self.range_root = build_synthetic_preprocessed_roots(self.base_dir)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_dense_frame_access_and_class_counts(self) -> None:
        point_ds = PreprocessedPointCloudDataset(self.point_root, source_subdirs="training", num_points=8)
        range_ds = PreprocessedRangeImageDataset(self.range_root, source_subdirs="training")

        point_frame = point_ds.get_dense_frame("segment_train", 100)
        range_frame = range_ds.get_frame_data("segment_train", 100)

        self.assertEqual(point_frame["xyz"].shape, (2, 4, 4, 3))
        self.assertEqual(range_frame["range_images"].shape, (2, 4, 4, 4))
        self.assertGreater(point_ds.compute_class_counts(23).sum(), 0)
        self.assertGreater(range_ds.compute_class_counts(23).sum(), 0)

    def test_datamodule_computes_class_weights(self) -> None:
        dm = WaymoLidarDataModule(
            data_dir=self.point_root,
            representation="point_clouds",
            batch_size=1,
            num_points=8,
            train_subdirs="training",
            val_subdirs="validation",
            test_subdirs="validation",
            num_workers=0,
            max_cached_segments=1,
        )
        dm.setup("fit")
        self.assertIsNotNone(dm.class_weights)
        self.assertEqual(dm.class_weights.shape[0], 23)

    def test_batch_contracts_for_all_models(self) -> None:
        point_ds = PreprocessedPointCloudDataset(self.point_root, source_subdirs="training", num_points=8)
        range_ds = PreprocessedRangeImageDataset(self.range_root, source_subdirs="training")

        point_frame = point_ds.get_dense_frame("segment_train", 100)
        range_frame = range_ds.get_frame_data("segment_train", 100)

        point_svm = LinearSVMPointClassifier(num_classes=23)
        point_prediction = point_svm.predict_segmented_pointcloud(
            point_frame=point_frame,
            sampling_steps=2,
        )
        self.assertEqual(sorted(point_prediction), ["points_xyz", "pred_labels", "true_labels", "valid_label"])
        self.assertEqual(point_prediction["points_xyz"].shape[-1], 3)

        range_unet = RangeImageUNetSegmenter(num_classes=23, base_channels=4, depth=2)
        range_prediction = range_unet.predict_segmented_pointcloud(
            point_frame=point_frame,
            range_frame=range_frame,
            sampling_steps=2,
        )
        self.assertEqual(sorted(range_prediction), ["points_xyz", "pred_labels", "true_labels", "valid_label"])
        self.assertEqual(range_prediction["points_xyz"].shape[-1], 3)

        range_diffusion = RangeImageDiffusionSegmenter(num_classes=23, base_channels=4, diffusion_steps=4)
        range_prediction = range_diffusion.predict_segmented_pointcloud(
            point_frame=point_frame,
            range_frame=range_frame,
            sampling_steps=2,
        )
        self.assertEqual(range_prediction["pred_labels"].ndim, 1)

        point_diffusion = PointCloudDiffusionSegmenter(num_classes=23, hidden_dim=32, depth=2, diffusion_steps=4)
        point_prediction = point_diffusion.predict_segmented_pointcloud(
            point_frame=point_frame,
            sampling_steps=2,
        )
        self.assertEqual(point_prediction["pred_labels"].ndim, 1)

        point_supervised = PointCloudSupervisedSegmenter(num_classes=23, hidden_dim=32, depth=2, backbone="edgeconv")
        point_prediction = point_supervised.predict_segmented_pointcloud(
            point_frame=point_frame,
            sampling_steps=2,
        )
        self.assertEqual(point_prediction["pred_labels"].ndim, 1)

    def test_diffusion_defaults_and_non_diffusion_ignores_sampling_steps(self) -> None:
        point_ds = PreprocessedPointCloudDataset(self.point_root, source_subdirs="training", num_points=8)
        range_ds = PreprocessedRangeImageDataset(self.range_root, source_subdirs="training")
        point_frame = point_ds.get_dense_frame("segment_train", 100)
        range_frame = range_ds.get_frame_data("segment_train", 100)

        point_diffusion = PointCloudDiffusionSegmenter(num_classes=23, hidden_dim=32, depth=2, diffusion_steps=4)
        point_diffusion.set_sampling_steps(3)
        prediction = point_diffusion.predict_segmented_pointcloud(point_frame=point_frame)
        self.assertEqual(prediction["pred_labels"].shape[0], prediction["points_xyz"].shape[0])

        range_diffusion = RangeImageDiffusionSegmenter(num_classes=23, base_channels=4, diffusion_steps=4)
        range_diffusion.set_sampling_steps(3)
        prediction = range_diffusion.predict_segmented_pointcloud(point_frame=point_frame, range_frame=range_frame)
        self.assertEqual(prediction["pred_labels"].shape[0], prediction["points_xyz"].shape[0])

        point_svm = LinearSVMPointClassifier(num_classes=23)
        prediction = point_svm.predict_segmented_pointcloud(point_frame=point_frame, sampling_steps=7)
        self.assertEqual(prediction["pred_labels"].shape[0], prediction["points_xyz"].shape[0])

        point_supervised = PointCloudSupervisedSegmenter(num_classes=23, hidden_dim=32, depth=2)
        prediction = point_supervised.predict_segmented_pointcloud(point_frame=point_frame, sampling_steps=7)
        self.assertEqual(prediction["pred_labels"].shape[0], prediction["points_xyz"].shape[0])

    def test_shared_stage_output_contract_for_all_models(self) -> None:
        point_dm = WaymoLidarDataModule(
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
        point_dm.setup("fit")
        point_batch = next(iter(point_dm.train_dataloader()))

        range_dm = WaymoLidarDataModule(
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
        range_dm.setup("fit")
        range_batch = next(iter(range_dm.train_dataloader()))

        models_and_batches = [
            (LinearSVMPointClassifier(num_classes=23), point_batch),
            (RangeImageUNetSegmenter(num_classes=23, base_channels=4, depth=2), range_batch),
            (PointCloudSupervisedSegmenter(num_classes=23, hidden_dim=32, depth=2), point_batch),
            (PointCloudDiffusionSegmenter(num_classes=23, hidden_dim=32, depth=2, diffusion_steps=4), point_batch),
            (RangeImageDiffusionSegmenter(num_classes=23, base_channels=4, diffusion_steps=4), range_batch),
        ]
        for model, batch in models_and_batches:
            output = model.compute_stage_output(batch, stage="train", prediction_mode="cheap", evaluation=False)
            self.assertEqual(sorted(output), ["batch_size", "labels", "loss", "metric_mask", "preds"])
            self.assertEqual(output["preds"].shape, output["labels"].shape)
            self.assertEqual(output["metric_mask"].shape, output["labels"].shape)

    def test_diffusion_validation_prediction_mode_switches_full_sampler_usage(self) -> None:
        dm = WaymoLidarDataModule(
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
        dm.setup("fit")
        batch = next(iter(dm.val_dataloader()))

        model = PointCloudDiffusionSegmenter(num_classes=23, hidden_dim=32, depth=2, diffusion_steps=4)
        with patch.object(model, "predict_point_labels", wraps=model.predict_point_labels) as full_predict:
            model.compute_stage_output(batch, stage="val", prediction_mode="cheap", evaluation=True)
            self.assertEqual(full_predict.call_count, 0)
            model.compute_stage_output(batch, stage="val", prediction_mode="full", evaluation=True)
            self.assertEqual(full_predict.call_count, 1)

    def test_diffusion_sampling_override_applies_to_full_validation_prediction(self) -> None:
        dm = WaymoLidarDataModule(
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
        dm.setup("fit")
        batch = next(iter(dm.val_dataloader()))

        model = PointCloudDiffusionSegmenter(num_classes=23, hidden_dim=32, depth=2, diffusion_steps=4)
        model.set_sampling_steps(3)
        with patch.object(model.ddpm, "sample", wraps=model.ddpm.sample) as sample_mock:
            model.compute_stage_output(batch, stage="val", prediction_mode="full", evaluation=True)
            self.assertEqual(sample_mock.call_args.kwargs["steps"], 3)

    def test_point_supervised_supports_both_backbones(self) -> None:
        dm = WaymoLidarDataModule(
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
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))

        for backbone in ("pointnet", "edgeconv"):
            model = PointCloudSupervisedSegmenter(num_classes=23, hidden_dim=32, depth=2, backbone=backbone)
            output = model.compute_stage_output(batch, stage="train", prediction_mode="cheap", evaluation=False)
            self.assertEqual(output["preds"].shape, output["labels"].shape)

    def test_geometry_only_switches_selected_input_dimensions(self) -> None:
        point_supervised = PointCloudSupervisedSegmenter(
            num_classes=23,
            hidden_dim=32,
            depth=2,
            geometry_only=False,
        )
        self.assertEqual(point_supervised.backbone.in_proj.in_features, 5)

        point_supervised_geom = PointCloudSupervisedSegmenter(
            num_classes=23,
            hidden_dim=32,
            depth=2,
            geometry_only=True,
        )
        self.assertEqual(point_supervised_geom.backbone.in_proj.in_features, 3)

        point_diffusion = PointCloudDiffusionSegmenter(
            num_classes=23,
            hidden_dim=32,
            depth=2,
            diffusion_steps=4,
            geometry_only=False,
        )
        self.assertEqual(point_diffusion.model.backbone.in_proj.in_features, 28)

        point_diffusion_legacy = PointCloudDiffusionSegmenter(
            num_classes=23,
            hidden_dim=32,
            depth=2,
            diffusion_steps=4,
            geometry_only=None,
        )
        self.assertEqual(point_diffusion_legacy.model.backbone.in_proj.in_features, 26)

        range_unet = RangeImageUNetSegmenter(num_classes=23, base_channels=4, depth=2, geometry_only=False)
        self.assertEqual(range_unet.model.encoders[0].conv1.conv.in_channels, 4)

        range_unet_geom = RangeImageUNetSegmenter(num_classes=23, base_channels=4, depth=2, geometry_only=True)
        self.assertEqual(range_unet_geom.model.encoders[0].conv1.conv.in_channels, 1)

        range_diffusion = RangeImageDiffusionSegmenter(num_classes=23, base_channels=4, diffusion_steps=4, geometry_only=True)
        self.assertEqual(range_diffusion.model.cond_encoder.net[0].in_channels, 1)


if __name__ == "__main__":
    unittest.main()
