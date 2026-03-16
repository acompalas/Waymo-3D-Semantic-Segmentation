import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.data import PreprocessedPointCloudDataset, PreprocessedRangeImageDataset, WaymoLidarDataModule
from src.models import PointCloudTaskModel, RangeImageTaskModel
from tests.helpers import build_synthetic_preprocessed_roots


class DataAndModelTests(unittest.TestCase):
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

    def test_dense_frame_access_and_class_counts(self) -> None:
        point_ds = PreprocessedPointCloudDataset(self.point_root, source_subdirs="training", num_points=8)
        range_ds = PreprocessedRangeImageDataset(self.range_root, source_subdirs="training")

        point_frame = point_ds.get_dense_frame("segment_train", 100)
        range_frame = range_ds.get_frame_data("segment_train", 100)

        self.assertEqual(point_frame["xyz"].shape, (2, 4, 4, 3))
        self.assertEqual(range_frame["range_images"].shape, (2, 4, 4, 4))
        self.assertGreater(point_ds.compute_class_counts(23).sum(), 0)
        self.assertGreater(range_ds.compute_class_counts(23).sum(), 0)

    def test_point_dataset_returns_only_sampled_valid_geometry_points(self) -> None:
        point_ds = PreprocessedPointCloudDataset(self.point_root, source_subdirs="training", num_points=8, seed=0)

        sample = point_ds[0]
        self.assertNotIn("valid_geometry", sample)
        self.assertTrue(bool(sample["valid_label"].all()))
        self.assertEqual(tuple(sample["points"].shape), (8, 3))

    def test_point_dataset_drops_frames_without_enough_valid_geometry(self) -> None:
        with self.assertRaises(ValueError):
            PreprocessedPointCloudDataset(self.point_root, source_subdirs="training", num_points=31)

    def test_datamodule_computes_class_weights(self) -> None:
        dm = self._point_dm()
        dm.setup("fit")
        self.assertIsNotNone(dm.class_weights)
        self.assertEqual(dm.class_weights.shape[0], 23)

    def test_batch_contracts_for_representative_models(self) -> None:
        point_ds = PreprocessedPointCloudDataset(self.point_root, source_subdirs="training", num_points=8)
        range_ds = PreprocessedRangeImageDataset(self.range_root, source_subdirs="training")

        point_frame = point_ds.get_dense_frame("segment_train", 100)
        range_frame = range_ds.get_frame_data("segment_train", 100)

        point_supervised = PointCloudTaskModel(
            num_classes=23,
            backbone="handcrafted",
            head="mlp",
            behavior="supervised",
        )
        point_prediction = point_supervised.predict_segmented_pointcloud(point_frame=point_frame, sampling_steps=2)
        self.assertEqual(sorted(point_prediction), ["points_xyz", "pred_labels", "true_labels", "valid_label"])

        point_diffusion = PointCloudTaskModel(
            num_classes=23,
            hidden_dim=32,
            depth=2,
            diffusion_steps=4,
            backbone="edgeconv",
            head="mlp",
            behavior="diffusion",
        )
        point_prediction = point_diffusion.predict_segmented_pointcloud(point_frame=point_frame, sampling_steps=2)
        self.assertEqual(point_prediction["pred_labels"].ndim, 1)

        range_supervised = RangeImageTaskModel(
            num_classes=23,
            base_channels=4,
            depth=2,
            backbone="unet",
            head="segmentation",
            behavior="supervised",
        )
        range_prediction = range_supervised.predict_segmented_pointcloud(point_frame=point_frame, range_frame=range_frame, sampling_steps=2)
        self.assertEqual(range_prediction["pred_labels"].ndim, 1)

        range_diffusion = RangeImageTaskModel(
            num_classes=23,
            base_channels=4,
            diffusion_steps=4,
            backbone="crossattn_unet",
            head="denoising",
            behavior="diffusion",
        )
        range_prediction = range_diffusion.predict_segmented_pointcloud(point_frame=point_frame, range_frame=range_frame, sampling_steps=2)
        self.assertEqual(range_prediction["pred_labels"].ndim, 1)

    def test_diffusion_defaults_and_non_diffusion_ignores_sampling_steps(self) -> None:
        point_ds = PreprocessedPointCloudDataset(self.point_root, source_subdirs="training", num_points=8)
        range_ds = PreprocessedRangeImageDataset(self.range_root, source_subdirs="training")
        point_frame = point_ds.get_dense_frame("segment_train", 100)
        range_frame = range_ds.get_frame_data("segment_train", 100)

        point_diffusion = PointCloudTaskModel(
            num_classes=23,
            hidden_dim=32,
            depth=2,
            diffusion_steps=4,
            backbone="pointnet",
            head="mlp",
            behavior="diffusion",
        )
        point_diffusion.set_sampling_steps(3)
        prediction = point_diffusion.predict_segmented_pointcloud(point_frame=point_frame)
        self.assertEqual(prediction["pred_labels"].shape[0], prediction["points_xyz"].shape[0])

        range_diffusion = RangeImageTaskModel(
            num_classes=23,
            base_channels=4,
            diffusion_steps=4,
            backbone="crossattn_unet",
            head="denoising",
            behavior="diffusion",
        )
        range_diffusion.set_sampling_steps(3)
        prediction = range_diffusion.predict_segmented_pointcloud(point_frame=point_frame, range_frame=range_frame)
        self.assertEqual(prediction["pred_labels"].shape[0], prediction["points_xyz"].shape[0])

        point_supervised = PointCloudTaskModel(
            num_classes=23,
            hidden_dim=32,
            depth=2,
            backbone="edgeconv",
            head="mlp",
            behavior="supervised",
        )
        prediction = point_supervised.predict_segmented_pointcloud(point_frame=point_frame, sampling_steps=7)
        self.assertEqual(prediction["pred_labels"].shape[0], prediction["points_xyz"].shape[0])

    def test_shared_stage_output_contract_for_representative_models(self) -> None:
        point_dm = self._point_dm()
        point_dm.setup("fit")
        point_batch = next(iter(point_dm.train_dataloader()))

        range_dm = self._range_dm()
        range_dm.setup("fit")
        range_batch = next(iter(range_dm.train_dataloader()))

        models_and_batches = [
            (
                PointCloudTaskModel(num_classes=23, backbone="handcrafted", head="mlp", behavior="supervised"),
                point_batch,
            ),
            (
                PointCloudTaskModel(num_classes=23, hidden_dim=32, depth=2, backbone="edgeconv", head="mlp", behavior="supervised"),
                point_batch,
            ),
            (
                PointCloudTaskModel(num_classes=23, hidden_dim=32, depth=2, diffusion_steps=4, backbone="pointnet", head="mlp", behavior="diffusion"),
                point_batch,
            ),
            (
                RangeImageTaskModel(num_classes=23, base_channels=4, depth=2, backbone="unet", head="segmentation", behavior="supervised"),
                range_batch,
            ),
            (
                RangeImageTaskModel(num_classes=23, base_channels=4, diffusion_steps=4, backbone="crossattn_unet", head="denoising", behavior="diffusion"),
                range_batch,
            ),
        ]
        for model, batch in models_and_batches:
            output = model.compute_stage_output(batch, stage="train", prediction_mode="cheap", evaluation=False)
            self.assertEqual(sorted(output), ["batch_size", "labels", "loss", "metric_mask", "preds"])
            self.assertEqual(output["preds"].shape, output["labels"].shape)
            self.assertEqual(output["metric_mask"].shape, output["labels"].shape)

    def test_diffusion_validation_prediction_mode_switches_full_sampler_usage(self) -> None:
        dm = self._point_dm()
        dm.setup("fit")
        batch = next(iter(dm.val_dataloader()))

        model = PointCloudTaskModel(
            num_classes=23,
            hidden_dim=32,
            depth=2,
            diffusion_steps=4,
            validation_prediction_mode="cheap",
            backbone="edgeconv",
            head="mlp",
            behavior="diffusion",
        )
        with patch.object(model.behavior_impl, "predict_point_labels", wraps=model.behavior_impl.predict_point_labels) as full_predict:
            model.compute_stage_output(batch, stage="val", prediction_mode="cheap", evaluation=True)
            self.assertEqual(full_predict.call_count, 0)
            model.compute_stage_output(batch, stage="val", prediction_mode="full", evaluation=True)
            self.assertEqual(full_predict.call_count, 1)

    def test_diffusion_sampling_override_applies_to_full_validation_prediction(self) -> None:
        dm = self._point_dm()
        dm.setup("fit")
        batch = next(iter(dm.val_dataloader()))

        model = PointCloudTaskModel(
            num_classes=23,
            hidden_dim=32,
            depth=2,
            diffusion_steps=4,
            backbone="pointnet",
            head="mlp",
            behavior="diffusion",
        )
        model.set_sampling_steps(3)
        with patch.object(model.behavior_impl.ddpm, "sample", wraps=model.behavior_impl.ddpm.sample) as sample_mock:
            model.compute_stage_output(batch, stage="val", prediction_mode="full", evaluation=True)
            self.assertEqual(sample_mock.call_args.kwargs["steps"], 3)

    def test_point_backbones_accept_xyz_and_validate_timestep_support(self) -> None:
        xyz = torch.randn(1, 8, 3)
        inputs = torch.randn(1, 8, 5)
        pointnet = PointCloudTaskModel(num_classes=23, backbone="pointnet", head="mlp", behavior="supervised")
        self.assertEqual(pointnet.backbone(inputs, xyz=xyz).shape[:2], (1, 8))
        with self.assertRaises(ValueError):
            pointnet.backbone(inputs, xyz=xyz, t=torch.ones(1, dtype=torch.long))

        pointnetpp = PointCloudTaskModel(num_classes=23, backbone="pointnetplusplus", head="mlp", behavior="supervised")
        self.assertEqual(pointnetpp.backbone(inputs, xyz=xyz).shape[:2], (1, 8))
        with self.assertRaises(ValueError):
            pointnetpp.backbone(inputs, xyz=xyz, t=torch.ones(1, dtype=torch.long))

        diffusion_model = PointCloudTaskModel(
            num_classes=23,
            hidden_dim=32,
            depth=2,
            diffusion_steps=4,
            backbone="edgeconv",
            head="mlp",
            behavior="diffusion",
        )
        diffusion_inputs = torch.randn(1, 8, 28)
        with self.assertRaises(ValueError):
            diffusion_model.backbone(diffusion_inputs, xyz=xyz)

    def test_handcrafted_backbone_uses_shared_head(self) -> None:
        model = PointCloudTaskModel(num_classes=23, backbone="handcrafted", head="mlp", behavior="supervised")
        self.assertEqual(model.head.net[-1].out_features, 23)
        self.assertEqual(model.backbone.output_dim, model.head.net[1].in_features)
        self.assertIsInstance(model.backbone.post_mlp, torch.nn.Identity)

        projected = PointCloudTaskModel(
            num_classes=23,
            backbone="handcrafted",
            head="mlp",
            behavior="supervised",
            proj_dim=64,
            proj_depth=2,
            proj_dropout=0.1,
        )
        self.assertEqual(projected.backbone.output_dim, 64)
        self.assertEqual(projected.head.net[1].in_features, 64)
        self.assertFalse(isinstance(projected.backbone.post_mlp, torch.nn.Identity))

    def test_geometry_only_switches_selected_input_dimensions(self) -> None:
        point_supervised = PointCloudTaskModel(
            num_classes=23,
            hidden_dim=32,
            depth=2,
            backbone="edgeconv",
            head="mlp",
            behavior="supervised",
            geometry_only=False,
        )
        self.assertEqual(point_supervised.backbone.in_proj.in_features, 5)

        point_supervised_geom = PointCloudTaskModel(
            num_classes=23,
            hidden_dim=32,
            depth=2,
            backbone="edgeconv",
            head="mlp",
            behavior="supervised",
            geometry_only=True,
        )
        self.assertEqual(point_supervised_geom.backbone.in_proj.in_features, 3)

        point_diffusion = PointCloudTaskModel(
            num_classes=23,
            hidden_dim=32,
            depth=2,
            diffusion_steps=4,
            backbone="edgeconv",
            head="mlp",
            behavior="diffusion",
            geometry_only=False,
        )
        self.assertEqual(point_diffusion.backbone.in_proj.in_features, 28)

        point_diffusion_geom = PointCloudTaskModel(
            num_classes=23,
            hidden_dim=32,
            depth=2,
            diffusion_steps=4,
            backbone="edgeconv",
            head="mlp",
            behavior="diffusion",
            geometry_only=True,
        )
        self.assertEqual(point_diffusion_geom.backbone.in_proj.in_features, 26)

        range_unet = RangeImageTaskModel(
            num_classes=23,
            base_channels=4,
            depth=2,
            backbone="unet",
            head="segmentation",
            behavior="supervised",
            geometry_only=False,
        )
        self.assertEqual(range_unet.backbone.encoders[0].conv1.conv.in_channels, 4)

        range_unet_geom = RangeImageTaskModel(
            num_classes=23,
            base_channels=4,
            depth=2,
            backbone="unet",
            head="segmentation",
            behavior="supervised",
            geometry_only=True,
        )
        self.assertEqual(range_unet_geom.backbone.encoders[0].conv1.conv.in_channels, 1)

        range_diffusion = RangeImageTaskModel(
            num_classes=23,
            base_channels=4,
            diffusion_steps=4,
            backbone="crossattn_unet",
            head="denoising",
            behavior="diffusion",
            geometry_only=True,
        )
        self.assertEqual(range_diffusion.backbone.cond_encoder.net[0].in_channels, 1)


if __name__ == "__main__":
    unittest.main()
