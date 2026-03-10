import tempfile
import unittest
from pathlib import Path

import torch

from src.data import PreprocessedPointCloudDataset, PreprocessedRangeImageDataset, WaymoLidarDataModule
from src.models import (
    LinearSVMPointClassifier,
    PointCloudDiffusionSegmenter,
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


if __name__ == "__main__":
    unittest.main()
