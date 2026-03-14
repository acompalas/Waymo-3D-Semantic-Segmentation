import unittest

from src.main import parse_args
from src.runtime import MODEL_REGISTRY


class RegistryCliTests(unittest.TestCase):
    def test_registry_has_expected_models(self) -> None:
        self.assertEqual(
            sorted(MODEL_REGISTRY),
            ["point_diffusion", "point_supervised", "point_svm", "range_diffusion", "range_unet"],
        )

    def test_train_parser_adds_model_specific_args(self) -> None:
        args = parse_args(
            [
                "train",
                "--model",
                "point_diffusion",
                "--backbone",
                "pointnet",
                "--diffusion-steps",
                "8",
                "--validation-prediction-mode",
                "full",
                "--no-auto-evaluate",
            ]
        )
        self.assertEqual(args.command, "train")
        self.assertEqual(args.model, "point_diffusion")
        self.assertEqual(args.backbone, "pointnet")
        self.assertEqual(args.diffusion_steps, 8)
        self.assertEqual(args.validation_prediction_mode, "full")
        self.assertFalse(args.auto_evaluate)

    def test_train_parser_adds_point_supervised_args(self) -> None:
        args = parse_args(
            [
                "train",
                "--model",
                "point_supervised",
                "--backbone",
                "pointnet",
                "--geometry-only",
                "--no-auto-evaluate",
            ]
        )
        self.assertEqual(args.command, "train")
        self.assertEqual(args.model, "point_supervised")
        self.assertEqual(args.backbone, "pointnet")
        self.assertTrue(args.geometry_only)

    def test_render_parser_accepts_range_model(self) -> None:
        args = parse_args(
            [
                "render",
                "--model",
                "range_unet",
                "--checkpoint",
                "fake.ckpt",
            ]
        )
        self.assertEqual(args.command, "render")
        self.assertEqual(args.model, "range_unet")

    def test_evaluate_parser_accepts_requested_splits(self) -> None:
        args = parse_args(
            [
                "evaluate",
                "--model",
                "point_svm",
                "--checkpoint",
                "fake.ckpt",
                "--splits",
                "train,val,test",
                "--max-batches",
                "3",
            ]
        )
        self.assertEqual(args.command, "evaluate")
        self.assertEqual(args.splits, "train,val,test")
        self.assertEqual(args.max_batches, 3)

    def test_existing_models_accept_geometry_only_flag(self) -> None:
        args = parse_args(
            [
                "train",
                "--model",
                "range_unet",
                "--geometry-only",
                "--no-auto-evaluate",
            ]
        )
        self.assertTrue(args.geometry_only)


if __name__ == "__main__":
    unittest.main()
