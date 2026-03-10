import unittest

from src.main import parse_args
from src.runtime import MODEL_REGISTRY


class RegistryCliTests(unittest.TestCase):
    def test_registry_has_expected_models(self) -> None:
        self.assertEqual(
            sorted(MODEL_REGISTRY),
            ["point_diffusion", "point_svm", "range_diffusion", "range_unet"],
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
            ]
        )
        self.assertEqual(args.command, "train")
        self.assertEqual(args.model, "point_diffusion")
        self.assertEqual(args.backbone, "pointnet")
        self.assertEqual(args.diffusion_steps, 8)

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


if __name__ == "__main__":
    unittest.main()
