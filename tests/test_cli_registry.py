import unittest

from src.main import parse_args
from src.runtime import backbone_choices, get_model_selection, head_choices, representation_choices


class RegistryCliTests(unittest.TestCase):
    def test_registry_has_expected_representations(self) -> None:
        self.assertEqual(representation_choices(), ["point_clouds", "range_images"])

    def test_point_supervised_parser_accepts_component_selection(self) -> None:
        args = parse_args(
            [
                "train",
                "--representation",
                "point_clouds",
                "--behavior",
                "supervised",
                "--backbone",
                "handcrafted",
                "--head",
                "mlp",
                "--proj-dim",
                "64",
                "--proj-depth",
                "2",
                "--geometry-only",
                "--no-auto-evaluate",
            ]
        )
        self.assertEqual(args.command, "train")
        self.assertEqual(args.representation, "point_clouds")
        self.assertEqual(args.behavior, "supervised")
        self.assertEqual(args.backbone, "handcrafted")
        self.assertEqual(args.head, "mlp")
        self.assertEqual(args.proj_dim, 64)
        self.assertEqual(args.proj_depth, 2)
        self.assertTrue(args.geometry_only)

    def test_point_diffusion_parser_adds_behavior_args(self) -> None:
        args = parse_args(
            [
                "train",
                "--representation",
                "point_clouds",
                "--behavior",
                "diffusion",
                "--backbone",
                "pointnet",
                "--head",
                "mlp",
                "--diffusion-steps",
                "8",
                "--validation-prediction-mode",
                "full",
                "--no-auto-evaluate",
            ]
        )
        self.assertEqual(args.diffusion_steps, 8)
        self.assertEqual(args.validation_prediction_mode, "full")

    def test_range_render_parser_accepts_requested_components(self) -> None:
        args = parse_args(
            [
                "render",
                "--representation",
                "range_images",
                "--behavior",
                "supervised",
                "--backbone",
                "unet",
                "--head",
                "segmentation",
                "--checkpoint",
                "fake.ckpt",
            ]
        )
        self.assertEqual(args.command, "render")
        self.assertEqual(args.backbone, "unet")

    def test_registry_filters_choices_by_behavior(self) -> None:
        self.assertEqual(backbone_choices("point_clouds", "diffusion"), ["edgeconv", "pointnet"])
        self.assertEqual(head_choices("range_images", "diffusion"), ["denoising"])

    def test_selection_builds_composite_model_id(self) -> None:
        selection = get_model_selection("point_clouds", "edgeconv", "mlp", "supervised")
        self.assertEqual(selection.model_id, "point_clouds__edgeconv__mlp__supervised")


if __name__ == "__main__":
    unittest.main()
