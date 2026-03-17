import unittest
from unittest.mock import patch

from src.main import _default_wandb_run_name, _generate_wandb_run_id, _wandb_name_timestamp, parse_args
from src.runtime import backbone_choices, get_model_selection, representation_choices


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
                "--hidden-dim",
                "64",
                "--depth",
                "2",
                "--dropout",
                "0.1",
                "--val-samples-per-segment",
                "3",
                "--geometry-only",
                "--weight-decay",
                "0.01",
                "--class-weight-alpha",
                "0.5",
                "--focal-loss-gamma",
                "2.0",
                "--no-auto-evaluate",
            ]
        )
        self.assertEqual(args.command, "train")
        self.assertEqual(args.representation, "point_clouds")
        self.assertEqual(args.behavior, "supervised")
        self.assertEqual(args.backbone, "handcrafted")
        self.assertEqual(args.hidden_dim, 64)
        self.assertEqual(args.depth, 2)
        self.assertEqual(args.dropout, 0.1)
        self.assertEqual(args.val_samples_per_segment, 3)
        self.assertEqual(args.log_pointcloud_count, 4)
        self.assertEqual(args.weight_decay, 0.01)
        self.assertEqual(args.class_weight_alpha, 0.5)
        self.assertEqual(args.focal_loss_gamma, 2.0)
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
                "--diffusion-steps",
                "8",
                "--no-auto-evaluate",
            ]
        )
        self.assertEqual(args.diffusion_steps, 8)

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
                "--checkpoint",
                "fake.ckpt",
            ]
        )
        self.assertEqual(args.command, "render")
        self.assertEqual(args.backbone, "unet")

    def test_explicit_dataset_paths_are_available_on_train_parser(self) -> None:
        args = parse_args(
            [
                "train",
                "--representation",
                "range_images",
                "--behavior",
                "supervised",
                "--backbone",
                "unet",
                "--point-data-dir",
                "custom/points",
                "--range-data-dir",
                "custom/range",
                "--no-auto-evaluate",
            ]
        )
        self.assertEqual(str(args.point_data_dir), "custom/points")
        self.assertEqual(str(args.range_data_dir), "custom/range")

    def test_legacy_data_dir_arg_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "train",
                    "--representation",
                    "point_clouds",
                    "--behavior",
                    "supervised",
                    "--backbone",
                    "handcrafted",
                    "--data-dir",
                    "custom/points",
                ]
            )

    def test_registry_filters_choices_by_behavior(self) -> None:
        self.assertEqual(backbone_choices("point_clouds", "diffusion"), ["edgeconv", "pointnet"])

    def test_selection_builds_composite_model_id(self) -> None:
        selection = get_model_selection("point_clouds", "edgeconv", "supervised")
        self.assertEqual(selection.model_id, "point_clouds__edgeconv__supervised")

    def test_default_wandb_run_name_uses_representation_backbone_and_timestamp(self) -> None:
        selection = get_model_selection("point_clouds", "edgeconv", "supervised")
        self.assertEqual(_default_wandb_run_name(selection, "20260316-154500"), "point_clouds-edgeconv-20260316-154500")

    def test_old_head_cli_arg_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(
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
                ]
            )

    def test_generate_wandb_run_id_returns_short_hex(self) -> None:
        with patch("src.main.uuid.uuid4") as uuid4_mock:
            uuid4_mock.return_value.hex = "abc123ef45678900"
            self.assertEqual(_generate_wandb_run_id(), "abc123ef")

    def test_wandb_name_timestamp_uses_expected_format(self) -> None:
        with patch("src.main.datetime") as datetime_mock:
            datetime_mock.now.return_value.strftime.return_value = "20260316-154500"
            self.assertEqual(_wandb_name_timestamp(), "20260316-154500")


if __name__ == "__main__":
    unittest.main()
