import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data import PreprocessedPointCloudDataset
from src.runtime.wandb_logging import confusion_table_rows, select_audit_examples, stage_metric_dict
from tests.helpers import build_synthetic_preprocessed_roots


class WandbLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tmpdir.name)
        self.point_root, _ = build_synthetic_preprocessed_roots(self.base_dir)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_confusion_rows_exclude_class_zero(self) -> None:
        confusion = np.array(
            [
                [5, 1, 0],
                [2, 3, 4],
                [0, 6, 7],
            ],
            dtype=np.int64,
        )
        rows, names = confusion_table_rows(confusion, ["undefined", "car", "pedestrian"])

        self.assertEqual(names, ["car", "pedestrian"])
        self.assertEqual(rows[0], ["car", "car", 3])
        self.assertEqual(rows[1], ["car", "pedestrian", 4])

    def test_stage_metric_dict_uses_class_names(self) -> None:
        payload = {
            "metrics": {
                "loss": 1.0,
                "acc": 0.5,
                "mIoU": 0.25,
                "IoU_class_1": 0.3,
                "IoU_class_2": 0.2,
            },
            "confusion_matrix": [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
        }
        metrics = stage_metric_dict("val", payload, ["undefined", "car", "pedestrian"], prefix="final")

        self.assertIn("final_val_mIoU", metrics)
        self.assertIn("final_val_IoU_car", metrics)
        self.assertIn("final_val_IoU_pedestrian", metrics)

    def test_select_audit_examples_returns_exact_requested_count(self) -> None:
        dataset = PreprocessedPointCloudDataset(self.point_root, source_subdirs="validation", num_points=8, seed=0)
        examples = select_audit_examples(dataset, count=4)

        self.assertEqual(len(examples), 4)
        self.assertTrue(all(example.segment == "segment_val" for example in examples))
        self.assertTrue(all(example.timestamp == 101 for example in examples))


if __name__ == "__main__":
    unittest.main()
