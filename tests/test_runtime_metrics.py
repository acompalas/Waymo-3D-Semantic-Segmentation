import unittest

import torch

from src.runtime.metrics import iou_metrics_from_confusion_matrix, masked_accuracy, update_confusion_matrix
from src.runtime.stage_eval import StageReportAccumulator


class RuntimeMetricsTests(unittest.TestCase):
    def test_confusion_matrix_and_accuracy_respect_metric_mask(self) -> None:
        confmat = torch.zeros((4, 4), dtype=torch.long)
        preds = torch.tensor([[0, 2, 1, 3]])
        labels = torch.tensor([[0, 2, 1, 1]])
        metric_mask = torch.tensor([[False, True, True, True]])

        acc = masked_accuracy(preds, labels, metric_mask)
        update_confusion_matrix(confmat, preds, labels, metric_mask, num_classes=4)

        self.assertAlmostEqual(float(acc.item()), 2.0 / 3.0, places=6)
        self.assertEqual(confmat[2, 2].item(), 1)
        self.assertEqual(confmat[1, 1].item(), 1)
        self.assertEqual(confmat[1, 3].item(), 1)
        self.assertEqual(confmat[0].sum().item(), 0)

    def test_miou_excludes_class_zero(self) -> None:
        confmat = torch.tensor(
            [
                [5, 0, 0],
                [0, 3, 1],
                [0, 2, 2],
            ],
            dtype=torch.long,
        )
        metrics = iou_metrics_from_confusion_matrix(confmat, ignore_class_zero=True)

        self.assertAlmostEqual(float(metrics["per_class_iou"][0].item()), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["per_class_iou"][1].item()), 0.5, places=6)
        self.assertAlmostEqual(float(metrics["per_class_iou"][2].item()), 0.4, places=6)
        self.assertAlmostEqual(float(metrics["miou"].item()), 0.45, places=6)

    def test_stage_report_accumulator_emits_stage_metrics_and_confusion(self) -> None:
        accumulator = StageReportAccumulator("test", num_classes=4)
        accumulator.consume(
            {
                "loss": torch.tensor(2.0),
                "preds": torch.tensor([[0, 1, 2, 1]]),
                "labels": torch.tensor([[0, 1, 2, 2]]),
                "metric_mask": torch.tensor([[False, True, True, True]]),
                "batch_size": 2,
            }
        )
        summary = accumulator.summary()

        self.assertEqual(summary["stage"], "test")
        self.assertIn("loss", summary["metrics"])
        self.assertIn("acc", summary["metrics"])
        self.assertIn("mIoU", summary["metrics"])
        self.assertIn("IoU_class_3", summary["metrics"])
        self.assertEqual(len(summary["confusion_matrix"]), 4)


if __name__ == "__main__":
    unittest.main()
