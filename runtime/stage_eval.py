from typing import Any

import torch

from .metrics import empty_confusion_matrix, iou_metrics_from_confusion_matrix, masked_accuracy, scalarize_metric_tensor, update_confusion_matrix


REQUIRED_STAGE_OUTPUT_KEYS = {
    "loss",
    "preds",
    "labels",
    "metric_mask",
    "batch_size",
}


def validate_stage_output(output: dict[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_STAGE_OUTPUT_KEYS.difference(output)
    if missing:
        raise ValueError(f"Stage output missing keys: {sorted(missing)}")

    preds = output["preds"]
    labels = output["labels"]
    metric_mask = output["metric_mask"]
    if not isinstance(preds, torch.Tensor) or not isinstance(labels, torch.Tensor) or not isinstance(metric_mask, torch.Tensor):
        raise TypeError("Stage output tensors must be torch.Tensor instances.")
    if preds.shape != labels.shape or preds.shape != metric_mask.shape:
        raise ValueError(
            "Stage output shapes must match: "
            f"preds={tuple(preds.shape)}, labels={tuple(labels.shape)}, metric_mask={tuple(metric_mask.shape)}"
        )
    return output


def consume_stage_output(confmat: torch.Tensor, output: dict[str, Any], *, num_classes: int) -> torch.Tensor:
    output = validate_stage_output(output)
    acc = masked_accuracy(output["preds"], output["labels"], output["metric_mask"])
    update_confusion_matrix(
        confmat,
        output["preds"],
        output["labels"],
        output["metric_mask"],
        num_classes=int(num_classes),
    )
    return acc


class StageReportAccumulator:
    def __init__(self, stage: str, num_classes: int) -> None:
        self.stage = str(stage)
        self.num_classes = int(num_classes)
        self.confmat = empty_confusion_matrix(num_classes)
        self.loss_sum = 0.0
        self.acc_sum = 0.0
        self.weight = 0

    def consume(self, output: dict[str, Any]) -> None:
        output = validate_stage_output(output)
        batch_size = max(1, int(output["batch_size"]))
        acc = consume_stage_output(self.confmat, output, num_classes=self.num_classes)
        self.loss_sum += scalarize_metric_tensor(output["loss"]) * batch_size
        self.acc_sum += scalarize_metric_tensor(acc) * batch_size
        self.weight += batch_size

    def summary(self) -> dict[str, Any]:
        denom = max(1, self.weight)
        metrics = {
            "loss": self.loss_sum / denom,
            "acc": self.acc_sum / denom,
        }
        iou_metrics = iou_metrics_from_confusion_matrix(self.confmat, ignore_class_zero=True)
        metrics["mIoU"] = scalarize_metric_tensor(iou_metrics["miou"])
        for idx, value in enumerate(iou_metrics["per_class_iou"]):
            metrics[f"IoU_class_{idx}"] = scalarize_metric_tensor(value)
        return {
            "stage": self.stage,
            "metrics": metrics,
            "confusion_matrix": self.confmat.detach().cpu().tolist(),
        }
