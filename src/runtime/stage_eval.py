from typing import Any

import torch

from .metrics import empty_confusion_matrix, iou_metrics_from_confusion_matrix, masked_accuracy, scalarize_metric_tensor, update_confusion_matrix


REQUIRED_STAGE_OUTPUT_KEYS = {
    "loss",
    "batch_size",
}

PREDICTION_STAGE_OUTPUT_KEYS = {
    "preds",
    "labels",
    "metric_mask",
}


def validate_stage_output(output: dict[str, Any], *, require_predictions: bool = False) -> dict[str, Any]:
    missing = REQUIRED_STAGE_OUTPUT_KEYS.difference(output)
    if missing:
        raise ValueError(f"Stage output missing keys: {sorted(missing)}")

    available_prediction_keys = [key for key in PREDICTION_STAGE_OUTPUT_KEYS if key in output]
    if require_predictions and len(available_prediction_keys) != len(PREDICTION_STAGE_OUTPUT_KEYS):
        missing_prediction_keys = sorted(PREDICTION_STAGE_OUTPUT_KEYS.difference(output))
        raise ValueError(f"Stage output missing prediction keys: {missing_prediction_keys}")
    if available_prediction_keys:
        if len(available_prediction_keys) != len(PREDICTION_STAGE_OUTPUT_KEYS):
            raise ValueError(
                "Stage output must provide either all prediction tensors or none: "
                f"got {sorted(available_prediction_keys)}"
            )
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


def stage_output_metric_weight(output: dict[str, Any]) -> int:
    output = validate_stage_output(output)
    if "metric_mask" not in output:
        return max(1, int(output["batch_size"]))
    return int(output["metric_mask"].bool().sum().item())


def consume_stage_output(confmat: torch.Tensor, output: dict[str, Any], *, num_classes: int) -> torch.Tensor | None:
    output = validate_stage_output(output)
    if "preds" not in output:
        return None
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
        output = validate_stage_output(output, require_predictions=True)
        metric_weight = stage_output_metric_weight(output)
        acc = consume_stage_output(self.confmat, output, num_classes=self.num_classes)
        if acc is None:
            raise ValueError("Stage report accumulation requires prediction tensors.")
        if metric_weight <= 0:
            return
        self.loss_sum += scalarize_metric_tensor(output["loss"]) * metric_weight
        self.acc_sum += scalarize_metric_tensor(acc) * metric_weight
        self.weight += metric_weight

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
            "metrics": metrics,
            "confusion_matrix": self.confmat.detach().cpu().tolist(),
        }
