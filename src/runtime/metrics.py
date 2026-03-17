import torch


def empty_confusion_matrix(num_classes: int, *, device: torch.device | None = None) -> torch.Tensor:
    return torch.zeros((int(num_classes), int(num_classes)), dtype=torch.long, device=device)


def masked_accuracy(preds: torch.Tensor, labels: torch.Tensor, metric_mask: torch.Tensor) -> torch.Tensor:
    metric_mask = metric_mask.bool()
    if not bool(metric_mask.any()):
        return preds.new_tensor(0.0, dtype=torch.float32)
    hits = preds[metric_mask] == labels[metric_mask]
    return hits.float().mean()


def update_confusion_matrix(
    confmat: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
    metric_mask: torch.Tensor,
    *,
    num_classes: int,
) -> None:
    metric_mask = metric_mask.bool()
    if not bool(metric_mask.any()):
        return

    preds = preds[metric_mask].reshape(-1).long()
    labels = labels[metric_mask].reshape(-1).long()
    keep = (
        (labels > 0)
        & (labels < int(num_classes))
        & (preds >= 0)
        & (preds < int(num_classes))
    )
    if not bool(keep.any()):
        return

    flat = labels[keep] * int(num_classes) + preds[keep]
    bincount = torch.bincount(
        flat,
        minlength=int(num_classes) * int(num_classes),
    ).reshape(int(num_classes), int(num_classes))
    confmat += bincount.to(device=confmat.device)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.full_like(denominator, fill_value=-1.0, dtype=torch.float32)
    valid = denominator > 0
    values[valid] = numerator[valid] / denominator[valid].clamp_min(1e-6)
    return values, valid


def confusion_metrics_from_confusion_matrix(confmat: torch.Tensor, *, ignore_class_zero: bool = True) -> dict[str, torch.Tensor]:
    conf = confmat.to(dtype=torch.float32)
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp

    iou, iou_valid = _safe_ratio(tp, tp + fp + fn)
    precision, precision_valid = _safe_ratio(tp, tp + fp)
    recall, recall_valid = _safe_ratio(tp, tp + fn)

    if ignore_class_zero and iou_valid.numel() > 0:
        iou_valid[0] = False
    if ignore_class_zero and precision_valid.numel() > 0:
        precision_valid[0] = False
    if ignore_class_zero and recall_valid.numel() > 0:
        recall_valid[0] = False

    zero = conf.new_tensor(0.0)
    miou = iou[iou_valid].mean() if bool(iou_valid.any()) else zero
    mean_precision = precision[precision_valid].mean() if bool(precision_valid.any()) else zero
    mean_recall = recall[recall_valid].mean() if bool(recall_valid.any()) else zero
    return {
        "per_class_iou": iou,
        "miou": miou,
        "per_class_precision": precision,
        "mean_precision": mean_precision,
        "per_class_recall": recall,
        "mean_recall": mean_recall,
    }


def iou_metrics_from_confusion_matrix(confmat: torch.Tensor, *, ignore_class_zero: bool = True) -> dict[str, torch.Tensor]:
    metrics = confusion_metrics_from_confusion_matrix(confmat, ignore_class_zero=ignore_class_zero)
    return {"per_class_iou": metrics["per_class_iou"], "miou": metrics["miou"]}


def scalarize_metric_tensor(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
