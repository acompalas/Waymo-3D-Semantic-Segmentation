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


def iou_metrics_from_confusion_matrix(confmat: torch.Tensor, *, ignore_class_zero: bool = True) -> dict[str, torch.Tensor]:
    conf = confmat.to(dtype=torch.float32)
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    union = tp + fp + fn

    iou = torch.full_like(union, fill_value=-1.0, dtype=torch.float32)
    valid = union > 0
    iou[valid] = tp[valid] / union[valid].clamp_min(1e-6)

    metric_valid = valid.clone()
    if ignore_class_zero and metric_valid.numel() > 0:
        metric_valid[0] = False
    miou = iou[metric_valid].mean() if bool(metric_valid.any()) else conf.new_tensor(0.0)
    return {"per_class_iou": iou, "miou": miou}


def scalarize_metric_tensor(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
