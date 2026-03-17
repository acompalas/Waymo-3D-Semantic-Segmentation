import torch


def sanitize_metric_name(name: str) -> str:
    sanitized = "".join(ch if str(ch).isalnum() else "_" for ch in str(name).strip().lower())
    sanitized = "_".join(part for part in sanitized.split("_") if part)
    return sanitized or "unnamed"


def prefixed_metric_name(name: str, *, prefix: str | None = None) -> str:
    if prefix is None or not str(prefix).strip():
        return str(name)
    return f"{str(prefix).strip()}_{str(name)}"


def metric_stage_label(stage: str) -> str:
    values = {
        "train": "train",
        "val": "validation",
        "test": "test",
    }
    return values.get(str(stage), str(stage))


def pointcloud_stage_label(stage: str) -> str:
    values = {
        "train": "train",
        "val": "val",
        "test": "test",
    }
    return values.get(str(stage), str(stage))


def loss_accuracy_section_key(stage: str, metric_name: str, *, prefix: str | None = None) -> str:
    child = prefixed_metric_name(f"{str(stage)}_{str(metric_name)}", prefix=prefix)
    return f"Losses, epoch, accuracies/{child}"


def confusion_family_section_key(
    stage: str,
    family: str,
    metric_name: str,
    *,
    prefix: str | None = None,
) -> str:
    section = f"{metric_stage_label(stage)} {str(family)}"
    child = prefixed_metric_name(metric_name, prefix=prefix)
    return f"{section}/{child}"


def pointcloud_section_key(stage: str, name: str, *, prefix: str | None = None) -> str:
    section = f"{pointcloud_stage_label(stage)} pointclouds"
    child = prefixed_metric_name(name, prefix=prefix)
    return f"{section}/{child}"


def confusion_matrix_section_key(stage: str, *, prefix: str | None = None) -> str:
    child = prefixed_metric_name(stage, prefix=prefix)
    return f"confusion matrices/{child}"


def resolve_runtime_device(device_arg: str) -> torch.device:
    key = str(device_arg).lower()
    if key == "cpu":
        return torch.device("cpu")
    if key in {"gpu", "cuda"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return torch.device("mps" if has_mps else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
