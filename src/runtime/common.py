import torch


def sanitize_metric_name(name: str) -> str:
    sanitized = "".join(ch if str(ch).isalnum() else "_" for ch in str(name).strip().lower())
    sanitized = "_".join(part for part in sanitized.split("_") if part)
    return sanitized or "unnamed"


def wandb_stage_name(stage: str) -> str:
    values = {
        "train": "train",
        "val": "valid",
        "test": "test",
    }
    return values.get(str(stage), str(stage))


def live_metric_key(stage: str, metric_name: str) -> str:
    stage_name = wandb_stage_name(stage)
    return f"{stage_name}_metrics/{stage_name}_{str(metric_name)}"


def per_class_metric_key(stage: str, family: str, class_label: str) -> str:
    stage_name = wandb_stage_name(stage)
    return f"{stage_name}_{str(family)}/{sanitize_metric_name(class_label)}"


def live_pointcloud_key(stage: str, name: str) -> str:
    return f"{wandb_stage_name(stage)}_pointclouds/{str(name)}"


def live_confusion_matrix_key(stage: str) -> str:
    stage_name = wandb_stage_name(stage)
    return f"{stage_name}_confusion_matrices/{stage_name}"


def final_pointcloud_key(stage: str, name: str) -> str:
    return f"final_pointclouds/{wandb_stage_name(stage)}_{str(name)}"


def final_confusion_matrix_key(stage: str) -> str:
    return f"final_confusion_matrices/{wandb_stage_name(stage)}"


def final_evaluation_table_key(name: str) -> str:
    return f"final_evaluation/{str(name)}"


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
