import torch


def sanitize_metric_name(name: str) -> str:
    sanitized = "".join(ch if str(ch).isalnum() else "_" for ch in str(name).strip().lower())
    sanitized = "_".join(part for part in sanitized.split("_") if part)
    return sanitized or "unnamed"


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
