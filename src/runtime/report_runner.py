from pathlib import Path
from typing import Iterable

from lightning.pytorch.loggers import CSVLogger
import torch

from ..data import WaymoLidarDataModule
from .reporting import write_stage_report_bundle
from .stage_eval import StageReportAccumulator


def resolve_runtime_device(accelerator_arg: str) -> torch.device:
    key = str(accelerator_arg).lower()
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


def move_batch_to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    return batch


def prepare_model_for_reporting(model, datamodule: WaymoLidarDataModule, device: torch.device) -> None:
    model.to(device)
    if hasattr(model, "set_class_weights") and datamodule.class_weights is not None:
        model.set_class_weights(datamodule.class_weights.to(device))
    if hasattr(model, "prepare_runtime"):
        model.prepare_runtime()
    model.eval()


def parse_report_splits(raw: str) -> list[str]:
    values = [item.strip() for item in str(raw).split(",")]
    splits = [item for item in values if item]
    if not splits:
        raise ValueError("Expected at least one split.")
    unsupported = sorted(set(splits).difference({"train", "val", "test"}))
    if unsupported:
        raise ValueError(f"Unsupported splits: {unsupported}")
    return splits


def setup_datamodule_for_report(datamodule: WaymoLidarDataModule, splits: Iterable[str]) -> None:
    requested_splits = list(splits)
    try:
        datamodule.setup("fit")
    except ValueError:
        if any(split in {"train", "val"} for split in requested_splits):
            raise
    if "test" in requested_splits:
        datamodule.setup("test")


def evaluate_split(model, datamodule: WaymoLidarDataModule, *, split: str, device: torch.device) -> dict:
    loader = datamodule.report_dataloader(split)
    accumulator = StageReportAccumulator(split, num_classes=int(model.hparams.num_classes))
    with torch.no_grad():
        for batch in loader:
            stage_output = model.compute_stage_output(
                move_batch_to_device(batch, device),
                stage=split,
                prediction_mode="full",
                evaluation=True,
            )
            accumulator.consume(stage_output)
    return accumulator.summary()


def collect_stage_reports(model, datamodule: WaymoLidarDataModule, *, splits: Iterable[str], device: torch.device) -> dict[str, dict]:
    return {
        stage: evaluate_split(model, datamodule, split=stage, device=device)
        for stage in splits
    }


def build_report_payload(*, spec, checkpoint_path: Path, stage_reports: dict[str, dict]) -> dict:
    return {
        "model": spec.model_id,
        "representation": spec.representation,
        "checkpoint": str(checkpoint_path),
        "stages": stage_reports,
    }


def log_final_report_metrics(logger: CSVLogger, report_payload: dict, *, step: int) -> None:
    metrics: dict[str, float] = {}
    for stage, stage_payload in report_payload.get("stages", {}).items():
        for key, value in stage_payload.get("metrics", {}).items():
            metrics[f"final_{stage}_{key}"] = float(value)
    if metrics:
        logger.log_metrics(metrics, step=step)
        logger.save()


def write_report_bundle(
    *,
    spec,
    checkpoint_path: Path,
    stage_reports: dict[str, dict],
    output_dir: Path,
    report_name: str,
) -> dict:
    report_payload = build_report_payload(spec=spec, checkpoint_path=checkpoint_path, stage_reports=stage_reports)
    write_stage_report_bundle(output_dir, report_name=report_name, payload=report_payload)
    return report_payload
