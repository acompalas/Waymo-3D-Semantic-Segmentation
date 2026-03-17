import sys
from typing import Iterable

import torch
from tqdm.auto import tqdm

from ..data import WaymoLidarDataModule
from .common import resolve_runtime_device
from .stage_eval import StageReportAccumulator


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
    if hasattr(model, "set_class_names"):
        model.set_class_names(list(datamodule.class_names))
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


def _progress_batches(loader, *, split: str):
    total = None
    try:
        total = len(loader)
    except TypeError:
        total = None
    return tqdm(
        loader,
        total=total,
        desc=f"Evaluating {split}",
        unit="batch",
        leave=True,
        disable=not sys.stderr.isatty(),
    )


def evaluate_split(
    model,
    datamodule: WaymoLidarDataModule,
    *,
    split: str,
    device: torch.device,
    max_batches: int = 0,
) -> dict:
    loader = datamodule.report_dataloader(split)
    accumulator = StageReportAccumulator(split, num_classes=int(model.hparams.num_classes))
    batch_limit = max(0, int(max_batches))
    with torch.no_grad():
        for batch_idx, batch in enumerate(_progress_batches(loader, split=split)):
            if batch_limit and batch_idx >= batch_limit:
                break
            stage_output = model.compute_stage_output(
                move_batch_to_device(batch, device),
                evaluation=True,
            )
            accumulator.consume(stage_output)
    return accumulator.summary()


def evaluate_and_log_splits(
    logger,
    model,
    datamodule: WaymoLidarDataModule,
    *,
    splits: Iterable[str],
    device: torch.device,
    class_names: list[str],
    step: int,
    prefix: str | None = None,
    max_batches: int = 0,
) -> None:
    from .wandb_logging import log_stage_report_to_wandb

    for stage in splits:
        stage_payload = evaluate_split(model, datamodule, split=stage, device=device, max_batches=max_batches)
        log_stage_report_to_wandb(
            logger,
            stage=stage,
            stage_payload=stage_payload,
            class_names=class_names,
            step=step,
            prefix=prefix,
        )
