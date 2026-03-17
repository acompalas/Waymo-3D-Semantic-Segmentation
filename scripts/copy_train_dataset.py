#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = Path("/content/drive/MyDrive/ece271b/preprocessed")
DEFAULT_DEST_ROOT = REPO_ROOT / "data" / "preprocessed"
SEGMENT_SOURCE_FILENAME = "segment_source.json"
TRAIN_SOURCE_SUBDIR = "training"

DATASET_ALIASES = {
    "point_clouds": "point_clouds",
    "points": "point_clouds",
    "range_images": "range_images",
    "range": "range_images",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy just the training segments for one preprocessed dataset from a Drive-backed "
            "root into a local dataset root."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_ALIASES),
        help="Which dataset to copy.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root that contains the preprocessed point_clouds/ and range_images/ datasets.",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=DEFAULT_DEST_ROOT,
        help="Destination root for the copied dataset.",
    )
    parser.add_argument(
        "--source-subdir",
        type=str,
        default=TRAIN_SOURCE_SUBDIR,
        help="Source split name to keep. Defaults to training.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the destination dataset if it exists.")
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any, *, sort_keys: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n", encoding="utf-8")


def _normalize_dataset_name(value: str) -> str:
    normalized = DATASET_ALIASES.get(str(value).strip().lower())
    if normalized is None:
        choices = ", ".join(sorted(DATASET_ALIASES))
        raise ValueError(f"Unsupported dataset '{value}'. Expected one of: {choices}")
    return normalized


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output path exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def _filtered_segment_source(records: Any, source_subdir: str) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(records, list):
        raise ValueError("Expected segment_source.json to contain a list of records.")

    out_records: list[dict[str, Any]] = []
    kept_segments: list[str] = []
    seen_segments: set[str] = set()
    found_source = False

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Expected every segment_source record to be an object.")

        current_source = str(record.get("source_subdir", ""))
        original_segments = [str(segment) for segment in record.get("segments", [])]
        keep_segments = original_segments if current_source == source_subdir else []
        found_source = found_source or current_source == source_subdir

        for segment in keep_segments:
            if segment in seen_segments:
                raise ValueError(f"Segment '{segment}' appears multiple times in segment_source.json")
            seen_segments.add(segment)
            kept_segments.append(segment)

        filtered_record = dict(record)
        filtered_record["segments"] = keep_segments
        out_records.append(filtered_record)

    if not found_source:
        raise ValueError(f"Could not find source_subdir='{source_subdir}' in segment_source.json")
    if not kept_segments:
        raise ValueError(f"No segments found under source_subdir='{source_subdir}'")
    return out_records, kept_segments


def _copy_metadata_files(source_dir: Path, dest_dir: Path) -> None:
    for metadata_path in sorted(source_dir.glob("*.json")):
        if metadata_path.name == SEGMENT_SOURCE_FILENAME or not metadata_path.is_file():
            continue
        shutil.copy2(metadata_path, dest_dir / metadata_path.name)


def copy_train_split_dataset(
    *,
    dataset: str,
    source_root: Path,
    dest_root: Path,
    source_subdir: str = TRAIN_SOURCE_SUBDIR,
    overwrite: bool = False,
) -> Path:
    dataset_name = _normalize_dataset_name(dataset)
    source_dir = Path(source_root) / dataset_name
    dest_dir = Path(dest_root) / dataset_name
    segments_source_dir = source_dir / "segments"

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source dataset directory: {source_dir}")
    if not segments_source_dir.is_dir():
        raise FileNotFoundError(f"Missing source segments directory: {segments_source_dir}")

    segment_source_records = _read_json(source_dir / SEGMENT_SOURCE_FILENAME)
    filtered_records, kept_segments = _filtered_segment_source(segment_source_records, str(source_subdir))

    _prepare_output_dir(dest_dir, overwrite=bool(overwrite))
    _copy_metadata_files(source_dir, dest_dir)
    _write_json(dest_dir / SEGMENT_SOURCE_FILENAME, filtered_records, sort_keys=False)

    dest_segments_dir = dest_dir / "segments"
    dest_segments_dir.mkdir(parents=True, exist_ok=False)
    for segment in kept_segments:
        segment_source_dir = segments_source_dir / segment
        if not segment_source_dir.is_dir():
            raise FileNotFoundError(f"Segment listed in metadata but missing on disk: {segment_source_dir}")
        shutil.copytree(segment_source_dir, dest_segments_dir / segment, copy_function=shutil.copy2)

    return dest_dir


def main() -> None:
    args = parse_args()
    dest_dir = copy_train_split_dataset(
        dataset=str(args.dataset),
        source_root=Path(args.source_root),
        dest_root=Path(args.dest_root),
        source_subdir=str(args.source_subdir),
        overwrite=bool(args.overwrite),
    )
    print(f"Copied {args.source_subdir} {DATASET_ALIASES[str(args.dataset)]} dataset to: {dest_dir}")


if __name__ == "__main__":
    main()
