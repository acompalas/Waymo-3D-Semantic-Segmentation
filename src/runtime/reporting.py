import json
from pathlib import Path

import numpy as np


def write_json_report(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _heatmap_color(value: float) -> tuple[int, int, int]:
    v = float(np.clip(value, 0.0, 1.0))
    low = np.array([24, 36, 58], dtype=np.float32)
    high = np.array([235, 121, 64], dtype=np.float32)
    rgb = low + (high - low) * v
    return tuple(int(x) for x in rgb.tolist())


def _normalize_confusion(confusion: np.ndarray) -> np.ndarray:
    confusion = confusion.astype(np.float32, copy=False)
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(confusion, dtype=np.float32)
    np.divide(confusion, np.clip(row_sums, 1.0, None), out=normalized, where=row_sums > 0.0)
    return normalized


def save_confusion_matrix_image(
    confusion_matrix: list[list[int]] | np.ndarray,
    output_path: str | Path,
    *,
    normalize: bool,
) -> Path:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"")
        return output_path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = np.asarray(confusion_matrix, dtype=np.float32)
    values = _normalize_confusion(matrix) if normalize else matrix
    max_value = float(values.max()) if values.size else 0.0
    scale = max(max_value, 1e-6)

    cell = 20
    margin = 44
    size = int(values.shape[0])
    width = margin + size * cell + 1
    height = margin + size * cell + 1

    image = Image.new("RGB", (width, height), color=(250, 250, 252))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for row in range(size):
        for col in range(size):
            value = float(values[row, col])
            color = _heatmap_color(value / scale)
            x0 = margin + col * cell
            y0 = row * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline=(220, 220, 225))

    for idx in range(size):
        text = str(idx)
        x = margin + idx * cell + 4
        y = height - margin + 6
        draw.text((x, y), text, fill=(32, 32, 40), font=font)
        draw.text((8, idx * cell + 4), text, fill=(32, 32, 40), font=font)

    title = "Normalized confusion matrix" if normalize else "Raw confusion matrix"
    draw.text((8, height - margin + 24), title, fill=(32, 32, 40), font=font)
    image.save(output_path)
    return output_path


def write_stage_report_bundle(
    output_dir: str | Path,
    *,
    report_name: str,
    payload: dict,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    json_path = write_json_report(output_dir / f"{report_name}.json", payload)
    paths["json"] = json_path

    for stage, stage_payload in payload.get("stages", {}).items():
        confusion = stage_payload.get("confusion_matrix")
        if confusion is None:
            continue
        raw_path = save_confusion_matrix_image(confusion, output_dir / f"{stage}_confusion_raw.png", normalize=False)
        normalized_path = save_confusion_matrix_image(
            confusion,
            output_dir / f"{stage}_confusion_normalized.png",
            normalize=True,
        )
        paths[f"{stage}_raw"] = raw_path
        paths[f"{stage}_normalized"] = normalized_path
    return paths
