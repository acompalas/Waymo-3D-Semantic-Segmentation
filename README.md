# Unified LiDAR Segmentation Trainers

This repository exposes one package and one CLI for LiDAR semantic segmentation with explicit component selection.

## Architecture
- `representation`: `point_clouds` or `range_images`
- `backbone`: feature extractor or conditioner
- `head`: output projection module
- `behavior`: supervised or diffusion training/inference policy

Primary model classes:
- `PointCloudTaskModel`
- `RangeImageTaskModel`

Point-cloud backbones:
- `edgeconv`
- `pointnet`
- `pointnetplusplus`
- `handcrafted`

Range-image backbones:
- `unet`
- `crossattn_unet`

Heads:
- point: `mlp`
- range supervised: `segmentation`
- range diffusion: `denoising`

Behaviors:
- `supervised`
- `diffusion`

## Layout
- `src/data`: preprocessed dataset loading, dense frame access, datamodule, scene-aware batching
- `src/models`: task models, backbones, heads, behaviors, and shared diffusion utilities
- `src/runtime`: component registries and CLI dispatch metadata
- `src/tools`: rendering helpers
- `src/main.py`: public CLI entrypoint
- `data_pipeline`: preprocessing tools that build the range-image and dense point-cloud artifacts consumed by `src`

## Data
Default roots:
- `data/preprocessed/point_clouds`
- `data/preprocessed/range_images`

Both roots are expected to contain:
- `meta.json`
- `segment_source.json`
- `segments/<segment_name>/...`

Range-model rendering also uses the paired dense point-cloud artifacts from `data/preprocessed/point_clouds` so predictions can be visualized in 3D without falling back to raw parquet readers.

## Model Prediction Contract
All models expose one primary inference contract: produce a segmented point cloud for a frame.

```python
predict_segmented_pointcloud(
    *,
    point_frame: dict,
    range_frame: dict | None = None,
) -> dict
```

Returned keys:
- `points_xyz`
- `pred_labels`
- `true_labels`
- `valid_label`

Representation-specific behavior:
- point-cloud models consume `point_frame` directly
- range-image models consume `range_frame` and project predictions onto the paired dense point cloud in `point_frame`
- diffusion behaviors denoise internally with the model's trained diffusion schedule
- `--geometry-only` drops non-geometry inputs:
  - point-cloud models use `xyz` instead of `xyz + point_features`
  - range-image models use `range` instead of all 4 channels
- `--val-samples-per-segment` controls deterministic validation subsampling during training:
  - `0` uses the full validation split
  - positive values cap validation frames per segment
  - diffusion defaults to `1` when omitted
  - supervised defaults to `0` when omitted

## CLI
Train:

```bash
python -m src.main train \
  --representation point_clouds \
  --behavior supervised \
  --backbone handcrafted \
  --head mlp

python -m src.main train \
  --representation point_clouds \
  --behavior supervised \
  --backbone edgeconv \
  --head mlp

python -m src.main train \
  --representation point_clouds \
  --behavior diffusion \
  --backbone pointnet \
  --head mlp \
  --diffusion-steps 200 \
  --validation-prediction-mode full

python -m src.main train \
  --representation range_images \
  --behavior supervised \
  --backbone unet \
  --head segmentation

python -m src.main train \
  --representation range_images \
  --behavior diffusion \
  --backbone crossattn_unet \
  --head denoising \
  --diffusion-steps 200
```

Evaluate:

```bash
python -m src.main evaluate \
  --representation point_clouds \
  --behavior supervised \
  --backbone handcrafted \
  --head mlp \
  --checkpoint output/point_clouds__handcrafted__mlp__supervised/.../checkpoints/last.ckpt

python -m src.main evaluate \
  --representation range_images \
  --behavior diffusion \
  --backbone crossattn_unet \
  --head denoising \
  --checkpoint output/range_images__crossattn_unet__denoising__diffusion/.../checkpoints/last.ckpt \
  --sampling-steps 50
```

Render:

```bash
python -m src.main render \
  --representation point_clouds \
  --behavior supervised \
  --backbone handcrafted \
  --head mlp \
  --checkpoint output/point_clouds__handcrafted__mlp__supervised/.../checkpoints/last.ckpt

python -m src.main render \
  --representation range_images \
  --behavior supervised \
  --backbone unet \
  --head segmentation \
  --checkpoint output/range_images__unet__segmentation__supervised/.../checkpoints/last.ckpt
```

Common options:
- `--data-dir`: override the component selection's default preprocessed root
- `--train-subdirs`, `--val-subdirs`, `--test-subdirs`: choose source subsets from `segment_source.json`
- `--batch-size`, `--num-points`, `--num-workers`, `--seed`
- `--max-batches`: cap evaluation to the first N batches of each requested split; `0` means no limit
- `--output-dir`: training/evaluation output root
- `--auto-evaluate` / `--no-auto-evaluate`: control whether training automatically runs the final checkpoint report pass
- diffusion behaviors also accept `--validation-prediction-mode cheap|full`

Important behavior:
- dataset representation is inferred from `--representation`
- valid backbone/head combinations are derived from the runtime registries
- class balancing, masking, scene splitting, and checkpoint/runtime behavior come from the shared `src/data` and `src/models` stack
- evaluation metrics only use `valid_label` and exclude class `0`
- point-cloud frames with fewer than `--num-points` valid geometry elements are dropped during dataset construction
- when a point-cloud frame has more than `--num-points` valid geometry elements, subsampling prefers `valid_label=True` points before filling from the remaining geometry-valid points

## Metrics And Reports
All models use the same metric pipeline.

Per epoch during training:
- `train_loss`, `val_loss`, `test_loss` as available for the current loop
- `train_acc`, `val_acc`, `test_acc`
- `train_mIoU`, `val_mIoU`, `test_mIoU`
- `train_IoU_class_<k>`, `val_IoU_class_<k>`, `test_IoU_class_<k>`

Metric semantics:
- confusion matrices and IoU only count elements under `valid_label`
- class `0` is excluded from mIoU but still gets its own per-class IoU scalar
- diffusion behaviors use `cheap` or `full` validation predictions according to `--validation-prediction-mode`
- final report passes always use full denoising for diffusion behaviors

Training outputs:
- the usual Lightning CSV logs and checkpoints under the run directory
- when auto-evaluation is enabled:
  - `final_report.json`
  - `train_confusion_raw.png`, `train_confusion_normalized.png`
  - `val_confusion_raw.png`, `val_confusion_normalized.png`
  - `test_confusion_raw.png`, `test_confusion_normalized.png`

Evaluation outputs:
- `python -m src.main evaluate ...` writes a structured JSON report plus raw and normalized confusion-matrix PNGs for the requested `--splits`

Render environment:
- on Linux, the renderer sets `GDK_BACKEND=x11` and `XDG_SESSION_TYPE=x11` before importing Open3D

## Data Pipeline
The training and rendering code expects preprocessed artifacts, not raw Waymo parquet files. The repository includes two preprocessing scripts under `data_pipeline`:

- `data_pipeline/preprocess_range_images.py`
  - reads raw Waymo source subdirectories containing `lidar`, `lidar_segmentation`, and `lidar_calibration`
  - writes preprocessed range-image segments with calibration and optional segmentation labels
- `data_pipeline/preprocess_point_clouds.py`
  - reads the preprocessed range-image output
  - writes dense point-cloud tensors plus `valid_geometry` and `valid_label` masks

Typical workflow:

```bash
python data_pipeline/preprocess_range_images.py \
  --data-dir data/waymo_open_dataset_v_2_0_1 \
  --output-dir data/preprocessed/range_images \
  --overwrite

python data_pipeline/preprocess_point_clouds.py \
  --range-dir data/preprocessed/range_images \
  --output-dir data/preprocessed/point_clouds \
  --overwrite
```

## Extending With A New Component
1. Add the backbone, head, or behavior under `src/models`.
2. Register it in `src/runtime/registry.py`.
3. Declare its valid representation/behavior combinations.
4. If it reuses an existing batch contract, no dataloader changes are needed.
