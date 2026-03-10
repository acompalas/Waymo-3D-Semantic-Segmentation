# Unified LiDAR Segmentation Trainers

This repository exposes one package and one CLI for all supported LiDAR semantic segmentation models.

## Models
- `point_svm`: point-cloud baseline using handcrafted local geometry features plus a linear SVM-style head
- `range_unet`: supervised range-image U-Net
- `range_diffusion`: diffusion model in range-image space
- `point_diffusion`: diffusion model in point-cloud space with `edgeconv` or `pointnet` backbone

## Layout
- `src/data`: preprocessed dataset loading, dense frame access, datamodule, scene-aware batching
- `src/models`: all trainable model families, shared diffusion utilities, and representation-specific model bases
- `src/runtime`: model registry and CLI dispatch metadata
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

The shared method is:

```python
predict_segmented_pointcloud(
    *,
    point_frame: dict,
    range_frame: dict | None = None,
    sampling_steps: int | None = None,
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
- diffusion models denoise internally; when `sampling_steps` is omitted they use the model's trained diffusion schedule
- non-diffusion models ignore `sampling_steps`

## CLI
Train:

```bash
python -m src.main train --model point_svm
python -m src.main train --model range_unet
python -m src.main train --model range_diffusion --diffusion-steps 200
python -m src.main train --model point_diffusion --backbone edgeconv --diffusion-steps 200
```

Evaluate:

```bash
python -m src.main evaluate --model range_unet --checkpoint output/range_unet/.../checkpoints/last.ckpt
python -m src.main evaluate --model point_diffusion --checkpoint output/point_diffusion/.../checkpoints/last.ckpt --sampling-steps 50
```

Render:

```bash
python -m src.main render --model range_diffusion --checkpoint output/range_diffusion/.../checkpoints/last.ckpt --sampling-steps 50
python -m src.main render --model point_svm --checkpoint output/point_svm/.../checkpoints/last.ckpt
```

Common options:
- `--data-dir`: override the model’s default preprocessed root
- `--train-subdirs`, `--val-subdirs`, `--test-subdirs`: choose source subsets from `segment_source.json`
- `--batch-size`, `--num-points`, `--num-workers`, `--seed`
- `--output-dir`: training/evaluation output root

Important behavior:
- dataset representation is inferred from `--model`
- class balancing, masking, scene splitting, and checkpoint/runtime behavior come from the shared `src/data` and `src/models` stack
- evaluation metrics only use `valid_label` and exclude class `0`

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

Notes:
- `preprocess_range_images.py` can treat selected source subdirectories as unlabeled via `--unlabeled-subdirs`
- both scripts support `--num-workers` for segment-level parallelism
- the point-cloud preprocessing step depends on the range-image preprocessing output

## Extending With A New Model
1. Add the Lightning module under `src/models`.
2. Register it in [`src/runtime/registry.py`](/home/logan/Desktop/271b_final_project_2/src/runtime/registry.py).
3. Provide:
   - a stable model id
   - the required representation (`point_clouds` or `range_images`)
   - model-specific CLI args
   - a module factory
4. Implement the shared segmented point-cloud prediction contract in the appropriate representation base.

If the model uses an existing batch contract, no dataloader changes are needed.
