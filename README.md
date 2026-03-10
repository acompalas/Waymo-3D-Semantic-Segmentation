# Unified LiDAR Segmentation Trainers

This repository exposes one package and one CLI for all supported LiDAR semantic segmentation models.

## Models
- `point_svm`: point-cloud baseline using handcrafted local geometry features plus a linear SVM-style head
- `range_unet`: supervised range-image U-Net
- `range_diffusion`: diffusion model in range-image space
- `point_diffusion`: diffusion model in point-cloud space with `edgeconv` or `pointnet` backbone

## Layout
- `src/data`: preprocessed dataset loading, dense frame access, datamodule, scene-aware batching
- `src/models`: all trainable model families and shared diffusion utilities
- `src/runtime`: model registry and dispatch helpers
- `src/tools`: rendering helpers
- `src/main.py`: public CLI entrypoint

## Data
Default roots:
- `data/preprocessed/point_clouds`
- `data/preprocessed/range_images`

Both roots are expected to contain:
- `meta.json`
- `segment_source.json`
- `segments/<segment_name>/...`

Range-model rendering also uses the paired dense point-cloud artifacts from `data/preprocessed/point_clouds` so predictions can be visualized in 3D without falling back to raw parquet readers.

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
python -m src.main render --model point_svm --checkpoint output/point_svm/.../checkpoints/last.ckpt --render-num-points 16384
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

## Extending With A New Model
1. Add the Lightning module under `src/models`.
2. Register it in [`src/runtime/registry.py`](/home/logan/Desktop/271b_final_project_2/src/runtime/registry.py).
3. Provide:
   - a stable model id
   - the required representation (`point_clouds` or `range_images`)
   - model-specific CLI args
   - a module factory
   - a render prediction adapter if rendering is supported

If the model uses an existing batch contract, no dataloader changes are needed.
