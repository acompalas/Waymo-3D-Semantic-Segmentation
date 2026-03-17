# Unified LiDAR Segmentation Trainers

This repository trains, evaluates, and renders LiDAR semantic segmentation models over two representations:

- `point_clouds`: sampled 3D point sets
- `range_images`: dense LiDAR range-image grids projected back to point clouds for evaluation and rendering

The public model-selection surface is:

- `representation`
- `backbone`
- `behavior`

The CLI supports:

- `train`
- `evaluate`
- `render`

W&B is the reporting backend for metrics, confusion matrices, and audit point clouds.

## Dependencies

- Training and evaluation: `torch`, `lightning`, `wandb`, `numpy`
- Preprocessing: `numpy`, `polars`, `tqdm`
- Rendering: `open3d`, `Pillow`

A minimal pip install looks like:

```bash
pip install torch lightning wandb numpy polars tqdm open3d pillow
```

## Representations

| Representation | What It Uses | Best For | Notes |
| --- | --- | --- | --- |
| `point_clouds` | sampled dense point clouds with `xyz` plus optional per-point features | point-set models and neighborhood graph methods | backbones always receive `xyz`; `--geometry-only` drops non-geometry point features |
| `range_images` | LiDAR range-image tensors with range, intensity, elongation, and NLZ channel | UNet-style convolutional models over the LiDAR image grid | predictions are projected back onto the paired dense point cloud; `--geometry-only` keeps only range |

## Behaviors

| Behavior | Meaning |
| --- | --- |
| `direct` | direct semantic segmentation training with cross-entropy |
| `diffusion` | denoising diffusion training over soft labels; validation/test metrics use real denoising predictions |

## Available Backbones

### Point-cloud backbones

| Backbone | Supports | Summary |
| --- | --- | --- |
| `pointnet` | `direct`, `diffusion` | residual per-point MLP stack with global max-pooled context |
| `edgeconv` | `direct`, `diffusion` | KNN graph backbone with EdgeConv-style local updates |
| `pointnetplusplus` | `direct` | PointNet++ hierarchy with set abstraction and feature propagation |
| `handcrafted` | `direct` | multi-scale handcrafted geometric features with optional learned post-MLP |

### Range-image backbones

| Backbone | Supports | Summary |
| --- | --- | --- |
| `unet` | `direct` | convolutional UNet encoder-decoder over range images |
| `crossattn_unet` | `diffusion` | timestep-conditioned residual UNet with LiDAR conditioning and cross-attention |

## Quick Start

Train a direct point-cloud model:

```bash
python -m src.main train \
  --representation point_clouds \
  --behavior direct \
  --backbone edgeconv
```

Train a diffusion range-image model:

```bash
python -m src.main train \
  --representation range_images \
  --behavior diffusion \
  --backbone crossattn_unet \
  --diffusion-steps 200
```

Evaluate a checkpoint:

```bash
python -m src.main evaluate \
  --representation point_clouds \
  --behavior direct \
  --backbone handcrafted \
  --checkpoint output/point_clouds__handcrafted__direct/.../checkpoints/last.ckpt
```

Render a checkpoint:

```bash
python -m src.main render \
  --representation range_images \
  --behavior direct \
  --backbone unet \
  --checkpoint output/range_images__unet__direct/.../checkpoints/last.ckpt
```

## Colab Preprocessing

For a Google Colab workflow, use [`scripts/colab_preprocess.py`](scripts/colab_preprocess.py) with `%run` so Drive and GCP authentication happen in the notebook kernel:

```python
%cd /content/final_project
%run scripts/colab_preprocess.py \
  --drive-root MyDrive/ece271b \
  --range-workers 4 \
  --point-workers 4
```

The script always reads raw data from `gs://waymo_open_dataset_v_2_0_1/`, downloads `segmentation.proto` from the official Waymo GitHub repo, and writes the preprocessed datasets under your chosen Drive root.
By default it uses `--progress-style print`, which is easier to follow in Colab than `tqdm`. The underlying preprocessing CLIs also support `--progress-style tqdm` and `--progress-style none`.

## CLI Arguments

### Shared component-selection arguments

These are required for every command.

| Argument | Description | Default |
| --- | --- | --- |
| `--representation` | data/model representation; choices: `point_clouds`, `range_images` | required |
| `--behavior` | training/inference behavior; choices: `direct`, `diffusion` | required |
| `--backbone` | backbone name; valid choices depend on representation and behavior | required |

### `train` arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--point-data-dir` | preprocessed point-cloud dataset root | `data/preprocessed/point_clouds` |
| `--range-data-dir` | preprocessed range-image dataset root | `data/preprocessed/range_images` |
| `--train-subdirs` | comma-free source split name(s) used for training | `"training"` |
| `--val-subdirs` | explicit validation source split(s); empty means split train segments with `--val-fraction` | `""` |
| `--test-subdirs` | source split(s) used for test/final evaluation; explicit empty string holds out test segments from `--train-subdirs` | `"validation"` |
| `--batch-size` | batch size | `2` |
| `--num-points` | sampled point count for point-cloud datasets | `16384` |
| `--num-classes` | number of semantic classes | `23` |
| `--val-fraction` | fraction of train segments used for each derived holdout split when `--val-subdirs` and/or `--test-subdirs` is empty | `0.1` |
| `--num-workers` | dataloader worker count | `4` |
| `--worker-start-method` | multiprocessing start method | `"spawn"` |
| `--max-epochs` | training epochs | `20` |
| `--lr` | learning rate | `1e-3` |
| `--weight-decay` | optimizer weight decay | `1e-4` |
| `--class-weight-alpha` | class-weight exponent in `(1 / count)^alpha`; `0` disables weighting | `1.0` |
| `--focal-loss-gamma` | focal-loss gamma for direct training; `0` recovers cross-entropy | `0.0` |
| `--early-stopping-patience` | early stopping patience in epochs | `2` |
| `--early-stopping-min-delta` | minimum improvement for early stopping | `0.0` |
| `--train-segment-fraction` | fraction of train segments to keep | `1.0` |
| `--train-frame-fraction` | fraction of frames to keep within each selected train segment; values in `[0, 1]`, with at least one frame kept per segment | `1.0` |
| `--max-cached-segments` | segment cache size inside dataset loaders | `2` |
| `--accelerator` | Lightning accelerator setting | `"auto"` |
| `--devices` | Lightning device setting | `"auto"` |
| `--precision` | Lightning precision setting | `"bf16-mixed"` |
| `--seed` | random seed | `0` |
| `--output-dir` | local checkpoint / W&B cache root | `output` |
| `--wandb-project` | W&B project name | `"ece271b-final-project"` |
| `--wandb-entity` | W&B entity/team | `None` |
| `--wandb-run-name` | explicit W&B run name; if omitted, runtime generates `<representation>-<backbone>-YYYYMMDD-HHMMSS` | `None` |
| `--wandb-tags` | comma-separated W&B tags | `""` |
| `--log-pointcloud-count` | audit point clouds logged per split | `4` |
| `--geometry-only` | use only geometry channels | `False` |
| `--val-samples-per-segment` | deterministic validation subsample count per segment; `None` resolves to behavior-dependent default | `None` |
| `--auto-evaluate` | run final checkpoint evaluation automatically after training | `True` |
| `--no-auto-evaluate` | disable final automatic evaluation | `False` |
| `--float32-matmul-precision` | PyTorch float32 matmul precision mode; choices: `highest`, `high`, `medium` | `"medium"` |
| `--diffusion-steps` | diffusion schedule length; diffusion models only | `1000` |

### `evaluate` arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--checkpoint` | checkpoint path to evaluate | required |
| `--point-data-dir` | preprocessed point-cloud dataset root | `data/preprocessed/point_clouds` |
| `--range-data-dir` | preprocessed range-image dataset root | `data/preprocessed/range_images` |
| `--test-subdirs` | source split(s) used for evaluation data; explicit empty string holds out test segments from the training split used by the datamodule | `"validation"` |
| `--splits` | comma-separated splits to evaluate | `"test"` |
| `--max-batches` | cap evaluation to the first N batches per split; `0` means no cap | `0` |
| `--batch-size` | evaluation batch size | `2` |
| `--num-points` | sampled point count for point-cloud datasets | `16384` |
| `--num-classes` | number of semantic classes | `23` |
| `--num-workers` | dataloader worker count | `4` |
| `--worker-start-method` | multiprocessing start method | `"spawn"` |
| `--max-cached-segments` | segment cache size | `2` |
| `--accelerator` | Lightning/runtime accelerator setting | `"auto"` |
| `--devices` | Lightning/runtime device setting | `"auto"` |
| `--precision` | Lightning precision setting | `"bf16-mixed"` |
| `--seed` | random seed | `0` |
| `--output-dir` | local W&B cache root | `output/eval` |
| `--wandb-project` | W&B project name | `"ece271b-final-project"` |
| `--wandb-entity` | W&B entity/team | `None` |
| `--wandb-run-name` | explicit W&B run name | `None` |
| `--wandb-tags` | comma-separated W&B tags | `""` |
| `--log-pointcloud-count` | audit point clouds logged per split | `4` |
| `--float32-matmul-precision` | PyTorch float32 matmul precision mode; choices: `highest`, `high`, `medium` | `"medium"` |

### `render` arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--checkpoint` | checkpoint path to render | required |
| `--point-data-dir` | preprocessed point-cloud dataset root | `data/preprocessed/point_clouds` |
| `--range-data-dir` | preprocessed range-image dataset root | `data/preprocessed/range_images` |
| `--source-subdirs` | source split(s) used for rendering data | `"validation"` |
| `--segment` | explicit segment name to render | `None` |
| `--seed` | random seed | `0` |
| `--render-num-points` | sampled point count for rendering | `16384` |
| `--output-gif` | output GIF path | `output/render/predictions.gif` |
| `--fps` | GIF frame rate | `5.0` |
| `--max-frames` | max frames to render; `0` means no explicit cap | `0` |
| `--camera-preset` | camera preset; choices: `car_pov`, `top`, `topdown` | `"car_pov"` |
| `--width` | render width | `1280` |
| `--height` | render height | `720` |
| `--point-size` | rendered point size | `2.5` |
| `--device` | render-time device selection; choices: `auto`, `cuda`, `mps`, `cpu` | `"auto"` |
| `--float32-matmul-precision` | PyTorch float32 matmul precision mode; choices: `highest`, `high`, `medium` | `"medium"` |

## Backbone-Specific Training Arguments

These arguments are only added to the `train` CLI when the chosen backbone declares them.

### `pointnet`

| Argument | Description | Default |
| --- | --- | --- |
| `--hidden-dim` | backbone hidden width | `256` |
| `--depth` | number of residual MLP blocks | `6` |
| `--dropout` | dropout inside residual MLP blocks | `0.1` |

### `edgeconv`

| Argument | Description | Default |
| --- | --- | --- |
| `--hidden-dim` | backbone hidden width | `256` |
| `--depth` | number of EdgeConv blocks | `6` |
| `--dropout` | dropout inside blocks | `0.1` |
| `--knn-k` | KNN neighborhood size | `16` |

### `pointnetplusplus`

| Argument | Description | Default |
| --- | --- | --- |
| `--hidden-dim` | final feature width after feature propagation | `256` |
| `--dropout` | post-MLP dropout | `0.1` |

### `handcrafted`

| Argument | Description | Default |
| --- | --- | --- |
| `--knn-scales` | comma-separated neighborhood sizes used for handcrafted features | `"16,32,64"` |
| `--knn-query-chunk` | chunk size for KNN queries | `4096` |
| `--hidden-dim` | width of the optional learned post-MLP adapter | `0` |
| `--depth` | depth of the optional learned post-MLP adapter; `0` disables it | `0` |
| `--dropout` | dropout in the optional learned post-MLP adapter | `0.0` |

### `unet`

| Argument | Description | Default |
| --- | --- | --- |
| `--base-channels` | base UNet channel width | `32` |
| `--depth` | number of encoder levels | `4` |
| `--dropout` | ConvBlock dropout | `0.0` |

### `crossattn_unet`

| Argument | Description | Default |
| --- | --- | --- |
| `--base-channels` | base channel width | `32` |
| `--dropout` | residual block dropout | `0.1` |

## Preprocessing Pipeline

The training and rendering code expects preprocessed artifacts, not raw Waymo parquet files.

### Step 1: preprocess range images

Script: `data_pipeline/preprocess_range_images.py`

Purpose:

- reads raw Waymo source subdirectories
- writes preprocessed range-image segments with calibration, timestamps, optional segmentation labels, and per-frame class counts

Arguments:

| Argument | Description | Default |
| --- | --- | --- |
| `--data-dir` | raw Waymo dataset root | `waymo_open_dataset_v_2_0_1` |
| `--output-dir` | preprocessed range-image output root | `preprocessed/range_images` |
| `--proto-path` | optional path to the segmentation proto used to build `classes.json`; if omitted, uses `<data-dir>/segmentation.proto` | `None` |
| `--laser-id` | Waymo laser id to preprocess | `1` |
| `--labeled-subdirs` | source subdirs treated as labeled | `training validation` |
| `--unlabeled-subdirs` | source subdirs treated as unlabeled | empty |
| `--num-workers` | segment-level worker processes | `1` |
| `--no-progress` | disable progress bars | `False` |
| `--overwrite` | replace existing output directory | `False` |

Example:

```bash
python data_pipeline/preprocess_range_images.py \
  --data-dir data/waymo_open_dataset_v_2_0_1 \
  --output-dir data/preprocessed/range_images \
  --overwrite
```

### Step 2: build dense point clouds from range images

Script: `data_pipeline/preprocess_point_clouds.py`

Purpose:

- reads preprocessed range-image segments
- writes dense point-cloud tensors with explicit geometry, supervision masks, and copied per-frame class counts

Arguments:

| Argument | Description | Default |
| --- | --- | --- |
| `--range-dir` | preprocessed range-image root | `preprocessed/range_images` |
| `--output-dir` | dense point-cloud output root | `preprocessed/point_clouds` |
| `--segments` | optional explicit segment whitelist | `None` |
| `--num-workers` | segment-level worker processes | `1` |
| `--no-progress` | disable progress bars | `False` |
| `--overwrite` | replace existing output directory | `False` |

Example:

```bash
python data_pipeline/preprocess_point_clouds.py \
  --range-dir data/preprocessed/range_images \
  --output-dir data/preprocessed/point_clouds \
  --overwrite
```

## Notes For Training And Evaluation

- the CLI always accepts explicit dataset roots via `--point-data-dir` and `--range-data-dir`
- preprocessing stores per-frame class counts in both representations so datasets can read frame-level counts directly and aggregate class totals without rescanning labels
- `--val-samples-per-segment` controls deterministic validation subsampling
- when omitted, validation defaults are:
  - direct: full validation set
  - diffusion: `1` frame per segment
- metrics only use `valid_label`
- class `0` is excluded from mIoU and displayed confusion matrices
- W&B logs:
  - scalar metrics
  - train/val confusion matrices
  - audit point clouds
  - class legend table

## Adding A New Backbone

The intended extension path is small and consistent.

### Add a new point-cloud backbone

1. Create a module under `src/models/backbones/point/`.
2. Implement a class that:
   - exposes `forward(inputs, *, xyz, t=None)`
   - sets `self.output_dim`
3. Add an `add_<name>_backbone_args(parser)` function if the backbone needs CLI args.
   - this function should register the CLI args
   - it should also return the normalized parsed arg names, for example `("hidden_dim", "depth", "dropout")`
4. Create a `<NAME>_BACKBONE_SPEC` in that same file using the shared `BackboneSpec` helper.
   - the spec should declare `supported_behaviors`
   - it should provide the backbone builder
   - it should point at the backbone's CLI-arg function
5. Declare `supported_behaviors`, for example `("direct",)` or `("direct", "diffusion")`.
6. Add the spec constant to `POINT_BACKBONE_SPECS` in `src/models/backbones/point/registry.py`.

### Add a new range-image backbone

1. Create a module under `src/models/backbones/range/`.
2. Implement a class that:
   - exposes `forward(x, *, cond=None, t=None)`
   - sets `self.output_channels`
3. Add an `add_<name>_backbone_args(parser)` function if needed.
   - this function should register the CLI args
   - it should also return the normalized parsed arg names that belong to the backbone
4. Create a `<NAME>_BACKBONE_SPEC` in that same file using the shared `BackboneSpec` helper.
5. Add the spec constant to `RANGE_BACKBONE_SPECS` in `src/models/backbones/range/registry.py`.

### How backbone CLI args are wired

Backbone-specific CLI args and the `BackboneSpec` instance both live in the backbone file.

The selected backbone's module defines the two pieces of metadata the runtime needs:

- an `add_<name>_backbone_args(parser)` function to add CLI args
- a `BackboneSpec` instance that stores the parsed arg names and build function

That means the backbone file is the single source of truth for both:

- which CLI args exist
- which parsed arg names belong to that backbone

The registry just maps backbone names to those per-file spec instances.

In the common case, adding a new backbone requires only:

- one new backbone module
- one registry entry
