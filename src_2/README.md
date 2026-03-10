# ECE271B Final Project Work

3D diffusion starter pipeline for Waymo LiDAR semantic segmentation.

## Files
- `waymo_lidar_dataset_loader.py`: point-cloud dataset loader from Waymo parquet files
- `model_3d.py`: PointNet-style 3D denoiser with timestep conditioning
- `diffusion.py`: DDPM schedule, q/p sampling, masked weighted loss
- `train.py`: training entrypoint with checkpointing + metrics CSV
- `evaluate.py`: checkpoint evaluation (mIoU + confusion matrix JSON)
- `plot_metrics.py`: plots loss and mIoU curves from `outputs/metrics.csv`

## Data layout
Expected root (default `data/`):
- `data/lidar/*.parquet`
- `data/lidar_segmentation/*.parquet`
- `data/lidar_calibration/*.parquet`

## Quick start
```bash
python3 train.py --epochs 1 --batch-size 1 --num-points 512 --num-train-segments 2 --num-val-segments 1
python3 evaluate.py --checkpoint outputs/best_model.pt --num-points 512 --num-test-segments 1 --num-ddpm-steps 50
python3 plot_metrics.py
```

Backbone options:
- `--backbone edgeconv` (default, local KNN aggregation)
- `--backbone pointnet` (simpler pointwise/global baseline)
