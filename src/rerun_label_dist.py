"""
rerun_label_dist.py
===================
Reruns only the label distribution (section 5) with the corrected channel index.
Semantic label is channel 1, instance ID is channel 0.

    cd "D:/ECE 271B SemSeg Project"
    python src/rerun_label_dist.py
"""

import polars as pl
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_ROOT = Path("Waymo Data/training")
LIDAR_DIR = DATA_ROOT / "lidar"
SEG_DIR   = DATA_ROOT / "lidar_segmentation"

CLASS_NAMES = {
    0:  "TYPE_UNDEFINED",     1:  "TYPE_CAR",
    2:  "TYPE_TRUCK",         3:  "TYPE_BUS",
    4:  "TYPE_OTHER_VEHICLE", 5:  "TYPE_MOTORCYCLIST",
    6:  "TYPE_BICYCLIST",     7:  "TYPE_PEDESTRIAN",
    8:  "TYPE_SIGN",          9:  "TYPE_TRAFFIC_LIGHT",
    10: "TYPE_POLE",          11: "TYPE_CONSTRUCTION_CONE",
    12: "TYPE_BICYCLE",       13: "TYPE_MOTORCYCLE",
    14: "TYPE_BUILDING",      15: "TYPE_VEGETATION",
    16: "TYPE_TREE_TRUNK",    17: "TYPE_CURB",
    18: "TYPE_ROAD",          19: "TYPE_LANE_MARKER",
    20: "TYPE_OTHER_GROUND",  21: "TYPE_WALKABLE",
    22: "TYPE_SIDEWALK",
}

def reshape(values, shape_col, dtype=np.float32):
    return np.asarray(values, dtype=dtype).reshape(tuple(int(x) for x in shape_col))

lidar_by_stem = {f.stem: f for f in sorted(LIDAR_DIR.glob("*.parquet"))}
seg_by_stem   = {f.stem: f for f in sorted(SEG_DIR.glob("*.parquet"))}
common        = sorted(set(lidar_by_stem) & set(seg_by_stem))

label_counts   = defaultdict(int)
total_points   = 0
frames_counted = 0

print(f"Processing {len(common)} files (channel 1 = semantic label)...\n")

for i, stem in enumerate(common):
    print(f"  [{i+1:3d}/{len(common)}] {stem[:55]}", end="\r")

    lf_li = (
        pl.scan_parquet(str(lidar_by_stem[stem]))
        .filter(pl.col("key.laser_name") == 1)
        .select([
            "key.frame_timestamp_micros",
            "[LiDARComponent].range_image_return1.values",
            "[LiDARComponent].range_image_return1.shape",
        ])
        .collect()
    )
    lf_sg = (
        pl.scan_parquet(str(seg_by_stem[stem]))
        .filter(pl.col("key.laser_name") == 1)
        .select([
            "key.frame_timestamp_micros",
            "[LiDARSegmentationLabelComponent].range_image_return1.values",
            "[LiDARSegmentationLabelComponent].range_image_return1.shape",
        ])
        .collect()
    )

    joined = lf_li.join(lf_sg, on="key.frame_timestamp_micros", how="inner")

    for row in joined.iter_rows(named=True):
        li    = reshape(row["[LiDARComponent].range_image_return1.values"],
                        row["[LiDARComponent].range_image_return1.shape"])
        valid = li[:, :, 0] > 0

        sg = reshape(
            row["[LiDARSegmentationLabelComponent].range_image_return1.values"],
            row["[LiDARSegmentationLabelComponent].range_image_return1.shape"],
            dtype=np.int32,
        )
        # ch0 = instance_id (large arbitrary ints)
        # ch1 = semantic_label (0-22)
        labels = sg[:, :, 1][valid]

        for lbl, cnt in zip(*np.unique(labels, return_counts=True)):
            label_counts[int(lbl)] += int(cnt)
        total_points  += len(labels)
        frames_counted += 1

    del lf_li, lf_sg, joined

print(f"\n\n  Total labeled points : {total_points:,}")
print(f"  Frames counted       : {frames_counted}")
print(f"\n  {'Cls':>4}  {'Name':<25}  {'Count':>12}  {'Pct':>8}")
print(f"  {'─'*4}  {'─'*25}  {'─'*12}  {'─'*8}")

for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
    pct  = 100.0 * cnt / max(total_points, 1)
    name = CLASS_NAMES.get(lbl, f"UNKNOWN_{lbl}")
    print(f"  {lbl:>4}  {name:<25}  {cnt:>12,}  {pct:>7.2f}%")