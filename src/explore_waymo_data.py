"""
explore_waymo_data.py  — MEMORY SAFE VERSION
=============================================
Run from your project root:

    cd "D:/ECE 271B SemSeg Project"
    python src/explore_waymo_data.py

Memory strategy:
  - scan_parquet (lazy) for all metadata / key-column queries
  - Array data (range images, seg labels) is processed ONE FILE at a time
    so only ~one segment worth of data (~200 MB) lives in RAM at once
  - del + garbage collect after each file
"""

import polars as pl
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("Waymo Data/training")
LIDAR_DIR  = DATA_ROOT / "lidar"
SEG_DIR    = DATA_ROOT / "lidar_segmentation"
CALIB_DIR  = DATA_ROOT / "lidar_calibration"
OUT_JSON   = Path("src/data_exploration_report.json")

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
LASER_NAMES = {1:"TOP", 2:"FRONT", 3:"SIDE_LEFT", 4:"SIDE_RIGHT", 5:"REAR"}


def section(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")


def reshape(values, shape_col, dtype=np.float32):
    shape = tuple(int(x) for x in shape_col)
    return np.asarray(values, dtype=dtype).reshape(shape)


# ── 1. File counts ─────────────────────────────────────────────────────────────
def check_files():
    section("1. FILE COUNTS")
    lidar_files = sorted(LIDAR_DIR.glob("*.parquet"))
    seg_files   = sorted(SEG_DIR.glob("*.parquet"))
    calib_files = sorted(CALIB_DIR.glob("*.parquet"))

    lidar_stems = {f.stem for f in lidar_files}
    seg_stems   = {f.stem for f in seg_files}
    matched     = lidar_stems & seg_stems

    print(f"  lidar/               {len(lidar_files)} files")
    print(f"  lidar_segmentation/  {len(seg_files)} files")
    print(f"  lidar_calibration/   {len(calib_files)} files")
    print(f"  matched stems        {len(matched)}")
    if lidar_stems - seg_stems:
        print(f"  WARNING: lidar without seg: {lidar_stems - seg_stems}")
    if seg_stems - lidar_stems:
        print(f"  WARNING: seg without lidar: {seg_stems - lidar_stems}")

    return {
        "lidar": len(lidar_files), "segmentation": len(seg_files),
        "calibration": len(calib_files), "matched": len(matched),
        "lidar_files": [f.name for f in lidar_files],
        "seg_files":   [f.name for f in seg_files],
    }


# ── 2. Schema — scan ONE file, no data loaded ──────────────────────────────────
def check_schemas():
    section("2. PARQUET SCHEMAS (first file of each type)")
    for label, directory in [
        ("lidar", LIDAR_DIR),
        ("lidar_segmentation", SEG_DIR),
        ("lidar_calibration", CALIB_DIR),
    ]:
        first = sorted(directory.glob("*.parquet"))[0]
        lf = pl.scan_parquet(str(first))  # lazy — reads footer only
        print(f"\n  [{label}]  {first.name}")
        for name, dtype in zip(lf.schema.names(), lf.schema.dtypes()):
            print(f"    {name:<72} {dtype}")


# ── 3. Segment / timestamp inventory — key cols only ──────────────────────────
def check_segments():
    section("3. SEGMENT / TIMESTAMP INVENTORY")

    # key columns are tiny — safe to collect across all files
    seg_keys = (
        pl.scan_parquet(str(SEG_DIR / "*.parquet"))
        .select(["key.segment_context_name",
                 "key.frame_timestamp_micros",
                 "key.laser_name"])
        .collect()
    )

    segments      = seg_keys["key.segment_context_name"].unique().to_list()
    lasers_in_seg = seg_keys["key.laser_name"].unique().sort().to_list()
    frames_per_seg = (
        seg_keys
        .group_by("key.segment_context_name")
        .agg(pl.col("key.frame_timestamp_micros").n_unique().alias("n"))
        ["n"].to_list()
    )

    print(f"\n  Unique segments : {len(segments)}")
    print(f"  Frames/segment  — min:{min(frames_per_seg)}  max:{max(frames_per_seg)}  "
          f"mean:{np.mean(frames_per_seg):.1f}")
    print(f"  Lasers in seg   : {[LASER_NAMES.get(l,l) for l in lasers_in_seg]}")

    # Pick the segment with the most TOP-lidar labeled frames for the deep dive
    best_seg = (
        seg_keys
        .filter(pl.col("key.laser_name") == 1)
        .group_by("key.segment_context_name")
        .agg(pl.col("key.frame_timestamp_micros").n_unique().alias("n"))
        .sort("n", descending=True)
        ["key.segment_context_name"][0]
    )

    return {
        "n_segments": len(segments),
        "frames_per_seg_min": int(min(frames_per_seg)),
        "frames_per_seg_max": int(max(frames_per_seg)),
        "frames_per_seg_mean": float(np.mean(frames_per_seg)),
        "lasers_with_seg": lasers_in_seg,
        "best_segment_for_dive": best_seg,
    }


# ── 4. Deep dive — filter to ONE frame before collecting ──────────────────────
def deep_dive(chosen_segment):
    section(f"4. DEEP DIVE  {chosen_segment[:50]}...")

    # In v2.0.1 the parquet filenames match segment IDs
    lidar_path = LIDAR_DIR / f"{chosen_segment}.parquet"
    seg_path   = SEG_DIR   / f"{chosen_segment}.parquet"
    calib_path = CALIB_DIR / f"{chosen_segment}.parquet"

    # Fall back to wildcard scan + filter if filenames don't match
    lp = str(lidar_path) if lidar_path.exists() else str(LIDAR_DIR / "*.parquet")
    sp = str(seg_path)   if seg_path.exists()   else str(SEG_DIR   / "*.parquet")
    cp = str(calib_path) if calib_path.exists() else str(CALIB_DIR / "*.parquet")

    # Get available timestamps — key cols only
    ts_list = (
        pl.scan_parquet(sp)
        .filter(
            (pl.col("key.segment_context_name") == chosen_segment) &
            (pl.col("key.laser_name") == 1)
        )
        .select("key.frame_timestamp_micros")
        .unique()
        .sort("key.frame_timestamp_micros")
        .collect()
        ["key.frame_timestamp_micros"].to_list()
    )
    chosen_ts = int(ts_list[0])
    print(f"\n  Labeled TOP-lidar frames in segment : {len(ts_list)}")
    print(f"  Inspecting timestamp               : {chosen_ts}")

    # Collect ONLY the one frame we need
    one_lidar = (
        pl.scan_parquet(lp)
        .filter(
            (pl.col("key.segment_context_name") == chosen_segment) &
            (pl.col("key.frame_timestamp_micros") == chosen_ts)
        )
        .collect()
    )
    one_seg = (
        pl.scan_parquet(sp)
        .filter(
            (pl.col("key.segment_context_name") == chosen_segment) &
            (pl.col("key.frame_timestamp_micros") == chosen_ts)
        )
        .collect()
    )
    one_calib = (
        pl.scan_parquet(cp)
        .filter(pl.col("key.segment_context_name") == chosen_segment)
        .collect()
    )

    seg_laser_ids = one_seg["key.laser_name"].to_list()
    laser_report  = {}

    for laser_id in sorted(one_lidar["key.laser_name"].unique().to_list()):
        name = LASER_NAMES.get(laser_id, str(laser_id))
        lr   = one_lidar.filter(pl.col("key.laser_name") == laser_id).row(0, named=True)
        has_seg = laser_id in seg_laser_ids

        shape1 = tuple(int(x) for x in lr["[LiDARComponent].range_image_return1.shape"])
        ri1    = reshape(lr["[LiDARComponent].range_image_return1.values"], shape1)
        valid  = ri1[:, :, 0] > 0
        n_valid = int(valid.sum())
        total   = shape1[0] * shape1[1]

        ri2_vals = lr["[LiDARComponent].range_image_return2.values"]
        has_r2   = ri2_vals is not None and len(ri2_vals) > 0
        n_r2     = 0
        if has_r2:
            s2   = tuple(int(x) for x in lr["[LiDARComponent].range_image_return2.shape"])
            ri2  = reshape(ri2_vals, s2)
            n_r2 = int((ri2[:, :, 0] > 0).sum())

        print(f"\n  ── Laser {laser_id} ({name}) ──")
        print(f"     return1 shape : {shape1}  (H, W, C)")
        print(f"     valid pixels  : {n_valid:,} / {total:,}  ({100*n_valid/total:.1f}%)")
        print(f"     has return2   : {has_r2}   valid_r2: {n_r2:,}")
        print(f"     has seg labels: {has_seg}")

        if laser_id == 1:
            rng = ri1[:, :, 0][valid]
            ins = ri1[:, :, 1][valid] if shape1[2] > 1 else None
            print(f"     range — min:{rng.min():.2f}  max:{rng.max():.2f}  mean:{rng.mean():.2f} m")
            if ins is not None:
                print(f"     intens— min:{ins.min():.4f}  max:{ins.max():.4f}  mean:{ins.mean():.4f}")

        if has_seg:
            sr  = one_seg.filter(pl.col("key.laser_name") == laser_id).row(0, named=True)
            ss  = tuple(int(x) for x in sr["[LiDARSegmentationLabelComponent].range_image_return1.shape"])
            ok  = ss[:2] == shape1[:2]
            print(f"     seg shape     : {ss}  ch0=instance_id, ch1=semantic_label")
            print(f"     spatial align : {'✓ MATCH' if ok else '✗ MISMATCH'}")

        laser_report[name] = {
            "laser_id": laser_id, "ri_shape": list(shape1),
            "valid_points": n_valid, "fill_pct": round(100*n_valid/total, 2),
            "has_return2": has_r2, "valid_r2": n_r2, "has_seg": has_seg,
        }

    # Calibration
    tc = one_calib.filter(pl.col("key.laser_name") == 1)
    if len(tc):
        crow    = tc.row(0, named=True)
        inc_min = float(crow["[LiDARCalibrationComponent].beam_inclination.min"])
        inc_max = float(crow["[LiDARCalibrationComponent].beam_inclination.max"])
        iv      = crow["[LiDARCalibrationComponent].beam_inclination.values"]
        T       = np.asarray(crow["[LiDARCalibrationComponent].extrinsic.transform"],
                             dtype=np.float64).reshape(4,4)
        print(f"\n  ── Calibration (TOP laser) ──")
        print(f"     beam_inclination : {np.degrees(inc_min):.2f}° → {np.degrees(inc_max):.2f}°")
        print(f"     per-beam values  : {'yes len=' + str(len(iv)) if iv else 'no (use linear interp)'}")
        print(f"     extrinsic T:\n{T}")
        laser_report["_calibration"] = {
            "inc_min_deg": round(np.degrees(inc_min), 3),
            "inc_max_deg": round(np.degrees(inc_max), 3),
            "per_beam": iv is not None and len(iv) > 0,
        }

    del one_lidar, one_seg, one_calib
    return laser_report


# ── 5. Label distribution — ONE parquet file at a time ────────────────────────
def label_distribution():
    section("5. LABEL DISTRIBUTION (TOP lidar, all segments)")
    print("  Processing one file at a time — RAM stays low...\n")

    lidar_by_stem = {f.stem: f for f in sorted(LIDAR_DIR.glob("*.parquet"))}
    seg_by_stem   = {f.stem: f for f in sorted(SEG_DIR.glob("*.parquet"))}
    common        = sorted(set(lidar_by_stem) & set(seg_by_stem))

    label_counts   = defaultdict(int)
    total_points   = 0
    frames_counted = 0

    for i, stem in enumerate(common):
        print(f"  [{i+1:3d}/{len(common)}] {stem[:55]}", end="\r")

        # Lazy scan → filter to TOP only → select only the columns we need → collect
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
            li = reshape(row["[LiDARComponent].range_image_return1.values"],
                         row["[LiDARComponent].range_image_return1.shape"])
            valid = li[:, :, 0] > 0

            sg = reshape(
                row["[LiDARSegmentationLabelComponent].range_image_return1.values"],
                row["[LiDARSegmentationLabelComponent].range_image_return1.shape"],
                dtype=np.int32,
            )
            labels = sg[:, :, 1][valid]  # ch0=instance_id, ch1=semantic_label (per Waymo proto)
            for lbl, cnt in zip(*np.unique(labels, return_counts=True)):
                label_counts[int(lbl)] += int(cnt)
            total_points  += len(labels)
            frames_counted += 1

        del lf_li, lf_sg, joined  # free RAM immediately

    print(f"\n\n  Total labeled points : {total_points:,}")
    print(f"  Frames counted       : {frames_counted}")
    print(f"\n  {'Cls':>4}  {'Name':<25}  {'Count':>12}  {'Pct':>8}")
    print(f"  {'─'*4}  {'─'*25}  {'─'*12}  {'─'*8}")

    label_dist = {}
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct  = 100.0 * cnt / max(total_points, 1)
        name = CLASS_NAMES.get(lbl, f"UNKNOWN_{lbl}")
        print(f"  {lbl:>4}  {name:<25}  {cnt:>12,}  {pct:>7.2f}%")
        label_dist[lbl] = {"name": name, "count": cnt, "pct": round(pct, 4)}

    return label_dist, total_points, frames_counted


# ── 6. Design summary ──────────────────────────────────────────────────────────
def design_summary(laser_report, label_dist):
    section("6. DESIGN SUMMARY FOR DIFFUSION MODEL")
    top     = laser_report.get("TOP", {})
    h, w, c = top.get("ri_shape", [64, 2650, 4])
    fill    = top.get("fill_pct", 0)
    n_cls   = len([k for k in label_dist if int(k) != 0])

    print(f"""
  Range image (TOP lidar):
    H = {h}    rows  = laser beams (vertical resolution)
    W = {w}   cols  = azimuth steps (horizontal resolution)
    C = {c}    chans = [range, intensity, elongation, ...]
    fill ≈ {fill:.0f}%  (rest is empty sky / behind vehicle)

  Segmentation map  (same H×W):
    channel 0 = instance id     (not needed for semseg)
    channel 1 = semantic label  (int, 0–22)

  Diffusion model tensor shapes:
    x_cond  : ({h}, {w}, {c})        LiDAR features, fixed
    x_0     : ({h}, {w}, {n_cls+1})  one-hot label map  ({n_cls} classes + undefined)
    U-Net in: ({h}, {w}, {c+n_cls+1}) concat(x_t, x_cond)
    Loss mask: valid pixels only (range > 0, ≈{fill:.0f}% of pixels)

  Aspect ratio {w}:{h} ≈ {w//h}:1
    Power-of-2 H → {2**int(np.ceil(np.log2(h)))}
    Consider cropping W to 512 (drops far lateral returns, saves memory)
    """)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    report = {}

    report["files"]    = check_files()
    check_schemas()
    seg_info           = check_segments()
    report["segments"] = seg_info

    laser_report         = deep_dive(seg_info["best_segment_for_dive"])
    report["laser_dive"] = laser_report

    label_dist, total, frames        = label_distribution()
    report["label_distribution"]     = label_dist
    report["total_labeled_points"]   = total
    report["frames_counted"]         = frames

    design_summary(laser_report, label_dist)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved → {OUT_JSON}\n")


if __name__ == "__main__":
    main()