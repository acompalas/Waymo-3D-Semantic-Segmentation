import polars as pl
import numpy as np

# grab one seg file
seg = pl.read_parquet("Waymo Data/training/lidar_segmentation/<any_file>.parquet")
row = seg.filter(pl.col("key.laser_name") == 1).row(0, named=True)
shape = tuple(int(x) for x in row["[LiDARSegmentationLabelComponent].range_image_return1.shape"])
labels = np.array(row["[LiDARSegmentationLabelComponent].range_image_return1.values"]).reshape(shape)
print("Unique label values found:", sorted(np.unique(labels[:,:,0]).tolist()))
print("Unique instance values found:", sorted(np.unique(labels[:,:,1]).tolist())[:10], "...")