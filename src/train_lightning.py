import argparse

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from lightning_datamodule import WaymoLidarDataModule
from lightning_model import LinearSVMPointClassifier
from lightning_range_model import RangeImageUNetSegmenter


def _parse_scales(raw: str) -> tuple[int, ...]:
    items = [x.strip() for x in str(raw).split(",")]
    values = tuple(int(x) for x in items if x)
    if not values:
        raise ValueError("knn-scales must contain at least one integer")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="preprocessed/point_clouds")
    p.add_argument("--dataset-type", type=str, default="point_clouds", choices=["point_clouds", "range_images"])
    p.add_argument("--train-subdirs", type=str, default="training")
    p.add_argument(
        "--val-subdirs",
        type=str,
        default="",
        help="Comma-separated source subdirs for validation. Empty means split from train-subdirs via --val-fraction.",
    )
    p.add_argument(
        "--test-subdirs",
        type=str,
        default="validation",
        help="Comma-separated source subdirs for test-time reporting (default: validation).",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-points", type=int, default=327680)#16384)
    p.add_argument("--num-classes", type=int, default=23)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--worker-start-method", type=str, default="spawn")
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--svm-reg", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--knn-scales", type=str, default="16,32,64")
    p.add_argument("--knn-support-size", type=int, default=327680)#16384)
    p.add_argument("--knn-query-chunk", type=int, default=16384)
    p.add_argument("--unet-base-channels", type=int, default=32)
    p.add_argument("--unet-depth", type=int, default=4)
    p.add_argument("--unet-dropout", type=float, default=0.0)
    p.add_argument("--no-balanced-class-weights", action="store_true")
    p.add_argument("--early-stopping-patience", type=int, default=10)
    p.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    p.add_argument(
        "--train-segment-fraction",
        type=float,
        default=1.0,
        help="Use only this fraction of selected training segments (0,1]. Validation/test are unchanged.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    datamodule = WaymoLidarDataModule(
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        num_points=args.num_points,
        num_classes=args.num_classes,
        train_subdirs=args.train_subdirs,
        val_subdirs=args.val_subdirs,
        test_subdirs=args.test_subdirs,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        worker_start_method=args.worker_start_method,
        seed=args.seed,
        max_cached_segments=1,
        balanced_class_weights=not bool(args.no_balanced_class_weights),
        train_segment_fraction=args.train_segment_fraction,
    )
    if args.dataset_type == "point_clouds":
        model = LinearSVMPointClassifier(
            num_classes=args.num_classes,
            learning_rate=args.lr,
            svm_reg=args.svm_reg,
            margin=args.margin,
            knn_scales=_parse_scales(args.knn_scales),
            knn_support_size=args.knn_support_size,
            knn_query_chunk=args.knn_query_chunk,
            use_balanced_class_weights=not bool(args.no_balanced_class_weights),
        )
    elif args.dataset_type == "range_images":
        model = RangeImageUNetSegmenter(
            num_classes=args.num_classes,
            in_channels=4,
            base_channels=args.unet_base_channels,
            depth=args.unet_depth,
            dropout=args.unet_dropout,
            learning_rate=args.lr,
            use_balanced_class_weights=not bool(args.no_balanced_class_weights),
        )
    else:
        raise ValueError(f"Unsupported dataset_type: {args.dataset_type}")

    checkpoint_cb = ModelCheckpoint(
        monitor="val_mIoU",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch:02d}-{val_mIoU:.4f}",
    )
    early_stopping_cb = EarlyStopping(
        monitor="val_mIoU",
        mode="max",
        patience=int(args.early_stopping_patience),
        min_delta=float(args.early_stopping_min_delta),
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
