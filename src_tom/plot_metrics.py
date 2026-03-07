"""Plot training metrics from outputs/metrics.csv."""

import argparse
from pathlib import Path

import csv


def parse_args():
    parser = argparse.ArgumentParser(description="Plot train/val loss and mIoU curves.")
    parser.add_argument("--metrics-csv", type=str, default="outputs/metrics.csv")
    parser.add_argument("--out-dir", type=str, default="outputs")
    return parser.parse_args()


def read_metrics(path: Path):
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if k != "epoch" else int(v) for k, v in row.items()})
    return rows


def main():
    args = parse_args()
    metrics_path = Path(args.metrics_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_path}")

    rows = read_metrics(metrics_path)
    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    train_miou = [r["train_miou"] for r in rows]
    val_miou = [r["val_miou"] for r in rows]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plots.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="train_loss")
    ax.plot(epochs, val_loss, label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    loss_path = out_dir / "loss_curve.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_miou, label="train_mIoU")
    ax.plot(epochs, val_miou, label="val_mIoU")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mIoU")
    ax.legend()
    fig.tight_layout()
    miou_path = out_dir / "miou_curve.png"
    fig.savefig(miou_path, dpi=150)
    plt.close(fig)

    print(f"Saved {loss_path}")
    print(f"Saved {miou_path}")


if __name__ == "__main__":
    main()
