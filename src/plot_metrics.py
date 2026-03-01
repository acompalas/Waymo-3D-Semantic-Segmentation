"""
plot_metrics.py
===============
Read the Lightning CSVLogger metrics and save loss + mIoU charts.

Run after training:
    python src/plot_metrics.py

Saves to outputs/:
    loss_curve.png   — train loss + val loss over epochs
    miou_curve.png   — val mIoU over epochs
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_OUT_DIR  = Path("outputs")
DEFAULT_LOG_DIR  = DEFAULT_OUT_DIR / "logs"


def find_metrics_csv(log_dir: Path) -> Path:
    """Find the most recent metrics.csv in Lightning's versioned log dirs."""
    csvs = sorted(log_dir.glob("*/metrics.csv"))
    if not csvs:
        # Also check one level deeper (version_0/metrics.csv)
        csvs = sorted(log_dir.glob("**/metrics.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No metrics.csv found under {log_dir}. "
            "Make sure training has run at least one epoch."
        )
    return csvs[-1]  # most recent version


def smooth(values, weight=0.6):
    """Exponential moving average for smoother curves."""
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return np.array(smoothed)


def main():
    out_dir = DEFAULT_OUT_DIR
    log_dir = DEFAULT_LOG_DIR

    csv_path = find_metrics_csv(log_dir)
    print(f"Reading metrics from: {csv_path}")

    df = pd.read_csv(csv_path)

    # Lightning logs step-level and epoch-level rows mixed together.
    # Epoch-level rows have a non-null 'epoch' and the epoch-level metric columns.
    epoch_df = df.dropna(subset=["epoch"]).copy()
    epoch_df["epoch"] = epoch_df["epoch"].astype(int)

    # Aggregate per epoch (some metrics logged multiple times per epoch)
    agg = epoch_df.groupby("epoch").mean(numeric_only=True).reset_index()

    epochs = agg["epoch"].values

    # ── Loss curve ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    if "train/loss_epoch" in agg.columns:
        train_loss = agg["train/loss_epoch"].dropna()
        ax.plot(epochs[:len(train_loss)], smooth(train_loss.values),
                color="steelblue", linewidth=2, label="Train loss")

    if "val/loss" in agg.columns:
        val_loss = agg["val/loss"].dropna()
        ax.plot(epochs[:len(val_loss)], smooth(val_loss.values),
                color="tomato", linewidth=2, label="Val loss")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (MSE)", fontsize=12)
    ax.set_title("Training and Validation Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    loss_path = out_dir / "loss_curve.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {loss_path}")

    # ── mIoU curve ─────────────────────────────────────────────────────────────
    if "val/miou" in agg.columns:
        fig, ax = plt.subplots(figsize=(9, 5))

        miou = agg["val/miou"].dropna()
        ax.plot(epochs[:len(miou)], smooth(miou.values),
                color="seagreen", linewidth=2, label="Val mIoU")

        best_epoch = epochs[miou.values.argmax()]
        best_miou  = miou.values.max()
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.annotate(
            f"Best: {best_miou:.4f}\n(epoch {best_epoch})",
            xy=(best_epoch, best_miou),
            xytext=(best_epoch + max(1, len(epochs) * 0.05), best_miou * 0.95),
            fontsize=9,
            color="gray",
        )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("mIoU", fontsize=12)
        ax.set_title("Validation mIoU (proxy — fast single-step estimate)", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        miou_path = out_dir / "miou_curve.png"
        fig.savefig(miou_path, dpi=150)
        plt.close(fig)
        print(f"Saved → {miou_path}")
    else:
        print("val/miou not found in logs — skipping mIoU chart.")

    print("\nDone. Charts saved to outputs/")


if __name__ == "__main__":
    # Install pandas/matplotlib if missing
    try:
        import pandas
        import matplotlib
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "matplotlib"])
        import pandas
        import matplotlib
        import matplotlib.pyplot as plt

    main()