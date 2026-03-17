#!/usr/bin/env python3
from __future__ import annotations

"""
Minimal Colab helper for Waymo preprocessing.

Run this from a Colab notebook with `%run`:

    %cd /content/final_project
    %run scripts/colab_preprocess.py \
      --drive-root MyDrive/ece271b \
      --range-workers 4 \
      --point-workers 4
"""

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import urllib.request


REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVE_MOUNT = Path("/content/drive")
GCS_MOUNT = Path("/content/gcs")
WAYMO_BUCKET = "waymo_open_dataset_v_2_0_1"
SEGMENTATION_PROTO_URL = (
    "https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/"
    "refs/heads/master/src/waymo_open_dataset/protos/segmentation.proto"
)
SEGMENTATION_PROTO_PATH = Path("/content/segmentation.proto")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mount Colab storage and preprocess the Waymo dataset.")
    parser.add_argument("--drive-root", type=Path, default=Path("MyDrive/ece271b"))
    parser.add_argument("--range-workers", type=int, default=4)
    parser.add_argument("--point-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-python-deps", action="store_true")
    return parser.parse_args()


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def run_command(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"\n$ {shlex.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def ensure_drive_mount() -> None:
    if not in_colab():
        raise RuntimeError("This helper is intended to be run inside Google Colab.")

    from google.colab import drive

    DRIVE_MOUNT.mkdir(parents=True, exist_ok=True)
    print(f"Mounting Google Drive at {DRIVE_MOUNT} ...")
    drive.mount(str(DRIVE_MOUNT))


def ensure_google_auth() -> None:
    if not in_colab():
        raise RuntimeError("This helper is intended to be run inside Google Colab.")

    from google.colab import auth

    print("Authenticating to Google Cloud ...")
    auth.authenticate_user()


def install_gcsfuse() -> None:
    if shutil.which("gcsfuse") is not None:
        return

    release = (
        subprocess.run(
            ["lsb_release", "-c", "-s"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )
    keyring = Path("/usr/share/keyrings/cloud.google.asc")
    repo_line = f"deb [signed-by={keyring}] https://packages.cloud.google.com/apt gcsfuse-{release} main"
    repo_path = Path("/etc/apt/sources.list.d/gcsfuse.list")

    print("Installing gcsfuse ...")
    run_command(
        [
            "bash",
            "-lc",
            f"curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | "
            f"gpg --dearmor -o {shlex.quote(str(keyring))}",
        ]
    )
    repo_path.write_text(repo_line + "\n", encoding="utf-8")
    run_command(["apt-get", "update"])
    run_command(["apt-get", "install", "-y", "gcsfuse"])


def ensure_gcs_mount() -> None:
    GCS_MOUNT.mkdir(parents=True, exist_ok=True)
    if os.path.ismount(GCS_MOUNT):
        print(f"Reusing existing GCS mount at {GCS_MOUNT}.")
        return

    install_gcsfuse()
    print(f"Mounting gs://{WAYMO_BUCKET} at {GCS_MOUNT} ...")
    run_command(["gcsfuse", "--implicit-dirs", WAYMO_BUCKET, str(GCS_MOUNT)])


def maybe_install_python_deps() -> None:
    print("Installing minimal Python preprocessing dependencies ...")
    run_command([sys.executable, "-m", "pip", "install", "-q", "numpy", "polars", "tqdm"])


def download_segmentation_proto() -> Path:
    print(f"Downloading segmentation proto to {SEGMENTATION_PROTO_PATH} ...")
    urllib.request.urlretrieve(SEGMENTATION_PROTO_URL, SEGMENTATION_PROTO_PATH)
    return SEGMENTATION_PROTO_PATH


def preprocess_range_images(
    raw_data_dir: Path,
    output_dir: Path,
    proto_path: Path,
    num_workers: int,
    overwrite: bool,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "data_pipeline" / "preprocess_range_images.py"),
        "--data-dir",
        str(raw_data_dir),
        "--output-dir",
        str(output_dir),
        "--proto-path",
        str(proto_path),
        "--num-workers",
        str(num_workers),
    ]
    if overwrite:
        cmd.append("--overwrite")
    run_command(cmd, cwd=REPO_ROOT)


def preprocess_point_clouds(range_dir: Path, output_dir: Path, num_workers: int, overwrite: bool) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "data_pipeline" / "preprocess_point_clouds.py"),
        "--range-dir",
        str(range_dir),
        "--output-dir",
        str(output_dir),
        "--num-workers",
        str(num_workers),
    ]
    if overwrite:
        cmd.append("--overwrite")
    run_command(cmd, cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()
    if not in_colab():
        raise RuntimeError("This helper is intended to be run inside Google Colab.")

    ensure_drive_mount()
    ensure_google_auth()
    ensure_gcs_mount()
    if not args.skip_python_deps:
        maybe_install_python_deps()

    drive_root = (DRIVE_MOUNT / args.drive_root).resolve()
    drive_root.mkdir(parents=True, exist_ok=True)
    raw_data_dir = GCS_MOUNT
    range_output_dir = drive_root / "preprocessed" / "range_images"
    point_output_dir = drive_root / "preprocessed" / "point_clouds"
    proto_path = download_segmentation_proto()

    print("\nResolved paths:")
    print(f"  raw_data_dir:     {raw_data_dir}")
    print(f"  drive_root:       {drive_root}")
    print(f"  proto_path:       {proto_path}")
    print(f"  range_output_dir: {range_output_dir}")
    print(f"  point_output_dir: {point_output_dir}")

    preprocess_range_images(
        raw_data_dir=raw_data_dir,
        output_dir=range_output_dir,
        proto_path=proto_path,
        num_workers=int(args.range_workers),
        overwrite=bool(args.overwrite),
    )
    preprocess_point_clouds(
        range_dir=range_output_dir,
        output_dir=point_output_dir,
        num_workers=int(args.point_workers),
        overwrite=bool(args.overwrite),
    )

    print("\nColab preprocessing complete.")


if __name__ == "__main__":
    main()
