"""
Download DIV2K dataset for super-resolution training.

Usage
-----
    python download_data.py                    # train + val
    python download_data.py --split train      # only training set
    python download_data.py --split val        # only validation set
    python download_data.py --output my_data   # custom output directory

Downloads HR (high-resolution) images from the official DIV2K server,
extracts them, and places into the correct project folders.
"""

import argparse
import os
import sys
import zipfile
import hashlib
from pathlib import Path

try:
    from urllib.request import urlretrieve, Request, urlopen
    from urllib.error import URLError
except ImportError:
    pass


# ---------------------------------------------------------------------------
# DIV2K URLs and metadata
# ---------------------------------------------------------------------------
DATASETS = {
    "train": {
        "url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "filename": "DIV2K_train_HR.zip",
        "extract_dir": "DIV2K_train_HR",
        "target_dir": "data/train",
        "description": "DIV2K Training set (800 images, ~3.3 GB)",
    },
    "val": {
        "url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        "filename": "DIV2K_valid_HR.zip",
        "extract_dir": "DIV2K_valid_HR",
        "target_dir": "data/val",
        "description": "DIV2K Validation set (100 images, ~440 MB)",
    },
}

# Optional test sets for benchmarking
BENCHMARK_SETS = {
    "Set5": "https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip",
    "Set14": "https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip",
}


# ---------------------------------------------------------------------------
# Progress bar for downloads
# ---------------------------------------------------------------------------
class DownloadProgressBar:
    """Simple progress bar for urlretrieve."""

    def __init__(self, filename: str):
        self.filename = filename
        self.last_percent = -1

    def __call__(self, block_num, block_size, total_size):
        if total_size <= 0:
            print(f"\r  Downloading {self.filename}... {block_num * block_size / 1e6:.1f} MB", end="")
            return
        percent = min(int(block_num * block_size * 100 / total_size), 100)
        if percent != self.last_percent:
            self.last_percent = percent
            downloaded = block_num * block_size / 1e6
            total = total_size / 1e6
            bar_len = 40
            filled = int(bar_len * percent / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {percent:3d}% ({downloaded:.1f}/{total:.1f} MB)", end="")
            if percent == 100:
                print()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def check_connection(url: str) -> bool:
    """Quick connectivity check."""
    try:
        req = Request(url, method="HEAD")
        resp = urlopen(req, timeout=10)
        return resp.status == 200
    except Exception:
        return False


def download_file(url: str, dest_path: str) -> str:
    """Download a file with progress bar. Returns path to downloaded file."""
    if os.path.isfile(dest_path):
        print(f"  File already exists: {dest_path}")
        return dest_path

    print(f"  URL: {url}")
    filename = os.path.basename(dest_path)
    progress = DownloadProgressBar(filename)

    try:
        urlretrieve(url, dest_path, reporthook=progress)
    except URLError as e:
        print(f"\n  [ERROR] Download failed: {e}")
        print("  Try downloading manually from the URL above.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  [CANCELLED] Removing partial download...")
        if os.path.isfile(dest_path):
            os.remove(dest_path)
        sys.exit(1)

    return dest_path


def extract_zip(zip_path: str, extract_to: str) -> str:
    """Extract ZIP archive. Returns path to extracted directory."""
    print(f"  Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"  Extracted to {extract_to}")
    return extract_to


def move_images(src_dir: str, dst_dir: str) -> int:
    """Move/copy image files from src_dir to dst_dir. Returns count."""
    os.makedirs(dst_dir, exist_ok=True)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    count = 0

    for fname in sorted(os.listdir(src_dir)):
        if Path(fname).suffix.lower() in extensions:
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            if not os.path.isfile(dst):
                os.rename(src, dst)
            count += 1

    return count


def download_split(split: str, base_dir: str, keep_zip: bool = False):
    """Download and set up one split (train or val)."""
    info = DATASETS[split]
    print(f"\n{'='*60}")
    print(f"  {info['description']}")
    print(f"{'='*60}")

    target_dir = os.path.join(base_dir, info["target_dir"])

    # Check if already downloaded
    if os.path.isdir(target_dir):
        existing = [f for f in os.listdir(target_dir)
                    if Path(f).suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if len(existing) > 0:
            print(f"  Already have {len(existing)} images in {target_dir}")
            answer = input("  Re-download? [y/N]: ").strip().lower()
            if answer != "y":
                return

    # Download
    zip_path = os.path.join(base_dir, info["filename"])
    download_file(info["url"], zip_path)

    # Extract
    extract_zip(zip_path, base_dir)

    # Move images to target directory
    extracted = os.path.join(base_dir, info["extract_dir"])
    if os.path.isdir(extracted):
        count = move_images(extracted, target_dir)
        print(f"  ✓ {count} images moved to {target_dir}")
        # Clean up extracted directory
        import shutil
        shutil.rmtree(extracted, ignore_errors=True)
    else:
        print(f"  [WARN] Expected directory {extracted} not found after extraction")

    # Clean up ZIP
    if not keep_zip and os.path.isfile(zip_path):
        os.remove(zip_path)
        print(f"  Removed {info['filename']}")


def print_summary(base_dir: str):
    """Print summary of available data."""
    print(f"\n{'='*60}")
    print("  Dataset summary")
    print(f"{'='*60}")
    for split, info in DATASETS.items():
        target = os.path.join(base_dir, info["target_dir"])
        if os.path.isdir(target):
            imgs = [f for f in os.listdir(target)
                    if Path(f).suffix.lower() in {".png", ".jpg", ".jpeg"}]
            print(f"  {split:>5}: {len(imgs):>4} images  ({target})")
        else:
            print(f"  {split:>5}:    - not downloaded  ({target})")

    print(f"\nГотово! Для обучения выполните:")
    print(f"  python train.py --data_dir data/train --val_dir data/val --scale 2 --epochs 100")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download DIV2K dataset for super-resolution training"
    )
    parser.add_argument(
        "--split", type=str, default="all",
        choices=["all", "train", "val"],
        help="Which split to download (default: all)",
    )
    parser.add_argument(
        "--output", type=str, default=".",
        help="Base project directory (default: current dir)",
    )
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="Keep downloaded ZIP files after extraction",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(args.output)
    os.makedirs(os.path.join(base_dir, "data", "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data", "val"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data", "test"), exist_ok=True)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║         DIV2K Dataset Downloader                        ║")
    print("║         для Super-Resolution App                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Output directory: {base_dir}")

    # Connectivity check
    print("\n  Checking connection...")
    if not check_connection(DATASETS["val"]["url"]):
        print("  [ERROR] Cannot reach DIV2K server.")
        print("  Check your internet connection or download manually:")
        for split, info in DATASETS.items():
            print(f"    {split}: {info['url']}")
        sys.exit(1)
    print("  Connection OK ✓")

    # Download requested splits
    if args.split in ("all", "train"):
        download_split("train", base_dir, args.keep_zip)
    if args.split in ("all", "val"):
        download_split("val", base_dir, args.keep_zip)

    print_summary(base_dir)


if __name__ == "__main__":
    main()
