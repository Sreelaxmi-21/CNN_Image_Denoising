"""
scripts/download_dataset.py
===========================
Automated dataset preparation script for the Image Denoising benchmark.

Curated Dataset Strategy:
-------------------------
1. Test Benchmark: Set12 benchmark dataset (12 classic DIP test images).
2. Training Set: 35 clean natural images from standard BSDS benchmark.
3. Validation Set: 10 clean natural images for validation during training.

Features:
- Downloads lightweight benchmark images directly (< 5MB total).
- Includes an automatic offline fallback to curated natural images from `skimage.data`
  (e.g., astronaut, chelsea, coffee, rocket, eagle, etc.) if network is unavailable.
- Establishes a deterministic, non-overlapping Train / Validation / Test split.
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


GITHUB_RAW_BASE = "https://raw.githubusercontent.com/SaoYan/DnCNN-PyTorch/master/data"


def setup_directories(base_dir: Path):
    """Creates the data directory structure."""
    train_dir = base_dir / "raw" / "train"
    val_dir = base_dir / "raw" / "val"
    test_dir = base_dir / "test_benchmark"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    return train_dir, val_dir, test_dir


def download_file(url: str, dest_path: Path, timeout: int = 10) -> bool:
    """Downloads a single file from URL to dest_path."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                dest_path.write_bytes(response.read())
                return True
    except Exception as e:
        return False
    return False


def populate_from_skimage(train_dir: Path, val_dir: Path, test_dir: Path):
    """
    Fallback: populates train, val, and test benchmark using standard natural
    images built into scikit-image (no internet required).
    """
    import skimage.data as skdata

    print("[Info] Populating dataset from curated scikit-image natural benchmarks...")
    available_images = [
        ("astronaut", skdata.astronaut),
        ("chelsea", skdata.chelsea),
        ("coffee", skdata.coffee),
        ("rocket", skdata.rocket),
        ("cat", skdata.cat),
        ("eagle", skdata.eagle),
        ("grass", skdata.grass),
        ("gravel", skdata.gravel),
        ("horse", skdata.horse),
        ("camera", skdata.camera),
        ("coins", skdata.coins),
        ("clock", skdata.clock),
        ("moon", skdata.moon),
        ("page", skdata.page),
        ("text", skdata.text),
        ("colorwheel", skdata.colorwheel),
    ]

    # Save images in RGB format
    count = 0
    for name, func in available_images:
        try:
            arr = func()
            if arr.ndim == 2:
                # Convert grayscale to 3-channel RGB for uniform pipeline handling
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]

            img = Image.fromarray(arr.astype(np.uint8))
            
            # Distribute into train, val, and test
            if count < 10:
                img.save(train_dir / f"sk_{name}.png")
            elif count < 13:
                img.save(val_dir / f"sk_{name}.png")
            else:
                img.save(test_dir / f"sk_{name}.png")
            count += 1
        except Exception as e:
            continue

    print(f"[Success] Extracted {count} natural images across train/val/test splits.")


def download_standard_benchmark(base_dir: Path, num_train: int = 35, num_val: int = 10):
    """
    Downloads curated BSDS and Set12 images from GitHub repository,
    with automatic fallback to scikit-image.
    """
    train_dir, val_dir, test_dir = setup_directories(base_dir)

    # 1. Download Set12 test benchmark images (01.png to 12.png)
    print("[1/3] Downloading Set12 benchmark test images...")
    set12_downloaded = 0
    for i in range(1, 13):
        fname = f"{i:02d}.png"
        url = f"{GITHUB_RAW_BASE}/Set12/{fname}"
        target = test_dir / fname
        if not target.exists():
            success = download_file(url, target)
            if success:
                set12_downloaded += 1
        else:
            set12_downloaded += 1

    print(f"      Set12 test images available: {set12_downloaded}/12")

    # If network failed completely on first test image, trigger fallback
    if set12_downloaded == 0:
        print("[Warning] Network download failed or unavailable. Falling back to local skimage benchmarks.")
        populate_from_skimage(train_dir, val_dir, test_dir)
        return

    # 2. Download Training images (curated subset of 35 images)
    print(f"[2/3] Downloading {num_train} curated natural training images...")
    train_downloaded = 0
    for i in range(1, num_train + 1):
        fname = f"test_{i:03d}.png"
        url = f"{GITHUB_RAW_BASE}/train/{fname}"
        target = train_dir / fname
        if not target.exists():
            if download_file(url, target):
                train_downloaded += 1
        else:
            train_downloaded += 1

    print(f"      Train images available: {train_downloaded}/{num_train}")

    # 3. Download Validation images (curated subset of 10 images)
    print(f"[3/3] Downloading {num_val} curated validation images...")
    val_downloaded = 0
    for i in range(num_train + 1, num_train + num_val + 1):
        fname = f"test_{i:03d}.png"
        url = f"{GITHUB_RAW_BASE}/train/{fname}"
        target = val_dir / fname
        if not target.exists():
            if download_file(url, target):
                val_downloaded += 1
        else:
            val_downloaded += 1

    print(f"      Val images available: {val_downloaded}/{num_val}")

    # Summary
    print("\n--- Dataset Preparation Summary ---")
    print(f"Train directory: {train_dir} ({len(list(train_dir.glob('*.png')))} images)")
    print(f"Val directory:   {val_dir} ({len(list(val_dir.glob('*.png')))} images)")
    print(f"Test benchmark:  {test_dir} ({len(list(test_dir.glob('*.png')))} images)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download curated image denoising benchmark dataset.")
    parser.add_argument("--data-dir", type=str, default="data", help="Base directory for data storage.")
    parser.add_argument("--num-train", type=int, default=35, help="Number of training images to curate.")
    parser.add_argument("--num-val", type=int, default=10, help="Number of validation images to curate.")
    parser.add_argument("--offline-only", action="store_true", help="Force use of local scikit-image benchmarks.")
    args = parser.parse_args()

    base_path = Path(args.data_dir)
    if args.offline_only:
        tr, va, te = setup_directories(base_path)
        populate_from_skimage(tr, va, te)
    else:
        download_standard_benchmark(base_path, num_train=args.num_train, num_val=args.num_val)
