"""
tests/test_step1.py
===================
Automated verification tests for Step 1:
1. Configuration parsing and explicit dataset split verification.
2. Mathematical properties of AWGN:
   - Expected standard deviation: sigma / 255.0
   - Expected variance: (sigma / 255.0)^2  (NOT sigma^2)
   - Zero-mean: E[n] ≈ 0.0
3. Deterministic reproducibility across seeds.
4. Dynamic range clipping to [0.0, 1.0].
5. Dataset directory structure and Set12 benchmark integrity.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.noise import (
    add_gaussian_noise_numpy,
    add_gaussian_noise_torch,
    verify_noise_properties,
)


def test_config():
    print("[Test 1/5] Validating config.yaml and dataset specification...")
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    assert config_path.exists(), f"Missing config file at {config_path}"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Check sections
    required_keys = ["experiment", "dataset", "noise", "traditional_filters", "model", "training", "evaluation"]
    for k in required_keys:
        assert k in cfg, f"Missing key in config: {k}"

    # Explicit dataset split checks (Train=BSDS 35, Val=BSDS 10, Test=Set12 12)
    ds = cfg["dataset"]
    assert ds["train"]["num_images"] == 35, "Expected exactly 35 train images in config"
    assert ds["validation"]["num_images"] == 10, "Expected exactly 10 validation images in config"
    assert ds["test_benchmark"]["num_images"] == 12, "Expected exactly 12 test images in config"
    assert len(ds["test_benchmark"]["image_names"]) == 12, "Set12 image list must have 12 items"

    # Dataset specification document check
    spec_path = PROJECT_ROOT / "data" / "DATASET_SPEC.md"
    assert spec_path.exists(), f"Missing DATASET_SPEC.md at {spec_path}"

    print(" -> PASSED: config.yaml and DATASET_SPEC.md explicitly specify dataset splits without ambiguity.")


def test_noise_mathematical_properties():
    print("[Test 2/5] Validating mathematical properties of AWGN (Mean ~= 0, Var ~= (sigma/255)^2)...")
    for sigma in [15.0, 25.0, 50.0]:
        stats = verify_noise_properties(shape=(1, 3, 512, 512), sigma=sigma, seed=42)
        
        expected_std = stats["theoretical_sigma_norm"]
        expected_var = stats["theoretical_variance"]
        measured_std = stats["measured_std"]
        measured_var = stats["measured_var"]
        measured_mean = stats["measured_mean"]

        # Mathematical verification: expected_variance MUST be (sigma/255)^2
        assert abs(expected_var - (sigma / 255.0) ** 2) < 1e-12, "Mathematical formula mismatch for variance!"
        assert stats["is_valid"], f"AWGN statistics failed for sigma={sigma}: {stats}"

        print(f" -> Sigma {sigma:4.1f}: Expected Var=(sigma/255)^2={expected_var:.6f}, Measured Var={measured_var:.6f}")
        print(f"               Expected Std=(sigma/255)  ={expected_std:.6f}, Measured Std={measured_std:.6f}")
        print(f"               Measured Mean={measured_mean:+.6f} (Error: Var={stats['var_error']:.6f}, Std={stats['std_error']:.6f})")

    print(" -> PASSED: AWGN strictly follows N(0, (sigma/255)^2) with verified variance.")


def test_noise_reproducibility():
    print("[Test 3/5] Validating deterministic reproducibility with random seeds...")
    dummy_np = np.full((64, 64, 3), 0.5, dtype=np.float32)
    dummy_torch = torch.full((3, 64, 64), 0.5, dtype=torch.float32)

    # NumPy seed check
    noisy_np_1 = add_gaussian_noise_numpy(dummy_np, sigma=25, seed=123)
    noisy_np_2 = add_gaussian_noise_numpy(dummy_np, sigma=25, seed=123)
    noisy_np_diff_seed = add_gaussian_noise_numpy(dummy_np, sigma=25, seed=999)

    assert np.array_equal(noisy_np_1, noisy_np_2), "NumPy noise not deterministic with same seed!"
    assert not np.array_equal(noisy_np_1, noisy_np_diff_seed), "NumPy noise should differ with different seeds!"

    # PyTorch seed check
    noisy_torch_1 = add_gaussian_noise_torch(dummy_torch, sigma=25, seed=123)
    noisy_torch_2 = add_gaussian_noise_torch(dummy_torch, sigma=25, seed=123)
    noisy_torch_diff_seed = add_gaussian_noise_torch(dummy_torch, sigma=25, seed=999)

    assert torch.equal(noisy_torch_1, noisy_torch_2), "PyTorch noise not deterministic with same seed!"
    assert not torch.equal(noisy_torch_1, noisy_torch_diff_seed), "PyTorch noise should differ with different seeds!"

    print(" -> PASSED: Deterministic reproducibility verified for both NumPy and PyTorch.")


def test_noise_clipping_bounds():
    print("[Test 4/5] Validating dynamic range clipping [0.0, 1.0]...")
    extreme_np = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32).reshape(2, 2, 1)
    extreme_torch = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]], dtype=torch.float32)

    # High sigma to force severe overflow/underflow
    noisy_np = add_gaussian_noise_numpy(extreme_np, sigma=100, seed=42)
    noisy_torch = add_gaussian_noise_torch(extreme_torch, sigma=100, seed=42)

    assert noisy_np.min() >= 0.0 and noisy_np.max() <= 1.0, f"NumPy out of bounds: [{noisy_np.min()}, {noisy_np.max()}]"
    assert noisy_torch.min() >= 0.0 and noisy_torch.max() <= 1.0, f"Torch out of bounds: [{noisy_torch.min()}, {noisy_torch.max()}]"

    print(" -> PASSED: Dynamic range is strictly clamped to [0.0, 1.0].")


def test_dataset_integrity():
    print("[Test 5/5] Validating curated dataset integrity & files...")
    train_dir = PROJECT_ROOT / "data" / "raw" / "train"
    val_dir = PROJECT_ROOT / "data" / "raw" / "val"
    test_dir = PROJECT_ROOT / "data" / "test_benchmark"

    train_files = sorted(list(train_dir.glob("*.png")))
    val_files = sorted(list(val_dir.glob("*.png")))
    test_files = sorted(list(test_dir.glob("*.png")))

    print(f" -> Found {len(train_files)} train, {len(val_files)} val, {len(test_files)} Set12 test images.")
    assert len(train_files) == 35, f"Expected exactly 35 train images, got {len(train_files)}"
    assert len(val_files) == 10, f"Expected exactly 10 validation images, got {len(val_files)}"
    assert len(test_files) == 12, f"Expected exactly 12 Set12 test images, got {len(test_files)}"

    # Check Set12 filename list
    expected_set12_names = [f"{i:02d}.png" for i in range(1, 13)]
    actual_set12_names = [f.name for f in test_files]
    assert actual_set12_names == expected_set12_names, f"Set12 file mismatch: {actual_set12_names}"

    # Verify image readability
    sample_img = Image.open(test_dir / "01.png")
    assert sample_img.size == (256, 256), f"Unexpected dimensions for Cameraman: {sample_img.size}"
    print(f" -> Verified Set12 test image 01.png (Cameraman) with dimensions: {sample_img.size}")
    print(" -> PASSED: Curated dataset and Set12 benchmark verified.")


if __name__ == "__main__":
    print("=== RUNNING STEP 1 VERIFICATION SUITE ===\n")
    test_config()
    test_noise_mathematical_properties()
    test_noise_reproducibility()
    test_noise_clipping_bounds()
    test_dataset_integrity()
    print("\n=== ALL STEP 1 TESTS PASSED SUCCESSFULLY ===")
