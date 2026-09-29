"""
tests/test_step2.py
===================
Focused verification test suite for Step 2:
1. Traditional spatial filters (Gaussian, Median, Bilateral):
   - Input/output shape preservation
   - Dynamic range adherence [0.0, 1.0]
   - Correctness on both RGB and Grayscale images
2. Metrics Engine (PSNR, SSIM, Latency):
   - Identity tests: PSNR(x, x) == inf, SSIM(x, x) ≈ 1.0
   - Correct degradation on noisy images
   - Compatibility with both NumPy arrays and PyTorch tensors
3. End-to-end integration across Clean -> AWGN -> Filter -> Metrics.
"""

import math
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.noise import add_gaussian_noise_numpy
from src.traditional import (
    apply_gaussian_filter,
    apply_median_filter,
    apply_bilateral_filter,
    apply_all_traditional_filters,
)
from src.metrics import (
    calculate_psnr,
    calculate_ssim,
    evaluate_image_quality,
    benchmark_latency,
)


def test_filters_shape_and_range():
    print("[Test 1/5] Validating filter input/output dimensions and dynamic ranges...")
    # Test on synthetic RGB image
    rng = np.random.default_rng(42)
    sample_rgb = rng.uniform(0.0, 1.0, size=(128, 128, 3)).astype(np.float32)

    filters = [
        ("Gaussian", lambda img: apply_gaussian_filter(img, kernel_size=5, sigma=1.0)),
        ("Median", lambda img: apply_median_filter(img, kernel_size=3)),
        ("Bilateral", lambda img: apply_bilateral_filter(img, diameter=7, sigma_color=50, sigma_space=50)),
    ]

    for name, fn in filters:
        out = fn(sample_rgb)
        assert out.shape == sample_rgb.shape, f"{name} changed shape: {out.shape} vs {sample_rgb.shape}"
        assert out.dtype == np.float32, f"{name} wrong dtype: {out.dtype}"
        assert out.min() >= 0.0, f"{name} output < 0.0: min is {out.min()}"
        assert out.max() <= 1.0, f"{name} output > 1.0: max is {out.max()}"
        print(f" -> {name:10s}: shape={out.shape}, dtype={out.dtype}, range=[{out.min():.3f}, {out.max():.3f}] (Valid)")

    # Test on Grayscale image
    sample_gray = rng.uniform(0.0, 1.0, size=(64, 64)).astype(np.float32)
    for name, fn in filters:
        out_g = fn(sample_gray)
        assert out_g.shape == sample_gray.shape, f"{name} (gray) changed shape: {out_g.shape}"
        assert 0.0 <= out_g.min() and out_g.max() <= 1.0, f"{name} (gray) range error"

    print(" -> PASSED: Filters preserve dimensions and strictly maintain [0.0, 1.0] range.")


def test_identity_metrics():
    print("[Test 2/5] Validating PSNR and SSIM identity properties (image compared with itself)...")
    rng = np.random.default_rng(101)
    clean = rng.uniform(0.0, 1.0, size=(100, 100, 3)).astype(np.float32)

    # 1. PSNR of identical images must be infinite
    psnr_self = calculate_psnr(clean, clean, data_range=1.0)
    assert math.isinf(psnr_self), f"Expected PSNR(x, x) == inf, got {psnr_self}"

    # 2. SSIM of identical images must be exactly 1.0 (within float precision)
    ssim_self = calculate_ssim(clean, clean, data_range=1.0)
    assert abs(ssim_self - 1.0) < 1e-6, f"Expected SSIM(x, x) ≈ 1.0, got {ssim_self}"

    print(f" -> PSNR(x, x) = {psnr_self} (Effectively infinite)")
    print(f" -> SSIM(x, x) = {ssim_self:.6f} (~ 1.0)")
    print(" -> PASSED: Identity assertions verified.")


def test_metrics_on_tensor_and_numpy():
    print("[Test 3/5] Validating metric calculation with PyTorch tensors and NumPy arrays...")
    clean_np = np.ones((64, 64, 3), dtype=np.float32) * 0.5
    noisy_np = clean_np + 0.05 # Add known constant difference

    clean_torch = torch.from_numpy(clean_np).permute(2, 0, 1) # (C, H, W)
    noisy_torch = torch.from_numpy(noisy_np).permute(2, 0, 1)

    psnr_np = calculate_psnr(clean_np, noisy_np)
    psnr_torch = calculate_psnr(clean_torch, noisy_torch)
    assert abs(psnr_np - psnr_torch) < 1e-4, f"NumPy vs PyTorch PSNR mismatch: {psnr_np} vs {psnr_torch}"

    ssim_np = calculate_ssim(clean_np, noisy_np)
    ssim_torch = calculate_ssim(clean_torch, noisy_torch)
    assert abs(ssim_np - ssim_torch) < 1e-4, f"NumPy vs PyTorch SSIM mismatch: {ssim_np} vs {ssim_torch}"

    print(f" -> PSNR: NumPy={psnr_np:.2f} dB, PyTorch={psnr_torch:.2f} dB")
    print(f" -> SSIM: NumPy={ssim_np:.4f}, PyTorch={ssim_torch:.4f}")
    print(" -> PASSED: Metric calculations are identical across NumPy arrays and PyTorch tensors.")


def test_clean_noisy_denoised_pipeline():
    print("[Test 4/5] Validating end-to-end Clean -> AWGN -> Filter -> Metric pipeline...")
    # Load actual benchmark test image (01.png Cameraman)
    img_path = PROJECT_ROOT / "data" / "test_benchmark" / "01.png"
    assert img_path.exists(), f"Benchmark image missing: {img_path}"

    clean = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0

    # Inject moderate noise sigma=25
    noisy = add_gaussian_noise_numpy(clean, sigma=25.0, seed=42)

    # Initial metrics of noisy image
    noisy_metrics = evaluate_image_quality(clean, noisy)
    print(f" -> Noisy (sigma=25): PSNR = {noisy_metrics['psnr']:.2f} dB, SSIM = {noisy_metrics['ssim']:.4f}")
    assert noisy_metrics["psnr"] < 30.0, "Noisy PSNR should degrade substantially from infinity"
    assert noisy_metrics["ssim"] < 0.95, "Noisy SSIM should degrade substantially from 1.0"

    # Apply all three classical filters
    denoised_dict = apply_all_traditional_filters(noisy)

    for filter_name, denoised in denoised_dict.items():
        metrics = evaluate_image_quality(clean, denoised)
        assert not math.isnan(metrics["psnr"]), f"{filter_name} produced NaN PSNR"
        assert not math.isnan(metrics["ssim"]), f"{filter_name} produced NaN SSIM"
        assert 0.0 <= metrics["ssim"] <= 1.0, f"{filter_name} SSIM out of bounds: {metrics['ssim']}"
        print(f" -> {filter_name.capitalize():10s}: PSNR = {metrics['psnr']:.2f} dB, SSIM = {metrics['ssim']:.4f}")

    print(" -> PASSED: Real image denoising pipeline executed and evaluated with valid metrics.")


def test_latency_benchmarking():
    print("[Test 5/5] Validating CPU latency benchmarking module...")
    dummy_img = np.random.rand(128, 128, 3).astype(np.float32)

    # Benchmark Gaussian filter
    timing = benchmark_latency(apply_gaussian_filter, dummy_img, kernel_size=5, sigma=1.0, num_runs=5, warmup=1)
    assert timing["mean_ms"] > 0.0, "Latency cannot be zero or negative"
    assert timing["fps"] > 0.0, "FPS cannot be zero or negative"
    print(f" -> Gaussian filter latency on CPU: {timing['mean_ms']:.2f} +/- {timing['std_ms']:.2f} ms ({timing['fps']:.1f} FPS)")
    print(" -> PASSED: Latency benchmarking operates reliably.")


if __name__ == "__main__":
    print("=== RUNNING STEP 2 VERIFICATION SUITE ===\n")
    test_filters_shape_and_range()
    test_identity_metrics()
    test_metrics_on_tensor_and_numpy()
    test_clean_noisy_denoised_pipeline()
    test_latency_benchmarking()
    print("\n=== ALL STEP 2 TESTS PASSED SUCCESSFULLY ===")
