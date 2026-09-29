"""
scripts/demo_traditional_filters.py
===================================
Reproducible demonstration script evaluating traditional spatial filters
(Gaussian, Median, Bilateral) on benchmark test images at sigma = 15, 25, and 50.

All metrics (PSNR, SSIM, and execution time) are computed dynamically from
actual execution without hardcoded values.
"""

import sys
from pathlib import Path
import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.noise import add_gaussian_noise_numpy
from src.traditional import apply_all_traditional_filters, apply_gaussian_filter, apply_median_filter, apply_bilateral_filter
from src.metrics import evaluate_image_quality, benchmark_latency


def run_traditional_filter_demo(
    test_image_names = ("01.png", "02.png"),
    sigmas = (15, 25, 50),
    seed = 42
):
    # Load configuration
    cfg_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    test_dir = PROJECT_ROOT / cfg["dataset"]["test_benchmark"]["dir"]

    print("=" * 82)
    print("      TRADITIONAL SPATIAL FILTER BENCHMARK (GAUSSIAN vs MEDIAN vs BILATERAL)")
    print("=" * 82)
    print(f"Filter Parameters:")
    print(f" - Gaussian:  kernel_size={cfg['traditional_filters']['gaussian']['kernel_size']}, "
          f"sigma={cfg['traditional_filters']['gaussian']['sigma']}")
    print(f" - Median:    kernel_size={cfg['traditional_filters']['median']['kernel_size']}")
    print(f" - Bilateral: diameter={cfg['traditional_filters']['bilateral']['diameter']}, "
          f"sigma_color={cfg['traditional_filters']['bilateral']['sigma_color']}, "
          f"sigma_space={cfg['traditional_filters']['bilateral']['sigma_space']}")
    print("-" * 82)

    for img_name in test_image_names:
        img_path = test_dir / img_name
        if not img_path.exists():
            print(f"[Warning] Image {img_name} not found in {test_dir}. Skipping.")
            continue

        clean_img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        print(f"\nTarget Benchmark Image: {img_name} (Dimensions: {clean_img.shape[0]}x{clean_img.shape[1]} RGB)")
        print(f"{'Noise (sigma)':<15} | {'Filter':<12} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Latency (ms)':<14} | {'FPS':<6}")
        print("-" * 75)

        for sigma in sigmas:
            # Deterministic noise injection
            noisy_img = add_gaussian_noise_numpy(clean_img, sigma=sigma, seed=seed)
            noisy_metrics = evaluate_image_quality(clean_img, noisy_img)

            # Benchmark and evaluate noisy baseline
            print(f"sigma = {sigma:<8} | {'Noisy (None)':<12} | {noisy_metrics['psnr']:<10.2f} | {noisy_metrics['ssim']:<8.4f} | {'N/A':<14} | {'N/A':<6}")

            # Apply each filter and measure latency
            # 1. Gaussian
            g_cfg = cfg["traditional_filters"]["gaussian"]
            g_timing = benchmark_latency(apply_gaussian_filter, noisy_img, kernel_size=g_cfg["kernel_size"], sigma=g_cfg["sigma"], num_runs=5)
            g_denoised = apply_gaussian_filter(noisy_img, kernel_size=g_cfg["kernel_size"], sigma=g_cfg["sigma"])
            g_metrics = evaluate_image_quality(clean_img, g_denoised)
            print(f"{'':<15} | {'Gaussian':<12} | {g_metrics['psnr']:<10.2f} | {g_metrics['ssim']:<8.4f} | {g_timing['mean_ms']:<14.2f} | {g_timing['fps']:<6.1f}")

            # 2. Median
            m_cfg = cfg["traditional_filters"]["median"]
            m_timing = benchmark_latency(apply_median_filter, noisy_img, kernel_size=m_cfg["kernel_size"], num_runs=5)
            m_denoised = apply_median_filter(noisy_img, kernel_size=m_cfg["kernel_size"])
            m_metrics = evaluate_image_quality(clean_img, m_denoised)
            print(f"{'':<15} | {'Median':<12} | {m_metrics['psnr']:<10.2f} | {m_metrics['ssim']:<8.4f} | {m_timing['mean_ms']:<14.2f} | {m_timing['fps']:<6.1f}")

            # 3. Bilateral
            b_cfg = cfg["traditional_filters"]["bilateral"]
            b_timing = benchmark_latency(apply_bilateral_filter, noisy_img, diameter=b_cfg["diameter"], sigma_color=b_cfg["sigma_color"], sigma_space=b_cfg["sigma_space"], num_runs=5)
            b_denoised = apply_bilateral_filter(noisy_img, diameter=b_cfg["diameter"], sigma_color=b_cfg["sigma_color"], sigma_space=b_cfg["sigma_space"])
            b_metrics = evaluate_image_quality(clean_img, b_denoised)
            print(f"{'':<15} | {'Bilateral':<12} | {b_metrics['psnr']:<10.2f} | {b_metrics['ssim']:<8.4f} | {b_timing['mean_ms']:<14.2f} | {b_timing['fps']:<6.1f}")
            print("-" * 75)


if __name__ == "__main__":
    run_traditional_filter_demo(test_image_names=["01.png", "02.png"], sigmas=[15, 25, 50])
