"""
src/visualize.py
================
Publication-style visualization functions for the image denoising comparison.

Figures produced:
    1. comparison_grid_{image}_{sigma}.png  -- 6-panel: Original / Noisy /
                                               Gaussian / Median / Bilateral / CNN
    2. error_maps_{image}_{sigma}.png       -- Absolute difference maps
                                               |denoised - clean| x4
    3. psnr_barchart.png                    -- PSNR bar chart grouped by method,
                                               across sigma={15, 25, 50}
    4. ssim_barchart.png                    -- SSIM bar chart (same grouping)

All figures are saved to results/figures/.

Usage:
    from src.visualize import (
        plot_comparison_grid,
        plot_error_maps,
        plot_psnr_bar_chart,
        plot_ssim_bar_chart,
        generate_all_visualizations,
    )
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend: must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluate import (
    load_image_rgb_float32,
    load_cnn_model,
    cnn_denoise_image,
    load_config,
    compute_summary_table,
)
from src.metrics import calculate_psnr, calculate_ssim
from src.noise import add_gaussian_noise_numpy
from src.traditional import apply_all_traditional_filters


# =========================================================================
# Internal helpers
# =========================================================================

def _safe_psnr_label(psnr: float) -> str:
    """Formats PSNR value as a string, handling infinity gracefully."""
    if psnr == float("inf") or psnr > 99.0:
        return "inf"
    return f"{psnr:.2f}"


def _clip_for_display(img: np.ndarray) -> np.ndarray:
    """Clips float32 array to [0.0, 1.0] for matplotlib imshow."""
    return np.clip(img, 0.0, 1.0).astype(np.float32)


# =========================================================================
# 1. 6-panel comparison grid
# =========================================================================

def plot_comparison_grid(
    clean: np.ndarray,
    noisy: np.ndarray,
    gauss_out: np.ndarray,
    median_out: np.ndarray,
    bilateral_out: np.ndarray,
    cnn_out: np.ndarray,
    sigma: int,
    image_name: str,
    output_path: str,
    dpi: int = 120,
) -> None:
    """
    Saves a 2x3 publication-style panel figure showing all six images.

    Panel layout:
        [ Original ]  [ Noisy (sigma) ]  [ Gaussian ]
        [ Median   ]  [ Bilateral     ]  [ CNN       ]

    PSNR and SSIM vs clean are annotated below each denoised panel.

    Args:
        clean:         Clean reference image (H, W, 3) float32.
        noisy:         Noisy degraded image (H, W, 3) float32.
        gauss_out:     Gaussian-filtered output.
        median_out:    Median-filtered output.
        bilateral_out: Bilateral-filtered output.
        cnn_out:       CNN autoencoder output.
        sigma:         AWGN noise level used.
        image_name:    Image filename (used in title and saved filename).
        output_path:   Full path to save the PNG figure.
        dpi:           Figure DPI (default: 120).
    """
    noisy_psnr    = calculate_psnr(clean, noisy)
    gauss_psnr    = calculate_psnr(clean, gauss_out)
    median_psnr   = calculate_psnr(clean, median_out)
    bilateral_psnr = calculate_psnr(clean, bilateral_out)
    cnn_psnr      = calculate_psnr(clean, cnn_out)

    noisy_ssim    = calculate_ssim(clean, noisy)
    gauss_ssim    = calculate_ssim(clean, gauss_out)
    median_ssim   = calculate_ssim(clean, median_out)
    bilateral_ssim = calculate_ssim(clean, bilateral_out)
    cnn_ssim      = calculate_ssim(clean, cnn_out)

    panels = [
        (clean,         "Original (Clean)",  None,               None),
        (noisy,         f"Noisy (sigma={sigma})", noisy_psnr,    noisy_ssim),
        (gauss_out,     "Gaussian Filter",   gauss_psnr,         gauss_ssim),
        (median_out,    "Median Filter",     median_psnr,        median_ssim),
        (bilateral_out, "Bilateral Filter",  bilateral_psnr,     bilateral_ssim),
        (cnn_out,       "CNN Autoencoder",   cnn_psnr,           cnn_ssim),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f"Denoising Comparison  |  {image_name}  |  sigma = {sigma}",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, (img, title, psnr, ssim) in zip(axes.flat, panels):
        ax.imshow(_clip_for_display(img))
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
        ax.axis("off")

        if psnr is not None and ssim is not None:
            subtitle = f"PSNR={_safe_psnr_label(psnr)} dB  SSIM={ssim:.4f}"
            ax.text(
                0.5, -0.04, subtitle,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=8, color="#333333"
            )

    plt.tight_layout(pad=1.5)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# 2. Absolute difference / error maps
# =========================================================================

def plot_error_maps(
    clean: np.ndarray,
    gauss_out: np.ndarray,
    median_out: np.ndarray,
    bilateral_out: np.ndarray,
    cnn_out: np.ndarray,
    sigma: int,
    image_name: str,
    output_path: str,
    amplify: float = 5.0,
    dpi: int = 120,
) -> None:
    """
    Saves a 1x4 figure of absolute error maps |denoised - clean| for each method.

    Pixel values are amplified by `amplify` for visual clarity. Error maps
    are shown in grayscale; brighter pixels indicate larger residual errors.

    Args:
        clean:         Clean reference image (H, W, 3) float32.
        gauss_out:     Gaussian-filtered output.
        median_out:    Median-filtered output.
        bilateral_out: Bilateral-filtered output.
        cnn_out:       CNN autoencoder output.
        sigma:         AWGN noise level used.
        image_name:    Image filename (used in title).
        output_path:   Full path to save the PNG figure.
        amplify:       Scalar multiplier for visual amplification of errors.
        dpi:           Figure DPI (default: 120).
    """
    # Compute mean absolute error maps (averaged over RGB channels)
    def _error_map(img: np.ndarray) -> np.ndarray:
        err = np.abs(img.astype(np.float64) - clean.astype(np.float64))
        return np.mean(err, axis=2).astype(np.float32)  # (H, W)

    gauss_err    = _error_map(gauss_out)
    median_err   = _error_map(median_out)
    bilateral_err = _error_map(bilateral_out)
    cnn_err      = _error_map(cnn_out)

    panels = [
        (gauss_err,    "Gaussian Error",    calculate_psnr(clean, gauss_out)),
        (median_err,   "Median Error",      calculate_psnr(clean, median_out)),
        (bilateral_err,"Bilateral Error",   calculate_psnr(clean, bilateral_out)),
        (cnn_err,      "CNN Error",         calculate_psnr(clean, cnn_out)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"Absolute Error Maps  |  {image_name}  |  sigma = {sigma}  "
        f"(amplified x{amplify:.0f})",
        fontsize=11, fontweight="bold"
    )

    vmax = max(np.max(e) for e, _, _ in panels) * amplify
    vmax = max(vmax, 1e-6)  # Avoid zero-range colormap

    for ax, (err_map, title, psnr) in zip(axes, panels):
        im = ax.imshow(
            np.clip(err_map * amplify, 0.0, 1.0),
            cmap="hot", vmin=0.0, vmax=min(1.0, vmax)
        )
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")
        ax.text(
            0.5, -0.04,
            f"PSNR={_safe_psnr_label(psnr)} dB",
            transform=ax.transAxes,
            ha="center", va="top", fontsize=8, color="#333333"
        )

    plt.tight_layout(pad=1.5)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# 3. PSNR bar chart
# =========================================================================

def plot_psnr_bar_chart(
    summary: Dict[str, Dict[int, Dict[str, float]]],
    noise_levels: List[int],
    output_path: str,
    dpi: int = 120,
) -> None:
    """
    Saves a grouped bar chart of mean PSNR (dB) across methods and noise levels.

    Args:
        summary:      Dict[method -> Dict[sigma -> {psnr, ssim, latency_ms}]].
        noise_levels: Ordered list of sigma values shown on x-axis groups.
        output_path:  Full path to save the PNG figure.
        dpi:          Figure DPI (default: 120).
    """
    methods = ["Gaussian", "Median", "Bilateral", "CNN"]
    colors  = ["#4472C4", "#ED7D31", "#70AD47", "#FF0000"]  # Blue/Orange/Green/Red

    n_groups = len(noise_levels)
    n_bars   = len(methods)
    bar_w    = 0.18
    x        = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (method, color) in enumerate(zip(methods, colors)):
        psnr_vals = []
        for sigma in noise_levels:
            if method in summary and sigma in summary[method]:
                psnr_vals.append(summary[method][sigma]["psnr"])
            else:
                psnr_vals.append(0.0)

        offsets = x + (i - (n_bars - 1) / 2.0) * bar_w
        bars = ax.bar(offsets, psnr_vals, bar_w, label=method, color=color, alpha=0.88)

        # Value labels on top of each bar
        for bar, val in zip(bars, psnr_vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.15,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7, color="#222222"
                )

    ax.set_xlabel("AWGN Noise Level (sigma)", fontsize=11)
    ax.set_ylabel("Mean PSNR (dB)", fontsize=11)
    ax.set_title("PSNR Comparison Across Denoising Methods", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"sigma = {s}" for s in noise_levels], fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    psnr_min = min(
        summary[m][s]["psnr"]
        for m in methods if m in summary
        for s in noise_levels if s in summary[m]
    )
    ax.set_ylim(max(0, psnr_min - 3), None)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# 4. SSIM bar chart
# =========================================================================

def plot_ssim_bar_chart(
    summary: Dict[str, Dict[int, Dict[str, float]]],
    noise_levels: List[int],
    output_path: str,
    dpi: int = 120,
) -> None:
    """
    Saves a grouped bar chart of mean SSIM across methods and noise levels.

    Args:
        summary:      Dict[method -> Dict[sigma -> {psnr, ssim, latency_ms}]].
        noise_levels: Ordered list of sigma values shown on x-axis groups.
        output_path:  Full path to save the PNG figure.
        dpi:          Figure DPI (default: 120).
    """
    methods = ["Gaussian", "Median", "Bilateral", "CNN"]
    colors  = ["#4472C4", "#ED7D31", "#70AD47", "#FF0000"]

    n_groups = len(noise_levels)
    n_bars   = len(methods)
    bar_w    = 0.18
    x        = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (method, color) in enumerate(zip(methods, colors)):
        ssim_vals = []
        for sigma in noise_levels:
            if method in summary and sigma in summary[method]:
                ssim_vals.append(summary[method][sigma]["ssim"])
            else:
                ssim_vals.append(0.0)

        offsets = x + (i - (n_bars - 1) / 2.0) * bar_w
        bars = ax.bar(offsets, ssim_vals, bar_w, label=method, color=color, alpha=0.88)

        for bar, val in zip(bars, ssim_vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.003,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7, color="#222222"
                )

    ax.set_xlabel("AWGN Noise Level (sigma)", fontsize=11)
    ax.set_ylabel("Mean SSIM", fontsize=11)
    ax.set_title("SSIM Comparison Across Denoising Methods", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"sigma = {s}" for s in noise_levels], fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, 1.05)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# 5. Orchestrator: generate all figures from evaluation results
# =========================================================================

def generate_all_visualizations(
    results: List[Dict],
    config_path: str = "configs/config.yaml",
    checkpoint_path: Optional[str] = None,
    test_dir: Optional[str] = None,
    figures_dir: Optional[str] = None,
    max_grids: int = 3,
    noise_seed: int = 42,
) -> None:
    """
    Generates all Step 5 visualization figures from evaluation results.

    Produces:
        - comparison_grid_{img}_{sigma}.png for up to `max_grids` test images
        - error_maps_{img}_{sigma}.png for same images
        - psnr_barchart.png
        - ssim_barchart.png

    Args:
        results:         List of per-record result dicts from run_full_evaluation().
        config_path:     Path to YAML configuration file.
        checkpoint_path: Override CNN checkpoint path.
        test_dir:        Override test benchmark directory.
        figures_dir:     Override output figures directory.
        max_grids:       Maximum number of images for which comparison grids are made.
        noise_seed:      Seed for reproducible noisy image generation.
    """
    config = load_config(config_path)

    # Resolve checkpoint: caller arg → config evaluation.inference_checkpoint
    # → legacy training.checkpoint_dir/best_model.pth (backward-compat fallback).
    _ckpt = checkpoint_path or (
        config.get("evaluation", {}).get("inference_checkpoint")
        or (
            config.get("training", {}).get("checkpoint_dir", "results/checkpoints")
            + "/best_model.pth"
        )
    )
    _test_dir = test_dir or config.get("dataset", {}).get(
        "test_benchmark", {}
    ).get("dir", "data/test_benchmark")
    _noise_levels = config.get("noise", {}).get("levels", [15, 25, 50])
    _figures_dir = figures_dir or config.get("evaluation", {}).get(
        "figures_dir", "results/figures"
    )

    os.makedirs(_figures_dir, exist_ok=True)

    # Compute summary for bar charts
    summary = compute_summary_table(results, _noise_levels)

    # ------------------------------------------------------------------
    # Bar charts (do not need individual image data)
    # ------------------------------------------------------------------
    print("Generating PSNR bar chart...")
    plot_psnr_bar_chart(
        summary, _noise_levels,
        output_path=os.path.join(_figures_dir, "psnr_barchart.png")
    )

    print("Generating SSIM bar chart...")
    plot_ssim_bar_chart(
        summary, _noise_levels,
        output_path=os.path.join(_figures_dir, "ssim_barchart.png")
    )

    # ------------------------------------------------------------------
    # Per-image comparison grids and error maps
    # ------------------------------------------------------------------
    model = load_cnn_model(_ckpt, device="cpu")
    test_path = Path(_test_dir)
    image_files = sorted(
        [f for f in test_path.glob("*.png")] + [f for f in test_path.glob("*.jpg")]
    )[:max_grids]

    for img_path in image_files:
        clean_np = load_image_rgb_float32(str(img_path))
        img_name = img_path.stem  # e.g. "01"

        for sigma in _noise_levels:
            noisy_np = add_gaussian_noise_numpy(clean_np, sigma=sigma, seed=noise_seed)
            filters  = apply_all_traditional_filters(noisy_np, config)
            cnn_out  = cnn_denoise_image(model, noisy_np, device="cpu")

            grid_path = os.path.join(
                _figures_dir, f"comparison_grid_{img_name}_sigma{sigma}.png"
            )
            err_path  = os.path.join(
                _figures_dir, f"error_maps_{img_name}_sigma{sigma}.png"
            )

            print(f"  comparison_grid: {img_name} sigma={sigma}")
            plot_comparison_grid(
                clean=clean_np,
                noisy=noisy_np,
                gauss_out=filters["gaussian"],
                median_out=filters["median"],
                bilateral_out=filters["bilateral"],
                cnn_out=cnn_out,
                sigma=sigma,
                image_name=img_path.name,
                output_path=grid_path,
            )

            print(f"  error_maps:      {img_name} sigma={sigma}")
            plot_error_maps(
                clean=clean_np,
                gauss_out=filters["gaussian"],
                median_out=filters["median"],
                bilateral_out=filters["bilateral"],
                cnn_out=cnn_out,
                sigma=sigma,
                image_name=img_path.name,
                output_path=err_path,
            )

    print(f"\nAll figures saved to: {_figures_dir}")


# =========================================================================
# CLI entry point
# =========================================================================

if __name__ == "__main__":
    import argparse
    from src.evaluate import run_full_evaluation

    parser = argparse.ArgumentParser(
        description="Generate Step 5 visualization figures."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--max-grids", type=int, default=3,
                        help="Number of images to produce comparison grids for.")
    args = parser.parse_args()

    # Run evaluation first, then visualize
    results = run_full_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
    )

    generate_all_visualizations(
        results=results,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        figures_dir=args.figures_dir,
        max_grids=args.max_grids,
    )
