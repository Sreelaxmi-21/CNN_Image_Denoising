"""
src/evaluate.py
===============
Full comparative evaluation of all four denoising methods on the Set12 benchmark.

Methods evaluated:
    1. Gaussian Filter   (linear spatial convolution)
    2. Median Filter     (order-statistics non-linear)
    3. Bilateral Filter  (edge-preserving photometric + geometric)
    4. CNN Autoencoder   (deep learned reconstruction)

Evaluation protocol:
    - Dataset  : 12 images from Set12 benchmark (data/test_benchmark/)
    - Noise    : AWGN at sigma in {15, 25, 50}, fixed seed=42 per image per sigma
    - Metrics  : PSNR (dB), SSIM, Latency (ms)
    - Output   : results/tables/evaluation_results.csv
                 results/tables/evaluation_results.md

Usage:
    python src/evaluate.py                    # uses default config
    python src/evaluate.py --config configs/config.yaml
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

# --- Ensure project root is on sys.path ---------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.metrics import calculate_psnr, calculate_ssim, benchmark_latency
from src.models.autoencoder import CompactCNNAutoencoder, get_model
from src.noise import add_gaussian_noise_numpy
from src.traditional import (
    apply_bilateral_filter,
    apply_gaussian_filter,
    apply_median_filter,
)


# =========================================================================
# Helper utilities
# =========================================================================

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Loads YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_image_rgb_float32(path: str) -> np.ndarray:
    """
    Loads an image from disk as an RGB float32 array normalized to [0.0, 1.0].

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        np.ndarray: Shape (H, W, 3), dtype float32, values in [0.0, 1.0].
    """
    img_pil = Image.open(path).convert("RGB")
    img_np = np.array(img_pil, dtype=np.float32) / 255.0
    return img_np


def load_cnn_model(checkpoint_path: str, device: str = "cpu") -> CompactCNNAutoencoder:
    """
    Loads the trained CNN autoencoder weights from a checkpoint file.

    Args:
        checkpoint_path: Path to best_model.pth produced by src/train.py.
        device: Torch device string ('cpu' or 'cuda').

    Returns:
        CompactCNNAutoencoder: Model in eval mode with loaded weights.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If checkpoint structure is invalid.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"CNN checkpoint not found at: {checkpoint_path}\n"
            "Run src/train.py first to generate the checkpoint."
        )

    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"Invalid checkpoint format: 'model_state_dict' key missing in {checkpoint_path}"
        )

    model = get_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Log checkpoint identity so users can confirm the right model is loaded.
    epoch = checkpoint.get("epoch", "?")
    val_psnr = checkpoint.get("val_psnr", float("nan"))
    val_loss = checkpoint.get("val_loss", float("nan"))
    print(
        f"[evaluate] Loaded checkpoint: {checkpoint_path}\n"
        f"           Epoch={epoch}  val_psnr={val_psnr:.4f} dB  val_loss={val_loss:.6f}"
    )

    return model


def cnn_denoise_image(
    model: CompactCNNAutoencoder,
    noisy_np: np.ndarray,
    device: str = "cpu"
) -> np.ndarray:
    """
    Runs CNN autoencoder inference on a full-resolution noisy image.

    The model is fully convolutional (no FC layers) and accepts arbitrary
    spatial dimensions, so no tiling is required. Input (H, W, C) numpy
    array is converted to (1, C, H, W) tensor, forward-passed, and
    converted back to (H, W, C) float32.

    Args:
        model: Loaded CompactCNNAutoencoder in eval mode.
        noisy_np: Noisy image (H, W, 3) float32 in [0.0, 1.0].
        device: Torch device string.

    Returns:
        np.ndarray: Denoised image (H, W, 3) float32 in [0.0, 1.0].
    """
    # Preserve original dimensions and pad to a multiple of 4.
    original_h, original_w = noisy_np.shape[:2]

    tensor_in = torch.from_numpy(
        noisy_np.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    pad_h = (4 - original_h % 4) % 4
    pad_w = (4 - original_w % 4) % 4

    # Pad only the bottom and right edges.
    if pad_h or pad_w:
        tensor_in = F.pad(
            tensor_in,
            (0, pad_w, 0, pad_h),
            mode="replicate"
        )

    with torch.no_grad():
        tensor_out = model(tensor_in)

    # Remove padding and restore original image dimensions.
    tensor_out = tensor_out[:, :, :original_h, :original_w]

    denoised = tensor_out.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    return np.clip(denoised, 0.0, 1.0).astype(np.float32)


# =========================================================================
# Core evaluation logic
# =========================================================================

def evaluate_single_image(
    clean_np: np.ndarray,
    sigma: int,
    image_name: str,
    model: CompactCNNAutoencoder,
    config: dict,
    device: str = "cpu",
    noise_seed: int = 42,
    latency_runs: int = 5,
) -> Dict[str, object]:
    """
    Evaluates all four denoising methods on a single clean image at one noise level.

    Args:
        clean_np:    Clean reference image (H, W, 3) float32 in [0.0, 1.0].
        sigma:       AWGN standard deviation on [0, 255] scale.
        image_name:  Filename (used for logging).
        model:       Loaded CNN model in eval mode.
        config:      Project configuration dict.
        device:      Torch device string.
        noise_seed:  Fixed seed ensuring identical noisy input for all methods.
        latency_runs: Number of timed runs for latency benchmarking.

    Returns:
        dict with keys: image, sigma, method, psnr, ssim, latency_ms
        One dict per method, returned as a list from the caller.
    """
    # Generate noisy image with deterministic seed so all methods get
    # exactly the same degraded input for a fair comparison.
    noisy_np = add_gaussian_noise_numpy(clean_np, sigma=sigma, seed=noise_seed)

    results = []

    # ------------------------------------------------------------------
    # 1. Gaussian filter
    # ------------------------------------------------------------------
    def run_gaussian():
        params = config.get("traditional_filters", {}).get("gaussian", {})
        return apply_gaussian_filter(noisy_np, **params)

    gauss_out = run_gaussian()
    gauss_lat = benchmark_latency(run_gaussian, num_runs=latency_runs, warmup=2)

    results.append({
        "image": image_name,
        "sigma": sigma,
        "method": "Gaussian",
        "psnr": calculate_psnr(clean_np, gauss_out),
        "ssim": calculate_ssim(clean_np, gauss_out),
        "latency_ms": gauss_lat["mean_ms"],
    })

    # ------------------------------------------------------------------
    # 2. Median filter
    # ------------------------------------------------------------------
    def run_median():
        params = config.get("traditional_filters", {}).get("median", {})
        return apply_median_filter(noisy_np, **params)

    median_out = run_median()
    median_lat = benchmark_latency(run_median, num_runs=latency_runs, warmup=2)

    results.append({
        "image": image_name,
        "sigma": sigma,
        "method": "Median",
        "psnr": calculate_psnr(clean_np, median_out),
        "ssim": calculate_ssim(clean_np, median_out),
        "latency_ms": median_lat["mean_ms"],
    })

    # ------------------------------------------------------------------
    # 3. Bilateral filter
    # ------------------------------------------------------------------
    def run_bilateral():
        params = config.get("traditional_filters", {}).get("bilateral", {})
        return apply_bilateral_filter(noisy_np, **params)

    bilateral_out = run_bilateral()
    bilateral_lat = benchmark_latency(run_bilateral, num_runs=latency_runs, warmup=2)

    results.append({
        "image": image_name,
        "sigma": sigma,
        "method": "Bilateral",
        "psnr": calculate_psnr(clean_np, bilateral_out),
        "ssim": calculate_ssim(clean_np, bilateral_out),
        "latency_ms": bilateral_lat["mean_ms"],
    })

    # ------------------------------------------------------------------
    # 4. CNN Autoencoder
    # ------------------------------------------------------------------
    def run_cnn():
        return cnn_denoise_image(model, noisy_np, device=device)

    cnn_out = run_cnn()
    cnn_lat = benchmark_latency(run_cnn, num_runs=latency_runs, warmup=2)

    results.append({
        "image": image_name,
        "sigma": sigma,
        "method": "CNN",
        "psnr": calculate_psnr(clean_np, cnn_out),
        "ssim": calculate_ssim(clean_np, cnn_out),
        "latency_ms": cnn_lat["mean_ms"],
    })

    return results


def run_full_evaluation(
    config_path: str = "configs/config.yaml",
    checkpoint_path: Optional[str] = None,
    test_dir: Optional[str] = None,
    noise_levels: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    device: str = "cpu",
    latency_runs: int = 5,
    noise_seed: int = 42,
) -> List[Dict[str, object]]:
    """
    Runs the complete evaluation pipeline over all Set12 images and noise levels.

    Args:
        config_path:     Path to configs/config.yaml.
        checkpoint_path: Override path to CNN checkpoint (.pth).
        test_dir:        Override path to test benchmark directory.
        noise_levels:    Override noise sigma list (default from config).
        output_dir:      Override output directory for tables.
        device:          Torch device for CNN inference.
        latency_runs:    Number of timed latency runs per method.
        noise_seed:      Fixed random seed for reproducible noise.

    Returns:
        List of per-image-per-sigma-per-method result dicts.
    """
    config = load_config(config_path)

    # Resolve checkpoint: CLI --checkpoint arg → config evaluation.inference_checkpoint
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

    _noise_levels = noise_levels or config.get("noise", {}).get(
        "levels", [15, 25, 50]
    )

    _output_dir = output_dir or config.get("evaluation", {}).get(
        "tables_dir", "results/tables"
    )

    # Ensure output directory exists
    os.makedirs(_output_dir, exist_ok=True)

    # Load CNN model
    print(f"Loading CNN checkpoint: {_ckpt}")
    model = load_cnn_model(_ckpt, device=device)
    print(f"CNN model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Discover test images
    test_path = Path(_test_dir)
    image_files = sorted(
        [f for f in test_path.glob("*.png")] + [f for f in test_path.glob("*.jpg")]
    )

    if not image_files:
        raise FileNotFoundError(
            f"No PNG/JPG images found in test benchmark directory: {_test_dir}\n"
            "Run scripts/download_dataset.py to download Set12 benchmark images."
        )

    print(f"Found {len(image_files)} test images in {_test_dir}")
    print(f"Noise levels (sigma): {_noise_levels}")
    print(f"Device: {device}\n")

    all_results = []
    total_evaluations = len(image_files) * len(_noise_levels)
    completed = 0

    for img_path in image_files:
        clean_np = load_image_rgb_float32(str(img_path))
        img_name = img_path.name
        h, w = clean_np.shape[:2]
        print(f"  [{img_name}] size={w}x{h}")

        for sigma in _noise_levels:
            img_results = evaluate_single_image(
                clean_np=clean_np,
                sigma=sigma,
                image_name=img_name,
                model=model,
                config=config,
                device=device,
                noise_seed=noise_seed,
                latency_runs=latency_runs,
            )
            all_results.extend(img_results)
            completed += 1

            # Print progress per image+sigma
            for r in img_results:
                psnr_str = f"{r['psnr']:.2f}" if r['psnr'] != float('inf') else "inf"
                print(
                    f"    sigma={sigma:2d} | {r['method']:<10s} | "
                    f"PSNR={psnr_str:>6s} dB | SSIM={r['ssim']:.4f} | "
                    f"Lat={r['latency_ms']:.1f} ms"
                )

    print(f"\nEvaluation complete: {len(all_results)} records generated.")

    # Save results to CSV and Markdown
    csv_path = os.path.join(_output_dir, "evaluation_results.csv")
    md_path  = os.path.join(_output_dir, "evaluation_results.md")

    save_results_csv(all_results, csv_path)
    save_results_markdown(all_results, md_path, _noise_levels)

    print(f"\nResults saved:")
    print(f"  CSV : {csv_path}")
    print(f"  MD  : {md_path}")

    return all_results


# =========================================================================
# Output serialization
# =========================================================================

def save_results_csv(results: List[Dict], output_path: str) -> None:
    """
    Saves evaluation results to a structured CSV file.

    Columns: image, sigma, method, psnr, ssim, latency_ms

    Args:
        results:     List of per-record result dicts.
        output_path: Path to output .csv file.
    """
    fieldnames = ["image", "sigma", "method", "psnr", "ssim", "latency_ms"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            psnr_val = row["psnr"]
            writer.writerow({
                "image":       row["image"],
                "sigma":       row["sigma"],
                "method":      row["method"],
                "psnr":        f"{psnr_val:.4f}" if psnr_val != float("inf") else "inf",
                "ssim":        f"{row['ssim']:.4f}",
                "latency_ms":  f"{row['latency_ms']:.3f}",
            })


def compute_summary_table(
    results: List[Dict],
    noise_levels: List[int]
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Aggregates per-image results into mean PSNR/SSIM/Latency grouped by
    (method, sigma).

    Returns:
        Dict[method -> Dict[sigma -> {'psnr': float, 'ssim': float, 'latency_ms': float}]]
    """
    from collections import defaultdict
    accum: Dict = defaultdict(lambda: defaultdict(lambda: {"psnr": [], "ssim": [], "lat": []}))

    for row in results:
        m = row["method"]
        s = row["sigma"]
        psnr_val = row["psnr"]
        if psnr_val == float("inf"):
            psnr_val = 100.0  # Cap inf for averaging (identical images)
        accum[m][s]["psnr"].append(psnr_val)
        accum[m][s]["ssim"].append(row["ssim"])
        accum[m][s]["lat"].append(row["latency_ms"])

    summary = {}
    for method, sigma_dict in accum.items():
        summary[method] = {}
        for sigma, vals in sigma_dict.items():
            summary[method][sigma] = {
                "psnr":       float(np.mean(vals["psnr"])),
                "ssim":       float(np.mean(vals["ssim"])),
                "latency_ms": float(np.mean(vals["lat"])),
            }
    return summary


def save_results_markdown(
    results: List[Dict],
    output_path: str,
    noise_levels: List[int]
) -> None:
    """
    Saves a publication-style summary Markdown table grouped by noise level
    and method, showing mean PSNR, SSIM, and Latency across all test images.

    Args:
        results:      List of per-record result dicts.
        output_path:  Path to output .md file.
        noise_levels: Ordered list of sigma values.
    """
    methods = ["Gaussian", "Median", "Bilateral", "CNN"]
    summary = compute_summary_table(results, noise_levels)

    lines = [
        "# Denoising Evaluation Results — Set12 Benchmark",
        "",
        "**Dataset:** Set12 (12 standard benchmark images)  ",
        "**Noise:** Additive White Gaussian Noise (AWGN), seed=42  ",
        "**Metrics:** Mean across all 12 test images  ",
        "",
    ]

    for sigma in noise_levels:
        lines.append(f"## Noise Level: sigma = {sigma}")
        lines.append("")
        lines.append("| Method     | PSNR (dB) | SSIM   | Latency (ms) |")
        lines.append("|------------|-----------|--------|--------------|")

        for method in methods:
            if method in summary and sigma in summary[method]:
                v = summary[method][sigma]
                lines.append(
                    f"| {method:<10s} | {v['psnr']:>9.2f} | {v['ssim']:.4f} | "
                    f"{v['latency_ms']:>12.2f} |"
                )
            else:
                lines.append(f"| {method:<10s} | N/A       | N/A    | N/A          |")

        lines.append("")

    # Overall summary table (averaged across all sigma)
    lines.append("## Overall Mean (All Sigma Levels)")
    lines.append("")
    lines.append("| Method     | PSNR (dB) | SSIM   | Latency (ms) |")
    lines.append("|------------|-----------|--------|--------------|")

    for method in methods:
        if method not in summary:
            continue
        psnr_vals, ssim_vals, lat_vals = [], [], []
        for sigma in noise_levels:
            if sigma in summary[method]:
                v = summary[method][sigma]
                psnr_vals.append(v["psnr"])
                ssim_vals.append(v["ssim"])
                lat_vals.append(v["latency_ms"])
        if psnr_vals:
            lines.append(
                f"| {method:<10s} | {np.mean(psnr_vals):>9.2f} | "
                f"{np.mean(ssim_vals):.4f} | {np.mean(lat_vals):>12.2f} |"
            )

    lines.append("")
    lines.append(
        "> Results generated by `src/evaluate.py` from actual model inference."
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =========================================================================
# CLI entry point
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all denoising methods on Set12 benchmark."
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Override CNN checkpoint path."
    )
    parser.add_argument(
        "--test-dir", default=None,
        help="Override test benchmark directory."
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output tables directory."
    )
    parser.add_argument(
        "--latency-runs", type=int, default=5,
        help="Number of latency benchmark runs per method (default: 5)."
    )
    args = parser.parse_args()

    run_full_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        latency_runs=args.latency_runs,
    )


if __name__ == "__main__":
    main()
