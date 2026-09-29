"""
src/metrics.py
==============
Standardized quantitative evaluation metrics for digital image denoising:
1. PSNR (Peak Signal-to-Noise Ratio) [dB]
2. SSIM (Structural Similarity Index)
3. Latency Benchmarking (ms/image & FPS)

Designed for strict comparability across classical spatial filters and
deep neural network outputs under normalized data range [0.0, 1.0].
"""

import time
import math
from typing import Callable, Dict, Optional, Tuple, Union
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn


def _to_numpy(img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Converts PyTorch Tensor or NumPy array to a float64 NumPy array in [0.0, 1.0].
    Handles channel dimensions (C, H, W) -> (H, W, C) for PyTorch tensors.
    """
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            # Convert (C, H, W) -> (H, W, C)
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[2] == 1:
                arr = arr.squeeze(axis=2)
        elif arr.ndim == 4:
            raise ValueError("Expected single image tensor (C, H, W) or (H, W), not batch (B, C, H, W).")
    elif isinstance(img, np.ndarray):
        arr = img.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(img)}. Expected np.ndarray or torch.Tensor.")

    # Validate or normalize range
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float64) / 255.0
    else:
        arr = arr.astype(np.float64)

    return np.clip(arr, 0.0, 1.0)


def calculate_psnr(
    clean: Union[np.ndarray, torch.Tensor],
    denoised: Union[np.ndarray, torch.Tensor],
    data_range: float = 1.0
) -> float:
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Mathematical Formulation:
        MSE = (1 / N) * sum((I_clean - I_denoised)^2)
        PSNR = 10 * log10(data_range^2 / MSE)

    Args:
        clean: Reference ground-truth clean image.
        denoised: Denoised or noisy reconstructed image.
        data_range: The dynamic range of pixel values (default: 1.0 for [0, 1]).

    Returns:
        float: PSNR in dB. Returns float('inf') if images are identical (MSE == 0).
    """
    clean_arr = _to_numpy(clean)
    denoised_arr = _to_numpy(denoised)

    if clean_arr.shape != denoised_arr.shape:
        raise ValueError(f"Shape mismatch: clean {clean_arr.shape} vs denoised {denoised_arr.shape}")

    mse = float(np.mean((clean_arr - denoised_arr) ** 2))

    if mse < 1e-12:
        return float("inf")

    psnr = 10.0 * math.log10((data_range ** 2) / mse)
    return float(psnr)


def calculate_ssim(
    clean: Union[np.ndarray, torch.Tensor],
    denoised: Union[np.ndarray, torch.Tensor],
    data_range: float = 1.0
) -> float:
    """
    Calculates Structural Similarity Index (SSIM) based on luminance, contrast,
    and structural fidelity (Wang et al., 2004).

    Mathematical Formulation:
        SSIM(x, y) = [ (2*mu_x*mu_y + C1) * (2*sigma_xy + C2) ] /
                     [ (mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2) ]
        where C1 = (K1*L)^2, C2 = (K2*L)^2, K1=0.01, K2=0.03, L=data_range.

    Args:
        clean: Reference ground-truth clean image.
        denoised: Denoised or noisy reconstructed image.
        data_range: The dynamic range of pixel values (default: 1.0 for [0, 1]).

    Returns:
        float: SSIM value in [-1.0, 1.0], typically [0.0, 1.0].
    """
    clean_arr = _to_numpy(clean)
    denoised_arr = _to_numpy(denoised)

    if clean_arr.shape != denoised_arr.shape:
        raise ValueError(f"Shape mismatch: clean {clean_arr.shape} vs denoised {denoised_arr.shape}")

    # Determine channel configuration
    if clean_arr.ndim == 2:
        channel_axis = None
    elif clean_arr.ndim == 3:
        channel_axis = 2
    else:
        raise ValueError(f"Unexpected image dimensions: {clean_arr.ndim}")

    val = ssim_fn(
        clean_arr,
        denoised_arr,
        data_range=data_range,
        channel_axis=channel_axis
    )
    return float(val)


def evaluate_image_quality(
    clean: Union[np.ndarray, torch.Tensor],
    denoised: Union[np.ndarray, torch.Tensor],
    data_range: float = 1.0
) -> Dict[str, float]:
    """
    Convenience function computing both PSNR and SSIM in a single call.

    Returns:
        Dict[str, float]: {'psnr': float, 'ssim': float}
    """
    return {
        "psnr": calculate_psnr(clean, denoised, data_range=data_range),
        "ssim": calculate_ssim(clean, denoised, data_range=data_range),
    }


def benchmark_latency(
    func: Callable,
    *args,
    num_runs: int = 10,
    warmup: int = 2,
    **kwargs
) -> Dict[str, float]:
    """
    Accurately benchmarks inference / execution latency on CPU.

    Args:
        func: Callable function to benchmark (e.g. filter function or model forward).
        *args: Positional arguments passed to func.
        num_runs: Number of timed iterations.
        warmup: Number of initial untimed warmup iterations.
        **kwargs: Keyword arguments passed to func.

    Returns:
        Dict[str, float]:
            - 'mean_ms': Average latency in milliseconds
            - 'std_ms': Standard deviation in milliseconds
            - 'fps': Equivalent frames per second
    """
    # Warmup runs to stabilize CPU caches
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    timings = []
    for _ in range(num_runs):
        t_start = time.perf_counter()
        _ = func(*args, **kwargs)
        t_end = time.perf_counter()
        timings.append((t_end - t_start) * 1000.0) # in ms

    mean_ms = float(np.mean(timings))
    std_ms = float(np.std(timings))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "fps": fps
    }
