"""
src/noise.py
============
Controlled Additive White Gaussian Noise (AWGN) generator.

Mathematical Formulation:
-------------------------
For a clean natural image x normalized to [0.0, 1.0]:
    y = clip(x + n, 0.0, 1.0)
where:
    n ~ N(0, sigma_norm^2)
    sigma_norm = sigma / 255.0
    expected_variance = sigma_norm^2 = (sigma / 255.0)^2

Standard Digital Image Processing noise levels:
    - sigma = 15: sigma_norm = 15/255 ≈ 0.0588, Expected Variance = (15/255)^2 ≈ 0.003460
    - sigma = 25: sigma_norm = 25/255 ≈ 0.0980, Expected Variance = (25/255)^2 ≈ 0.009612
    - sigma = 50: sigma_norm = 50/255 ≈ 0.1961, Expected Variance = (50/255)^2 ≈ 0.038447
"""

from typing import Union, Tuple, Optional
import numpy as np
import torch


def add_gaussian_noise_numpy(
    image: np.ndarray,
    sigma: float,
    seed: Optional[int] = None,
    clip_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Adds zero-mean Additive White Gaussian Noise (AWGN) to a NumPy image array.

    Args:
        image: Clean input image in [0.0, 1.0], shape (H, W, C) or (H, W), dtype float32/float64.
        sigma: Standard deviation of noise on the [0, 255] scale (e.g. 15, 25, 50).
        seed: Optional integer seed for deterministic, reproducible noise generation.
        clip_range: Tuple of (min_val, max_val) to clip the resulting noisy image.

    Returns:
        np.ndarray: Noisy image clipped to clip_range with the same shape and float32 dtype.
    """
    if not (0.0 <= image.min() and image.max() <= 1.0):
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    sigma_norm = float(sigma) / 255.0

    if seed is not None:
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=sigma_norm, size=image.shape).astype(np.float32)
    else:
        noise = np.random.normal(loc=0.0, scale=sigma_norm, size=image.shape).astype(np.float32)

    noisy_image = np.clip(image + noise, clip_range[0], clip_range[1])
    return noisy_image.astype(np.float32)


def add_gaussian_noise_torch(
    tensor: torch.Tensor,
    sigma: float,
    seed: Optional[int] = None,
    clip_range: Tuple[float, float] = (0.0, 1.0)
) -> torch.Tensor:
    """
    Adds zero-mean Additive White Gaussian Noise (AWGN) to a PyTorch Tensor.

    Args:
        tensor: Clean input image tensor in [0.0, 1.0], shape (C, H, W) or (B, C, H, W).
        sigma: Standard deviation of noise on the [0, 255] scale (e.g. 15, 25, 50).
        seed: Optional integer seed for deterministic noise generation.
        clip_range: Tuple of (min_val, max_val) for clipping.

    Returns:
        torch.Tensor: Noisy image tensor clipped to clip_range with float32 dtype.
    """
    sigma_norm = float(sigma) / 255.0

    if seed is not None:
        generator = torch.Generator(device=tensor.device).manual_seed(seed)
        noise = torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype) * sigma_norm
    else:
        noise = torch.randn_like(tensor) * sigma_norm

    noisy_tensor = torch.clamp(tensor + noise, min=clip_range[0], max=clip_range[1])
    return noisy_tensor


def verify_noise_properties(
    shape: Tuple[int, ...] = (1, 3, 512, 512),
    sigma: float = 25.0,
    seed: int = 42
) -> dict:
    """
    Empirically verifies the mathematical properties:
    - Mean: E[n] ≈ 0.0
    - Standard Deviation: std(n) ≈ sigma / 255.0
    - Variance: Var[n] ≈ (sigma / 255.0)^2

    Args:
        shape: Shape of the noise tensor to sample for verification.
        sigma: Standard deviation on [0, 255] scale.
        seed: Random seed for sampling.

    Returns:
        dict: Measured statistics vs. theoretical values and validity flag.
    """
    sigma_norm = float(sigma) / 255.0
    expected_variance = sigma_norm ** 2

    rng = np.random.default_rng(seed)
    sample_noise = rng.normal(loc=0.0, scale=sigma_norm, size=shape).astype(np.float64)

    measured_mean = float(np.mean(sample_noise))
    measured_std = float(np.std(sample_noise))
    measured_var = float(np.var(sample_noise))

    mean_error = abs(measured_mean - 0.0)
    std_error = abs(measured_std - sigma_norm)
    var_error = abs(measured_var - expected_variance)

    # Tolerances scaled for sample size
    is_valid = (mean_error < 0.005) and (std_error < 0.005) and (var_error < 0.005)

    return {
        "sigma_255": sigma,
        "theoretical_sigma_norm": sigma_norm,
        "theoretical_variance": expected_variance,
        "measured_mean": measured_mean,
        "measured_std": measured_std,
        "measured_var": measured_var,
        "mean_error": mean_error,
        "std_error": std_error,
        "var_error": var_error,
        "is_valid": is_valid
    }
