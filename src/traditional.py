"""
src/traditional.py
==================
Classical spatial-domain image denoising filters:
1. Gaussian Filter (Linear smoothing)
2. Median Filter (Order-statistics non-linear)
3. Bilateral Filter (Edge-preserving photometric + geometric non-linear)

All filter functions accept images normalized in [0.0, 1.0] (RGB or Grayscale)
and return denoised images in the exact same format, shape, and range.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import cv2


def _ensure_normalized_float32(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Validates and normalizes input image to float32 in [0.0, 1.0].
    Returns normalized array and whether the original was uint8.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image to be a numpy.ndarray, got {type(image)}")

    was_uint8 = (image.dtype == np.uint8)
    if was_uint8:
        norm_img = image.astype(np.float32) / 255.0
    else:
        norm_img = image.astype(np.float32)
        if norm_img.max() > 1.0:
            norm_img = norm_img / 255.0

    norm_img = np.clip(norm_img, 0.0, 1.0)
    return norm_img, was_uint8


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Converts float32 [0.0, 1.0] image to uint8 [0, 255]."""
    return np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)


def apply_gaussian_filter(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Applies 2D Gaussian linear spatial filtering.

    Mathematical Formulation:
        I_gauss(p) = sum_{q in Omega} I(q) * G_sigma(||p - q||)
        where G_sigma(r) = (1 / (2*pi*sigma^2)) * exp(-r^2 / (2*sigma^2))

    Characteristics:
        - Linear spatial convolution filter.
        - Attenuates high-frequency noise effectively in smooth regions.
        - Blurs sharp boundaries, corners, and texture details.

    Args:
        image: Input image array (H, W, C) or (H, W) in [0.0, 1.0].
        kernel_size: Odd integer kernel size (default: 5).
        sigma: Standard deviation of Gaussian kernel in spatial domain (default: 1.0).

    Returns:
        np.ndarray: Denoised image in float32 within [0.0, 1.0].
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"Gaussian kernel_size must be an odd integer, got {kernel_size}")

    img_norm, _ = _ensure_normalized_float32(image)
    img_u8 = _to_uint8(img_norm)

    filtered_u8 = cv2.GaussianBlur(
        img_u8,
        ksize=(kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT
    )

    filtered = filtered_u8.astype(np.float32) / 255.0
    return np.clip(filtered, 0.0, 1.0)


def apply_median_filter(
    image: np.ndarray,
    kernel_size: int = 3
) -> np.ndarray:
    """
    Applies median order-statistics spatial filtering.

    Mathematical Formulation:
        I_median(p) = median { I(q) | q in Omega(p) }

    Characteristics:
        - Non-linear spatial filter.
        - Exceptional suppression of impulse / salt-and-pepper noise.
        - Preserves step edges better than Gaussian filtering, but can cause
          oil-painting artifacts or corner rounding under continuous AWGN.

    Args:
        image: Input image array (H, W, C) or (H, W) in [0.0, 1.0].
        kernel_size: Odd integer aperture linear size (default: 3).

    Returns:
        np.ndarray: Denoised image in float32 within [0.0, 1.0].
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"Median kernel_size must be an odd integer, got {kernel_size}")

    img_norm, _ = _ensure_normalized_float32(image)
    img_u8 = _to_uint8(img_norm)

    filtered_u8 = cv2.medianBlur(img_u8, ksize=kernel_size)

    filtered = filtered_u8.astype(np.float32) / 255.0
    return np.clip(filtered, 0.0, 1.0)


def apply_bilateral_filter(
    image: np.ndarray,
    diameter: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0
) -> np.ndarray:
    """
    Applies bilateral edge-preserving smoothing.

    Mathematical Formulation:
        I_bilateral(p) = (1 / W_p) * sum_{q in Omega} I(q) * G_space(||p - q||) * G_color(|I(p) - I(q)|)
        where:
            G_space: Spatial Gaussian measuring geometric closeness.
            G_color: Radiometric Gaussian measuring pixel photometric similarity.
            W_p: Normalization factor ensuring weights sum to 1.

    Characteristics:
        - Non-linear filter combining spatial and range domains.
        - Smooths noise in flat regions while preventing blur across prominent edges.
        - Higher computational complexity than linear spatial convolution.

    Args:
        image: Input image array (H, W, C) or (H, W) in [0.0, 1.0].
        diameter: Diameter of each pixel neighborhood (default: 7).
        sigma_color: Filter standard deviation in intensity space on [0, 255] scale (default: 50.0).
        sigma_space: Filter standard deviation in spatial coordinate space (default: 50.0).

    Returns:
        np.ndarray: Denoised image in float32 within [0.0, 1.0].
    """
    img_norm, _ = _ensure_normalized_float32(image)
    img_u8 = _to_uint8(img_norm)

    filtered_u8 = cv2.bilateralFilter(
        img_u8,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
        borderType=cv2.BORDER_REFLECT
    )

    filtered = filtered_u8.astype(np.float32) / 255.0
    return np.clip(filtered, 0.0, 1.0)


def apply_all_traditional_filters(
    image: np.ndarray,
    config: Optional[dict] = None
) -> Dict[str, np.ndarray]:
    """
    Runs Gaussian, Median, and Bilateral filters on the exact same input image.

    Args:
        image: Input noisy image array (H, W, C) or (H, W) in [0.0, 1.0].
        config: Optional dictionary overriding default filter hyperparameters.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - 'gaussian': Gaussian filtered output
            - 'median': Median filtered output
            - 'bilateral': Bilateral filtered output
    """
    # Default parameters aligned with configs/config.yaml
    g_params = {"kernel_size": 5, "sigma": 1.0}
    m_params = {"kernel_size": 3}
    b_params = {"diameter": 7, "sigma_color": 50.0, "sigma_space": 50.0}

    if config and "traditional_filters" in config:
        tf_cfg = config["traditional_filters"]
        if "gaussian" in tf_cfg:
            g_params.update(tf_cfg["gaussian"])
        if "median" in tf_cfg:
            m_params.update(tf_cfg["median"])
        if "bilateral" in tf_cfg:
            b_params.update(tf_cfg["bilateral"])

    denoised_gaussian = apply_gaussian_filter(image, **g_params)
    denoised_median = apply_median_filter(image, **m_params)
    denoised_bilateral = apply_bilateral_filter(image, **b_params)

    return {
        "gaussian": denoised_gaussian,
        "median": denoised_median,
        "bilateral": denoised_bilateral,
    }
