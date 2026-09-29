"""
src/dataset.py
==============
PyTorch Dataset and DataLoader for Image Denoising.

Key Capabilities:
-----------------
1. Patch-Based Extraction:
   Extracts 64x64 patches with stride 48 from natural images.
2. Dynamic Paired Noisy/Clean Generation:
   For training, patches are augmented and dynamically injected with
   controlled AWGN sampled from sigma in {15, 25, 50} (or configurable).
3. Deterministic Validation:
   Validation patches use fixed noise seeds so evaluation metrics
   (Loss, PSNR, SSIM) are strictly comparable across training epochs.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

from src.noise import add_gaussian_noise_torch


def extract_patches_from_image(
    image: np.ndarray,
    patch_size: int = 64,
    stride: int = 48
) -> List[np.ndarray]:
    """
    Slices an image into overlapping or non-overlapping patches.

    Args:
        image: Image array of shape (H, W, C) in [0.0, 1.0].
        patch_size: Spatial height and width of extracted square patches.
        stride: Step size between adjacent patch origins.

    Returns:
        List[np.ndarray]: Extracted patches, each of shape (patch_size, patch_size, C).
    """
    h, w, c = image.shape
    if h < patch_size or w < patch_size:
        raise ValueError(f"Image dimensions ({h}x{w}) smaller than patch_size ({patch_size})")

    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size, :]
            patches.append(patch.copy())

    return patches


class PatchDenoisingDataset(Dataset):
    """
    Paired Denoising Dataset operating on 64x64 patches.

    Returns:
        noisy_tensor: Shape (3, 64, 64) in [0.0, 1.0]
        clean_tensor: Shape (3, 64, 64) in [0.0, 1.0]
        sigma: Float representing the noise standard deviation on [0, 255] scale.
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        patch_size: int = 64,
        stride: int = 48,
        noise_levels: Tuple[float, ...] = (15.0, 25.0, 50.0),
        is_train: bool = True,
        augment: bool = True,
        seed: Optional[int] = 42
    ):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.noise_levels = tuple(float(s) for s in noise_levels)
        self.is_train = is_train
        self.augment = augment and is_train
        self.seed = seed

        # Collect image files
        image_files = sorted(list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg")))
        if not image_files:
            raise FileNotFoundError(f"No image files found in {self.image_dir}")

        # Extract and cache clean patches into memory as torch.FloatTensor
        raw_patches = []
        for img_path in image_files:
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
            patches = extract_patches_from_image(img, patch_size=self.patch_size, stride=self.stride)
            raw_patches.extend(patches)

        if not raw_patches:
            raise RuntimeError(f"Zero patches extracted from {self.image_dir}")

        # Store as float tensor with shape (N, 3, patch_size, patch_size)
        stacked = np.stack(raw_patches, axis=0) # (N, H, W, 3)
        self.clean_patches = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous() # (N, 3, H, W)

        # For validation, pre-assign deterministic noise levels and seeds
        if not self.is_train:
            val_rng = np.random.default_rng(self.seed if self.seed is not None else 42)
            self.val_sigmas = val_rng.choice(self.noise_levels, size=len(self.clean_patches))
            self.val_seeds = [int(val_rng.integers(1, 1000000)) for _ in range(len(self.clean_patches))]

    def __len__(self) -> int:
        return len(self.clean_patches)

    def _apply_augmentations(self, patch: torch.Tensor) -> torch.Tensor:
        """Random horizontal flip, vertical flip, and 90-degree rotations."""
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            patch = torch.flip(patch, dims=[2])
        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            patch = torch.flip(patch, dims=[1])
        # Random 90-degree rotation (k in {0, 1, 2, 3})
        k = int(torch.randint(0, 4, (1,)).item())
        if k > 0:
            patch = torch.rot90(patch, k=k, dims=[1, 2])
        return patch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        clean_patch = self.clean_patches[idx].clone()

        if self.is_train:
            if self.augment:
                clean_patch = self._apply_augmentations(clean_patch)

            # Dynamic noise level selection
            sigma_idx = int(torch.randint(0, len(self.noise_levels), (1,)).item())
            sigma = self.noise_levels[sigma_idx]

            # Dynamic AWGN injection
            noisy_patch = add_gaussian_noise_torch(clean_patch, sigma=sigma, seed=None)
        else:
            # Deterministic noise for validation
            sigma = float(self.val_sigmas[idx])
            seed = self.val_seeds[idx]
            noisy_patch = add_gaussian_noise_torch(clean_patch, sigma=sigma, seed=seed)

        return noisy_patch, clean_patch, sigma


def get_dataloaders(
    config_path: Union[str, Path] = "configs/config.yaml",
    custom_config: Optional[dict] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Instantiates training and validation DataLoaders using project configuration.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    if custom_config is not None:
        cfg = custom_config
    else:
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

    # Dataset parameters from config
    train_dir = Path(cfg["dataset"]["train"]["dir"])
    val_dir = Path(cfg["dataset"]["validation"]["dir"])
    patch_size = cfg["dataset"]["train"].get("patch_size", 64)
    stride = cfg["dataset"]["train"].get("stride", 48)
    batch_size = cfg["dataset"].get("batch_size", 32)
    num_workers = cfg["dataset"].get("num_workers", 0)
    noise_levels = cfg["noise"].get("levels", [15, 25, 50])
    seed = cfg["experiment"].get("seed", 42)

    # Training dataset and loader
    train_dataset = PatchDenoisingDataset(
        image_dir=train_dir,
        patch_size=patch_size,
        stride=stride,
        noise_levels=noise_levels,
        is_train=True,
        augment=True,
        seed=seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )

    # Validation dataset and loader (stride equal to patch_size for non-overlapping validation)
    val_dataset = PatchDenoisingDataset(
        image_dir=val_dir,
        patch_size=patch_size,
        stride=patch_size, # non-overlapping for validation
        noise_levels=noise_levels,
        is_train=False,
        augment=False,
        seed=seed
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )

    return train_loader, val_loader
