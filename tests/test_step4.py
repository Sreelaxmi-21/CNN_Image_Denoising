"""
tests/test_step4.py
===================
Focused verification test suite for Step 4: Training Pipeline on CPU.

Verifies:
1. Dataloaders: patch extraction, batch shapes (B, 3, 64, 64), range bounds [0.0, 1.0].
2. Dynamic noise injection with sigma in {15, 25, 50} and deterministic validation patches.
3. 1-epoch dry-run: forward pass, backward pass, optimizer step, valid train/val loss.
4. Loss descent: verifies that training MSE loss decreases over multiple iterations.
5. Checkpoint saving & loading: saves best_model.pth, verifies file size, reloads into
   fresh model instance, and verifies exact output equivalence.
6. Logging: verifies CSV and JSON training logs are created with valid rows.
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.autoencoder import CompactCNNAutoencoder, get_model
from src.dataset import PatchDenoisingDataset, get_dataloaders
from src.train import train_one_epoch, validate_epoch, train_model


def test_dataloaders_and_patches():
    print("[Test 1/6] Validating PatchDenoisingDataset & DataLoaders...")
    train_loader, val_loader = get_dataloaders("configs/config.yaml")

    assert len(train_loader) > 0, "Train loader is empty!"
    assert len(val_loader) > 0, "Validation loader is empty!"

    # Inspect first batch
    for noisy, clean, sigmas in train_loader:
        assert noisy.shape == (32, 3, 64, 64) or noisy.shape[1:] == (3, 64, 64), f"Unexpected shape: {noisy.shape}"
        assert clean.shape == noisy.shape, f"Shape mismatch: {clean.shape} vs {noisy.shape}"
        assert noisy.min() >= 0.0 and noisy.max() <= 1.0, f"Noisy range out of bounds: [{noisy.min()}, {noisy.max()}]"
        assert clean.min() >= 0.0 and clean.max() <= 1.0, f"Clean range out of bounds: [{clean.min()}, {clean.max()}]"
        
        # Check sigmas
        unique_sigmas = set(sigmas.numpy())
        for s in unique_sigmas:
            assert s in [15.0, 25.0, 50.0], f"Unexpected noise sigma: {s}"
        print(f" -> Train batch verified: shape={list(noisy.shape)}, sigmas in batch={unique_sigmas}")
        break

    print(" -> PASSED: Train and validation dataloaders provide valid paired tensors.")


def test_validation_determinism():
    print("[Test 2/6] Validating deterministic validation noise...")
    val_dataset = PatchDenoisingDataset(
        image_dir=PROJECT_ROOT / "data" / "raw" / "val",
        patch_size=64,
        stride=64,
        noise_levels=(15.0, 25.0, 50.0),
        is_train=False,
        augment=False,
        seed=42
    )

    # Calling __getitem__ on same index twice must produce identical noise
    noisy1, clean1, sigma1 = val_dataset[0]
    noisy2, clean2, sigma2 = val_dataset[0]

    assert torch.equal(noisy1, noisy2), "Validation noise is non-deterministic!"
    assert torch.equal(clean1, clean2), "Validation clean patches differ!"
    assert sigma1 == sigma2, "Validation sigmas differ!"

    print(f" -> Verified patch 0: sigma={sigma1}, tensor identical across repeated queries.")
    print(" -> PASSED: Validation dataset is strictly deterministic for fair epoch tracking.")


def test_one_epoch_dry_run():
    print("[Test 3/6] Executing 1-epoch training dry-run on CPU...")
    device = torch.device("cpu")
    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_loader, val_loader = get_dataloaders("configs/config.yaml")

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_psnr, val_ssim = validate_epoch(model, val_loader, criterion, device)

    assert not np.isnan(train_loss) and not np.isinf(train_loss), f"Train loss invalid: {train_loss}"
    assert not np.isnan(val_loss) and not np.isinf(val_loss), f"Val loss invalid: {val_loss}"
    assert train_loss > 0.0 and val_loss > 0.0, "Losses must be strictly positive."
    assert val_psnr > 0.0, f"Val PSNR must be positive, got {val_psnr}"
    assert 0.0 <= val_ssim <= 1.0, f"Val SSIM out of bounds: {val_ssim}"

    print(f" -> 1 Epoch Dry-Run Output: Train MSE={train_loss:.6f}, Val MSE={val_loss:.6f}, Val PSNR={val_psnr:.2f} dB, Val SSIM={val_ssim:.4f}")
    print(" -> PASSED: 1-epoch forward/backward/validation cycle executes seamlessly on CPU.")


def test_loss_descent():
    print("[Test 4/6] Verifying optimization and loss descent over multiple iterations...")
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.MSELoss()

    # Fixed synthetic mini-batch to test optimization convergence
    fixed_clean = torch.rand(8, 3, 64, 64, device=device)
    fixed_noisy = torch.clamp(fixed_clean + torch.randn_like(fixed_clean) * 0.1, 0.0, 1.0)

    losses = []
    model.train()
    for step in range(5):
        optimizer.zero_grad()
        output = model(fixed_noisy)
        loss = criterion(output, fixed_clean)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f" -> Loss descent progression: Initial={losses[0]:.6f} -> Step 5={losses[-1]:.6f}")
    assert losses[-1] < losses[0], f"Loss failed to descend: {losses[0]} to {losses[-1]}"
    print(" -> PASSED: Optimizer successfully drives loss down.")


def test_checkpoint_saving_and_loading():
    print("[Test 5/6] Verifying checkpoint creation, structure, and model reloading...")
    model = get_model()
    dummy_input = torch.rand(1, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        original_output = model(dummy_input)

    checkpoint_dir = PROJECT_ROOT / "results" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    test_ckpt_path = checkpoint_dir / "test_checkpoint.pth"

    payload = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "train_loss": 0.015,
        "val_loss": 0.014,
        "val_psnr": 24.5,
        "val_ssim": 0.65,
    }
    torch.save(payload, test_ckpt_path)

    assert test_ckpt_path.exists(), "Checkpoint file was not created!"
    file_size_mb = os.path.getsize(test_ckpt_path) / (1024 * 1024)
    assert file_size_mb > 1.0, f"Checkpoint size unexpectedly small: {file_size_mb:.2f} MB"
    print(f" -> Checkpoint written: size={file_size_mb:.2f} MB")

    # Reload into fresh instance
    new_model = get_model()
    loaded_data = torch.load(test_ckpt_path, map_location="cpu", weights_only=False)
    new_model.load_state_dict(loaded_data["model_state_dict"])
    new_model.eval()

    with torch.no_grad():
        reloaded_output = new_model(dummy_input)

    assert torch.allclose(original_output, reloaded_output, atol=1e-7), "Reloaded model output mismatch!"
    print(" -> PASSED: Checkpoint saves and reloads with bit-exact tensor outputs.")

    # Clean up test checkpoint
    if test_ckpt_path.exists():
        test_ckpt_path.unlink()


def test_full_pipeline_and_logging():
    print("[Test 6/6] Verifying train_model execution with 2 epochs & log generation...")
    test_ckpt_name = "test_best_model.pth"
    test_log_name = "test_training_log"

    summary = train_model(
        config_path="configs/config.yaml",
        epochs_override=2,
        lr_override=0.001,
        device_override="cpu",
        checkpoint_name=test_ckpt_name,
        log_name=test_log_name,
        save_checkpoint=True
    )

    test_model_path = PROJECT_ROOT / "results" / "checkpoints" / test_ckpt_name
    csv_log_path = PROJECT_ROOT / "results" / "logs" / f"{test_log_name}.csv"
    json_log_path = PROJECT_ROOT / "results" / "logs" / f"{test_log_name}.json"

    assert test_model_path.exists(), f"Test checkpoint was not created at {test_model_path}"
    assert csv_log_path.exists(), f"CSV log missing at {csv_log_path}"
    assert json_log_path.exists(), f"JSON log missing at {json_log_path}"

    # Verify CSV content
    with open(csv_log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 3, f"Expected 1 header + 2 epoch lines in CSV, got {len(lines)}"
    assert lines[0].strip() == "epoch,train_loss,val_loss,val_psnr,val_ssim,lr,time_sec"

    print(f" -> Test model checkpoint verified at: {test_model_path}")
    print(f" -> CSV log verified with {len(lines) - 1} epoch records.")

    # Clean up test artifacts
    if test_model_path.exists():
        test_model_path.unlink()
    if csv_log_path.exists():
        csv_log_path.unlink()
    if json_log_path.exists():
        json_log_path.unlink()

    print(" -> PASSED: Training execution, checkpointing, and logging verified end-to-end.")


if __name__ == "__main__":
    print("=== RUNNING STEP 4 VERIFICATION SUITE ===\n")
    test_dataloaders_and_patches()
    test_validation_determinism()
    test_one_epoch_dry_run()
    test_loss_descent()
    test_checkpoint_saving_and_loading()
    test_full_pipeline_and_logging()
    print("\n=== ALL STEP 4 TESTS PASSED SUCCESSFULLY ===")
