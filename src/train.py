"""
src/train.py
============
CPU-friendly training loop for the Compact CNN Denoising Autoencoder.

Key Features:
-------------
1. Pure MSE reconstruction loss (E[||f(y) - x||^2]).
2. Adam optimizer with learning-rate scheduling.
3. Validation monitoring tracking Loss, PSNR (dB), and SSIM.
4. Model checkpointing: saves best model weights to results/checkpoints/best_model.pth.
5. Comprehensive epoch logging to CSV and JSON formats in results/logs/.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.autoencoder import CompactCNNAutoencoder, get_model
from src.dataset import get_dataloaders
from src.metrics import calculate_psnr, calculate_ssim


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Trains the model for one epoch and returns the average MSE loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for noisy, clean, _ in dataloader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        batch_size = noisy.size(0)

        optimizer.zero_grad()
        reconstructed = model(noisy)
        loss = criterion(reconstructed, clean)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return float(epoch_loss)


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Evaluates the model on validation patches.
    Returns:
        Tuple[float, float, float]: (val_loss, avg_val_psnr, avg_val_ssim)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    psnr_scores = []
    ssim_scores = []

    with torch.no_grad():
        for noisy, clean, _ in dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            batch_size = noisy.size(0)

            reconstructed = model(noisy)
            loss = criterion(reconstructed, clean)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            # Evaluate metrics on sample level
            for i in range(batch_size):
                clean_sample = clean[i].cpu()
                recon_sample = reconstructed[i].cpu()
                p = calculate_psnr(clean_sample, recon_sample)
                s = calculate_ssim(clean_sample, recon_sample)
                if not np.isinf(p):
                    psnr_scores.append(p)
                ssim_scores.append(s)

    val_loss = running_loss / total_samples if total_samples > 0 else 0.0
    avg_psnr = float(np.mean(psnr_scores)) if psnr_scores else 0.0
    avg_ssim = float(np.mean(ssim_scores)) if ssim_scores else 0.0

    return float(val_loss), avg_psnr, avg_ssim


def train_model(
    config_path: Union[str, Path] = "configs/config.yaml",
    epochs_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    device_override: Optional[str] = "cpu",
    checkpoint_name: str = "best_model.pth",
    log_name: str = "training_log",
    save_checkpoint: bool = True
) -> Dict:
    """
    Master training execution function.

    Args:
        config_path: Path to YAML configuration.
        epochs_override: Optional integer to override configured epoch count.
        lr_override: Optional float to override configured learning rate.
        device_override: Execution device (default 'cpu').
        checkpoint_name: Filename for saved checkpoint (default 'best_model.pth').
        log_name: Base name for log files (default 'training_log').
        save_checkpoint: Whether to save model weights to disk.

    Returns:
        Dict: Training history and benchmark summary.
    """
    cfg_path = Path(config_path)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Set seeds for deterministic training
    seed = cfg["experiment"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device_override if device_override else cfg["experiment"].get("device", "cpu"))

    # Training parameters
    epochs = epochs_override if epochs_override is not None else cfg["training"].get("epochs", 15)
    lr = lr_override if lr_override is not None else cfg["training"].get("learning_rate", 0.001)
    weight_decay = float(cfg["training"].get("weight_decay", 1e-5))

    # Directories
    checkpoint_dir = PROJECT_ROOT / cfg["training"].get("checkpoint_dir", "results/checkpoints")
    log_dir = PROJECT_ROOT / cfg["training"].get("log_dir", "results/logs")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / checkpoint_name
    csv_log_path = log_dir / f"{log_name}.csv"
    json_log_path = log_dir / f"{log_name}.json"

    # Data loaders
    train_loader, val_loader = get_dataloaders(config_path=cfg_path)

    # Model, Optimizer, Loss, Scheduler
    model = get_model(
        in_channels=cfg["model"].get("in_channels", 3),
        out_channels=cfg["model"].get("out_channels", 3)
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    eta_min = float(cfg["training"].get("min_learning_rate", 1e-5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    print("=" * 80)
    print(f" TRAINING SETUP: {cfg['experiment'].get('title')}")
    print(f" - Device: {device} | Total Parameters: {model.count_parameters()['trainable_parameters']:,}")
    print(f" - Epochs: {epochs} | Batch Size: {cfg['dataset'].get('batch_size', 32)} | Initial LR: {lr}")
    print(f" - Checkpoint Target: {checkpoint_path}")
    print("=" * 80)
    print(f"{'Epoch':<8} | {'Train MSE':<12} | {'Val MSE':<12} | {'Val PSNR (dB)':<14} | {'Val SSIM':<10} | {'LR':<10} | {'Time (s)':<8}")
    print("-" * 80)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
        "lr": [],
        "epoch_time": []
    }

    best_val_loss = float("inf")
    start_epoch = 1

    # Resume from an existing checkpoint if one is available
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if lr_override is not None:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_override

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if epochs > scheduler.T_max:
                scheduler.T_max = epochs

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))

        print(f"Resuming from epoch {start_epoch}")
        print(f"Previous best validation loss: {best_val_loss:.6f}")

        # Restore previous JSON history if available
        if json_log_path.exists():
            try:
                with open(json_log_path, "r", encoding="utf-8") as f:
                    prev_json = json.load(f)
                if "history" in prev_json and isinstance(prev_json["history"], dict):
                    prev_hist = prev_json["history"]
                    epochs_list = prev_hist.get("epoch", [])
                    valid_indices = [i for i, ep in enumerate(epochs_list) if ep < start_epoch]
                    for k in history.keys():
                        if k in prev_hist and isinstance(prev_hist[k], list):
                            history[k] = [prev_hist[k][i] for i in valid_indices if i < len(prev_hist[k])]
            except Exception as e:
                print(f"Warning: Could not restore previous JSON history: {e}")

        # Fallback: restore history from CSV if JSON history was empty
        if not history["epoch"] and csv_log_path.exists():
            try:
                with open(csv_log_path, "r", encoding="utf-8") as f:
                    csv_lines = f.readlines()
                for line in csv_lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 7 and parts[0].isdigit():
                        ep = int(parts[0])
                        if ep < start_epoch:
                            history["epoch"].append(ep)
                            history["train_loss"].append(float(parts[1]))
                            history["val_loss"].append(float(parts[2]))
                            history["val_psnr"].append(float(parts[3]))
                            history["val_ssim"].append(float(parts[4]))
                            history["lr"].append(float(parts[5]))
                            history["epoch_time"].append(float(parts[6]))
            except Exception as e:
                print(f"Warning: Could not restore history from CSV: {e}")

    # Prepare CSV log file (preserve previous records if resuming)
    csv_header = "epoch,train_loss,val_loss,val_psnr,val_ssim,lr,time_sec\n"
    if start_epoch > 1 and csv_log_path.exists():
        with open(csv_log_path, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()
        preserved_lines = [csv_header]
        for line in existing_lines[1:]:
            parts = line.strip().split(",")
            if parts and parts[0].isdigit():
                if int(parts[0]) < start_epoch:
                    preserved_lines.append(line if line.endswith("\n") else line + "\n")
        with open(csv_log_path, "w", encoding="utf-8") as f:
            f.writelines(preserved_lines)
    elif not csv_log_path.exists() or start_epoch == 1:
        with open(csv_log_path, "w", encoding="utf-8") as f:
            f.write(csv_header)

    start_total_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_psnr, val_ssim = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0

        # Update history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)
        history["lr"].append(current_lr)
        history["epoch_time"].append(elapsed)

        # Log to CSV
        with open(csv_log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_psnr:.4f},{val_ssim:.4f},{current_lr:.6e},{elapsed:.2f}\n")

        print(f"{epoch:<8d} | {train_loss:<12.6f} | {val_loss:<12.6f} | {val_psnr:<14.2f} | {val_ssim:<10.4f} | {current_lr:<10.2e} | {elapsed:<8.2f}")

        # Checkpoint saving on improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_checkpoint:
                checkpoint_payload = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                    "config": cfg
                }
                torch.save(checkpoint_payload, checkpoint_path)

    total_training_time = time.time() - start_total_time
    cumulative_time = sum(history["epoch_time"]) if history["epoch_time"] else total_training_time

    # Save JSON history
    summary_data = {
        "experiment": cfg["experiment"]["title"],
        "total_epochs": max(epochs, len(history["epoch"])),
        "best_val_loss": best_val_loss,
        "total_training_time_seconds": cumulative_time,
        "checkpoint_file": str(checkpoint_path),
        "history": history
    }
    with open(json_log_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("=" * 80)
    print(f"[Done] Training complete in {total_training_time:.2f} seconds.")
    print(f"       Best Validation Loss: {best_val_loss:.6f}")
    if save_checkpoint:
        print(f"       Saved Checkpoint:     {checkpoint_path} ({os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB)")
    print(f"       Metrics Logged to:    {csv_log_path}")
    print("=" * 80)

    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN Denoising Autoencoder on CPU.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (default: cpu).")
    args = parser.parse_args()

    train_model(
        config_path=args.config,
        epochs_override=args.epochs,
        lr_override=args.lr,
        device_override=args.device
    )
