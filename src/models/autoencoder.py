"""
src/models/autoencoder.py
=========================
Compact Deep Convolutional Autoencoder with Skip Connections for Image Denoising.

Architecture Diagram & Tensor Shape Progression:
------------------------------------------------
Input:                     (B,   3, 64, 64)  [RGB, normalized 0.0 - 1.0]
  │
  ├─── Conv1 Block ──────> (B,  32, 64, 64)  ───── Skip Connection 1 ─────────┐
  │                                                                            │
  ├─── Conv2 Block ──────> (B,  64, 32, 32)  ───── Skip Connection 2 ──┐       │
  │                                                                    │       │
  ├─── Conv3 Block ──────> (B, 128, 16, 16)                            │       │
  │                                                                    │       │
  ├─── Latent Bottleneck ─> (B, 128, 16, 16)                           │       │
  │                                                                    │       │
  ├─── Deconv1 ──────────> (B,  64, 32, 32)                            │       │
  │          │                                                         │       │
  │          └── Concat with Conv2 Skip ─> (B, 128, 32, 32) <──────────┘       │
  │          └── Fuse1 Projection Layer ──> (B,  64, 32, 32)                   │
  │                                                                            │
  ├─── Deconv2 ──────────> (B,  32, 64, 64)                                    │
  │          │                                                                 │
  │          └── Concat with Conv1 Skip ─> (B,  64, 64, 64) <──────────────────┘
  │          └── Fuse2 Projection Layer ──> (B,  32, 64, 64)
  │
  └─── Output Layer ─────> (B,   3, 64, 64)  [Sigmoid activation, [0.0, 1.0]]

Skip Connection Rationale & Dimensional Adjustment:
---------------------------------------------------
1. Concatenation of Deconv1 (64 ch) and Conv2 (64 ch) yields 128 channels.
   Since Deconv2 requires 64 input channels, a projection block `fuse1`
   (Conv2d 128 -> 64 + BatchNorm + ReLU) is implemented to adapt channels
   and learn optimal feature blending between latent and spatial features.
2. Concatenation of Deconv2 (32 ch) and Conv1 (32 ch) yields 64 channels.
   Since the output conv requires 32 input channels, a projection block `fuse2`
   (Conv2d 64 -> 32 + BatchNorm + ReLU) is implemented to adapt channels.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn


class CompactCNNAutoencoder(nn.Module):
    """
    Compact Convolutional Autoencoder with Symmetrical Skip Connections.

    Designed specifically for B.Tech Digital Image Processing research:
    - Lightweight parameter footprint (< 450k parameters).
    - Runs efficiently on CPU.
    - Symmetrical downsampling and upsampling with skip connections to preserve
      fine structural edges lost during bottleneck compression.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # ==========================================
        # 1. ENCODER
        # ==========================================
        # Conv1: 3 -> 32 (spatial size: 64x64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Conv2: 32 -> 64 (spatial downsampling: 64x64 -> 32x32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Conv3: 64 -> 128 (spatial downsampling: 32x32 -> 16x16)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # ==========================================
        # 2. LATENT BOTTLENECK
        # ==========================================
        # Conv: 128 -> 128 (spatial size: 16x16)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # ==========================================
        # 3. DECODER & SKIP CONNECTIONS
        # ==========================================
        # Deconv1: Transposed Conv 128 -> 64 (spatial upsampling: 16x16 -> 32x32)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )

        # Fuse1 Projection Layer:
        # Concatenates Deconv1 (64) + Conv2 Skip (64) = 128 channels.
        # Projects 128 -> 64 channels so Deconv2 receives valid 64-channel input.
        self.fuse1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Deconv2: Transposed Conv 64 -> 32 (spatial upsampling: 32x32 -> 64x64)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )

        # Fuse2 Projection Layer:
        # Concatenates Deconv2 (32) + Conv1 Skip (32) = 64 channels.
        # Projects 64 -> 32 channels so Output Conv receives valid 32-channel input.
        self.fuse2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # ==========================================
        # 4. OUTPUT RECONSTRUCTION LAYER
        # ==========================================
        # Conv: 32 -> 3 (spatial size: 64x64) + Sigmoid activation for [0.0, 1.0]
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

        # Deterministic weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Applies Kaiming Normal initialization for robust gradient propagation."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward denoising pass.

        Args:
            x: Noisy input tensor of shape (B, 3, H, W) normalized to [0.0, 1.0].

        Returns:
            torch.Tensor: Denoised reconstruction of shape (B, 3, H, W) in [0.0, 1.0].
        """
        # Encoder forward pass with skip feature retention
        e1 = self.conv1(x)    # Shape: (B, 32, H,   W)
        e2 = self.conv2(e1)   # Shape: (B, 64, H/2, W/2)
        e3 = self.conv3(e2)   # Shape: (B, 128, H/4, W/4)

        # Bottleneck latent representation
        b = self.bottleneck(e3) # Shape: (B, 128, H/4, W/4)

        # Decoder Stage 1
        d1 = self.deconv1(b)  # Shape: (B, 64, H/2, W/2)
        # Skip connection 2: concatenate d1 with e2
        cat1 = torch.cat([d1, e2], dim=1) # Shape: (B, 128, H/2, W/2)
        f1 = self.fuse1(cat1)             # Shape: (B,  64, H/2, W/2)

        # Decoder Stage 2
        d2 = self.deconv2(f1) # Shape: (B, 32, H, W)
        # Skip connection 1: concatenate d2 with e1
        cat2 = torch.cat([d2, e1], dim=1) # Shape: (B, 64, H, W)
        f2 = self.fuse2(cat2)             # Shape: (B, 32, H, W)

        # Output projection
        out = self.out_conv(f2) # Shape: (B, 3, H, W)
        return out

    def forward_with_intermediates(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns intermediate feature maps for shape inspection and verification.
        """
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        b = self.bottleneck(e3)
        d1 = self.deconv1(b)
        cat1 = torch.cat([d1, e2], dim=1)
        f1 = self.fuse1(cat1)
        d2 = self.deconv2(f1)
        cat2 = torch.cat([d2, e1], dim=1)
        f2 = self.fuse2(cat2)
        out = self.out_conv(f2)

        return {
            "input": x,
            "conv1": e1,
            "conv2": e2,
            "conv3": e3,
            "bottleneck": b,
            "deconv1": d1,
            "concat1": cat1,
            "fuse1": f1,
            "deconv2": d2,
            "concat2": cat2,
            "fuse2": f2,
            "output": out,
        }

    def count_parameters(self) -> Dict[str, int]:
        """Computes total and trainable parameter counts programmatically."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "non_trainable_parameters": non_trainable
        }


def get_model(in_channels: int = 3, out_channels: int = 3) -> CompactCNNAutoencoder:
    """Factory function instantiating the autoencoder on CPU."""
    model = CompactCNNAutoencoder(in_channels=in_channels, out_channels=out_channels)
    model.to("cpu")
    return model
