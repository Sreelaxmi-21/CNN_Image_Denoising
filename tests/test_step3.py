"""
tests/test_step3.py
===================
Focused verification test suite for Step 3: CNN Autoencoder Architecture.

Verifies:
1. Model instantiation.
2. CPU execution.
3. Forward pass with tensor of shape (2, 3, 64, 64).
4. Output shape equals input shape.
5. Output range is within [0.0, 1.0].
6. No NaN or Inf values in output.
7. Skip-connection feature map dimensions are explicitly valid.
8. Trainable parameter count is finite and non-zero (measured programmatically).
9. Backpropagation works using MSE loss.
10. Gradients are successfully produced on all trainable parameters.
11. Model accepts alternative batch sizes (e.g., B=1, B=4) dynamically.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.autoencoder import CompactCNNAutoencoder, get_model


def test_step3_model_instantiation():
    """Test 1: Model instantiation & nn.Module check."""
    model = get_model(in_channels=3, out_channels=3)
    assert isinstance(model, nn.Module), "Model is not an instance of nn.Module"


def test_step3_device_cpu():
    """Test 2: CPU execution check."""
    model = get_model(in_channels=3, out_channels=3)
    device = next(model.parameters()).device
    assert device.type == "cpu", f"Expected CPU device, got {device}"


def test_step3_forward_pass_execution():
    """Test 3: Forward pass execution with batch input."""
    torch.manual_seed(42)
    model = get_model(in_channels=3, out_channels=3)
    batch_input = torch.rand(2, 3, 64, 64, dtype=torch.float32, device="cpu")
    model.eval()
    with torch.no_grad():
        output = model(batch_input)
    assert output is not None, "Model forward pass returned None"
    assert output.shape == (2, 3, 64, 64), f"Unexpected output shape: {output.shape}"


def test_step3_output_shape_matches_input():
    """Test 4: Output shape equals input shape."""
    torch.manual_seed(42)
    model = get_model(in_channels=3, out_channels=3)
    batch_input = torch.rand(2, 3, 64, 64, dtype=torch.float32, device="cpu")
    model.eval()
    with torch.no_grad():
        output = model(batch_input)
    assert output.shape == batch_input.shape, f"Shape mismatch: {output.shape} vs {batch_input.shape}"


def test_step3_output_range_bounded():
    """Test 5: Output range bounded within [0.0, 1.0]."""
    torch.manual_seed(42)
    model = get_model(in_channels=3, out_channels=3)
    batch_input = torch.rand(2, 3, 64, 64, dtype=torch.float32, device="cpu")
    model.eval()
    with torch.no_grad():
        output = model(batch_input)
    min_val = float(output.min())
    max_val = float(output.max())
    assert min_val >= 0.0, f"Output contains values < 0.0 (min={min_val})"
    assert max_val <= 1.0, f"Output contains values > 1.0 (max={max_val})"


def test_step3_output_no_nan_or_inf():
    """Test 6: No NaN or Inf values in output."""
    torch.manual_seed(42)
    model = get_model(in_channels=3, out_channels=3)
    batch_input = torch.rand(2, 3, 64, 64, dtype=torch.float32, device="cpu")
    model.eval()
    with torch.no_grad():
        output = model(batch_input)
    assert not torch.isnan(output).any(), "Output contains NaN values!"
    assert not torch.isinf(output).any(), "Output contains Inf values!"


def test_step3_skip_connection_dimensions():
    """Test 7: Explicit Skip-Connection Dimension Verification."""
    torch.manual_seed(42)
    model = get_model(in_channels=3, out_channels=3)
    batch_input = torch.rand(2, 3, 64, 64, dtype=torch.float32, device="cpu")
    model.eval()
    intermediates = model.forward_with_intermediates(batch_input)

    expected_shapes = {
        "input": (2, 3, 64, 64),
        "conv1": (2, 32, 64, 64),
        "conv2": (2, 64, 32, 32),
        "conv3": (2, 128, 16, 16),
        "bottleneck": (2, 128, 16, 16),
        "deconv1": (2, 64, 32, 32),
        "concat1": (2, 128, 32, 32),
        "fuse1": (2, 64, 32, 32),
        "deconv2": (2, 32, 64, 64),
        "concat2": (2, 64, 64, 64),
        "fuse2": (2, 32, 64, 64),
        "output": (2, 3, 64, 64),
    }

    for name, expected in expected_shapes.items():
        actual = tuple(intermediates[name].shape)
        assert actual == expected, f"Layer {name} shape mismatch: expected {expected}, got {actual}"

    # Verify Skip 2 concatenation compatibility (Deconv1 + Conv2)
    assert intermediates["deconv1"].shape[2:] == intermediates["conv2"].shape[2:], "Deconv1 and Conv2 spatial mismatch!"
    # Verify Skip 1 concatenation compatibility (Deconv2 + Conv1)
    assert intermediates["deconv2"].shape[2:] == intermediates["conv1"].shape[2:], "Deconv2 and Conv1 spatial mismatch!"


def test_step3_parameter_counts():
    """Test 8: Trainable parameter count programmatically measured."""
    model = get_model(in_channels=3, out_channels=3)
    param_info = model.count_parameters()
    total_p = param_info["total_parameters"]
    trainable_p = param_info["trainable_parameters"]
    assert trainable_p > 0, "Trainable parameters count is 0!"
    assert trainable_p == total_p, "Some parameters are not trainable!"
    assert total_p == 426435, f"Expected 426,435 parameters, got {total_p}"


def test_step3_mse_loss_backpropagation():
    """Test 9: Backpropagation with pure MSE reconstruction loss."""
    torch.manual_seed(42)
    model = get_model(in_channels=3, out_channels=3)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    train_input = torch.rand(2, 3, 64, 64, device="cpu")
    train_target = torch.rand(2, 3, 64, 64, device="cpu")

    train_out = model(train_input)
    criterion = nn.MSELoss()
    loss = criterion(train_out, train_target)
    loss.backward()

    loss_val = float(loss.item())
    assert not torch.isnan(loss) and not torch.isinf(loss), f"Loss is invalid: {loss_val}"
    assert loss_val > 0.0, f"MSE loss must be positive, got {loss_val}"


def test_step3_gradients_on_all_trainable_parameters():
    """Test 10: Non-zero gradient verification on all trainable parameter tensors."""
    torch.manual_seed(42)
    model = get_model(in_channels=3, out_channels=3)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    train_input = torch.rand(2, 3, 64, 64, device="cpu")
    train_target = torch.rand(2, 3, 64, 64, device="cpu")

    train_out = model(train_input)
    criterion = nn.MSELoss()
    loss = criterion(train_out, train_target)
    loss.backward()

    missing_grads = []
    zero_grads = []
    nan_grads = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                missing_grads.append(name)
            elif torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                nan_grads.append(name)
            elif param.grad.abs().sum().item() == 0.0:
                zero_grads.append(name)

    assert len(missing_grads) == 0, f"Missing gradients on: {missing_grads}"
    assert len(nan_grads) == 0, f"NaN/Inf gradients on: {nan_grads}"
    assert len(zero_grads) == 0, f"Zero gradients on: {zero_grads}"
    assert len(list(model.parameters())) == 21, f"Expected 21 parameter tensors, got {len(list(model.parameters()))}"


def test_step3_dynamic_batch_sizes():
    """Test 11: Multi-batch size compatibility (B=1 and B=4)."""
    torch.manual_seed(42)
    model = get_model(in_channels=3, out_channels=3)
    model.eval()
    with torch.no_grad():
        out_b1 = model(torch.rand(1, 3, 64, 64, device="cpu"))
        out_b4 = model(torch.rand(4, 3, 64, 64, device="cpu"))
    assert out_b1.shape == (1, 3, 64, 64), f"B=1 shape mismatch: {out_b1.shape}"
    assert out_b4.shape == (4, 3, 64, 64), f"B=4 shape mismatch: {out_b4.shape}"


def run_all_step3_tests():
    """Procedural test runner for direct script execution."""
    print("=== RUNNING STEP 3 VERIFICATION SUITE ===\n")

    print("[Test 1/11] Instantiating CompactCNNAutoencoder...")
    test_step3_model_instantiation()
    print(" -> PASSED: Model instantiated successfully.")

    print("[Test 2/11] Verifying execution device is CPU...")
    test_step3_device_cpu()
    print(" -> PASSED: Model resides strictly on cpu.")

    print("[Test 3/11] Executing forward pass with input shape (2, 3, 64, 64)...")
    test_step3_forward_pass_execution()
    print(" -> PASSED: Forward pass completed.")

    print("[Test 4/11] Checking output shape matches input shape...")
    test_step3_output_shape_matches_input()
    print(" -> PASSED: Output shape (2, 3, 64, 64) perfectly matches input shape.")

    print("[Test 5/11] Verifying output values are bounded within [0.0, 1.0]...")
    test_step3_output_range_bounded()
    print(" -> PASSED: Output values strictly bounded within [0.0, 1.0].")

    print("[Test 6/11] Checking for NaN or Inf values in output...")
    test_step3_output_no_nan_or_inf()
    print(" -> PASSED: Output tensor is numerically clean (no NaN/Inf).")

    print("[Test 7/11] Explicitly verifying skip-connection layer dimensions...")
    test_step3_skip_connection_dimensions()
    print(" -> PASSED: All skip-connection dimensions and projections are mathematically sound.")

    print("[Test 8/11] Programmatically measuring model parameter count...")
    test_step3_parameter_counts()
    print(" -> PASSED: Trainable parameter count is finite, non-zero, and programmatically verified (426,435).")

    print("[Test 9/11] Testing backpropagation with pure MSE reconstruction loss...")
    test_step3_mse_loss_backpropagation()
    print(" -> PASSED: MSE loss backpropagation operates without errors.")

    print("[Test 10/11] Verifying non-zero gradients on every trainable layer...")
    test_step3_gradients_on_all_trainable_parameters()
    print(" -> PASSED: Gradient backpropagation verified across all 21 parameter tensors.")

    print("[Test 11/11] Verifying dynamic batch-size compatibility (B=1 and B=4)...")
    test_step3_dynamic_batch_sizes()
    print(" -> PASSED: Dynamic batch handling verified.")

    print("\n=== ALL STEP 3 TESTS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_all_step3_tests()
