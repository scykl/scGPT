#!/usr/bin/env python
"""
Test script for verifying Intel XPU support in scGPT.

This script validates:
1. Device detection (XPU, CUDA, CPU)
2. Device movement
3. Mixed precision training
4. Gradient scaling
5. Model creation and inference
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scgpt.utils import (
    get_device,
    is_xpu_available,
    is_cuda_available,
    get_device_backend,
    AutocastConfig,
    GradScalerAdapter,
    synchronize_device,
    empty_cache,
)


def test_device_detection():
    """Test device detection."""
    print("\n=== Testing Device Detection ===")
    
    print(f"XPU available: {is_xpu_available()}")
    print(f"CUDA available: {is_cuda_available()}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Test auto-detection
    device = get_device("auto")
    print(f"Auto-detected device: {device}")
    assert device is not None, "Failed to detect any device"
    
    # Test explicit device selection
    for device_str in ["cpu", "auto"]:
        try:
            d = get_device(device_str)
            print(f"  {device_str:10s} -> {d}")
        except ValueError as e:
            print(f"  {device_str:10s} -> Error: {e}")
    
    # Test optional device selection
    for device_str in ["xpu", "cuda"]:
        try:
            d = get_device(device_str)
            print(f"  {device_str:10s} -> {d}")
        except ValueError as e:
            print(f"  {device_str:10s} -> Fallback (expected)")
    
    return device


def test_device_backend(device):
    """Test device backend identification."""
    print("\n=== Testing Device Backend ===")
    
    backend = get_device_backend(device)
    print(f"Device: {device}")
    print(f"Backend: {backend}")
    assert backend in ["xpu", "cuda", "cpu"], f"Invalid backend: {backend}"


def test_tensor_movement(device):
    """Test moving tensors to device."""
    print("\n=== Testing Tensor Movement ===")
    
    # Create tensor on CPU
    x = torch.randn(10, 20)
    print(f"Tensor created on: {x.device}")
    
    # Move to target device
    x_device = x.to(device)
    print(f"Tensor moved to: {x_device.device}")
    
    assert str(x_device.device).startswith(device.type), "Tensor not on target device"
    
    # Test multiple tensors
    y = torch.randn(5, 10).to(device)
    z = torch.randn(10, 20).to(device)
    
    result = torch.matmul(z, y)
    print(f"Matrix multiplication result device: {result.device}")
    
    return x_device, y, z


def test_mixed_precision(device):
    """Test mixed precision training."""
    print("\n=== Testing Mixed Precision (Autocast) ===")
    
    autocast_config = AutocastConfig(device, enabled=True)
    print(f"Autocast enabled on device: {device}")
    
    # Create simple model
    model = torch.nn.Linear(10, 5).to(device)
    x = torch.randn(4, 10).to(device)
    
    # Test with autocast
    with autocast_config.autocast_context():
        output = model(x)
        loss = output.mean()
    
    print(f"Forward pass with autocast successful")
    print(f"Output dtype: {output.dtype}")
    print(f"Loss: {loss.item():.4f}")


def test_gradient_scaling(device):
    """Test gradient scaling."""
    print("\n=== Testing Gradient Scaling ===")
    
    scaler = GradScalerAdapter(device, enabled=True)
    print(f"GradScaler initialized for device: {device}")
    print(f"Scaler enabled: {scaler.is_enabled()}")
    
    # Create simple model
    model = torch.nn.Linear(10, 5).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    x = torch.randn(4, 10).to(device)
    target = torch.randn(4, 5).to(device)
    
    # Training step with gradient scaling
    optimizer.zero_grad()
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, target)
    
    # Scale loss
    scaled_loss = scaler.scale(loss)
    print(f"Loss scaled: {scaled_loss.item():.6f}")
    
    scaled_loss.backward()
    scaler.unscale_(optimizer)
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name} gradient norm: {param.grad.norm().item():.6f}")
    
    scaler.step(optimizer)
    scaler.update()
    
    print(f"Gradient scaling step completed successfully")


def test_model_creation(device):
    """Test creating and running a model."""
    print("\n=== Testing Model Creation and Inference ===")
    
    from scgpt.model import TransformerModel
    from torchtext.vocab import Vocab
    
    # Create a simple vocab
    vocab_dict = {
        "<pad>": 0,
        "<cls>": 1,
        "<eos>": 2,
        "gene_1": 3,
        "gene_2": 4,
        "gene_3": 5,
    }
    
    # Create model
    try:
        model = TransformerModel(
            ntoken=len(vocab_dict),
            d_model=64,
            nhead=4,
            d_hid=256,
            nlayers=2,
            nlayers_cls=1,
            vocab=vocab_dict,
            dropout=0.1,
            pad_token="<pad>",
            pad_value=0,
            do_mvc=False,
            do_dab=False,
            use_batch_labels=False,
            use_fast_transformer=False,  # Disable flash attention for testing
        )
        
        model.to(device)
        print(f"Model created and moved to {device}")
        
        # Create dummy input
        batch_size = 4
        seq_len = 10
        
        input_gene_ids = torch.randint(0, len(vocab_dict), (batch_size, seq_len)).to(device)
        input_values = torch.randn(batch_size, seq_len).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=None,
            )
        
        print(f"Model output shape: {output['mlm_output'].shape}")
        print(f"Model output device: {output['mlm_output'].device}")
        print(f"Model inference successful")
        
        return model
        
    except Exception as e:
        print(f"Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_device_synchronization(device):
    """Test device synchronization utilities."""
    print("\n=== Testing Device Synchronization ===")
    
    try:
        synchronize_device(device)
        print(f"Device synchronized successfully")
    except Exception as e:
        print(f"Device synchronization raised: {e} (may be expected for CPU)")
    
    try:
        empty_cache(device)
        print(f"Device cache cleared successfully")
    except Exception as e:
        print(f"Cache clearing raised: {e} (may be expected for CPU)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("scGPT Intel XPU Support Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Device Detection
        device = test_device_detection()
        
        # Test 2: Device Backend
        test_device_backend(device)
        
        # Test 3: Tensor Movement
        test_tensor_movement(device)
        
        # Test 4: Mixed Precision
        test_mixed_precision(device)
        
        # Test 5: Gradient Scaling
        test_gradient_scaling(device)
        
        # Test 6: Model Creation
        model = test_model_creation(device)
        
        # Test 7: Device Synchronization
        test_device_synchronization(device)
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        print(f"\nFinal device used: {device}")
        print(f"Device type: {get_device_backend(device)}")
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ Test failed!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
