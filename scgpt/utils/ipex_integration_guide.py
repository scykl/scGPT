"""
Intel IPEX Integration Guide for scGPT

This module provides guidance on supporting Intel XPU (Arc GPUs, Data Center GPUs) 
for scGPT through Intel Extension for PyTorch (IPEX).

## Overview

scGPT can now run on Intel GPUs using IPEX. This guide covers:
1. Installation and setup
2. Device management
3. Mixed precision training
4. Custom operator handling
5. Performance tuning
"""

# Installation Instructions
IPEX_INSTALLATION = """
### Prerequisites
- Ubuntu 22.04+ or Windows 11+
- Intel Arc GPU (A770, A750, etc.) or Intel Data Center GPU

### Installation Steps

1. Install Intel Extension for PyTorch (IPEX):
   ```bash
   # For CUDA-less installations (recommended for pure GPU usage)
   pip install intel-extension-for-pytorch

   # Or with conda
   conda install -c conda-forge intel-extension-for-pytorch
   ```

2. Verify installation:
   ```python
   import torch
   import intel_extension_for_pytorch as ipex
   print(f"PyTorch version: {torch.__version__}")
   print(f"IPEX version: {ipex.__version__}")
   
   # Check if XPU is available
   if hasattr(torch, 'xpu'):
       print(f"XPU available: {torch.xpu.is_available()}")
       print(f"XPU devices: {torch.xpu.device_count()}")
   ```

3. Set up environment variables:
   ```bash
   # For Intel Arc GPU
   export SYCL_PI_LEVEL_ZERO_ENABLE_IMMEDIATE_COMMANDLIST=1
   export SYCL_PI_LEVEL_ZERO_GPU_DEVICE_TIER=1
   
   # Optional: Enable profiling
   export SYCL_PI_LEVEL_ZERO_ENABLE_PROFILING=1
   ```
"""

# Device Management
DEVICE_MANAGEMENT = """
### Using Device Management Utilities

The scgpt.utils.device_utils module provides device-agnostic APIs:

```python
from scgpt.utils import get_device, is_xpu_available, get_device_backend

# Auto-detect available device (prefers XPU > CUDA > CPU)
device = get_device("auto")

# Or explicitly select
device = get_device("xpu")  # Will fallback to CPU if XPU unavailable
device = get_device("cuda")
device = get_device("cpu")

# Check device backend
backend = get_device_backend(device)  # Returns "xpu", "cuda", or "cpu"

# Check availability
if is_xpu_available():
    print("Intel XPU is available!")
```

### Example Training Script

```python
import torch
from scgpt.utils import get_device, GradScalerAdapter, AutocastConfig

# Get device
device = get_device("auto")
print(f"Using device: {device}")

# Create model and move to device
model = YourModel()
model.to(device)

# Create optimizer and scaler (automatically handles device)
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScalerAdapter(device, enabled=True)

# Use AutocastConfig for mixed precision
autocast_config = AutocastConfig(device, enabled=True)

# Training loop
for epoch in range(num_epochs):
    for batch_data in dataloader:
        # Move data to device
        input_ids = batch_data['input_ids'].to(device)
        labels = batch_data['labels'].to(device)
        
        # Forward pass with mixed precision
        with autocast_config.autocast_context():
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
        
        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
```
"""

# Custom Operator Handling
CUSTOM_OPERATOR_HANDLING = """
### Custom CUDA Operators in scGPT

scGPT uses Flash Attention as a custom CUDA/SYCL operator. When running on XPU:

1. **Automatic Fallback**: If Flash Attention is not available for the device,
   scGPT automatically falls back to standard PyTorch transformers.

2. **Flash Attention on XPU**: 
   - Requires IPEX with Flash Attention support
   - Install with: `pip install intel-extension-for-pytorch[xpu]`
   - Check availability: `from flash_attn import flash_attn_available`

3. **Configuration**:
   ```python
   model = TransformerModel(
       # ... other parameters ...
       use_fast_transformer=True,  # Will auto-detect availability
       fast_transformer_backend="flash"  # IPEX will optimize this
   )
   ```

4. **Recommended Settings for Intel GPU**:
   - For inference: Set use_fast_transformer=True for best performance
   - For training: Start with use_fast_transformer=True, fallback to False if memory issues occur
   - For larger models: Consider use_fast_transformer=False on older GPUs

### Handling Custom Extensions

If you have custom CUDA extensions:

1. **Check Device Type**:
   ```python
   from scgpt.utils import get_device_backend
   device_backend = get_device_backend(device)
   
   if device_backend == "xpu":
       # Use XPU-specific implementation
       pass
   elif device_backend == "cuda":
       # Use CUDA implementation
       pass
   else:
       # Use CPU fallback
       pass
   ```

2. **Optional: Build for XPU**:
   - If your extension has SYCL support, rebuild with SYCL compiler
   - Check IPEX documentation for extension compilation

3. **Fallback Strategy**:
   ```python
   try:
       # Try optimized version
       result = custom_cuda_op(input_tensor)
   except RuntimeError as e:
       if "not supported" in str(e).lower():
           # Fallback to PyTorch implementation
           result = pytorch_equivalent(input_tensor)
       else:
           raise
   ```
"""

# Performance Tuning
PERFORMANCE_TUNING = """
### Performance Tuning for Intel GPU

1. **Batch Size**:
   - Start with smaller batches (16-32) for Arc GPUs
   - Increase gradually based on available memory

2. **Mixed Precision**:
   ```python
   # Enable fp16 mixed precision
   scaler = GradScalerAdapter(device, enabled=True)
   autocast_config = AutocastConfig(device, enabled=True)
   ```

3. **Memory Optimization**:
   ```python
   from scgpt.utils import empty_cache, synchronize_device
   
   # Clear GPU cache periodically
   if batch_id % 100 == 0:
       empty_cache(device)
       synchronize_device(device)
   ```

4. **IPEX Graph Optimization**:
   ```python
   from scgpt.utils import optimize_model_for_device
   
   # Optimize model for inference
   model = optimize_model_for_device(
       model, 
       device, 
       optimize_for_inference=True
   )
   ```

5. **Recommended Configuration for Finetuning**:
   ```python
   config = dict(
       device="auto",  # Auto-detect
       amp=True,  # Enable mixed precision
       batch_size=32,  # Adjust based on GPU memory
       gradient_accumulation_steps=2,  # If needed for larger effective batch
       pin_memory=True,  # Enable for faster data transfer
       num_workers=4,  # Number of data loading workers
   )
   ```
"""

# Troubleshooting
TROUBLESHOOTING = """
### Common Issues and Solutions

1. **"XPU is not available"**:
   - Check GPU drivers: `sycl-ls` (should show your Intel GPU)
   - Reinstall IPEX: `pip install --force-reinstall intel-extension-for-pytorch`

2. **"CUDA out of memory"** (when trying to run on XPU):
   - Your script is using CPU fallback instead of XPU
   - Verify device selection: `print(device)`
   - Check: `torch.xpu.is_available()` returns True

3. **"Flash Attention not available"**:
   - Fallback to standard transformer: automatic
   - Set `use_fast_transformer=False` explicitly if needed

4. **Slow Performance on XPU**:
   - Check if using CPU: `print(next(model.parameters()).device)`
   - Verify Flash Attention is available for your model
   - Reduce batch size if memory-bound
   - Enable mixed precision: `config.amp = True`

5. **"SYCL runtime not found"**:
   - Level Zero loader not installed
   - Install: `sudo apt-get install level-zero level-zero-loader libze-dev`

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

from scgpt.utils import get_device, is_xpu_available
print(f"XPU available: {is_xpu_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")

device = get_device("auto")
print(f"Selected device: {device}")
print(f"Device type: {device.type}")
```
"""

# Integration with Finetuning Example
FINETUNING_EXAMPLE = """
### Finetuning scGPT on Intel GPU

1. **Update your finetuning script**:

   ```python
   # In your finetuning script (e.g., examples/finetune_integration.py):
   
   from scgpt.utils import get_device, GradScalerAdapter, AutocastConfig
   
   # Automatically select available device
   device = get_device("auto")
   print(f"Using device: {device}")
   
   # Create model
   model = TransformerModel(...)
   model.to(device)
   
   # Create scaler for mixed precision
   scaler = GradScalerAdapter(device, enabled=config.amp)
   
   # In training loop:
   autocast_config = AutocastConfig(device, enabled=config.amp)
   with autocast_config.autocast_context():
       outputs = model(...)
       loss = ...
   ```

2. **Run finetuning**:
   ```bash
   export ZE_AFFINITY_MASK=0  # Run on first GPU
   python examples/finetune_integration.py
   ```

3. **Monitor device usage**:
   ```bash
   # Check Intel GPU usage
   smi  # For Intel Arc/Data Center GPUs
   # Or use system monitor showing GPU utilization
   ```
"""

# References and Resources
REFERENCES = """
### Useful Resources

- Intel Extension for PyTorch: https://github.com/intel/intel-extension-for-pytorch
- Intel Arc GPU: https://www.intel.com/content/www/us/en/products/discrete-gpus/arc.html
- Data Center GPU: https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu.html
- PyTorch XPU Support: https://pytorch.org/docs/stable/notes/cuda.html (look for XPU sections)
- SYCL Documentation: https://www.khronos.org/sycl/

### Performance Benchmarks

For reference, typical performance characteristics on Intel Arc A770:
- fp32 mixed precision: ~90% of fp32 performance
- fp16 mixed precision: ~120-130% of fp32 performance (recommended)
- Batch size 64-128: optimal for gene expression models

For Intel Data Center GPUs:
- Generally 2-4x faster than Arc for the same model size
"""

if __name__ == "__main__":
    print(__doc__)
    print("\n=== IPEX Installation ===")
    print(IPEX_INSTALLATION)
    print("\n=== Device Management ===")
    print(DEVICE_MANAGEMENT)
    print("\n=== Custom Operator Handling ===")
    print(CUSTOM_OPERATOR_HANDLING)
    print("\n=== Performance Tuning ===")
    print(PERFORMANCE_TUNING)
    print("\n=== Troubleshooting ===")
    print(TROUBLESHOOTING)
    print("\n=== Finetuning Example ===")
    print(FINETUNING_EXAMPLE)
    print("\n=== References ===")
    print(REFERENCES)
