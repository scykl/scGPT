# Intel XPU (Arc/Data Center GPU) Support for scGPT

This document describes the Intel XPU support implementation for scGPT, enabling training and inference on Intel Arc GPUs, Data Center GPUs, and other Intel accelerators.

## Quick Start

### 1. Install IPEX

```bash
# For Intel GPU support
pip install intel-extension-for-pytorch
```

### 2. Update Your Training Script

```python
from scgpt.utils import get_device, GradScalerAdapter, AutocastConfig

# Auto-select available device (XPU > CUDA > CPU)
device = get_device("auto")

# Create model and move to device
model = YourModel()
model.to(device)

# Use cross-device compatible GradScaler
scaler = GradScalerAdapter(device, enabled=True)

# Use cross-device compatible autocast
autocast_config = AutocastConfig(device, enabled=True)

# Training loop
for batch in dataloader:
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    with autocast_config.autocast_context():
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

### 3. Run Your Script

```bash
# Run with Intel GPU (XPU)
export SYCL_PI_LEVEL_ZERO_GPU_DEVICE_TIER=1
python your_training_script.py
```

## Implementation Details

### What Changed

#### 1. Device Management (`scgpt/utils/device_utils.py`)

New utilities for cross-device support:

- **`get_device(device_str)`**: Auto-detect or manually select device
  - `"auto"`: Prefer XPU > CUDA > CPU
  - `"xpu"`, `"cuda"`, `"cpu"`: Explicit selection
  - Falls back gracefully if device unavailable

- **`is_xpu_available()`**: Check if Intel IPEX/XPU is available

- **`get_device_backend(device)`**: Get backend name ("xpu", "cuda", "cpu")

- **`AutocastConfig`**: Device-agnostic mixed precision context
  - Supports CUDA AMP (float16 in device)
  - Supports XPU mixed precision (via IPEX)
  - CPU fallback (no precision change)

- **`GradScalerAdapter`**: Unified gradient scaling interface
  - Uses CUDA GradScaler for CUDA
  - Uses IPEX GradScaler for XPU
  - No scaling for CPU

- **`optimize_model_for_device(model, device)`**: Optimize model for target device
  - Applies IPEX optimizations for XPU
  - CUDA models use standard PyTorch

- **`optimize_optimizer_for_device(optimizer, device)`**: Optimize optimizer
  - Applies IPEX optimizer fusion for XPU

#### 2. Modified Files

**`scgpt/trainer.py`**:
- Replaced `torch.cuda.amp.autocast()` with `AutocastConfig(device, enabled=config.amp).autocast_context()`
- Works with device-agnostic scaler passed as parameter

**`scgpt/tasks/cell_emb.py`**:
- Updated `embed_data()` to use `get_device("auto")` instead of hardcoded CUDA
- Replaced `torch.cuda.amp.autocast` with `AutocastConfig`
- Default device changed from `"cuda"` to `"auto"` for better portability

**`examples/finetune_integration.py`**:
- Updated device initialization to use `get_device("auto")`
- Replaced `torch.cuda.amp.GradScaler` with `GradScalerAdapter`
- Replaced `torch.cuda.amp.autocast` with `AutocastConfig`

**`scgpt/utils/__init__.py`**:
- Exported device management utilities

#### 3. Key Design Decisions

1. **Backward Compatibility**: All changes are backward compatible. Existing code using `device` parameter still works.

2. **Graceful Fallback**: If XPU is not available, automatically falls back to CUDA or CPU.

3. **No Model Changes**: Model architecture remains unchanged. All optimizations are applied externally.

4. **Custom Operators**: Flash Attention automatically falls back to standard Transformer if not available for the device.

## Custom CUDA Operators

### Flash Attention Handling

Flash Attention is a custom CUDA operator. When running on XPU:

1. **Automatic Detection**: The code checks `flash_attn_available` before using Flash Attention
2. **Fallback**: If not available for XPU, standard PyTorch Transformer is used
3. **Performance**: IPEX provides optimized Flash Attention for XPU when available

```python
# In model.py, Flash Attention is already handled:
try:
    from flash_attn.flash_attention import FlashMHA
    flash_attn_available = True
except ImportError:
    flash_attn_available = False

# Usage in model:
if use_fast_transformer:
    if not flash_attn_available:
        warnings.warn("flash_attn not available, using standard transformer")
        use_fast_transformer = False
```

### Handling Other Custom Operators

If you have other custom CUDA operators:

```python
from scgpt.utils import get_device_backend

device_backend = get_device_backend(device)

if device_backend == "xpu":
    # Use XPU-optimized version or PyTorch fallback
    result = pytorch_fallback(input)
elif device_backend == "cuda":
    # Use CUDA-specific implementation
    result = cuda_optimized(input)
else:
    # Use CPU version
    result = cpu_version(input)
```

## Performance Optimization for Intel GPU

### Mixed Precision Training

Intel XPU supports mixed precision training (bfloat16 and float16):

```python
# Enable in config
config.amp = True

# Automatically handled by GradScalerAdapter and AutocastConfig
scaler = GradScalerAdapter(device, enabled=config.amp)
autocast_config = AutocastConfig(device, enabled=config.amp)
```

### Memory Optimization

```python
from scgpt.utils import empty_cache, synchronize_device

# Periodically clear memory
if step % 100 == 0:
    empty_cache(device)
    synchronize_device(device)
```

### Model Optimization

```python
from scgpt.utils import optimize_model_for_device, optimize_optimizer_for_device

# Optimize model for inference
model = optimize_model_for_device(model, device, optimize_for_inference=True)

# Optimize optimizer for training
optimizer = optimize_optimizer_for_device(optimizer, device)
```

## Testing and Validation

### Test Device Detection

```python
from scgpt.utils import get_device, is_xpu_available, is_cuda_available

# Check what's available
print(f"XPU available: {is_xpu_available()}")
print(f"CUDA available: {is_cuda_available()}")

# Get device
device = get_device("auto")
print(f"Using device: {device}")
```

### Test Mixed Precision

```python
from scgpt.utils import AutocastConfig

device = get_device("auto")
autocast_config = AutocastConfig(device, enabled=True)

# Should work on any device
with autocast_config.autocast_context():
    print(f"Autocast context active on {device}")
```

### Test Gradient Scaling

```python
from scgpt.utils import GradScalerAdapter

device = get_device("auto")
scaler = GradScalerAdapter(device, enabled=True)

loss = model(x)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
scaler.step(optimizer)
scaler.update()

print(f"Gradient scaling worked on {device}")
```

## Known Limitations

1. **Custom CUDA Kernels**: Third-party CUDA kernels may not work on XPU without SYCL implementation
2. **Older PyTorch**: Requires PyTorch >= 1.13 with IPEX for XPU support
3. **Windows Support**: XPU on Windows is limited; Linux recommended

## Environment Variables

### Intel GPU Setup

```bash
# Use Level Zero backend (recommended)
export SYCL_DEVICE_TYPE=GPU
export SYCL_PI_LEVEL_ZERO_ENABLE_IMMEDIATE_COMMANDLIST=1

# For Intel Arc GPUs specifically
export SYCL_PI_LEVEL_ZERO_GPU_DEVICE_TIER=1

# Disable immediate command list if experiencing issues
# export SYCL_PI_LEVEL_ZERO_ENABLE_IMMEDIATE_COMMANDLIST=0
```

### Debugging

```bash
# Enable SYCL debugging
export SYCL_PI_LEVEL_ZERO_ENABLE_PROFILING=1

# Verbose output
export SYCL_DEBUG_TRACE_PI=1
export SYCL_PI_LEVEL_ZERO_SYNC_UR_MEM_OBJ_WITH_NATIVE_MEM=1
```

## References

- **Intel Extension for PyTorch (IPEX)**: https://github.com/intel/intel-extension-for-pytorch
- **Intel Arc GPUs**: https://www.intel.com/content/www/us/en/products/discrete-gpus/arc.html
- **Level Zero Driver**: https://github.com/oneapi-src/level-zero
- **Flash Attention**: https://github.com/HazyResearch/flash-attention

## Contributing

If you encounter issues with Intel GPU support:

1. Check device availability: `python -c \"import torch; print(hasattr(torch, 'xpu'))\"`
2. Test IPEX installation: `python -c \"import intel_extension_for_pytorch as ipex; print(ipex.__version__)\"`
3. Run debug script to identify issues
4. Report detailed error messages with device info

## License

Same as scGPT (MIT License)
