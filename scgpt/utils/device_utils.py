"""
Device management utilities for supporting multiple hardware backends.
Supports CUDA, CPU, and Intel XPU (with IPEX).
"""

import torch
import warnings
from typing import Optional, Union, Tuple
from contextlib import contextmanager


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Get a torch device object with support for multiple backends.
    
    Args:
        device_str: Device specification. Options:
            - None or "auto": Auto-detect (XPU > CUDA > CPU)
            - "cuda": Use CUDA (falls back to CPU if unavailable)
            - "cpu": Use CPU
            - "xpu": Use Intel XPU (falls back to CPU if unavailable)
            - "0", "1", etc.: Specific device index (CUDA default)
    
    Returns:
        torch.device: The requested device
    
    Raises:
        ValueError: If invalid device string is provided
    """
    if device_str is None or device_str == "auto":
        # Auto-detect: prefer XPU > CUDA > CPU
        if is_xpu_available():
            return torch.device("xpu")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    device_lower = str(device_str).lower()
    
    if device_lower == "xpu":
        if is_xpu_available():
            return torch.device("xpu")
        else:
            warnings.warn(
                "XPU not available, falling back to CPU. "
                "Please ensure Intel IPEX is installed."
            )
            return torch.device("cpu")
    
    elif device_lower == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            warnings.warn(
                "CUDA not available, falling back to CPU. "
                "Please ensure PyTorch with CUDA support is installed."
            )
            return torch.device("cpu")
    
    elif device_lower == "cpu":
        return torch.device("cpu")
    
    elif device_lower.isdigit():
        # Device index like "0", "1"
        device_id = int(device_lower)
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device_id}")
        else:
            warnings.warn(
                f"CUDA not available, falling back to CPU. "
                f"Requested device index {device_id} is ignored."
            )
            return torch.device("cpu")
    
    else:
        # Try to parse as torch.device directly
        try:
            return torch.device(device_str)
        except RuntimeError as e:
            raise ValueError(f"Invalid device string: {device_str}") from e


def is_xpu_available() -> bool:
    """
    Check if Intel XPU backend is available.

    Returns:
        bool: True if XPU is available (via IPEX or PyTorch built-in XPU support)
    """
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available"):
        return torch.xpu.is_available()

    try:
        import intel_extension_for_pytorch as ipex
        return hasattr(ipex, "__version__")
    except ImportError:
        return False


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def get_device_backend(device: torch.device) -> str:
    """
    Get the backend name for a device.
    
    Args:
        device: torch.device object
    
    Returns:
        str: Backend name ("xpu", "cuda", or "cpu")
    """
    device_type = str(device.type).lower()
    if device_type in ["xpu", "cuda", "cpu"]:
        return device_type
    return "cpu"


class AutocastConfig:
    """Configuration for automatic mixed precision (AMP) autocast."""
    
    def __init__(self, device: torch.device, enabled: bool = False):
        """
        Initialize autocast configuration.
        
        Args:
            device: torch.device to use
            enabled: Whether to enable mixed precision
        """
        self.device = device
        self.enabled = enabled
        self.device_backend = get_device_backend(device)
    
    @contextmanager
    def autocast_context(self):
        """
        Context manager for autocast based on device type.

        Usage:
            config = AutocastConfig(device, enabled=True)
            with config.autocast_context():
                # operations here
        """
        if not self.enabled:
            yield
            return

        if self.device_backend == "cuda":
            with torch.cuda.amp.autocast():
                yield
        elif self.device_backend == "xpu":
            if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
                with torch.amp.autocast("xpu", enabled=True):
                    yield
            else:
                try:
                    import intel_extension_for_pytorch as ipex
                    with torch.autocast(device_type="xpu", dtype=torch.float16):
                        yield
                except ImportError:
                    warnings.warn(
                        "XPU autocast not available, running without mixed precision"
                    )
                    yield
        else:
            yield


class GradScalerAdapter:
    """
    Adapter for GradScaler that works with multiple backends.
    
    Provides a unified interface for gradient scaling across CUDA, XPU, and CPU.
    """
    
    def __init__(self, device: torch.device, enabled: bool = False):
        """
        Initialize gradient scaler adapter.
        
        Args:
            device: torch.device to use
            enabled: Whether to enable gradient scaling
        """
        self.device = device
        self.enabled = enabled
        self.device_backend = get_device_backend(device)
        self._init_scaler()
    
    def _init_scaler(self):
        """Initialize the appropriate scaler for the backend."""
        if not self.enabled:
            self.scaler = None
            return

        if self.device_backend == "cuda":
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        elif self.device_backend == "xpu":
            try:
                import intel_extension_for_pytorch as ipex
                if hasattr(ipex, "GradScaler"):
                    self.scaler = ipex.GradScaler(enabled=True)
                else:
                    self.scaler = torch.amp.GradScaler("xpu", enabled=True)
            except ImportError:
                if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
                    self.scaler = torch.amp.GradScaler("xpu", enabled=True)
                else:
                    warnings.warn(
                        "XPU GradScaler not available, gradient scaling disabled"
                    )
                    self.scaler = None
        else:
            self.scaler = None
    
    def scale(self, loss):
        """Scale loss for backward pass."""
        if self.scaler is None:
            return loss
        return self.scaler.scale(loss)
    
    def unscale_(self, optimizer):
        """Unscale gradients."""
        if self.scaler is None:
            return
        if self.device_backend == "xpu":
            return
        self.scaler.unscale_(optimizer)
    
    def step(self, optimizer):
        """Optimizer step with scaled loss."""
        if self.scaler is None:
            optimizer.step()
        elif self.device_backend == "xpu":
            optimizer.step()
        else:
            self.scaler.step(optimizer)
    
    def update(self):
        """Update the scale for next iteration."""
        if self.scaler is None:
            return
        if self.device_backend == "xpu":
            return
        self.scaler.update()
    
    def get_scale(self):
        """Get current scale."""
        if self.scaler is None:
            return 1.0
        return self.scaler.get_scale()
    
    def is_enabled(self):
        """Check if scaler is enabled."""
        return self.enabled and self.scaler is not None


def optimize_model_for_device(
    model: torch.nn.Module, 
    device: torch.device,
    optimize_for_inference: bool = False
) -> torch.nn.Module:
    """
    Optimize model for specific device backend.
    
    Args:
        model: PyTorch model to optimize
        device: Target device
        optimize_for_inference: Whether to optimize for inference instead of training
    
    Returns:
        Optimized model
    """
    device_backend = get_device_backend(device)
    
    if device_backend == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
            # Optimize model for XPU
            if optimize_for_inference:
                model = ipex.optimize(model, dtype=torch.float32)
            else:
                # For training, use mixed precision
                model = ipex.optimize(
                    model,
                    dtype=torch.float32,
                    level="O1"  # Mixed precision
                )
        except ImportError:
            warnings.warn(
                "IPEX not available, model optimization skipped"
            )
    
    return model


def optimize_optimizer_for_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> torch.optim.Optimizer:
    """
    Optimize optimizer for specific device backend.
    
    Args:
        optimizer: PyTorch optimizer
        device: Target device
    
    Returns:
        Optimized optimizer (may be the same object or a new one)
    """
    device_backend = get_device_backend(device)
    
    if device_backend == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
            # IPEX provides optimizer fusion
            if hasattr(ipex, "optimize_optimizer"):
                optimizer = ipex.optimize_optimizer(
                    optimizer,
                    dtype=torch.float32
                )
        except ImportError:
            pass
    
    return optimizer


def synchronize_device(device: torch.device):
    """
    Synchronize device to ensure all operations are complete.
    
    Args:
        device: torch.device to synchronize
    """
    device_backend = get_device_backend(device)
    
    if device_backend == "cuda":
        torch.cuda.synchronize()
    elif device_backend == "xpu":
        try:
            torch.xpu.synchronize()
        except Exception:
            pass  # Fallback if synchronize is not available


def empty_cache(device: torch.device):
    """
    Empty device cache to free memory.
    
    Args:
        device: torch.device to clear cache for
    """
    device_backend = get_device_backend(device)
    
    if device_backend == "cuda":
        torch.cuda.empty_cache()
    elif device_backend == "xpu":
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass  # Fallback if empty_cache is not available


class DeviceContext:
    """
    Context manager for device-specific settings.
    
    Usage:
        with DeviceContext(device) as ctx:
            # Device-specific code here
            pass
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.device_backend = get_device_backend(device)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        synchronize_device(self.device)
        return False
