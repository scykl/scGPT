from .util import *
from .device_utils import (
    get_device,
    is_xpu_available,
    is_cuda_available,
    get_device_backend,
    AutocastConfig,
    GradScalerAdapter,
    optimize_model_for_device,
    optimize_optimizer_for_device,
    synchronize_device,
    empty_cache,
    DeviceContext,
)
