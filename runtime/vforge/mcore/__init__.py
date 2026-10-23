# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

import vforge.mcore.tensor_parallel
import vforge.mcore.utils
from vforge.mcore import parallel_state
from vforge.mcore.distributed import DistributedDataParallel
from vforge.mcore.inference_params import InferenceParams
from vforge.mcore.model_parallel_config import ModelParallelConfig
from vforge.mcore.package_info import (
    __contact_emails__,
    __contact_names__,
    __description__,
    __download_url__,
    __homepage__,
    __keywords__,
    __license__,
    __package_name__,
    __repository_url__,
    __shortversion__,
    __version__,
)
from vforge.mcore.timers import Timers
from vforge.mcore.utils import is_torch_min_version

# Alias parallel_state as mpu, its legacy name
mpu = parallel_state

__all__ = [
    "parallel_state",
    "tensor_parallel",
    "utils",
    "DistributedDataParallel",
    "InferenceParams",
    "ModelParallelConfig",
    "Timers",
    "__contact_emails__",
    "__contact_names__",
    "__description__",
    "__download_url__",
    "__homepage__",
    "__keywords__",
    "__license__",
    "__package_name__",
    "__repository_url__",
    "__shortversion__",
    "__version__",
]

from .safe_globals import register_safe_globals, safe_load_from_bytes

if is_torch_min_version("2.6a0"):
    register_safe_globals()

# Avoid direct usage of unsafe `torch.storage._load_from_bytes` (weights_only=False)
# Use safe implementation with weights_only=True
torch.storage._load_from_bytes = safe_load_from_bytes
