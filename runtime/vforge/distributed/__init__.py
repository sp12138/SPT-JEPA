"""Distributed runtime primitives."""

from vforge.distributed.expert_parallel import (
    Expert,
    ExpertBankSpec,
    ExpertLayout,
    ExpertParallelPlan,
)
from vforge.distributed.mcore_data_parallel import MCoreDataParallelConfig
from vforge.distributed.parallel_state import (
    ParallelConfig,
    ParallelState,
    destroy_parallel_state,
    get_parallel_state,
    init_parallel_state,
)
from vforge.distributed.parallelize import (
    FSDP2Config,
    FSDPModulePlan,
    MixedPrecisionConfig,
    ParallelizeConfig,
    apply_replicated_data_parallel,
    parallelize_model,
    unwrap_replicated_data_parallel,
)

# A model author's interface to tensor parallelism, re-exported here because it
# is a contract rather than an internal: these are written in model code, the way
# `parallel.tp_size` and `parallel.engine` are written in a YAML. The styles
# themselves are torch's, imported straight from `torch.distributed.tensor
# .parallel`, so there is no vforge spelling of them to learn.
from vforge.distributed.tensor_parallel import (
    ShardedRMSNorm,
    build_linear,
    mark_tp_sharded,
    tensor_parallel_enabled,
)

__all__ = [
    "Expert",
    "ExpertBankSpec",
    "ExpertLayout",
    "ExpertParallelPlan",
    "FSDP2Config",
    "FSDPModulePlan",
    "MixedPrecisionConfig",
    "MCoreDataParallelConfig",
    "ParallelConfig",
    "ParallelState",
    "ParallelizeConfig",
    "ShardedRMSNorm",
    "apply_replicated_data_parallel",
    "build_linear",
    "destroy_parallel_state",
    "get_parallel_state",
    "init_parallel_state",
    "mark_tp_sharded",
    "parallelize_model",
    "tensor_parallel_enabled",
    "unwrap_replicated_data_parallel",
]
