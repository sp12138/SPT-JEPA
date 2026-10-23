"""MCore replicated data parallelism over vForge-owned process groups.

The adapter keeps MCore's global parallel state unused.  vForge constructs the
topology, injects the required groups, and owns the gradient lifecycle around
the otherwise upstream DDP implementation.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from vforge.distributed.parallel_state import get_parallel_state
from vforge.utils.gradients import set_main_gradient_active

_DATA_PARALLEL_BACKEND = "_vforge_data_parallel_backend"
_GRADIENT_SEEN = "_vforge_gradient_seen"
_GRADIENT_HOOK = "_vforge_gradient_seen_hook"
_PROCESS_GROUP_COLLECTION = "_vforge_mcore_process_groups"
_MIN_BUCKET_SIZE = 40_000_000


@dataclass(frozen=True)
class MCoreDataParallelConfig:
    """Communication policy for MCore DDP and DistributedOptimizer."""

    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = True


def apply_mcore_data_parallel(
    model: nn.Module,
    *,
    config: MCoreDataParallelConfig | None = None,
) -> nn.Module:
    """Wrap a materialized model in overlapping MCore DDP over DP x CP."""
    policy = config or MCoreDataParallelConfig()
    state = get_parallel_state()
    if state.config.dp_mode != "ddp":
        return model
    if any(tensor.is_meta for tensor in (*model.parameters(), *model.buffers())):
        raise RuntimeError("MCore DDP requires a materialized model")
    if not dist.is_initialized() or state.dp_group is None or state.dp_cp_group is None:
        raise RuntimeError(
            "MCore distributed optimizer requires torch.distributed initialization "
            "and explicit DP and DP x CP process groups, including at size one"
        )

    from vforge.mcore.distributed import DistributedDataParallelConfig
    from vforge.mcore.distributed.distributed_data_parallel import DistributedDataParallel
    from vforge.mcore.optimizer.distrib_optimizer import DistributedOptimizer
    from vforge.mcore.process_groups_config import ProcessGroupCollection
    from vforge.mcore.transformer.transformer_config import TransformerConfig

    parameter = next(model.parameters(), None)
    parameter_dtype = parameter.dtype if parameter is not None else torch.float32
    topology = state.config
    config = TransformerConfig(
        num_layers=1,
        hidden_size=max(1, topology.tp_size),
        num_attention_heads=max(1, topology.tp_size),
        tensor_model_parallel_size=topology.tp_size,
        pipeline_model_parallel_size=topology.pp_size,
        context_parallel_size=topology.cp_size,
        params_dtype=parameter_dtype,
        bf16=parameter_dtype == torch.bfloat16,
        fp16=parameter_dtype == torch.float16,
    )
    # MCore's default grows the bucket floor for very large DP groups. Resolve
    # it before computing the DistributedOptimizer layout so DDP and optimizer
    # agree on identical bucket boundaries.
    overlap_enabled = policy.overlap_grad_reduce or policy.overlap_param_gather
    bucket_size = (
        max(_MIN_BUCKET_SIZE, 1_000_000 * state.dp_cp_group.size()) if overlap_enabled else None
    )
    ddp_config = DistributedDataParallelConfig(
        # DistributedOptimizer consumes reduced shards directly and maintains
        # FP32 master parameters, so its communication buffer need not match
        # the BF16 model parameter dtype used by the former torch bridge.
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=policy.overlap_grad_reduce,
        overlap_param_gather=policy.overlap_param_gather,
        use_distributed_optimizer=True,
        bucket_size=bucket_size,
    )
    singleton_group = state.rank_singleton_group
    if singleton_group is None:
        raise RuntimeError("MCore DDP requires vForge's rank-singleton process group")
    tensor_parallel_group = state.attention_tp_group
    groups = ProcessGroupCollection(
        dp=state.dp_group,
        dp_cp=state.dp_cp_group,
        # No expert parallelism is exposed by this milestone. Reusing the
        # ordinary reduction domain avoids another otherwise-unused group.
        expt_dp=state.dp_cp_group,
        # MCore DDP requires concrete TP/PP/EP groups even when those
        # dimensions have size one.  The singleton is an identity group, not a
        # reduction domain; real gradient synchronization remains on DP x CP.
        tp=tensor_parallel_group if tensor_parallel_group is not None else singleton_group,
        pp=singleton_group,
        ep=singleton_group,
        # With one distributed-optimizer instance, gradient statistics must
        # cover both the DP x CP shards and all TP shards. vForge's supported
        # topology has no PP/EP yet, so that domain is WORLD.
        intra_dist_opt=dist.group.WORLD,
        mp=tensor_parallel_group if tensor_parallel_group is not None else singleton_group,
        tp_ep_pp=tensor_parallel_group if tensor_parallel_group is not None else singleton_group,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    full_param_layout = DistributedOptimizer.compute_full_param_layout(
        trainable_parameters,
        bucket_size,
        state.dp_cp_group.size(),
        ddp_config,
        expert_data_parallel_world_size=state.dp_cp_group.size(),
    )
    wrapped = DistributedDataParallel(
        config=config,
        ddp_config=ddp_config,
        module=model,
        disable_bucketing=not overlap_enabled,
        pg_collection=groups,
        full_param_layout=full_param_layout,
    )
    for parameter in wrapped.parameters():
        if not parameter.requires_grad:
            continue
        setattr(parameter, _GRADIENT_SEEN, False)
        set_main_gradient_active(parameter, False)
        hook = parameter.register_hook(_mark_gradient_seen(parameter))
        setattr(parameter, _GRADIENT_HOOK, hook)
    setattr(wrapped, _DATA_PARALLEL_BACKEND, "mcore")
    setattr(wrapped, _PROCESS_GROUP_COLLECTION, groups)
    # Match torch DDP's initialization contract. DP x CP peers own replicas of
    # the same TP shard and must begin from identical values.
    wrapped.broadcast_params()
    return wrapped


def prepare_mcore_data_parallel_backward(model: nn.Module) -> None:
    """Clear MCore's persistent buffers and per-step used-parameter markers."""
    if not is_mcore_data_parallel(model):
        return
    model.zero_grad_buffer()
    # MCore learns each bucket's expected hook counts from its first backward.
    # Seed zero counts for globally unused parameters so they remain part of
    # that stable pattern without preventing used parameters from dispatching
    # their bucket's reduce-scatter as soon as backward reaches them.
    for bucket_group in model.bucket_groups + model.expert_parallel_bucket_groups:
        if not bucket_group.ddp_config.overlap_grad_reduce:
            continue
        for parameter in bucket_group.params:
            bucket_group.per_param_grad_ready_counts.setdefault(parameter, 0)
    for parameter in model.parameters():
        if not hasattr(parameter, "main_grad"):
            continue
        setattr(parameter, _GRADIENT_SEEN, False)
        set_main_gradient_active(parameter, False)


def finish_mcore_data_parallel_gradient_sync(model: nn.Module) -> None:
    """Reduce gradients, then agree which parameters participated globally.

    MCore owns the gradient collective: backward hooks dispatch each ready
    bucket's asynchronous reduce-scatter over DP x CP, and
    ``finish_grad_sync`` waits for any outstanding work. The mask reduction
    below does not reduce gradients; it records which optimizer shards must
    remain inactive for parameters unused on every replica.
    """
    if not is_mcore_data_parallel(model):
        return
    # Actual DDP gradient synchronization. MCore has already accumulated local
    # autograd results into its contiguous main_grad buffers via backward hooks.
    model.finish_grad_sync()
    parameters = [parameter for parameter in model.parameters() if hasattr(parameter, "main_grad")]
    if not parameters:
        return
    device = parameters[0].device
    # A main_grad view exists (and is zero-initialized) even for an unused
    # parameter. If any DP x CP replica used the parameter, every replica must
    # activate its reduced optimizer shard; otherwise all replicas keep the
    # shard gradient absent so weight decay and optimizer state do not update it.
    active = torch.tensor(
        [
            bool(getattr(parameter, _GRADIENT_SEEN, False))
            or bool(getattr(parameter, "grad_added_to_main_grad", False))
            for parameter in parameters
        ],
        device=device,
        dtype=torch.uint8,
    )
    dist.all_reduce(active, op=dist.ReduceOp.MAX, group=model.dp_cp_group)
    for parameter, used in zip(parameters, active.tolist(), strict=True):
        set_main_gradient_active(parameter, bool(used))


def finish_mcore_data_parallel_parameter_sync(model: nn.Module) -> None:
    """Make every post-update parameter shard visible on every DP x CP peer.

    Normal training deliberately does not call this after an optimizer step:
    MCore's forward pre-hooks gather one bucket at a time and overlap subsequent
    gathers with forward compute. Callers that immediately inspect the complete
    updated model, such as parity diagnostics, need this explicit synchronization.
    """
    if not is_mcore_data_parallel(model) or not model.ddp_config.overlap_param_gather:
        return
    model.start_param_sync(force_sync=True)


def is_mcore_data_parallel(model: nn.Module) -> bool:
    return getattr(model, _DATA_PARALLEL_BACKEND, None) == "mcore"


def mcore_process_group_collection(model: nn.Module):
    """Return the vForge-owned groups injected into an MCore DDP wrapper."""
    if not is_mcore_data_parallel(model):
        raise TypeError("MCore process groups require an MCore DDP model")
    return getattr(model, _PROCESS_GROUP_COLLECTION)


def _mark_gradient_seen(parameter: nn.Parameter):
    parameter_ref = weakref.ref(parameter)

    def hook(gradient: torch.Tensor) -> torch.Tensor:
        current_parameter = parameter_ref()
        if current_parameter is not None:
            setattr(current_parameter, _GRADIENT_SEEN, True)
        return gradient

    return hook


__all__ = [
    "apply_mcore_data_parallel",
    "MCoreDataParallelConfig",
    "finish_mcore_data_parallel_gradient_sync",
    "finish_mcore_data_parallel_parameter_sync",
    "is_mcore_data_parallel",
    "mcore_process_group_collection",
    "prepare_mcore_data_parallel_backward",
]
