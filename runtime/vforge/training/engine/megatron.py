"""Megatron-Core training engine.

Owns the dimensions Megatron brings: tensor and context parallelism, MCore
replicated data parallelism, and MCore's distributed optimizer. Pipeline
parallelism remains future work. Ulysses is rejected rather than combined with
them. The two shard a sequence by different means --
Ulysses trades whole attention heads between ranks over a group of its own,
Megatron's sequence parallelism rides the tensor parallel group and costs no
ranks -- and nothing here reconciles them.

A model states how each projection splits and nothing about who implements it,
so this engine is what turns those declarations into mcore layers: they are a
different class whose weight is allocated already narrow, and the only way to get
one is to build it and put it where the plain projection was. It then declares
the shards at the state-dict boundary, which is what lets reading and writing
weights stay the fsdp2 path unchanged.

Neither step reaches mcore at import time. Both live in
``vforge.distributed.tensor_parallel``, which defers its mcore import, because
mcore pulls in triton at package import and that would make this engine
unloadable on machines that can still usefully run tp=1.

This engine deliberately uses MCore DDP and DistributedOptimizer even when the
DP and model-parallel groups each contain one rank. The explicit size-one
groups keep that lifecycle identical to larger topologies and avoid a separate
torch-optimizer path.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn

from vforge.distributed.mcore_data_parallel import (
    apply_mcore_data_parallel,
    finish_mcore_data_parallel_gradient_sync,
    prepare_mcore_data_parallel_backward,
)
from vforge.distributed.parallelize import materialize_meta_module, parallelize_model
from vforge.distributed.tensor_parallel import (
    apply_mcore_tensor_parallel,
    install_tp_state_dict_hooks,
    restore_mcore_tensor_parallel_parameter_attributes,
    validate_mcore_tensor_parallel_parameter_attributes,
)
from vforge.models.loading import cast_module_parameters
from vforge.training.diagnostics import validate_gradient_norm_dtype
from vforge.training.engine.base import (
    ComputeStep,
    EnginePolicy,
    EngineRunContext,
    EngineStepResult,
    TrainingEngine,
)
from vforge.training.profiling import record_region
from vforge.training.recipe import OptimizerSpec, SchedulerSpec, StepContext


class MegatronEngine(TrainingEngine):
    """Megatron-Core engine with tensor and context parallelism."""

    name = "megatron"

    def __init__(self, policy: EnginePolicy) -> None:
        parallel = policy.parallelize.parallel
        validate_gradient_norm_dtype(
            policy.gradient_norm_dtype,
            engine=self.name,
            tp_size=parallel.tp_size,
        )
        if parallel.ep_size > 1:
            raise NotImplementedError("EP requires the fsdp2 engine")
        if parallel.usp_size > 1:
            raise NotImplementedError(
                "the megatron engine does not run Ulysses; its own sequence "
                "parallelism is a different mechanism riding the tp group. got "
                f"usp_size={parallel.usp_size}"
            )
        if parallel.pp_size > 1:
            raise NotImplementedError(
                "the megatron engine has no pipeline parallel implementation yet; got "
                f"pp_size={parallel.pp_size}"
            )
        if parallel.cp_size > 1 and (parallel.dp_mode != "ddp" or policy.parallelize.fsdp2.enable):
            raise NotImplementedError(
                "the first MCore context-parallel implementation uses replicated "
                "DDP and cannot be combined with FSDP2; set parallel.dp_mode=ddp "
                "and parallelize.fsdp2.enable=false"
            )
        if policy.gradient_accumulation_steps != 1:
            raise NotImplementedError(
                "the megatron engine runs a single microbatch per step until it drives "
                f"a pipeline schedule; got {policy.gradient_accumulation_steps}"
            )
        if parallel.tp_size > 1 and policy.parallelize.fsdp2.enable:
            # Both want to say how a weight is split, in different vocabularies.
            # fully_shard replaces every parameter with a DTensor placed on the
            # data mesh -- it does so even when that mesh holds one rank, where
            # it shards nothing -- and what it wraps is already mcore's tp
            # slice. The state dict then describes the data axis and cannot also
            # describe the tp one, so a checkpoint would record a shard as if it
            # were whole. Expressing both needs a two-dimensional mesh that
            # nothing here builds yet, so refuse rather than write that.
            raise NotImplementedError(
                "tensor parallelism and fsdp2 cannot both shard this model yet: "
                "fsdp2 would re-wrap mcore's tp shards as DTensors on the data "
                f"mesh and hide the tp axis from every checkpoint. got tp_size="
                f"{parallel.tp_size} with parallelize.fsdp2.enable=true; set it "
                "false to run tp on its own"
            )
        if parallel.dp_mode != "ddp" or policy.parallelize.fsdp2.enable:
            raise NotImplementedError(
                "the megatron engine owns MCore DDP and DistributedOptimizer; "
                "set parallel.dp_mode=ddp and parallelize.fsdp2.enable=false"
            )
        self.policy = policy

    def prepare_model(
        self,
        model: nn.Module,
        *,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    ) -> nn.Module:
        if self.policy.parallelize.parallel.cp_size > 1:
            from vforge.distributed.context_parallel import initialize_context_parallel

            initialize_context_parallel("mcore")
        # This engine's own tensor parallelism, then the hooks that make the
        # shards it produced legible to a checkpoint. Hooks sit on modules, so
        # they survive the to_empty() that materializes the model later and
        # replaces every Parameter object.
        apply_mcore_tensor_parallel(model)
        install_tp_state_dict_hooks(model)
        return parallelize_model(
            model,
            self.policy.parallelize,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        )

    def materialize_model(
        self,
        model: nn.Module,
        device: torch.device | str,
    ) -> nn.Module:
        """Allocate local shards, then complete MCore's parameter contract."""
        model = materialize_meta_module(model, device)
        precision = self.policy.parallelize.fsdp2.mixed_precision
        if precision.enable and precision.param_dtype is not None:
            parameter_dtype = {
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "float16": torch.float16,
                "fp32": torch.float32,
                "float32": torch.float32,
            }[precision.param_dtype]
            # Match FSDP2's mixed-precision contract: ``param_dtype`` changes
            # parameter storage/all-gather precision, while model-authored
            # buffers retain their original dtype unless ``buffer_dtype`` is
            # configured separately. In particular, Wan's non-persistent RoPE
            # tables are FP64; casting the whole module to BF16 compounds a
            # small positional-encoding error through every transformer block.
            cast_module_parameters(model, parameter_dtype)
        restore_mcore_tensor_parallel_parameter_attributes(model)
        validate_mcore_tensor_parallel_parameter_attributes(model)
        return model

    def prepare_data_parallel(self, model: nn.Module) -> nn.Module:
        return apply_mcore_data_parallel(
            model,
            config=self.policy.parallelize.mcore_ddp,
        )

    def configure_optimizers(
        self,
        model: nn.Module,
        optimizer_spec: OptimizerSpec | None,
        scheduler_spec: SchedulerSpec | None,
    ):
        if optimizer_spec is None:
            raise ValueError(
                "the megatron engine requires an engine-neutral OptimizerSpec; "
                "a torch-only optimizer_factory cannot construct MCore DistributedOptimizer"
            )
        from vforge.training.mcore_optimizer import build_mcore_distributed_optimizer

        return build_mcore_distributed_optimizer(
            model,
            optimizer_spec,
            scheduler_spec,
            clip_grad=self.policy.gradient_clip_val,
        )

    def initialize_optimizer_from_pretrained(
        self,
        model: nn.Module,
        optimizer: Any,
        sources: tuple[Any, ...],
    ) -> None:
        """Preserve HF FP32 weights in MCore's DP-sharded master parameters."""
        from vforge.models.loading import load_pretrained_optimizer_main_param_state
        from vforge.training.mcore_optimizer import MCoreDistributedOptimizer

        if not isinstance(optimizer, MCoreDistributedOptimizer):
            raise TypeError("MegatronEngine requires its MCore distributed optimizer")
        checkpoint_model = model.module if hasattr(model, "module") else model
        state_dict = load_pretrained_optimizer_main_param_state(checkpoint_model, sources)
        optimizer.reload_main_params_from_pretrained(state_dict)

    def run_update(
        self,
        *,
        model: nn.Module,
        microbatches: Iterator[Any],
        step: int,
        context: StepContext,
        compute_step: ComputeStep,
        optimizer: Any,
        scheduler: Any,
        run: EngineRunContext,
        update_target: nn.Module | None = None,
    ) -> EngineStepResult:
        policy = self.policy
        load_started = time.perf_counter()
        batch = next(microbatches)
        dataloader_s = time.perf_counter() - load_started
        update_target = model if update_target is None else update_target
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            prepare_mcore_data_parallel_backward(update_target)

        if run.offload_manager is not None:
            run.offload_manager.begin_step()
        forward_started = time.perf_counter()
        output = self.run_forward(
            lambda: compute_step(model, batch, step, context),
            run=run,
        )
        if output.backward_losses is not None:
            raise ValueError("lazy backward losses require the FSDP2 engine")
        forward_at = time.perf_counter()

        if not policy.run_backward:
            if run.offload_manager is not None:
                run.offload_manager.abort_step()
            return EngineStepResult(
                output=output,
                diagnostics={},
                forward_s=forward_at - forward_started,
                backward_s=0.0,
                optimizer_s=0.0,
                completed_at=forward_at,
                dataloader_s=dataloader_s,
            )

        if optimizer is None:
            raise RuntimeError("run_backward=True requires an optimizer")
        with record_region(run.profiler, "vforge/backward_update"):
            from vforge.training.mcore_optimizer import (
                MCoreDistributedOptimizer,
                backward_mcore_update,
            )

            if not isinstance(optimizer, MCoreDistributedOptimizer):
                raise TypeError("MegatronEngine requires its MCore distributed optimizer")
            update = backward_mcore_update(
                loss=output.loss,
                model=update_target,
                optimizer=optimizer,
                scheduler=scheduler,
                usp_size=context.parallel_state.config.usp_size,
                parity=policy.parity,
                precision_context=run.precision_context,
                offload_manager=run.offload_manager,
                parameter_bucket=run.parameter_bucket,
                diagnostics_config=policy.diagnostics,
                reduction=run.logical_reduction,
                # CP gathers the complete output for the loss, but its gather
                # backward returns only this rank's sequence slice. MCore DDP
                # averages those disjoint contributions over DP x CP; scale by
                # CP so the CP dimension sums while DP continues to average.
                backward_scale=context.parallel_state.config.cp_size,
                finish_gradient_sync=lambda: finish_mcore_data_parallel_gradient_sync(
                    update_target
                ),
            )
        return EngineStepResult(
            output=output,
            diagnostics=update.diagnostics,
            forward_s=forward_at - forward_started,
            backward_s=update.backward_at - forward_at,
            optimizer_s=update.optimizer_at - update.backward_at,
            completed_at=update.optimizer_at,
            dataloader_s=dataloader_s,
        )


__all__ = ["MegatronEngine"]
