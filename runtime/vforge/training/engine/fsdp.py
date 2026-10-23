"""FSDP2 training engine.

Torch-native throughout: both dimensions it can shard along are expressed as
DTensor, so a weight split by tensor parallelism and then by data parallelism
carries both axes at once. That is what lets the two combine here and not under
megatron, and it is also why nothing in this file has to teach the checkpoint
layer anything -- DTensor is the only vocabulary it already reads.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn

from vforge.distributed.parallelize import (
    apply_replicated_data_parallel,
    materialize_meta_module,
    parallelize_model,
)
from vforge.distributed.tensor_parallel import apply_torch_tensor_parallel
from vforge.training.diagnostics import (
    clip_and_step,
    gradient_clipping_fingerprint,
    run_backward,
    validate_gradient_norm_dtype,
)
from vforge.training.engine.base import (
    ComputeStep,
    EnginePolicy,
    EngineRunContext,
    EngineStepResult,
    TrainingEngine,
)
from vforge.training.output_aggregation import StepOutputAggregator
from vforge.training.profiling import record_region
from vforge.training.recipe import StepContext


class FsdpEngine(TrainingEngine):
    """Torch-native engine: FSDP2 sharding with Ulysses handled in the model."""

    name = "fsdp2"

    def __init__(self, policy: EnginePolicy) -> None:
        parallel = policy.parallelize.parallel
        validate_gradient_norm_dtype(
            policy.gradient_norm_dtype,
            engine=self.name,
            tp_size=parallel.tp_size,
        )
        if parallel.ep_size > 1:
            parallel.normalized(
                parallel.dp_size
                * parallel.usp_size
                * parallel.tp_size
                * parallel.cp_size
                * parallel.pp_size
            )
        if parallel.pp_size > 1:
            raise NotImplementedError(
                "the fsdp2 engine has no pipeline parallel implementation; got "
                f"pp_size={parallel.pp_size}"
            )
        if parallel.cp_size > 1:
            raise NotImplementedError(
                "the fsdp2 context-parallel backend is not implemented yet; "
                f"got cp_size={parallel.cp_size}. Use engine=megatron or set cp_size=1"
            )
        if parallel.dp_mode == "ddp" and parallel.dp_size > 1 and parallel.tp_size > 1:
            raise NotImplementedError(
                "the fsdp2 engine cannot combine torch-native tensor parallelism "
                "with replicated DDP: TP parameters are DTensors, which PyTorch "
                "DDP cannot flatten for parameter synchronization. Use fsdp2 data "
                "sharding, or select the megatron engine for TP + DDP"
            )
        self.policy = policy

    def prepare_model(
        self,
        model: nn.Module,
        *,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    ) -> nn.Module:
        # This engine's own tensor parallelism; the shared step below then shards
        # what is left over the data mesh.
        apply_torch_tensor_parallel(model)
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
        """Materialize DTensor/FSDP parameters without changing their layouts."""
        return materialize_meta_module(model, device)

    def prepare_data_parallel(self, model: nn.Module) -> nn.Module:
        return apply_replicated_data_parallel(model)

    def configure_optimizers(self, model, optimizer_spec, scheduler_spec):
        del model, optimizer_spec, scheduler_spec
        return None

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
        """Execute one FSDP optimizer update.

        ``model`` is the forward root while ``update_target`` owns clipping and
        diagnostics. They differ when a composite model updates one independent
        FSDP root.
        """

        policy = self.policy
        gradient_clipping_fingerprint(run.gradient_clipper, policy.gradient_norm_dtype)
        update_target = model if update_target is None else update_target
        if policy.run_backward and optimizer is None:
            raise RuntimeError("run_backward=True requires an optimizer")
        count = policy.gradient_accumulation_steps
        outputs = StepOutputAggregator(count)
        forward_s = 0.0
        backward_s = 0.0
        dataloader_s = 0.0
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        for _ in range(count):
            load_started = time.perf_counter()
            batch = next(microbatches)
            dataloader_s += time.perf_counter() - load_started
            if run.offload_manager is not None:
                run.offload_manager.begin_step()
            forward_started = time.perf_counter()
            output = self.run_forward(
                lambda batch=batch: compute_step(model, batch, step, context),
                run=run,
            )
            forward_s += time.perf_counter() - forward_started
            if count > 1 and output.backward_losses is not None:
                raise ValueError("lazy backward losses require gradient_accumulation_steps=1")
            if policy.run_backward:
                backward_started = time.perf_counter()
                with record_region(run.profiler, "vforge/backward_update"):
                    # Sync every backward, retaining only sharded gradients.
                    run_backward(
                        loss=output.loss,
                        backward_scale=1.0 / count,
                        offload_manager=run.offload_manager,
                        backward_losses=output.backward_losses,
                        backward_forward_context=run.precision_context,
                    )
                backward_s += time.perf_counter() - backward_started
            elif run.offload_manager is not None:
                run.offload_manager.abort_step()
            outputs.add(output)
            del output, batch
        output = outputs.finish()

        optimizer_s = 0.0
        diagnostics = {}
        if policy.run_backward:
            optimizer_started = time.perf_counter()
            with record_region(run.profiler, "vforge/backward_update"):
                diagnostics = clip_and_step(
                    model=update_target,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    usp_size=context.parallel_state.config.usp_size,
                    gradient_clip_val=policy.gradient_clip_val,
                    parity=policy.parity,
                    parameter_bucket=run.parameter_bucket,
                    diagnostics_config=policy.diagnostics,
                    # The root filter uses parameter identity and also applies
                    # to a separately optimized submodule.
                    reduction=run.logical_reduction,
                    gradient_clipper=run.gradient_clipper,
                )
            optimizer_s = time.perf_counter() - optimizer_started
        return EngineStepResult(
            output=output,
            diagnostics=diagnostics,
            forward_s=forward_s,
            backward_s=backward_s,
            optimizer_s=optimizer_s,
            completed_at=time.perf_counter(),
            dataloader_s=dataloader_s,
        )


__all__ = ["FsdpEngine"]
