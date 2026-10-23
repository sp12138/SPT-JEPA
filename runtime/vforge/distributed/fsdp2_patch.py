"""Compatibility patches for released FSDP2 implementations."""

from __future__ import annotations

import functools
import inspect
from typing import Any

import torch

_PATCH_MARKER = "_vforge_ac_mixed_precision_patch"


def validate_gradient_divide_factor(
    module: torch.nn.Module, *, factor: float, mesh_size: int, mp_policy: Any = None
) -> None:
    """Reject the PyTorch 2.13 singleton FSDP double-division path."""
    if mesh_size != 1 or factor == 1:
        return
    reduce_dtype = (
        mp_policy.reduce_dtype or mp_policy.param_dtype if mp_policy is not None else None
    )
    if any(
        (reduce_dtype or parameter.dtype) not in (torch.float32, torch.bfloat16)
        for parameter in module.parameters()
    ):
        raise NotImplementedError(
            "custom gradient division with FSDP size 1 requires "
            "float32 or bfloat16 gradient reduction"
        )


def ensure_activation_checkpoint_mixed_precision_compatibility() -> bool:
    """Patch affected FSDP2 hooks and return whether a patch was installed.

    PyTorch versions containing pytorch/pytorch#182580 are detected by their
    ``FSDPState._cast_forward_inputs`` helper and are left untouched. Older
    implementations are patched only when their hook source matches the known
    vulnerable shape; unknown layouts fail closed instead of modifying private
    internals speculatively.
    """
    import torch.distributed.fsdp._fully_shard._fsdp_state as fsdp_state

    state_cls = fsdp_state.FSDPState
    if getattr(state_cls, _PATCH_MARKER, False):
        return False
    if _has_upstream_fix(state_cls):
        return False
    _validate_vulnerable_layout(state_cls, fsdp_state)

    original_pre_forward = state_cls._pre_forward
    original_post_forward = state_cls._post_forward

    @functools.wraps(original_pre_forward)
    def _pre_forward(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        is_recompute = self._training_state == fsdp_state.TrainingState.PRE_BACKWARD
        args, kwargs = original_pre_forward(self, module, args, kwargs)
        if is_recompute and self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype:
            with torch.profiler.record_function("FSDP::cast_forward_inputs"):
                cast_fn = functools.partial(
                    fsdp_state._cast_fp_tensor,
                    self._mp_policy.param_dtype,
                )
                args = fsdp_state._apply_to_tensors(cast_fn, args)
                kwargs = fsdp_state._apply_to_tensors(cast_fn, kwargs)
        return args, kwargs

    @functools.wraps(original_post_forward)
    def _post_forward(
        self,
        module: torch.nn.Module,
        inputs: Any,
        output: Any,
    ) -> Any:
        is_recompute = self._training_state == fsdp_state.TrainingState.PRE_BACKWARD
        output = original_post_forward(self, module, inputs, output)
        if is_recompute and self._mp_policy.output_dtype is not None:
            with torch.profiler.record_function("FSDP::cast_forward_outputs"):
                output = fsdp_state._apply_to_tensors(
                    functools.partial(
                        fsdp_state._cast_fp_tensor,
                        self._mp_policy.output_dtype,
                    ),
                    output,
                )
        return output

    state_cls._pre_forward = _pre_forward
    state_cls._post_forward = _post_forward
    setattr(state_cls, _PATCH_MARKER, True)
    return True


def _has_upstream_fix(state_cls: type) -> bool:
    if not hasattr(state_cls, "_cast_forward_inputs"):
        return False
    source = _method_source(state_cls._pre_forward)
    pre_backward = source.split("TrainingState.PRE_BACKWARD", maxsplit=1)[-1]
    return "_cast_forward_inputs" in pre_backward.split("TrainingState.FORWARD", maxsplit=1)[0]


def _validate_vulnerable_layout(state_cls: type, fsdp_state) -> None:
    required_module_attrs = (
        "TrainingState",
        "_apply_to_tensors",
        "_cast_fp_tensor",
    )
    missing = [name for name in required_module_attrs if not hasattr(fsdp_state, name)]
    pre_source = _method_source(state_cls._pre_forward)
    post_source = _method_source(state_cls._post_forward)
    vulnerable_pre = (
        "TrainingState.PRE_BACKWARD" in pre_source and "return args, kwargs" in pre_source
    )
    vulnerable_post = "TrainingState.PRE_BACKWARD" in post_source and "return output" in post_source
    if missing or not vulnerable_pre or not vulnerable_post:
        details = (
            f"missing={missing}, vulnerable_pre={vulnerable_pre}, vulnerable_post={vulnerable_post}"
        )
        raise RuntimeError(
            "Unsupported FSDP2 implementation for activation-checkpoint "
            f"mixed-precision compatibility patch ({details}); "
            f"torch={torch.__version__}. Upgrade to a build containing "
            "pytorch/pytorch#182580."
        )


def _method_source(method) -> str:
    try:
        return inspect.getsource(method)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "Cannot inspect FSDP2 private hook source; refusing to install "
            f"compatibility patch for torch={torch.__version__}."
        ) from exc
