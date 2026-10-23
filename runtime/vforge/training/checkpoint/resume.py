"""Fingerprint and rank-local state helpers for training resume."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

import torch.distributed as dist

from vforge.training.checkpoint.manifest import CheckpointManifest
from vforge.training.diagnostics import gradient_clipping_fingerprint
from vforge.training.parity import ParityStepConfig


def qualified_type(value: Any) -> str | None:
    if value is None:
        return None
    return f"{type(value).__module__}.{type(value).__qualname__}"


def component_types(value: Any, *, attribute: str) -> str | dict[str, str | None] | None:
    """Fingerprint one component or a structurally named collection."""

    declared = getattr(value, attribute, value)
    if isinstance(declared, Mapping):
        return {str(name): qualified_type(item) for name, item in declared.items()}
    return qualified_type(declared)


def execution_fingerprint(
    *,
    seed: int,
    precision: str,
    autocast: bool,
    gradient_accumulation_steps: int,
    gradient_clip_val: float,
    parity: ParityStepConfig | None,
    optimizer: Any,
    scheduler: Any,
    parallelize: Any,
    gradient_norm_dtype: str = "float32",
    gradient_clipper: Any = None,
) -> dict[str, Any]:
    """Describe execution choices that affect bit-exact continuation."""
    fsdp2 = parallelize.fsdp2
    mixed_precision = fsdp2.mixed_precision
    return {
        "seed": seed,
        "precision": precision,
        "autocast": autocast,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clip_val": gradient_clip_val,
        **gradient_clipping_fingerprint(gradient_clipper, gradient_norm_dtype),
        "parity": (
            None
            if parity is None
            else {
                "optimizer_step_mode": parity.optimizer_step_mode,
                "scale_grads_before_optimizer_step": (parity.scale_grads_before_optimizer_step),
                "quantize_after_optimizer_step": (parity.quantize_after_optimizer_step),
                "emit_legacy_probe_aliases": parity.emit_legacy_probe_aliases,
            }
        ),
        "optimizer_type": component_types(optimizer, attribute="optimizers"),
        "scheduler_type": component_types(scheduler, attribute="schedulers"),
        "parallelize": {
            "gradient_checkpointing": parallelize.gradient_checkpointing,
            "fsdp2": {
                "enable": fsdp2.enable,
                "reshard_after_forward": fsdp2.reshard_after_forward,
                "basic_modules": list(fsdp2.basic_modules),
                "leaf_module_paths": list(fsdp2.leaf_module_paths),
                "mixed_precision": {
                    "enable": mixed_precision.enable,
                    "param_dtype": mixed_precision.param_dtype,
                    "reduce_dtype": mixed_precision.reduce_dtype,
                    "output_dtype": mixed_precision.output_dtype,
                    "buffer_dtype": mixed_precision.buffer_dtype,
                    "cast_forward_inputs": mixed_precision.cast_forward_inputs,
                    "preserve_module_paths": list(mixed_precision.preserve_module_paths),
                    "preserve_forward_input_paths": list(
                        mixed_precision.preserve_forward_input_paths
                    ),
                    "contract_mode": mixed_precision.contract_mode,
                },
            },
        },
    }


def parallel_fingerprint(parallelize: Any) -> dict[str, Any]:
    """Describe the normalized distributed topology for a checkpoint."""
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    parallel = parallelize.parallel.normalized(world_size)
    return {
        "dp_size": parallel.dp_size,
        "dp_replicate_size": parallel.dp_replicate_size,
        "dp_shard_size": parallel.dp_shard_size,
        "usp_size": parallel.usp_size,
        "cp_size": parallel.cp_size,
        "tp_size": parallel.tp_size,
        "pp_size": parallel.pp_size,
        "dp_mode": parallel.dp_mode,
        "include_usp_in_fsdp": parallel.include_usp_in_fsdp,
        "extra_parallel_names": list(parallel.extra_parallel_names),
        "extra_parallel_sizes": list(parallel.extra_parallel_sizes),
        "device_type": parallel.device_type,
    }


def warn_global_zero(message: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def validate_resume_fingerprints(
    manifest: CheckpointManifest,
    *,
    execution_policy: dict[str, Any],
    topology: dict[str, Any],
    fingerprints: dict[str, Any] | None = None,
) -> bool:
    """Return whether a full checkpoint still has a bit-exact contract."""
    if manifest.state != "full":
        return True
    bit_exact = True
    if manifest.execution_policy != execution_policy:
        raise RuntimeError(
            "strict resume rejected an execution policy mismatch. "
            f"policy: saved={manifest.execution_policy!r}, "
            f"current={execution_policy!r}"
        )
    if manifest.topology and manifest.topology != topology:
        raise RuntimeError(
            "strict resume rejected a parallel topology mismatch. "
            f"topology: saved={manifest.topology!r}, current={topology!r}"
        )
    for name, current in (fingerprints or {}).items():
        saved = manifest.fingerprints.get(name)
        if saved != current:
            raise RuntimeError(
                f"strict resume rejected a {name} fingerprint mismatch. "
                f"saved={saved!r}, current={current!r}"
            )
    return bit_exact


def build_dataloader_resume_contract(
    loader: Any,
    application_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe loader properties that identify its resumable sample stream."""
    sampler = getattr(loader, "sampler", None)
    dataset = getattr(loader, "dataset", None)
    try:
        dataset_length = len(dataset) if dataset is not None else None
    except TypeError:
        dataset_length = None
    try:
        loader_length = len(loader)
    except TypeError:
        loader_length = None
    contract = {
        "loader_type": qualified_type(loader),
        "loader_length": loader_length,
        "dataset_type": qualified_type(dataset),
        "dataset_length": dataset_length,
        "batch_size": getattr(loader, "batch_size", None),
        "drop_last": getattr(loader, "drop_last", None),
        "sampler_type": qualified_type(sampler),
        "sampler_seed": getattr(sampler, "seed", None),
    }
    if application_contract is not None:
        if not isinstance(application_contract, dict):
            raise TypeError("dataloader_resume_contract must return a dictionary")
        contract["application"] = application_contract
    return contract


def validate_bit_exact_dataloader(
    loader: Any,
    supports_bit_exact_resume: Callable[[Any], bool | None],
) -> None:
    explicit = supports_bit_exact_resume(loader)
    if explicit is not None:
        if not explicit:
            raise RuntimeError("application rejected bit-exact dataloader resume")
        return
    num_workers = int(getattr(loader, "num_workers", 0) or 0)
    if num_workers > 0:
        raise RuntimeError(
            "full DCP resume with DataLoader workers requires the "
            "application to explicitly implement "
            "supports_bit_exact_dataloader_resume(loader)"
        )


def validate_dataloader_resume_contract(
    saved: Any,
    current: dict[str, Any],
    *,
    bit_exact: bool,
) -> bool:
    if not isinstance(saved, dict):
        raise ValueError("full DCP is missing the dataloader contract")
    if saved != current:
        if bit_exact:
            raise RuntimeError(
                "strict resume rejected a dataloader contract mismatch. "
                f"contract: saved={saved!r}, current={current!r}"
            )
        warn_global_zero(
            "full DCP dataloader contract changed; the saved loader state "
            "will be restored best-effort, but the sample stream is not bit-exact. "
            f"contract: saved={saved!r}, current={current!r}"
        )
        return False
    return bit_exact


def restore_scheduler(scheduler: Any, state: Any) -> None:
    if scheduler is None:
        if state is not None:
            raise RuntimeError(
                "checkpoint contains scheduler state but no scheduler was configured"
            )
        return
    if state is None:
        raise RuntimeError("full DCP resume requires scheduler state for the configured scheduler")
    scheduler.load_state_dict(state)


def fast_forward_scheduler(scheduler: Any, start_step: int) -> None:
    if scheduler is None or start_step <= 0:
        return
    for _ in range(int(start_step)):
        scheduler.step()


__all__ = [
    "build_dataloader_resume_contract",
    "execution_fingerprint",
    "fast_forward_scheduler",
    "parallel_fingerprint",
    "qualified_type",
    "restore_scheduler",
    "validate_bit_exact_dataloader",
    "validate_dataloader_resume_contract",
    "validate_resume_fingerprints",
    "warn_global_zero",
]
