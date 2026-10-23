"""Numerical operation entry points.

Public attention callables remain available via ``from vforge.ops import attention``.
Process-level
backend policy is installed once via :func:`initialize_ops`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from vforge.ops._attention import (
    AttentionCallPolicy,
    SegmentedSparsity,
    TeacherForcingSparsity,
    attention_debug_enabled,
    get_flash_attention_capability,
    segmented_available,
    set_attention_debug,
    set_flash_attention_backend,
    set_flash_attention_deterministic,
    set_sdpa_backend,
    teacher_forcing_available,
    teacher_forcing_deterministic_available,
    teacher_forcing_sparsity_stats,
    unified_attention,
    unified_attention_varlen,
)
from vforge.ops._attention import (
    get_attn_implementation as _detect_attn_implementation,
)
from vforge.ops.config import OpsConfig

if TYPE_CHECKING:
    from vforge.distributed.parallel_state import ParallelState

_OPS_CONFIG: OpsConfig | None = None
_RESOLVED_BACKEND: str | None = None


def initialize_ops(
    config: OpsConfig,
    *,
    device: torch.device,
    parallel_state: ParallelState,
) -> str:
    """Install the process-level ops policy and return the resolved backend id.

    Must run after the rank device is bound. Re-applying the same config is
    idempotent; a conflicting config raises.
    """
    del parallel_state  # reserved for future Ulysses-aware ops validation
    global _OPS_CONFIG, _RESOLVED_BACKEND

    if _OPS_CONFIG is not None:
        if _OPS_CONFIG != config:
            raise RuntimeError(
                "initialize_ops() called with a conflicting OpsConfig; "
                "one process may only host one immutable ops policy"
            )
        assert _RESOLVED_BACKEND is not None
        return _RESOLVED_BACKEND

    set_attention_debug(config.debug)
    set_flash_attention_deterministic(config.deterministic)
    set_flash_attention_backend(config.backend)
    torch.use_deterministic_algorithms(config.deterministic, warn_only=False)
    torch.backends.cudnn.deterministic = config.deterministic
    if config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
    if config.backend in {"sdpa", "sdpa_math"}:
        set_sdpa_backend(config.backend)
    if config.blas_backend == "cublas" and device.type == "cuda":
        torch.backends.cuda.preferred_blas_library("cublas")
    resolved = _resolve_backend(config, device=device)
    _OPS_CONFIG = config
    _RESOLVED_BACKEND = resolved
    return resolved


def reset_ops() -> None:
    """Clear process ops policy. Intended for tests and process teardown."""
    global _OPS_CONFIG, _RESOLVED_BACKEND
    _OPS_CONFIG = None
    _RESOLVED_BACKEND = None
    set_attention_debug(False)
    set_sdpa_backend("sdpa")


def get_ops_config() -> OpsConfig | None:
    return _OPS_CONFIG


def get_resolved_attention_backend() -> str | None:
    return _RESOLVED_BACKEND


def get_attn_implementation() -> str:
    """Return the installed backend, or hardware capability before init."""
    return _RESOLVED_BACKEND or _detect_attn_implementation()


def _resolve_backend(config: OpsConfig, *, device: torch.device) -> str:
    if config.backend != "auto":
        if config.backend in {"sdpa", "sdpa_math"}:
            return config.backend
        if device.type != "cuda":
            raise RuntimeError(
                f"attention.backend={config.backend!r} requires a CUDA device, got {device!r}"
            )
        # Explicit FA backends are validated lazily on first kernel use by the
        # existing flash loader; fail fast only on obvious device mismatch here.
        return config.backend

    if device.type != "cuda":
        if not config.allow_fallback:
            raise RuntimeError(
                "attention.backend=auto could not select a flash backend on "
                f"{device!r} and allow_fallback=false"
            )
        return "sdpa"

    capability = get_flash_attention_capability()
    backend_id = capability.get("backend_id")
    if isinstance(backend_id, str) and backend_id:
        return backend_id
    if config.allow_fallback:
        return "sdpa"
    raise RuntimeError(
        "attention.backend=auto found no flash-attn backend and allow_fallback=false"
    )


def _use_sdpa() -> bool:
    if _RESOLVED_BACKEND is None:
        raise RuntimeError("vforge ops are not initialized; call initialize_runtime() first")
    return _RESOLVED_BACKEND in {"sdpa", "sdpa_math"}


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    *,
    ulysses: bool = False,
    distributed: bool = False,
    seq_real_len: int | None = None,
    sparsity: TeacherForcingSparsity | SegmentedSparsity | None = None,
    call_policy: AttentionCallPolicy | None = None,
) -> torch.Tensor:
    """Run dense attention under the immutable process ops policy.

    ``sparsity`` selects the teacher-forcing path: it describes the whole mask,
    so it is mutually exclusive with ``attn_mask`` and ``is_causal``. Under
    Ulysses the spec stays global -- the all-to-all hands every rank the full
    sequence with a slice of the heads, so the mask the kernel sees is
    unsharded.
    """
    if distributed or ulysses:
        from vforge.distributed.parallel_state import get_parallel_state

        state = get_parallel_state()
        if state.cp_enabled:
            if call_policy is not None and call_policy.backend != "configured":
                raise NotImplementedError(
                    "context parallel attention does not support per-call backend overrides"
                )
            if seq_real_len is not None:
                raise NotImplementedError(
                    "context parallel attention does not support padded sequences yet"
                )
            from vforge.distributed.context_parallel import context_parallel_attention

            return context_parallel_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
        usp_size = int(state.config.usp_size)
        if usp_size > 1:
            from functools import partial

            from vforge.distributed.ulysses import ulysses_attention

            return ulysses_attention(
                q,
                k,
                v,
                group=state.usp_group,
                world_size=usp_size,
                attention_fn=partial(_local_attention, call_policy=call_policy),
                varlen_attention_fn=partial(
                    _local_varlen_attention,
                    call_policy=call_policy,
                ),
                attn_mask=attn_mask,
                seq_real_len=seq_real_len,
                is_causal=is_causal,
                sparsity=sparsity,
            )
    return _local_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        is_causal=is_causal,
        sparsity=sparsity,
        call_policy=call_policy,
    )


def _local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    sparsity: TeacherForcingSparsity | SegmentedSparsity | None = None,
    call_policy: AttentionCallPolicy | None = None,
) -> torch.Tensor:
    if call_policy is not None and call_policy.cast_qk_to_v:
        q, k = q.to(v.dtype), k.to(v.dtype)
    return unified_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        is_causal=is_causal,
        native_mode=_use_sdpa(),
        sparsity=sparsity,
        call_policy=call_policy,
    )


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    *,
    sparsity: TeacherForcingSparsity | SegmentedSparsity | None = None,
    priority: tuple[torch.nn.attention.SDPBackend, ...] | None = None,
    contiguous: bool = True,
) -> torch.Tensor:
    """Run one local attention call through PyTorch SDPA.

    This bypasses the process-local flash backend without enabling distributed
    layout transforms. It is suitable for rectangular or additive-mask calls
    that cannot use the configured flash kernel.
    """
    return _local_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        is_causal=is_causal,
        sparsity=sparsity,
        call_policy=AttentionCallPolicy.sdpa(priority, contiguous=contiguous),
    )


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    real_len: int,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run padded-varlen attention under the process ops policy."""
    return _local_varlen_attention(
        q,
        k,
        v,
        real_len,
        is_causal=is_causal,
    )


def _local_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    real_len: int,
    is_causal: bool = False,
    call_policy: AttentionCallPolicy | None = None,
) -> torch.Tensor:
    if call_policy is not None and call_policy.backend != "configured":
        raise NotImplementedError("varlen attention does not support per-call backend overrides")
    return unified_attention_varlen(
        q,
        k,
        v,
        real_len,
        is_causal=is_causal,
        native_mode=_use_sdpa(),
    )


__all__ = [
    "OpsConfig",
    "AttentionCallPolicy",
    "SegmentedSparsity",
    "TeacherForcingSparsity",
    "attention",
    "attention_debug_enabled",
    "get_attn_implementation",
    "get_flash_attention_capability",
    "get_ops_config",
    "get_resolved_attention_backend",
    "initialize_ops",
    "reset_ops",
    "sdpa_attention",
    "segmented_available",
    "set_attention_debug",
    "teacher_forcing_available",
    "teacher_forcing_deterministic_available",
    "teacher_forcing_sparsity_stats",
    "unified_attention",
    "unified_attention_varlen",
    "varlen_attention",
]
