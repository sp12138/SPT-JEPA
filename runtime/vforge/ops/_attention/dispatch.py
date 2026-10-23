"""Hardware-aware attention with optional preserved-kernel offload."""

from __future__ import annotations

import os
from threading import Lock

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from vforge.ops._attention.flash import (
    AttentionBackend,
    AttentionCallMetadata,
    load_fa3_backend,
    load_fa4_backend,
    run_flash_attention,
    run_segmented_attention,
    run_teacher_forcing_attention,
)
from vforge.ops._attention.policy import AttentionCallPolicy
from vforge.ops._attention.segmented import SegmentedSparsity, segmented_dense_mask
from vforge.ops._attention.teacher_forcing import (
    TeacherForcingSparsity,
    teacher_forcing_dense_mask,
)


def _detect_gpu_kind() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        device_name = torch.cuda.get_device_name(torch.cuda.current_device()).lower()
    except Exception:
        return None
    if "h200" in device_name:
        return "h200"
    if "h20" in device_name:
        return "h20"
    if "b200" in device_name:
        return "b200"
    if "b300" in device_name:
        return "b300"
    return None


_active_flash_backend: AttentionBackend | None = None
_flash_attn_import_error: BaseException | None = None
_backend_resolution_attempted = False
_backend_resolution_lock = Lock()
_GPU_KIND_UNRESOLVED = object()
_gpu_kind: str | None | object = _GPU_KIND_UNRESOLVED
_FLASH_ATTENTION_DETERMINISTIC = False
_REQUESTED_FLASH_BACKEND: str | None = None
_SDPA_PRIORITY: tuple[SDPBackend, ...] | None = None
_FLASH_BACKEND_SPECS = {
    "h200": ("flash_attention_3",),
    "h20": ("flash_attention_3",),
    "b200": ("flash_attention_4",),
    "b300": ("flash_attention_4",),
}

_DEBUG_ENABLED = os.environ.get("VFORGE_ATTENTION_DEBUG", "").strip() == "1"


def set_attention_debug(enabled: bool) -> None:
    """Enable or disable rank-0 attention debug prints."""
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = bool(enabled)


def attention_debug_enabled() -> bool:
    return _DEBUG_ENABLED


def set_flash_attention_deterministic(enabled: bool) -> None:
    """Configure FA4's deterministic backward kernel before first use."""
    global _FLASH_ATTENTION_DETERMINISTIC
    if _active_flash_backend is not None:
        active = bool(getattr(_active_flash_backend, "deterministic", False))
        if active != bool(enabled):
            raise RuntimeError(
                "FlashAttention deterministic mode must be configured before "
                "the backend is resolved"
            )
        return
    _FLASH_ATTENTION_DETERMINISTIC = bool(enabled)


def set_flash_attention_backend(backend: str | None) -> None:
    """Pin which flash backend to load, overriding the GPU-kind default.

    Without this the loader picks purely off the device name, so asking for
    ``flash_attention_4`` on a Hopper box would record FA4 as the policy while
    quietly loading FA3. Pinning makes the mismatch fail at load instead.
    """
    global _REQUESTED_FLASH_BACKEND
    requested = None if backend in (None, "auto", "sdpa", "sdpa_math") else backend
    if _active_flash_backend is not None:
        if requested is not None and _active_flash_backend.backend_id != requested:
            raise RuntimeError(
                "the flash backend must be selected before it is resolved; "
                f"{_active_flash_backend.backend_id} is already active"
            )
        return
    _REQUESTED_FLASH_BACKEND = requested


def set_sdpa_backend(backend: str) -> None:
    """Select the kernel used when the process policy routes attention to SDPA."""
    global _SDPA_PRIORITY
    if backend == "sdpa":
        _SDPA_PRIORITY = None
    elif backend == "sdpa_math":
        _SDPA_PRIORITY = (SDPBackend.MATH,)
    else:
        raise ValueError(f"unsupported SDPA backend policy: {backend!r}")


def _is_rank_zero() -> bool:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def _debug_attention(
    *,
    backend: AttentionBackend,
    path: str,
    kind: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> None:
    if not _DEBUG_ENABLED or not _is_rank_zero():
        return
    print(
        "VFORGE_ATTENTION_DEBUG "
        f"backend={backend.backend_id} path={path} kind={kind} "
        f"q={tuple(q.shape)}/{q.dtype} k={tuple(k.shape)}/{k.dtype} "
        f"v={tuple(v.shape)}/{v.dtype} causal={is_causal}",
        flush=True,
    )


def _get_gpu_kind() -> str | None:
    global _gpu_kind
    if _gpu_kind is _GPU_KIND_UNRESOLVED:
        _gpu_kind = _detect_gpu_kind()
    return _gpu_kind


def _resolve_flash_backend() -> AttentionBackend | None:
    """Resolve the hardware-specific backend once, on first explicit use."""
    global _active_flash_backend
    global _backend_resolution_attempted
    global _flash_attn_import_error

    if _active_flash_backend is not None:
        return _active_flash_backend
    with _backend_resolution_lock:
        if _active_flash_backend is not None:
            return _active_flash_backend
        if _backend_resolution_attempted:
            return None

        _backend_resolution_attempted = True

        def fa4_loader() -> AttentionBackend:
            if _FLASH_ATTENTION_DETERMINISTIC:
                return load_fa4_backend(deterministic=True)
            return load_fa4_backend()

        if _REQUESTED_FLASH_BACKEND == "flash_attention_3":
            loader = load_fa3_backend
        elif _REQUESTED_FLASH_BACKEND == "flash_attention_4":
            loader = fa4_loader
        else:
            gpu_kind = _get_gpu_kind()
            if gpu_kind in {"h200", "h20"}:
                loader = load_fa3_backend
            elif gpu_kind in {"b200", "b300"}:
                loader = fa4_loader
            else:
                return None

        if loader is load_fa3_backend and _FLASH_ATTENTION_DETERMINISTIC:
            raise RuntimeError(
                "runtime.deterministic=true is unsupported by FlashAttention-3: "
                "its preserved private backward only supports nondeterministic execution"
            )
        try:
            _active_flash_backend = loader()
        except (ImportError, ModuleNotFoundError) as exc:
            _flash_attn_import_error = exc
        return _active_flash_backend


def _require_flash_backend() -> AttentionBackend:
    backend = _resolve_flash_backend()
    if backend is not None:
        return backend
    gpu_kind = _get_gpu_kind()
    if gpu_kind is not None and _flash_attn_import_error is not None:
        raise ImportError(
            f"{gpu_kind.upper()} missing its flash-attn interface"
        ) from _flash_attn_import_error
    raise RuntimeError("flash attention is unavailable; enable native attention mode")


def get_flash_attention_capability() -> dict[str, object]:
    """Return backend availability and public kernel callables for integrations."""
    backend = _resolve_flash_backend()
    if backend is None:
        return {
            "FLASH_ATTN_3_AVAILABLE": False,
            "FLASH_ATTN_4_AVAILABLE": False,
            "flash_attn_func": None,
            "flash_attn_varlen_func": None,
            "backend_id": None,
        }
    return {
        "FLASH_ATTN_3_AVAILABLE": backend.backend_id == "flash_attention_3",
        "FLASH_ATTN_4_AVAILABLE": backend.backend_id == "flash_attention_4",
        "flash_attn_func": backend.dense_func,
        "flash_attn_varlen_func": backend.varlen_func,
        "backend_id": backend.backend_id,
    }


def get_attn_implementation() -> str:
    spec = _FLASH_BACKEND_SPECS.get(_get_gpu_kind())
    return "sdpa" if spec is None else spec[0]


def _validate_uniform_tail_mask(
    attn_mask: torch.Tensor,
    *,
    batch: int,
    sequence_length: int,
) -> torch.Tensor:
    """Return a ``[batch, sequence_length]`` boolean keep-prefix mask."""
    if attn_mask.dtype != torch.bool or attn_mask.shape[-1] != sequence_length:
        raise ValueError(
            "flash attention masks must be boolean uniform-tail masks with "
            f"last dimension {sequence_length}, got "
            f"shape={tuple(attn_mask.shape)} dtype={attn_mask.dtype}"
        )
    flattened = attn_mask.reshape(-1, sequence_length)
    if flattened.shape[0] == 1 and batch != 1:
        flattened = flattened.expand(batch, -1)
    if flattened.shape[0] != batch:
        raise ValueError(
            "flash attention mask batch does not match attention input: "
            f"mask_batch={flattened.shape[0]}, batch={batch}"
        )
    has_hole = ((~flattened[:, :-1]) & flattened[:, 1:]).any()
    message = (
        "flash attention mask must contain one contiguous True prefix followed by False padding"
    )
    if flattened.device.type == "cuda" and hasattr(torch, "_assert_async"):
        # Keep the production path asynchronous; malformed masks fail on the
        # device without a per-block host synchronization.
        torch._assert_async(~has_hole, message)
    elif bool(has_hole.item()):
        raise ValueError(message)
    return flattened.contiguous()


def _pack_uniform_tail(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep_mask: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    AttentionCallMetadata,
    torch.Tensor,
]:
    """Pack padded Q/K/V into varlen tensors; seqused is intentionally unused."""
    batch, sequence_length, heads, head_dim = q.shape
    lengths = keep_mask.sum(dim=-1, dtype=torch.int32)
    flat_mask = keep_mask.reshape(-1)
    q_packed = q.reshape(batch * sequence_length, heads, head_dim)[flat_mask]
    k_packed = k.reshape(batch * sequence_length, heads, head_dim)[flat_mask]
    v_packed = v.reshape(batch * sequence_length, heads, head_dim)[flat_mask]
    cu_seqlens = torch.empty(batch + 1, device=q.device, dtype=torch.int32)
    cu_seqlens[0] = 0
    cu_seqlens[1:] = lengths.cumsum(0)
    metadata = AttentionCallMetadata(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        # FlashAttention accepts an upper bound. Reusing the padded length
        # avoids synchronizing CUDA merely to read ``lengths.max()``.
        max_seqlen_q=sequence_length,
        max_seqlen_k=sequence_length,
    )
    return q_packed, k_packed, v_packed, metadata, flat_mask


def _scatter_uniform_tail(
    packed: torch.Tensor,
    *,
    batch: int,
    sequence_length: int,
    heads: int,
    head_dim: int,
    flat_mask: torch.Tensor,
) -> torch.Tensor:
    out = packed.new_zeros(batch * sequence_length, heads, head_dim)
    out[flat_mask] = packed
    return out.reshape(batch, sequence_length, heads, head_dim)


def _run_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    metadata: AttentionCallMetadata | None,
    is_causal: bool,
) -> torch.Tensor:
    """Dispatch one dense or packed-varlen FlashAttention call."""
    backend = _require_flash_backend()
    kind = "dense" if metadata is None else "varlen"
    output, preserved = run_flash_attention(
        backend,
        q,
        k,
        v,
        metadata=metadata,
        is_causal=is_causal,
    )
    _debug_attention(
        backend=backend,
        path="checkpoint" if preserved else "direct",
        kind=kind,
        q=q,
        k=k,
        v=v,
        is_causal=is_causal,
    )
    return output


def _run_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    priority: tuple[SDPBackend, ...] | None = None,
    contiguous: bool = True,
) -> torch.Tensor:
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    if contiguous:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
    if priority is not None:
        with sdpa_kernel(list(priority), set_priority=True):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
    else:
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
    out = out.transpose(1, 2)
    return out.contiguous() if contiguous else out


def _run_teacher_forcing_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sparsity: TeacherForcingSparsity,
) -> torch.Tensor:
    """Dispatch one teacher-forcing sparse FA4 call."""
    expected = sparsity.seqlen
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.shape[1] != expected:
            raise ValueError(
                f"teacher-forcing attention is self-attention over the doubled "
                f"sequence: {name} has seqlen {tensor.shape[1]}, sparsity "
                f"describes {expected}"
            )
    backend = _require_flash_backend()
    need_bwd = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
    output, preserved = run_teacher_forcing_attention(
        backend,
        q,
        k,
        v,
        sparsity=sparsity,
        need_bwd=need_bwd,
    )
    _debug_attention(
        backend=backend,
        path="checkpoint" if preserved else "direct",
        kind="teacher_forcing",
        q=q,
        k=k,
        v=v,
        is_causal=False,
    )
    return output


def _run_segmented_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sparsity: SegmentedSparsity,
) -> torch.Tensor:
    """Dispatch rectangular segmented attention after exact shape validation."""
    if q.shape[1] != sparsity.seqlen_q:
        raise ValueError(
            f"query seqlen {q.shape[1]} does not match segmented spec {sparsity.seqlen_q}"
        )
    for name, tensor in (("key", k), ("value", v)):
        if tensor.shape[1] != sparsity.seqlen_kv:
            raise ValueError(
                f"{name} seqlen {tensor.shape[1]} does not match segmented spec "
                f"{sparsity.seqlen_kv}"
            )
    backend = _require_flash_backend()
    need_bwd = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
    output, preserved = run_segmented_attention(
        backend,
        q,
        k,
        v,
        sparsity=sparsity,
        need_bwd=need_bwd,
    )
    _debug_attention(
        backend=backend,
        path="checkpoint" if preserved else "direct",
        kind="segmented",
        q=q,
        k=k,
        v=v,
        is_causal=False,
    )
    return output


def unified_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    native_mode: bool = False,
    sparsity: TeacherForcingSparsity | SegmentedSparsity | None = None,
    call_policy: AttentionCallPolicy | None = None,
) -> torch.Tensor:
    call_policy = call_policy or AttentionCallPolicy()
    use_sdpa = native_mode or call_policy.backend == "sdpa"
    sdpa_priority = call_policy.sdpa_priority if call_policy.backend == "sdpa" else _SDPA_PRIORITY
    if sparsity is not None:
        if attn_mask is not None or is_causal:
            raise ValueError(
                "sparsity already describes the whole mask; pass neither "
                "attn_mask nor is_causal alongside it"
            )
        if use_sdpa:
            dense_mask = (
                segmented_dense_mask(sparsity, device=q.device)
                if isinstance(sparsity, SegmentedSparsity)
                else teacher_forcing_dense_mask(sparsity, device=q.device)
            )
            return _run_sdpa(
                q,
                k,
                v,
                attn_mask=dense_mask,
                is_causal=False,
                priority=sdpa_priority,
                contiguous=call_policy.sdpa_contiguous,
            )
        if isinstance(sparsity, SegmentedSparsity):
            return _run_segmented_attention(q, k, v, sparsity=sparsity)
        return _run_teacher_forcing_attention(q, k, v, sparsity=sparsity)

    if use_sdpa:
        return _run_sdpa(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            priority=sdpa_priority,
            contiguous=call_policy.sdpa_contiguous,
        )

    if attn_mask is None:
        return _run_flash_attention(
            q,
            k,
            v,
            metadata=None,
            is_causal=is_causal,
        )

    batch, sequence_length, heads, head_dim = q.shape
    keep_mask = _validate_uniform_tail_mask(
        attn_mask,
        batch=batch,
        sequence_length=sequence_length,
    )
    q_packed, k_packed, v_packed, metadata, flat_mask = _pack_uniform_tail(
        q,
        k,
        v,
        keep_mask,
    )
    out_packed = _run_flash_attention(
        q_packed,
        k_packed,
        v_packed,
        metadata=metadata,
        is_causal=is_causal,
    )
    return _scatter_uniform_tail(
        out_packed,
        batch=batch,
        sequence_length=sequence_length,
        heads=heads,
        head_dim=head_dim,
        flat_mask=flat_mask,
    )


def unified_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    real_len: int,
    is_causal: bool = False,
    native_mode: bool = False,
) -> torch.Tensor:
    """Run attention without trailing sequence padding."""
    batch, padded_len, heads, head_dim = q.shape
    if real_len == padded_len:
        return unified_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            native_mode=native_mode,
        )
    if real_len <= 0 or real_len > padded_len:
        raise ValueError(f"real_len={real_len} must be in (0, padded_len={padded_len}]")

    if native_mode:
        out_real = unified_attention(
            q[:, :real_len].contiguous(),
            k[:, :real_len].contiguous(),
            v[:, :real_len].contiguous(),
            is_causal=is_causal,
            native_mode=True,
        )
        pad = padded_len - real_len
        return F.pad(out_real, (0, 0, 0, 0, 0, pad)) if pad else out_real

    q_real = (
        q[:, :real_len]
        .contiguous()
        .reshape(
            batch * real_len,
            heads,
            head_dim,
        )
    )
    k_real = (
        k[:, :real_len]
        .contiguous()
        .reshape(
            batch * real_len,
            heads,
            head_dim,
        )
    )
    v_real = (
        v[:, :real_len]
        .contiguous()
        .reshape(
            batch * real_len,
            heads,
            head_dim,
        )
    )
    cu_seqlens = torch.arange(
        batch + 1,
        device=q.device,
        dtype=torch.int32,
    ) * int(real_len)
    metadata = AttentionCallMetadata(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=int(real_len),
        max_seqlen_k=int(real_len),
    )
    out_real = _run_flash_attention(
        q_real,
        k_real,
        v_real,
        metadata=metadata,
        is_causal=is_causal,
    )
    out_real = out_real.reshape(batch, real_len, heads, head_dim)
    pad = padded_len - real_len
    return F.pad(out_real, (0, 0, 0, 0, 0, pad)) if pad else out_real
