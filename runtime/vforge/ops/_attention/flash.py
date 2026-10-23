"""FlashAttention FA3/FA4 backends.

Dense, packed-varlen and teacher-forcing sparse are the supported kernel inputs.
``seqused`` is not supported yet; uniform-tail masks are packed to varlen in
the public attention layer. Seqused can be added later if packing becomes a
performance bottleneck.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from vforge.checkpointing.offload import ActivationOffloadManager
from vforge.ops._attention.segmented import (
    SegmentedSparsity,
    build_segmented_metadata,
    segmented_mask_mod_cute,
)
from vforge.ops._attention.teacher_forcing import (
    TeacherForcingSparsity,
    build_metadata,
    mask_mod_cute,
)


@dataclass(frozen=True)
class AttentionCallMetadata:
    """Kernel metadata retained when checkpoint replay skips attention."""

    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None
    mask_mod: Callable | None = None
    block_sparse_tensors: object | None = None

    def __post_init__(self) -> None:
        has_mask = self.mask_mod is not None
        has_sparse_tensors = self.block_sparse_tensors is not None
        if has_mask != has_sparse_tensors:
            raise ValueError(
                "sparse attention metadata requires both mask_mod and block_sparse_tensors"
            )

    @property
    def is_sparse(self) -> bool:
        return self.mask_mod is not None

    def direct_kwargs(self) -> dict[str, object]:
        if self.is_sparse:
            return {
                "mask_mod": self.mask_mod,
                "block_sparse_tensors": self.block_sparse_tensors,
            }
        values = {
            "cu_seqlens_q": self.cu_seqlens_q,
            "cu_seqlens_k": self.cu_seqlens_k,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_k": self.max_seqlen_k,
        }
        return {key: value for key, value in values.items() if value is not None}


CheckpointAttentionMetadata = AttentionCallMetadata | None


def _unwrap_public_result(result: object) -> torch.Tensor:
    return result[0] if isinstance(result, tuple) else result  # type: ignore[index]


def _signature_parameters(operation: Callable) -> dict[str, inspect.Parameter]:
    # ``torch.library.custom_op`` returns a CustomOpDef whose public
    # ``__call__`` is only ``(*args, **kwargs)``. Its original function is
    # retained as ``_init_fn`` and carries the dialect-defining signature.
    candidates = (getattr(operation, "_init_fn", None), operation)
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            parameters = inspect.signature(candidate).parameters
        except (TypeError, ValueError):
            continue
        if any(
            parameter.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
            for parameter in parameters.values()
        ):
            return dict(parameters)
    return {}


def _fa3_split_window(forward_op: Callable, backward_op: Callable) -> bool:
    forward_parameters = _signature_parameters(forward_op)
    backward_parameters = _signature_parameters(backward_op)
    has_split_window = all(
        name in parameters
        for parameters in (forward_parameters, backward_parameters)
        for name in ("window_size_left", "window_size_right")
    )
    has_tuple_window = all(
        "window_size" in parameters for parameters in (forward_parameters, backward_parameters)
    )
    if has_split_window == has_tuple_window:
        raise RuntimeError("unsupported FA3 private fwd/bwd signatures")
    return has_split_window


def _require_private_parameters(
    operation: Callable,
    *,
    operation_name: str,
    required: tuple[str, ...],
) -> None:
    parameters = _signature_parameters(operation)
    if not parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        return
    missing = tuple(name for name in required if name not in parameters)
    if missing:
        raise RuntimeError(
            f"{operation_name} does not support required parameters {missing}; "
            "upgrade the FlashAttention runtime"
        )


@dataclass(frozen=True)
class FA3Backend:
    """FlashAttention-3 public kernels plus private checkpoint ops."""

    dense_func: Callable
    varlen_func: Callable
    forward_op: Callable
    backward_op: Callable
    split_window: bool = field(init=False)
    backend_id: str = field(default="flash_attention_3", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "split_window",
            _fa3_split_window(self.forward_op, self.backward_op),
        )

    def run_dense(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        return _unwrap_public_result(self.dense_func(q, k, v, causal=is_causal))

    def run_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        metadata: AttentionCallMetadata,
        is_causal: bool,
    ) -> torch.Tensor:
        return _unwrap_public_result(
            self.varlen_func(
                q,
                k,
                v,
                **metadata.direct_kwargs(),
                causal=is_causal,
            )
        )

    def run_teacher_forcing(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        sparsity: TeacherForcingSparsity,
        need_bwd: bool,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "teacher-forcing attention needs FA4's mask_mod kernels; this "
            "process resolved FlashAttention-3. Run on an SM100-class device."
        )

    def run_segmented(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        sparsity: SegmentedSparsity,
        need_bwd: bool,
    ) -> torch.Tensor:
        del q, k, v, sparsity, need_bwd
        raise NotImplementedError(
            "segmented sparse attention requires FlashAttention-4 on an SM100-class device"
        )

    def checkpoint_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        metadata: CheckpointAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if metadata is not None and metadata.is_sparse:
            raise NotImplementedError(
                "FlashAttention-3 does not support mask_mod/block-sparse preservation"
            )
        cu_q = None if metadata is None else metadata.cu_seqlens_q
        cu_k = None if metadata is None else metadata.cu_seqlens_k
        max_q = None if metadata is None else metadata.max_seqlen_q
        max_k = None if metadata is None else metadata.max_seqlen_k
        # seqused slots remain None: packed-varlen only for now.
        positional = (
            q,
            k,
            v,
            None,
            None,
            None,
            None,
            cu_q,
            cu_k,
            None,
            None,
            None,
            max_q,
            max_k,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            is_causal,
        )
        window_kwargs = (
            {"window_size_left": -1, "window_size_right": -1}
            if self.split_window
            else {"window_size": (-1, -1)}
        )
        result = self.forward_op(
            *positional,
            **window_kwargs,
            softcap=0.0,
        )
        return result[0], result[1]

    def checkpoint_backward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        dout: torch.Tensor,
        auxiliary: torch.Tensor,
        *,
        is_causal: bool,
        metadata: CheckpointAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if metadata is not None and metadata.is_sparse:
            raise NotImplementedError(
                "FlashAttention-3 does not support mask_mod/block-sparse preservation"
            )
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        cu_q = None if metadata is None else metadata.cu_seqlens_q
        cu_k = None if metadata is None else metadata.cu_seqlens_k
        max_q = None if metadata is None else metadata.max_seqlen_q
        max_k = None if metadata is None else metadata.max_seqlen_k
        positional = (
            dout,
            q,
            k,
            v,
            out,
            auxiliary,
            cu_q,
            cu_k,
            None,
            None,
            max_q,
            max_k,
            dq,
            dk,
            dv,
            None,
            is_causal,
        )
        window_kwargs = (
            {"window_size_left": -1, "window_size_right": -1}
            if self.split_window
            else {"window_size": (-1, -1)}
        )
        result = self.backward_op(
            *positional,
            **window_kwargs,
            softcap=0.0,
            deterministic=False,
        )
        if isinstance(result, tuple) and len(result) >= 3:
            return result[0], result[1], result[2]
        return dq, dk, dv


@dataclass(frozen=True)
class FA4Backend:
    """FlashAttention-4 public kernels plus private checkpoint ops."""

    dense_func: Callable
    varlen_func: Callable
    forward_op: Callable
    backward_op: Callable
    deterministic: bool = False
    backend_id: str = field(default="flash_attention_4", init=False)

    def run_dense(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        kwargs = {"causal": is_causal}
        if self.deterministic:
            kwargs["deterministic"] = True
        return _unwrap_public_result(self.dense_func(q, k, v, **kwargs))

    def run_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        metadata: AttentionCallMetadata,
        is_causal: bool,
    ) -> torch.Tensor:
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        kwargs = {
            **metadata.direct_kwargs(),
            "causal": is_causal,
        }
        if self.deterministic:
            kwargs["deterministic"] = True
        return _unwrap_public_result(self.varlen_func(q, k, v, **kwargs))

    def run_teacher_forcing(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        sparsity: TeacherForcingSparsity,
        need_bwd: bool,
    ) -> torch.Tensor:
        # FSDP mixed precision can leave RMSNorm q/k in FP32 while the value
        # projection is BF16. FA4 4.0 requires all three inputs to share a
        # dtype; Taichu's DeepSpeed path presents them in the value dtype.
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        # Metadata is broadcast over batch and heads, so it is built once per
        # (spec, device) and cached; only the tile lists reach the GPU.
        forward_tensors, backward_tensors = build_metadata(
            sparsity,
            device=q.device,
            need_bwd=need_bwd,
            deterministic=self.deterministic,
        )
        return _unwrap_public_result(
            self.dense_func(
                q,
                k,
                v,
                deterministic=self.deterministic,
                mask_mod=mask_mod_cute(sparsity),
                block_sparse_tensors=forward_tensors,
                block_sparse_tensors_bwd=backward_tensors,
            )
        )

    def run_segmented(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        sparsity: SegmentedSparsity,
        need_bwd: bool,
    ) -> torch.Tensor:
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        forward_tensors, backward_tensors = build_segmented_metadata(
            sparsity,
            device=q.device,
            need_bwd=need_bwd,
            deterministic=self.deterministic,
        )
        return _unwrap_public_result(
            self.dense_func(
                q,
                k,
                v,
                deterministic=self.deterministic,
                mask_mod=segmented_mask_mod_cute(sparsity),
                block_sparse_tensors=forward_tensors,
                block_sparse_tensors_bwd=backward_tensors,
            )
        )

    def checkpoint_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
        metadata: CheckpointAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metadata_kwargs = {} if metadata is None else metadata.direct_kwargs()
        result = self.forward_op(
            q,
            k,
            v,
            **metadata_kwargs,
            causal=is_causal,
            window_size_left=None,
            window_size_right=None,
            return_lse=True,
        )
        return result[0], result[1]

    def checkpoint_backward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        dout: torch.Tensor,
        auxiliary: torch.Tensor,
        *,
        is_causal: bool,
        metadata: CheckpointAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        metadata_kwargs = {} if metadata is None else metadata.direct_kwargs()
        return self.backward_op(
            q,
            k,
            v,
            out,
            dout,
            auxiliary,
            **metadata_kwargs,
            softmax_scale=None,
            causal=is_causal,
            softcap=0.0,
            window_size_left=None,
            window_size_right=None,
            deterministic=self.deterministic,
        )


AttentionBackend = FA3Backend | FA4Backend


class _FlashAttentionOffloadFunction(torch.autograd.Function):
    """Concrete FA forward/backward with checkpoint replay preservation."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        manager: ActivationOffloadManager,
        backend: AttentionBackend,
        is_causal: bool,
        metadata: CheckpointAttentionMetadata,
        backward_metadata: CheckpointAttentionMetadata,
    ) -> torch.Tensor:
        ctx.save_for_backward(q, k, v)
        if manager.checkpoint_is_recompute():
            return manager.restore_next_checkpoint_output()

        out, auxiliary = backend.checkpoint_forward(
            q,
            k,
            v,
            is_causal=is_causal,
            metadata=metadata,
        )
        manager.capture_checkpoint_tensors(
            ctx,
            out,
            auxiliary,
            metadata if backward_metadata is None else backward_metadata,
        )
        ctx.manager = manager
        ctx.backend = backend
        ctx.is_causal = is_causal
        ctx.set_materialize_grads(False)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor | None):
        manager = ctx.manager
        try:
            q, k, v = ctx.saved_tensors
            out, auxiliary, metadata = manager.prepare_checkpoint_backward(ctx)
            if dout is None:
                dout = torch.zeros_like(out)
            dq, dk, dv = ctx.backend.checkpoint_backward(
                q,
                k,
                v,
                out,
                dout,
                auxiliary,
                is_causal=ctx.is_causal,
                metadata=metadata,
            )
        finally:
            try:
                manager.release_checkpoint_tensors(ctx)
            finally:
                ctx.manager = None
                ctx.backend = None
        return dq, dk, dv, None, None, None, None, None


def run_flash_attention(
    backend: AttentionBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    metadata: AttentionCallMetadata | None,
    is_causal: bool,
) -> tuple[torch.Tensor, bool]:
    """Run dense/varlen FA, preserving selected checkpoint invocations."""
    manager = ActivationOffloadManager.claim_checkpoint_call()
    if manager is not None:
        return (
            _FlashAttentionOffloadFunction.apply(
                q,
                k,
                v,
                manager,
                backend,
                is_causal,
                metadata,
                None,
            ),
            True,
        )
    if metadata is None:
        return backend.run_dense(q, k, v, is_causal=is_causal), False
    return (
        backend.run_varlen(
            q,
            k,
            v,
            metadata=metadata,
            is_causal=is_causal,
        ),
        False,
    )


def run_teacher_forcing_attention(
    backend: AttentionBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sparsity: TeacherForcingSparsity,
    need_bwd: bool,
) -> tuple[torch.Tensor, bool]:
    """Run FA4 sparse attention through the same concrete offload Function."""
    manager = ActivationOffloadManager.claim_checkpoint_call()
    if manager is not None:
        if not isinstance(backend, FA4Backend):
            raise NotImplementedError("teacher-forcing preservation requires FlashAttention-4")
        _require_private_parameters(
            backend.forward_op,
            operation_name="FA4 private sparse forward",
            required=("mask_mod", "block_sparse_tensors", "return_lse"),
        )
        _require_private_parameters(
            backend.backward_op,
            operation_name="FA4 private sparse backward",
            required=("mask_mod", "block_sparse_tensors", "deterministic"),
        )
        forward_tensors, backward_tensors = build_metadata(
            sparsity,
            device=q.device,
            need_bwd=need_bwd,
            deterministic=backend.deterministic,
        )
        if backward_tensors is None:
            raise RuntimeError("teacher-forcing preservation requires backward metadata")
        mask_mod = mask_mod_cute(sparsity)
        forward_metadata = AttentionCallMetadata(
            mask_mod=mask_mod,
            block_sparse_tensors=forward_tensors,
        )
        backward_metadata = AttentionCallMetadata(
            mask_mod=mask_mod,
            block_sparse_tensors=backward_tensors,
        )
        return (
            _FlashAttentionOffloadFunction.apply(
                q,
                k,
                v,
                manager,
                backend,
                False,
                forward_metadata,
                backward_metadata,
            ),
            True,
        )
    return (
        backend.run_teacher_forcing(
            q,
            k,
            v,
            sparsity=sparsity,
            need_bwd=need_bwd,
        ),
        False,
    )


def run_segmented_attention(
    backend: AttentionBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sparsity: SegmentedSparsity,
    need_bwd: bool,
) -> tuple[torch.Tensor, bool]:
    """Run segmented FA4 through the concrete offload Function when selected."""
    if isinstance(backend, FA4Backend):
        q = q.to(v.dtype)
        k = k.to(v.dtype)
    manager = ActivationOffloadManager.claim_checkpoint_call()
    if manager is not None:
        if not isinstance(backend, FA4Backend):
            raise NotImplementedError("segmented preservation requires FlashAttention-4")
        _require_private_parameters(
            backend.forward_op,
            operation_name="FA4 private segmented forward",
            required=("mask_mod", "block_sparse_tensors", "return_lse"),
        )
        _require_private_parameters(
            backend.backward_op,
            operation_name="FA4 private segmented backward",
            required=("mask_mod", "block_sparse_tensors", "deterministic"),
        )
        forward_tensors, backward_tensors = build_segmented_metadata(
            sparsity,
            device=q.device,
            need_bwd=need_bwd,
            deterministic=backend.deterministic,
        )
        if backward_tensors is None:
            raise RuntimeError("segmented preservation requires backward metadata")
        mask_mod = segmented_mask_mod_cute(sparsity)
        forward_metadata = AttentionCallMetadata(
            mask_mod=mask_mod,
            block_sparse_tensors=forward_tensors,
        )
        backward_metadata = AttentionCallMetadata(
            mask_mod=mask_mod,
            block_sparse_tensors=backward_tensors,
        )
        return (
            _FlashAttentionOffloadFunction.apply(
                q,
                k,
                v,
                manager,
                backend,
                False,
                forward_metadata,
                backward_metadata,
            ),
            True,
        )
    return (
        backend.run_segmented(
            q,
            k,
            v,
            sparsity=sparsity,
            need_bwd=need_bwd,
        ),
        False,
    )


def load_fa3_backend() -> FA3Backend:
    from flash_attn_interface import (
        _flash_attn_backward,
        _flash_attn_forward,
        flash_attn_func,
        flash_attn_varlen_func,
    )

    return FA3Backend(
        flash_attn_func,
        flash_attn_varlen_func,
        _flash_attn_forward,
        _flash_attn_backward,
    )


def load_fa4_backend(*, deterministic: bool = False) -> FA4Backend:
    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

    return FA4Backend(
        flash_attn_func,
        flash_attn_varlen_func,
        _flash_attn_fwd,
        _flash_attn_bwd,
        deterministic=deterministic,
    )
