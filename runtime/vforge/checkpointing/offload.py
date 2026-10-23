"""Saved-tensor CPU offload for activation-heavy model regions."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from weakref import finalize

import torch


@dataclass(frozen=True)
class ActivationOffloadConfig:
    """Configure saved-tensor CPU offload and transfer overlap."""

    min_tensor_bytes: int = 1 << 20
    pin_memory: bool = True
    max_cpu_bytes: int | None = None
    async_transfer: bool = False


def _normalize_call_indices(
    indices: tuple[int, ...] | None,
    *,
    field_name: str,
) -> tuple[int, ...] | None:
    if indices is None:
        return None
    normalized = tuple(sorted(set(int(index) for index in indices)))
    if any(index < 0 for index in normalized):
        raise ValueError(f"{field_name} must contain non-negative indices")
    return normalized


@dataclass(frozen=True)
class AttentionOffloadPolicy:
    """Configure Native checkpoint-aware attention offload."""

    enable: bool = False
    config: ActivationOffloadConfig = field(default_factory=ActivationOffloadConfig)
    preserve_fa_call_indices: tuple[int, ...] | None = (0,)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preserve_fa_call_indices",
            _normalize_call_indices(
                self.preserve_fa_call_indices,
                field_name="preserve_fa_call_indices",
            ),
        )


@dataclass
class ActivationOffloadStats:
    packed_tensors: int = 0
    packed_bytes: int = 0
    restored_tensors: int = 0
    restored_bytes: int = 0
    kept_tensors: int = 0
    kept_bytes: int = 0
    peak_live_bytes: int = 0

    def as_dict(self) -> dict[str, int | float]:
        gib = float(1024**3)
        return {
            "packed_tensors": self.packed_tensors,
            "packed_bytes": self.packed_bytes,
            "packed_gb": self.packed_bytes / gib,
            "restored_tensors": self.restored_tensors,
            "restored_bytes": self.restored_bytes,
            "restored_gb": self.restored_bytes / gib,
            "kept_tensors": self.kept_tensors,
            "kept_bytes": self.kept_bytes,
            "kept_gb": self.kept_bytes / gib,
            "peak_live_bytes": self.peak_live_bytes,
            "peak_live_gb": self.peak_live_bytes / gib,
        }


@dataclass
class _OffloadedTensor:
    device: torch.device
    tensor: torch.Tensor
    num_bytes: int
    d2h_event: torch.cuda.Event | None = None
    restored: torch.Tensor | None = None
    h2d_event: torch.cuda.Event | None = None


class _CheckpointBlockState:
    """All mutable state owned by one checkpointed block invocation."""

    def __init__(
        self,
        manager: ActivationOffloadManager,
        preserve_call_indices: tuple[int, ...] | None,
    ) -> None:
        self.manager = manager
        self.preserve_call_indices = preserve_call_indices
        self.contexts: list[Any] = []
        self.previous: _CheckpointBlockState | None = None
        self.prefetched = False
        self.recompute = False
        self.call_index = 0
        self.claim_pending = False
        self.recompute_index = 0
        self.prefetch_advanced = False

    def begin_execution(self, *, recompute: bool) -> None:
        self.recompute = recompute
        self.call_index = 0
        self.claim_pending = False
        if recompute:
            self.recompute_index = 0
            self.prefetch_advanced = False
            self.ensure_backward_prefetch()

    def claim_next(self) -> bool:
        if self.claim_pending:
            raise RuntimeError("selected checkpoint call was not consumed")
        index = self.call_index
        self.call_index += 1
        selected = self.preserve_call_indices is None or index in self.preserve_call_indices
        self.claim_pending = selected
        return selected

    def consume_claim(self) -> None:
        if not self.claim_pending:
            raise RuntimeError("checkpoint operation did not claim its call")
        self.claim_pending = False

    def append_context(
        self,
        ctx: Any,
        out: torch.Tensor,
        auxiliary: torch.Tensor,
        metadata: object = None,
    ) -> None:
        ctx._vforge_offload_block = self
        ctx._vforge_packed_out = self.manager.pack(out)
        ctx._vforge_packed_auxiliary = self.manager.pack(auxiliary)
        ctx._vforge_offload_metadata = metadata
        ctx._vforge_restored_out = None
        ctx._vforge_offload_released = False
        self.contexts.append(ctx)

    def next_recompute_context(self) -> Any:
        if self.recompute_index >= len(self.contexts):
            raise RuntimeError("checkpoint activation cache underflow during recompute")
        ctx = self.contexts[self.recompute_index]
        self.recompute_index += 1
        return ctx

    def restore_out(self, ctx: Any) -> torch.Tensor:
        if ctx._vforge_restored_out is None:
            ctx._vforge_restored_out = self.manager.unpack(ctx._vforge_packed_out)
        return ctx._vforge_restored_out

    def packed_tensors(self) -> tuple[_OffloadedTensor, ...]:
        return tuple(
            packed
            for ctx in self.contexts
            for packed in (ctx._vforge_packed_out, ctx._vforge_packed_auxiliary)
            if isinstance(packed, _OffloadedTensor)
        )

    def register_for_backward_prefetch(self) -> None:
        if self.contexts:
            self.manager._register_checkpoint_block(self)

    def ensure_backward_prefetch(self) -> None:
        if self.manager.config.async_transfer:
            self.manager._schedule_checkpoint_block(self)

    def advance_backward_prefetch(self) -> None:
        if self.prefetch_advanced:
            return
        if self.previous is not None:
            self.manager._schedule_checkpoint_block(self.previous)
        self.prefetch_advanced = True

    def release(self, ctx: Any) -> None:
        if ctx._vforge_offload_released:
            return
        self.manager.release(ctx._vforge_packed_out, ctx._vforge_packed_auxiliary)
        ctx._vforge_packed_out = None
        ctx._vforge_packed_auxiliary = None
        ctx._vforge_offload_metadata = None
        ctx._vforge_offload_released = True
        if all(item._vforge_offload_released for item in self.contexts):
            self.contexts.clear()


_ACTIVE_CHECKPOINT: ContextVar[_CheckpointBlockState | None] = ContextVar(
    "vforge_activation_offload_checkpoint",
    default=None,
)


@contextmanager
def _checkpoint_block_context(
    block: _CheckpointBlockState,
    *,
    recompute: bool,
) -> Iterator[None]:
    if _ACTIVE_CHECKPOINT.get() is not None:
        raise RuntimeError(
            "nested manager-aware checkpoint is unsupported; remove either "
            "the outer or inner checkpoint wrapper"
        )
    block.begin_execution(recompute=recompute)
    token = _ACTIVE_CHECKPOINT.set(block)
    try:
        yield
        if not recompute:
            block.register_for_backward_prefetch()
    finally:
        _ACTIVE_CHECKPOINT.reset(token)


class ActivationOffloadManager:
    """Coordinate checkpoint preservation, CPU transfer, and prefetch."""

    def __init__(
        self,
        config: ActivationOffloadConfig | None = None,
        *,
        preserve_call_indices: tuple[int, ...] | None = (0,),
    ) -> None:
        self.config = config or ActivationOffloadConfig()
        self.preserve_call_indices = _normalize_call_indices(
            preserve_call_indices,
            field_name="preserve_call_indices",
        )
        if self.config.async_transfer and not self.config.pin_memory:
            raise ValueError("async activation offload requires pinned CPU memory")
        self._stats = ActivationOffloadStats()
        self._lock = Lock()
        self._live_bytes = 0
        self._d2h_streams: dict[int, torch.cuda.Stream] = {}
        self._h2d_streams: dict[int, torch.cuda.Stream] = {}
        self._checkpoint_blocks: list[_CheckpointBlockState] = []
        self._step_active = False
        self._packed_signatures: dict[
            tuple[tuple[int, ...], tuple[int, ...], str],
            list[int],
        ] = {}

    def checkpoint_contexts(
        self,
    ) -> tuple[AbstractContextManager, AbstractContextManager]:
        """Build checkpoint forward/recompute contexts for one invocation."""
        return self._checkpoint_contexts()

    def _checkpoint_contexts(
        self,
    ) -> tuple[AbstractContextManager, AbstractContextManager]:
        block = _CheckpointBlockState(self, self.preserve_call_indices)
        return (
            _checkpoint_block_context(
                block,
                recompute=False,
            ),
            _checkpoint_block_context(
                block,
                recompute=True,
            ),
        )

    @classmethod
    def claim_checkpoint_call(cls) -> ActivationOffloadManager | None:
        """Claim the next selected operator call from the active invocation."""
        block = _ACTIVE_CHECKPOINT.get()
        if block is None or not block.claim_next():
            return None
        return block.manager

    def checkpoint_is_recompute(self) -> bool:
        """Whether this manager is executing a checkpoint replay."""
        return self._active_checkpoint().recompute

    def capture_checkpoint_tensors(
        self,
        ctx: Any,
        out: torch.Tensor,
        auxiliary: torch.Tensor,
        metadata: object = None,
    ) -> None:
        """Pack one selected operator's tensors onto its autograd context."""
        block = self._active_checkpoint()
        block.consume_claim()
        block.append_context(ctx, out, auxiliary, metadata)

    def restore_next_checkpoint_output(self) -> torch.Tensor:
        """Restore the next selected output during checkpoint replay."""
        block = self._active_checkpoint()
        block.consume_claim()
        return block.restore_out(block.next_recompute_context())

    def prepare_checkpoint_backward(
        self,
        ctx: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        """Restore tensors from an operator's original autograd context."""
        block = self._checkpoint_block(ctx)
        out = block.restore_out(ctx)
        auxiliary = self.unpack(ctx._vforge_packed_auxiliary)
        block.advance_backward_prefetch()
        return out, auxiliary, ctx._vforge_offload_metadata

    def release_checkpoint_tensors(self, ctx: Any) -> None:
        """Release tensors captured on an operator's autograd context."""
        block = self._checkpoint_block(ctx)
        ctx._vforge_restored_out = None
        block.release(ctx)

    def _active_checkpoint(self) -> _CheckpointBlockState:
        block = _ACTIVE_CHECKPOINT.get()
        if block is None or block.manager is not self:
            raise RuntimeError("activation offload manager has no active checkpoint invocation")
        return block

    def _checkpoint_block(self, ctx: Any) -> _CheckpointBlockState:
        block = getattr(ctx, "_vforge_offload_block", None)
        if not isinstance(block, _CheckpointBlockState) or block.manager is not self:
            raise RuntimeError("autograd context does not belong to this offload manager")
        return block

    def pack(self, tensor: torch.Tensor) -> torch.Tensor | _OffloadedTensor:
        """Offload one explicitly selected tensor."""
        return self._pack(tensor)

    def unpack(self, packed: torch.Tensor | _OffloadedTensor) -> torch.Tensor:
        """Restore one tensor returned by :meth:`pack`."""
        return self._unpack(packed)

    def _register_checkpoint_block(self, block: _CheckpointBlockState) -> None:
        if not self.config.async_transfer:
            return
        if not self._step_active:
            raise RuntimeError("activation offload group registered outside begin_step()")
        block.previous = self._checkpoint_blocks[-1] if self._checkpoint_blocks else None
        self._checkpoint_blocks.append(block)

    def begin_step(self) -> None:
        """Start one forward/backward generation."""
        if self._step_active or self._checkpoint_blocks:
            raise RuntimeError("previous activation offload step was not finalized")
        if self._live_bytes:
            raise RuntimeError(
                "previous activation offload CPU tensors are still live: "
                f"{self._live_bytes / 1024**3:.3f} GiB"
            )
        self._step_active = True

    def start_backward_prefetch(self) -> None:
        """Prime the last forward block before backward begins."""
        if self._checkpoint_blocks:
            self._schedule_checkpoint_block(self._checkpoint_blocks[-1])

    def finish_backward_prefetch(self) -> None:
        """Release the per-forward ordering registry after backward."""
        self._clear_checkpoint_blocks()
        self._step_active = False

    def abort_step(self) -> None:
        """Drop registry and prefetched GPU copies after no-backward/error."""
        self._clear_checkpoint_blocks()
        self._step_active = False

    def _schedule_checkpoint_block(self, block: _CheckpointBlockState) -> None:
        if block.prefetched:
            return
        for packed in block.packed_tensors():
            self._start_restore(packed)
        block.prefetched = True

    def _clear_checkpoint_blocks(self) -> None:
        for block in self._checkpoint_blocks:
            self.release(*block.packed_tensors())
            block.contexts.clear()
        self._checkpoint_blocks.clear()

    def release(self, *packed_tensors: object) -> None:
        """Drop prefetched GPU copies after the owning backward finishes."""
        for packed in packed_tensors:
            if isinstance(packed, _OffloadedTensor):
                packed.restored = None
                packed.h2d_event = None

    def pop_stats(self) -> dict[str, object]:
        with self._lock:
            stats = self._stats
            self._stats = ActivationOffloadStats()
            signatures = self._packed_signatures
            self._packed_signatures = {}
            current_live_bytes = self._live_bytes
        result = stats.as_dict()
        result["current_live_bytes"] = current_live_bytes
        result["current_live_gb"] = current_live_bytes / float(1024**3)
        result["packed_signatures"] = [
            {
                "shape": list(shape),
                "stride": list(stride),
                "dtype": dtype,
                "count": values[0],
                "bytes": values[1],
            }
            for (shape, stride, dtype), values in sorted(
                signatures.items(),
                key=lambda item: item[1][1],
                reverse=True,
            )[:32]
        ]
        return result

    def _pack(self, tensor: torch.Tensor) -> torch.Tensor | _OffloadedTensor:
        num_bytes = tensor.numel() * tensor.element_size()
        should_offload = (
            tensor.device.type == "cuda"
            and tensor.layout == torch.strided
            and num_bytes >= self.config.min_tensor_bytes
            and type(tensor.grad_fn).__name__ != "TBackward0"
        )
        if not should_offload:
            with self._lock:
                self._stats.kept_tensors += 1
                self._stats.kept_bytes += num_bytes
            return tensor.detach()

        with self._lock:
            projected_bytes = self._live_bytes + num_bytes
            if (
                self.config.max_cpu_bytes is not None
                and projected_bytes > self.config.max_cpu_bytes
            ):
                raise RuntimeError(
                    "activation offload CPU budget exceeded: "
                    f"requested={num_bytes / 1024**3:.3f} GiB, "
                    f"live={self._live_bytes / 1024**3:.3f} GiB, "
                    f"limit={self.config.max_cpu_bytes / 1024**3:.3f} GiB"
                )
            self._live_bytes = projected_bytes
            self._stats.packed_tensors += 1
            self._stats.packed_bytes += num_bytes
            self._stats.peak_live_bytes = max(
                self._stats.peak_live_bytes,
                self._live_bytes,
            )
            signature = (
                tuple(tensor.shape),
                tuple(tensor.stride()),
                str(tensor.dtype),
            )
            signature_stats = self._packed_signatures.setdefault(
                signature,
                [0, 0],
            )
            signature_stats[0] += 1
            signature_stats[1] += num_bytes

        source = tensor.detach()
        try:
            with torch.no_grad():
                if self.config.pin_memory:
                    cpu_tensor = torch.empty_strided(
                        source.size(),
                        source.stride(),
                        dtype=source.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    if self.config.async_transfer:
                        source_stream = torch.cuda.current_stream(source.device)
                        d2h_stream = self._stream_for(
                            self._d2h_streams,
                            source.device,
                        )
                        d2h_stream.wait_stream(source_stream)
                        with torch.cuda.stream(d2h_stream):
                            cpu_tensor.copy_(source, non_blocking=True)
                            d2h_event = torch.cuda.Event()
                            d2h_event.record(d2h_stream)
                        source.record_stream(d2h_stream)
                    else:
                        cpu_tensor.copy_(source, non_blocking=False)
                        d2h_event = None
                else:
                    cpu_tensor = source.to("cpu")
                    d2h_event = None
        except Exception:
            with self._lock:
                self._live_bytes -= num_bytes
            raise
        finalize(cpu_tensor, self._release_cpu_bytes, num_bytes)
        return _OffloadedTensor(
            tensor.device,
            cpu_tensor,
            num_bytes,
            d2h_event=d2h_event,
        )

    def _unpack(self, packed: torch.Tensor | _OffloadedTensor) -> torch.Tensor:
        if isinstance(packed, torch.Tensor):
            return packed
        if self.config.async_transfer:
            self._start_restore(packed)
            current_stream = torch.cuda.current_stream(packed.device)
            current_stream.wait_event(packed.h2d_event)
            packed.restored.record_stream(current_stream)
            return packed.restored

        restored = packed.tensor.to(packed.device, non_blocking=False)
        with self._lock:
            self._stats.restored_tensors += 1
            self._stats.restored_bytes += packed.num_bytes
        return restored

    def _start_restore(self, packed: _OffloadedTensor) -> None:
        if packed.restored is not None:
            return
        h2d_stream = self._stream_for(self._h2d_streams, packed.device)
        if packed.d2h_event is not None:
            h2d_stream.wait_event(packed.d2h_event)
        with torch.cuda.stream(h2d_stream), torch.no_grad():
            restored = torch.empty_strided(
                packed.tensor.size(),
                packed.tensor.stride(),
                dtype=packed.tensor.dtype,
                device=packed.device,
            )
            restored.copy_(packed.tensor, non_blocking=True)
            h2d_event = torch.cuda.Event()
            h2d_event.record(h2d_stream)
        packed.restored = restored
        packed.h2d_event = h2d_event
        with self._lock:
            self._stats.restored_tensors += 1
            self._stats.restored_bytes += packed.num_bytes

    @staticmethod
    def _stream_for(
        streams: dict[int, torch.cuda.Stream],
        device: torch.device,
    ) -> torch.cuda.Stream:
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        stream = streams.get(device_index)
        if stream is None:
            stream = torch.cuda.Stream(device=device_index)
            streams[device_index] = stream
        return stream

    def _release_cpu_bytes(self, num_bytes: int) -> None:
        with self._lock:
            self._live_bytes -= num_bytes
