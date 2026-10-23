"""LTX reference/clean KV state with immutable views for delayed backward.

Reference keys are pre-RoPE; generated-history keys are post-RoPE. Under USP,
entries hold the full valid sequence for local heads, excluding token padding. Appends
replace tuples, never mutate saved tensors. A denoising forward captures a
view so activation recomputation cannot observe future blocks or ref positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

Pair = tuple[torch.Tensor, torch.Tensor]


class CacheMode(Enum):
    PREFILL_REF = "prefill_ref"
    PREFILL_CLEAN = "prefill_clean"
    DENOISE = "denoise"


@dataclass
class KVCache:
    reference: Pair | None = None
    clean: tuple[Pair, ...] = ()
    readonly: bool = False
    dtype: torch.dtype | None = None
    window_tokens: int = 0
    sink_tokens: int = 0
    _sink: Pair | None = None
    _ring: Pair | None = None
    _cursor: int = 0

    def write(self, key: torch.Tensor, value: torch.Tensor, mode: CacheMode) -> None:
        if self.readonly:
            raise RuntimeError("cannot mutate a denoising KV snapshot")
        if key.shape != value.shape:
            raise ValueError("KV shapes must match")
        # The reference writes into preallocated latent-dtype buffers. RoPE
        # and CUDA RMSNorm can produce FP32 keys even during BF16 forward.
        pair = tuple(self._store_tensor(tensor) for tensor in (key, value))
        if mode == CacheMode.PREFILL_REF:
            if self.reference is not None or self.clean:
                raise RuntimeError("reference must be written once before clean history")
            self.reference = pair
        elif mode == CacheMode.PREFILL_CLEAN:
            if self.reference is None:
                raise RuntimeError("prefill reference before committing history")
            if self.window_tokens:
                self._write_window(pair)
            else:
                self.clean = (*self.clean, pair)
        else:
            raise ValueError("denoising cannot write KV history")

    def __post_init__(self):
        if min(self.window_tokens, self.sink_tokens) < 0:
            raise ValueError("KV window and sink sizes must be non-negative")
        if self.sink_tokens and not self.window_tokens:
            raise ValueError("fixed sink requires a bounded KV window")

    def _write_window(self, pair: Pair) -> None:
        # Copy-on-write preserves snapshots captured by attention/recomputation.
        sink_length = 0 if self._sink is None else self._sink[0].shape[1]
        take = min(self.sink_tokens - sink_length, pair[0].shape[1])
        if take:
            prefix = tuple(t[:, :take].clone() for t in pair)
            self._sink = (
                prefix
                if self._sink is None
                else tuple(
                    torch.cat((old, new), dim=1)
                    for old, new in zip(self._sink, prefix, strict=True)
                )
            )
            pair = tuple(t[:, take:] for t in pair)
        if pair[0].shape[1]:
            used = 0 if self._ring is None else self._ring[0].shape[1]
            buffers = [t.new_zeros(t.shape[0], self.window_tokens, *t.shape[2:]) for t in pair]
            if self._ring is not None:
                for buffer, old in zip(buffers, self._ring, strict=True):
                    buffer[:, :used] = old
            offset = 0
            while offset < pair[0].shape[1]:
                count = min(self.window_tokens - self._cursor, pair[0].shape[1] - offset)
                for buffer, value in zip(buffers, pair, strict=True):
                    buffer[:, self._cursor : self._cursor + count] = value[
                        :, offset : offset + count
                    ]
                self._cursor = (self._cursor + count) % self.window_tokens
                offset += count
            used = min(self.window_tokens, used + pair[0].shape[1])
            self._ring = tuple(t[:, :used].clone() for t in buffers)
        # Match the reference ring's physical order, not chronological sorting.
        self.clean = tuple(pair for pair in (self._sink, self._ring) if pair is not None)

    def _store_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        value = tensor.detach().to(dtype=self.dtype)
        # A trimmed USP tail can be a contiguous view that still owns the full
        # padded storage (notably for a one-token reference). Cache only valid KV.
        if value.untyped_storage().nbytes() > value.numel() * value.element_size():
            value = value.clone()
        return value

    def snapshot(self) -> KVCache:
        return KVCache(self.reference, self.clean, readonly=True, dtype=self.dtype)

    def shard_reference_heads(self, *, heads: int, head_dim: int, size: int, rank: int) -> None:
        """Keep local heads after an unsharded, no-grad reference prefill."""
        if self.readonly or self.reference is None or self.clean:
            raise RuntimeError("shard reference heads immediately after reference prefill")
        if min(heads, head_dim, size) < 1 or not 0 <= rank < size:
            raise ValueError("invalid reference head shard geometry")
        if any(value.shape[-1] != heads * head_dim for value in self.reference):
            raise ValueError("reference KV must contain the full unsharded heads")
        if size == 1:
            return
        shards = []
        for value in self.reference:
            value = value.unflatten(-1, (heads, head_dim))
            value = torch.nn.functional.pad(value, (0, 0, 0, (-heads) % size))
            value = value.chunk(size, dim=2)[rank].flatten(2).contiguous()
            shards.append(self._store_tensor(value))
        self.reference = tuple(shards)


@dataclass
class LayerKVCache:
    video_self: KVCache
    audio_self: KVCache
    video_cond_pe: Pair | None = None
    audio_cond_pe: Pair | None = None

    def snapshot(self) -> LayerKVCache:
        return LayerKVCache(
            self.video_self.snapshot(),
            self.audio_self.snapshot(),
            self.video_cond_pe,
            self.audio_cond_pe,
        )


class ModelKVCache:
    def __init__(
        self,
        num_layers: int,
        *,
        dtype: torch.dtype | None = None,
        video_window_tokens: int = 0,
        audio_window_tokens: int = 0,
        video_sink_tokens: int = 0,
        audio_sink_tokens: int = 0,
    ):
        if num_layers < 1:
            raise ValueError("cache requires at least one layer")
        self.layers = [
            LayerKVCache(
                KVCache(
                    dtype=dtype, window_tokens=video_window_tokens, sink_tokens=video_sink_tokens
                ),
                KVCache(
                    dtype=dtype, window_tokens=audio_window_tokens, sink_tokens=audio_sink_tokens
                ),
            )
            for _ in range(num_layers)
        ]

    def set_cond_pe(self, video_pe: Pair | None, audio_pe: Pair | None) -> None:
        for layer in self.layers:
            layer.video_cond_pe = video_pe
            layer.audio_cond_pe = audio_pe
