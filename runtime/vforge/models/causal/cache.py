"""Fixed-capacity KV storage for incremental block-causal inference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class CausalKVCacheLayer:
    """One transformer's K/V buffers with optional sink/recent eviction."""

    key: torch.Tensor
    value: torch.Tensor
    committed_tokens: int = 0
    local_tokens: int = 0

    @property
    def capacity(self) -> int:
        return int(self.key.shape[1])

    def update_and_attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        write_start: int,
        commit: bool,
        attention_fn: AttentionFn,
        frame_tokens: int = 1,
        local_window_frames: int = -1,
        sink_frames: int = 0,
        history_tokens: int | None = None,
    ) -> torch.Tensor:
        """Overwrite current slots and attend committed history plus this write.

        ``history_tokens`` pins the committed extent instead of reading the live
        counter, which lets a caller replay one forward against the exact
        history it originally saw.
        """
        write_start = int(write_start)
        write_end = write_start + int(key.shape[1])
        frame_tokens = int(frame_tokens)
        local_window_frames = int(local_window_frames)
        sink_frames = int(sink_frames)
        if key.shape != value.shape:
            raise ValueError(
                f"key and value shapes must match, got {tuple(key.shape)} and {tuple(value.shape)}"
            )
        if key.dtype != value.dtype:
            raise TypeError(f"key and value dtypes must match, got {key.dtype} and {value.dtype}")
        if query.shape[:2] != key.shape[:2]:
            raise ValueError(
                "query and key must describe the same current block; "
                f"got {tuple(query.shape[:2])} and {tuple(key.shape[:2])}"
            )
        if write_start < 0:
            raise ValueError(f"KV write start must be non-negative, got {write_start}")
        committed_tokens = self.committed_tokens if history_tokens is None else int(history_tokens)
        if local_window_frames == -1 and write_end > self.capacity:
            raise ValueError(
                f"KV write [{write_start}, {write_end}) exceeds capacity {self.capacity}"
            )
        if write_start > committed_tokens:
            raise ValueError(
                "KV writes cannot skip uncommitted absolute slots; "
                f"start={write_start}, committed={committed_tokens}"
            )
        if self.key.shape[0] != key.shape[0] or self.key.shape[2:] != key.shape[2:]:
            raise ValueError(
                f"KV write shape {tuple(key.shape)} is incompatible with cache "
                f"{tuple(self.key.shape)}"
            )
        if self.key.dtype != key.dtype:
            if self.committed_tokens:
                raise TypeError(
                    "KV compute dtype changed after history was committed: "
                    f"cache={self.key.dtype}, current={key.dtype}"
                )
            self.key = self.key.to(dtype=key.dtype)
            self.value = self.value.to(dtype=value.dtype)

        if frame_tokens < 1:
            raise ValueError(f"frame_tokens must be positive, got {frame_tokens}")
        if local_window_frames == 0 or local_window_frames < -1:
            raise ValueError(
                f"local_window_frames must be -1 or positive; got {local_window_frames}"
            )
        if sink_frames < 0:
            raise ValueError(f"sink_frames must be non-negative, got {sink_frames}")
        if local_window_frames == -1:
            attend_end = max(committed_tokens, write_end)
            cached_prefix = self.key[:, :write_start]
            cached_suffix = self.key[:, write_end:attend_end]
            value_prefix = self.value[:, :write_start]
            value_suffix = self.value[:, write_end:attend_end]
            attend_key = torch.cat((cached_prefix, key, cached_suffix), dim=1)
            attend_value = torch.cat((value_prefix, value, value_suffix), dim=1)
            output = attention_fn(
                query,
                attend_key,
                attend_value,
            )
            if commit:
                with torch.no_grad():
                    self.key[:, write_start:write_end].copy_(key.detach())
                    self.value[:, write_start:write_end].copy_(value.detach())
                self.committed_tokens = attend_end
                self.local_tokens = attend_end
            return output

        max_tokens = min(local_window_frames * frame_tokens, self.capacity)
        if key.shape[1] > max_tokens:
            raise ValueError(
                f"KV window must fit the current block; block={key.shape[1]} window={max_tokens}"
            )
        sink_tokens = min(
            sink_frames * frame_tokens,
            max_tokens - int(key.shape[1]),
        )
        history_key = self.key[:, : self.local_tokens]
        history_value = self.value[:, : self.local_tokens]
        combined_key = torch.cat([history_key, key], dim=1)
        combined_value = torch.cat([history_value, value], dim=1)
        if combined_key.shape[1] > max_tokens:
            recent_tokens = max_tokens - sink_tokens
            if sink_tokens:
                combined_key = torch.cat(
                    [combined_key[:, :sink_tokens], combined_key[:, -recent_tokens:]],
                    dim=1,
                )
                combined_value = torch.cat(
                    [combined_value[:, :sink_tokens], combined_value[:, -recent_tokens:]],
                    dim=1,
                )
            else:
                combined_key = combined_key[:, -max_tokens:]
                combined_value = combined_value[:, -max_tokens:]
        output = attention_fn(
            query,
            combined_key,
            combined_value,
        )
        if commit:
            local_tokens = int(combined_key.shape[1])
            self.key[:, :local_tokens].copy_(combined_key)
            self.value[:, :local_tokens].copy_(combined_value)
            self.local_tokens = local_tokens
            self.committed_tokens = max(self.committed_tokens, write_end)
        return output


@dataclass
class CausalKVCache:
    """Per-layer KV cache stack owned by one guidance branch."""

    layers: list[CausalKVCacheLayer]

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_index: int) -> CausalKVCacheLayer:
        return self.layers[layer_index]

    @property
    def committed_tokens(self) -> int:
        if not self.layers:
            return 0
        committed = self.layers[0].committed_tokens
        if any(layer.committed_tokens != committed for layer in self.layers[1:]):
            raise RuntimeError("KV cache layers have inconsistent committed token counts")
        return committed

    def reset(self) -> None:
        for layer in self.layers:
            layer.committed_tokens = 0
            layer.local_tokens = 0


def allocate_kv_cache(
    *,
    num_layers: int,
    batch_size: int,
    capacity: int,
    num_heads: int,
    head_dim: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> CausalKVCache:
    """Allocate a zeroed fixed-capacity cache for every transformer layer."""
    dimensions = {
        "num_layers": num_layers,
        "batch_size": batch_size,
        "capacity": capacity,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }
    invalid = {name: int(value) for name, value in dimensions.items() if int(value) < 1}
    if invalid:
        raise ValueError(f"KV cache dimensions must be positive, got {invalid}")
    shape = (int(batch_size), int(capacity), int(num_heads), int(head_dim))
    return CausalKVCache(
        layers=[
            CausalKVCacheLayer(
                key=torch.zeros(shape, device=device, dtype=dtype),
                value=torch.zeros(shape, device=device, dtype=dtype),
            )
            for _ in range(int(num_layers))
        ]
    )


__all__ = ["AttentionFn", "CausalKVCache", "CausalKVCacheLayer", "allocate_kv_cache"]
