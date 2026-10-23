from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vforge.models.causal import (
    IncrementalKVAttentionContext,
    IncrementalKVExecutor,
    IncrementalKVGuidanceCaches,
    allocate_incremental_kv_guidance_caches,
    allocate_kv_cache,
    use_incremental_kv_cache,
)


def _cache(capacity: int = 4):
    return allocate_kv_cache(
        num_layers=1,
        batch_size=1,
        capacity=capacity,
        num_heads=1,
        head_dim=1,
        device="cpu",
        dtype=torch.float32,
    )


def test_attention_context_resolves_layer_from_call_context() -> None:
    cache = _cache()
    context = IncrementalKVAttentionContext(layer_index=0)

    with use_incremental_kv_cache(cache):
        assert context.resolve_cache_layer() is cache[0]


def _update(
    layer,
    keys: list[float],
    *,
    write_start: int,
    commit: bool = False,
    attended: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> None:
    key = torch.tensor(keys).view(1, -1, 1, 1)

    def attention(query, cached_key, cached_value):
        if attended is not None:
            attended.append((cached_key.clone(), cached_value.clone()))
        return query

    layer.update_and_attend(
        torch.zeros_like(key),
        key,
        key + 10,
        write_start=write_start,
        commit=commit,
        attention_fn=attention,
    )


def test_absolute_kv_slots_append_overwrite_and_commit() -> None:
    cache = _cache()
    layer = cache[0]
    attended: list[tuple[torch.Tensor, torch.Tensor]] = []

    _update(layer, [1.0, 2.0], write_start=0, commit=True, attended=attended)
    assert cache.committed_tokens == 2
    torch.testing.assert_close(attended[-1][0].flatten(), torch.tensor([1.0, 2.0]))

    _update(layer, [9.0], write_start=2, attended=attended)
    assert cache.committed_tokens == 2
    torch.testing.assert_close(attended[-1][0].flatten(), torch.tensor([1.0, 2.0, 9.0]))

    _update(layer, [7.0], write_start=2, attended=attended)
    assert cache.committed_tokens == 2
    torch.testing.assert_close(attended[-1][0].flatten(), torch.tensor([1.0, 2.0, 7.0]))
    assert layer.key[0, 2, 0, 0].item() == 0.0

    _update(layer, [5.0], write_start=2, commit=True, attended=attended)
    assert cache.committed_tokens == 3
    assert layer.key[0, 2, 0, 0].item() == 5.0


def test_uncommitted_kv_is_differentiable_without_persisting_graph_state() -> None:
    layer = _cache()[0]
    key = torch.tensor([3.0]).view(1, 1, 1, 1).requires_grad_()
    value = torch.tensor([4.0]).view(1, 1, 1, 1).requires_grad_()

    output = layer.update_and_attend(
        torch.ones_like(key),
        key,
        value,
        write_start=0,
        commit=False,
        attention_fn=lambda _query, cached_key, cached_value: cached_key + cached_value,
    )
    output.sum().backward()

    torch.testing.assert_close(key.grad, torch.ones_like(key))
    torch.testing.assert_close(value.grad, torch.ones_like(value))
    assert layer.committed_tokens == 0
    assert not layer.key.requires_grad
    assert not layer.value.requires_grad
    assert torch.count_nonzero(layer.key) == 0
    assert torch.count_nonzero(layer.value) == 0


def test_absolute_kv_slots_reject_gaps_and_capacity_overflow() -> None:
    layer = _cache(capacity=2)[0]
    with pytest.raises(ValueError, match="cannot skip"):
        _update(layer, [0.0], write_start=1)
    with pytest.raises(ValueError, match="exceeds capacity"):
        _update(layer, [0.0], write_start=2)


def test_sliding_window_preserves_sink_and_recent_tokens() -> None:
    layer = _cache(capacity=4)[0]
    attended: list[tuple[torch.Tensor, torch.Tensor]] = []

    for index, value in enumerate((1.0, 2.0, 3.0, 4.0, 5.0)):
        key = torch.tensor([value]).view(1, 1, 1, 1)

        def attention(query, cached_key, cached_value):
            attended.append((cached_key.clone(), cached_value.clone()))
            return query

        layer.update_and_attend(
            torch.zeros_like(key),
            key,
            key + 10,
            write_start=index,
            commit=True,
            attention_fn=attention,
            frame_tokens=1,
            local_window_frames=4,
            sink_frames=1,
        )

    torch.testing.assert_close(
        attended[-1][0].flatten(),
        torch.tensor([1.0, 3.0, 4.0, 5.0]),
    )
    assert layer.committed_tokens == 5
    assert layer.local_tokens == 4


def test_short_clip_shrinks_configured_window_to_cache_capacity() -> None:
    layer = _cache(capacity=3)[0]
    attended: list[tuple[torch.Tensor, torch.Tensor]] = []

    key = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1)

    def attention(query, cached_key, cached_value):
        attended.append((cached_key.clone(), cached_value.clone()))
        return query

    layer.update_and_attend(
        torch.zeros_like(key),
        key,
        key + 10,
        write_start=0,
        commit=True,
        attention_fn=attention,
        frame_tokens=1,
        local_window_frames=30,
        sink_frames=9,
    )

    torch.testing.assert_close(attended[-1][0].flatten(), key.flatten())
    assert layer.committed_tokens == 3
    assert layer.local_tokens == 3


def test_guidance_cache_reset_clears_logical_history() -> None:
    cache = _cache()
    _update(cache[0], [1.0, 2.0], write_start=0, commit=True)

    cache.reset()

    assert cache.committed_tokens == 0
    assert cache[0].local_tokens == 0


class _NeutralExecutorTransformer(nn.Module):
    def __init__(self, patch_size=(1, 1, 1)) -> None:
        super().__init__()
        self.config = SimpleNamespace(patch_size=patch_size)
        attention = nn.Module()
        attention.heads = 2
        attention.inner_dim = 6
        block = nn.Module()
        block.attn1 = attention
        self.blocks = nn.ModuleList([block, block])
        self.weight = nn.Parameter(torch.ones(()))

    @staticmethod
    def rope(hidden_states):
        tokens = hidden_states.shape[2] * hidden_states.shape[3] * hidden_states.shape[4]
        positions = torch.arange(tokens).view(1, tokens, 1, 1)
        return positions, positions


def test_neutral_guidance_cache_lifecycle_isolated_by_branch() -> None:
    executor = IncrementalKVExecutor(
        _NeutralExecutorTransformer(),
        block_forward=lambda _model, hidden, *_args, **_kwargs: hidden,
        usp_size=1,
    )
    caches = allocate_incremental_kv_guidance_caches(
        executor,
        batch_size=1,
        num_cache_tokens=4,
        device="cpu",
        dtype=torch.float32,
        use_cfg=True,
    )

    assert caches.conditional is not caches.unconditional
    with pytest.raises(RuntimeError, match="separate caches"):
        IncrementalKVGuidanceCaches(
            conditional=caches.conditional,
            unconditional=caches.conditional,
        )
