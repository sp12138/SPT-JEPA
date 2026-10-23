from functools import partial

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from vforge.checkpointing.offload import ActivationOffloadManager
from vforge.ops._attention import dispatch as attention
from vforge.ops._attention.flash import (
    FA3Backend,
    FA4Backend,
)


class _Controller(ActivationOffloadManager):
    def pack(self, tensor):
        return tensor.detach().clone()

    def unpack(self, packed):
        return packed.clone()


def test_unified_attention_preserves_first_call_only(monkeypatch):
    calls = {"high_level": 0, "forward": 0, "backward": 0}

    def high_level(q, k, v, *, causal=False):
        assert not causal
        calls["high_level"] += 1
        # Keep one saved tensor so non-reentrant checkpoint must replay this
        # branch, matching real FlashAttention autograd behavior.
        return q.square() + k + v

    def low_level_forward(q, k, v, **_kwargs):
        calls["forward"] += 1
        return q + k + v, torch.zeros(q.shape[:-1], dtype=torch.float32)

    def low_level_backward(_q, _k, _v, _out, dout, _lse, **_kwargs):
        calls["backward"] += 1
        return dout, dout, dout

    monkeypatch.setattr(
        attention,
        "_active_flash_backend",
        FA4Backend(
            high_level,
            high_level,
            low_level_forward,
            low_level_backward,
        ),
    )

    def block(q, k, v):
        self_attention = attention.unified_attention(q, k, v)
        cross_attention = attention.unified_attention(q, k, v)
        return self_attention + cross_attention

    inputs = [torch.randn(2, requires_grad=True) for _ in range(3)]
    output = checkpoint(
        block,
        *inputs,
        use_reentrant=False,
        context_fn=_Controller().checkpoint_contexts,
    )
    output.sum().backward()

    assert calls == {"high_level": 2, "forward": 1, "backward": 1}
    assert torch.equal(
        inputs[0].grad,
        torch.ones_like(inputs[0]) + 2 * inputs[0].detach(),
    )
    assert torch.equal(inputs[1].grad, torch.full_like(inputs[1], 2))
    assert torch.equal(inputs[2].grad, torch.full_like(inputs[2], 2))


def test_unified_varlen_attention_preserves_boundaries(monkeypatch):
    calls = {"high_level": 0, "forward": 0, "backward": 0}
    seen_cu_seqlens = []

    def high_level(*_args, **_kwargs):
        calls["high_level"] += 1
        raise AssertionError("checkpoint path should use the low-level binding")

    def low_level_forward(q, k, v, **kwargs):
        calls["forward"] += 1
        seen_cu_seqlens.append(("forward", kwargs["cu_seqlens_q"].cpu().tolist()))
        return q + k + v, torch.zeros(q.shape[:-1], dtype=torch.float32)

    def low_level_backward(_q, _k, _v, _out, dout, _lse, **kwargs):
        calls["backward"] += 1
        seen_cu_seqlens.append(("backward", kwargs["cu_seqlens_q"].cpu().tolist()))
        return dout, dout, dout

    monkeypatch.setattr(
        attention,
        "_active_flash_backend",
        FA4Backend(
            high_level,
            high_level,
            low_level_forward,
            low_level_backward,
        ),
    )

    inputs = [torch.randn(2, 5, 1, 4, requires_grad=True) for _ in range(3)]
    output = checkpoint(
        partial(attention.unified_attention_varlen, real_len=3),
        *inputs,
        use_reentrant=False,
        context_fn=_Controller().checkpoint_contexts,
    )
    output.sum().backward()

    assert calls == {"high_level": 0, "forward": 1, "backward": 1}
    assert seen_cu_seqlens == [
        ("forward", [0, 3, 6]),
        ("backward", [0, 3, 6]),
    ]
    assert torch.count_nonzero(output[:, 3:]) == 0
    for value in inputs:
        assert torch.equal(value.grad[:, :3], torch.ones_like(value.grad[:, :3]))
        assert torch.count_nonzero(value.grad[:, 3:]) == 0


def test_unified_mask_attention_packs_to_varlen_metadata(monkeypatch):
    calls = {"high_level": 0, "forward": 0, "backward": 0}
    seen = []

    def high_level(*_args, **_kwargs):
        calls["high_level"] += 1
        raise AssertionError("checkpoint path should use the low-level binding")

    def low_level_forward(q, k, v, **kwargs):
        calls["forward"] += 1
        seen.append(
            (
                "forward",
                tuple(q.shape),
                kwargs["cu_seqlens_q"].cpu().tolist(),
            )
        )
        return q + k + v, torch.zeros(q.shape[:-1], dtype=torch.float32)

    def low_level_backward(_q, _k, _v, _out, dout, _lse, **kwargs):
        calls["backward"] += 1
        seen.append(
            (
                "backward",
                tuple(_q.shape),
                kwargs["cu_seqlens_q"].cpu().tolist(),
            )
        )
        return (
            torch.ones_like(_q),
            torch.ones_like(_k),
            torch.ones_like(_v),
        )

    monkeypatch.setattr(
        attention,
        "_active_flash_backend",
        FA4Backend(
            high_level,
            high_level,
            low_level_forward,
            low_level_backward,
        ),
    )

    inputs = [torch.randn(2, 5, 1, 4, requires_grad=True) for _ in range(3)]
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    ).view(2, 1, 1, 5)
    output = checkpoint(
        partial(attention.unified_attention, attn_mask=mask),
        *inputs,
        use_reentrant=False,
        context_fn=_Controller().checkpoint_contexts,
    )
    output.sum().backward()

    assert calls == {"high_level": 0, "forward": 1, "backward": 1}
    assert seen == [
        ("forward", (7, 1, 4), [0, 3, 7]),
        ("backward", (7, 1, 4), [0, 3, 7]),
    ]
    for batch, real_len in enumerate((3, 4)):
        assert torch.count_nonzero(output[batch, real_len:]) == 0
        for value in inputs:
            assert torch.count_nonzero(value.grad[batch, real_len:]) == 0


def test_fa3_uniform_tail_mask_packs_to_public_varlen(monkeypatch):
    calls = {"public": 0, "forward": 0, "backward": 0}
    seen = {}

    def public(q, k, v, **kwargs):
        calls["public"] += 1
        seen["q_shape"] = tuple(q.shape)
        seen["cu_seqlens_q"] = kwargs["cu_seqlens_q"].tolist()
        return q + k + v

    def forward(
        *_args,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
    ):
        calls["forward"] += 1
        raise AssertionError("direct path must use public varlen")

    def backward(
        *_args,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        deterministic=False,
    ):
        calls["backward"] += 1
        raise AssertionError("direct path must use public varlen")

    monkeypatch.setattr(
        attention,
        "_active_flash_backend",
        FA3Backend(public, public, forward, backward),
    )
    inputs = [torch.randn(2, 5, 1, 4, requires_grad=True) for _ in range(3)]
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    ).view(2, 1, 1, 5)

    output = attention.unified_attention(*inputs, attn_mask=mask)
    output.sum().backward()

    assert calls == {"public": 1, "forward": 0, "backward": 0}
    assert seen == {
        "q_shape": (7, 1, 4),
        "cu_seqlens_q": [0, 3, 7],
    }
    for batch, real_len in enumerate((3, 4)):
        assert torch.count_nonzero(output[batch, real_len:]) == 0
        for value in inputs:
            assert torch.count_nonzero(value.grad[batch, real_len:]) == 0


def test_unified_mask_attention_rejects_non_prefix_mask():
    inputs = [torch.randn(1, 4, 1, 2) for _ in range(3)]
    mask = torch.tensor([True, False, True, False]).view(1, 1, 1, 4)

    with pytest.raises(ValueError, match="contiguous True prefix"):
        attention.unified_attention(
            *inputs,
            attn_mask=mask,
        )
