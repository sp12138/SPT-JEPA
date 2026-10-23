import torch

import vforge.ops as ops
from vforge.ops._attention import dispatch as attention
from vforge.ops._attention.flash import FA4Backend
from vforge.ops.config import OpsConfig


def _private_forward_should_not_run(*_args, **_kwargs):
    raise AssertionError("direct attention must not use preservation forward")


def _private_backward_should_not_run(*_args, **_kwargs):
    raise AssertionError("direct attention must not use preservation backward")


def test_direct_dense_uses_public_dense_function(monkeypatch):
    calls = []

    def dense(q, k, v, *, causal=False):
        calls.append(("dense", causal))
        return q + k + v

    def varlen(*_args, **_kwargs):
        raise AssertionError("dense attention must not use the varlen function")

    monkeypatch.setattr(
        attention,
        "_active_flash_backend",
        FA4Backend(
            dense,
            varlen,
            _private_forward_should_not_run,
            _private_backward_should_not_run,
        ),
    )
    inputs = [torch.randn(2, 4, 1, 3) for _ in range(3)]

    output = attention.unified_attention(*inputs, is_causal=True)

    assert calls == [("dense", True)]
    assert torch.equal(output, sum(inputs))


def test_direct_packed_varlen_uses_public_varlen_function(monkeypatch):
    seen = {}

    def dense(*_args, **_kwargs):
        raise AssertionError("packed varlen attention must not use dense")

    def varlen(q, k, v, **kwargs):
        seen.update(kwargs)
        return q + k + v

    monkeypatch.setattr(
        attention,
        "_active_flash_backend",
        FA4Backend(
            dense,
            varlen,
            _private_forward_should_not_run,
            _private_backward_should_not_run,
        ),
    )
    inputs = [torch.randn(2, 5, 1, 3) for _ in range(3)]

    output = attention.unified_attention_varlen(
        *inputs,
        real_len=3,
        is_causal=True,
    )

    assert seen["causal"] is True
    assert seen["max_seqlen_q"] == 3
    assert seen["max_seqlen_k"] == 3
    assert seen["cu_seqlens_q"].tolist() == [0, 3, 6]
    assert seen["cu_seqlens_k"].tolist() == [0, 3, 6]
    assert output.shape == inputs[0].shape
    assert torch.count_nonzero(output[:, 3:]) == 0


def test_direct_uniform_tail_mask_packs_to_varlen(monkeypatch):
    seen = {}

    def dense(*_args, **_kwargs):
        raise AssertionError("uniform-tail attention must not use dense")

    def varlen(q, k, v, **kwargs):
        seen["q_shape"] = tuple(q.shape)
        seen.update(kwargs)
        return q + k + v

    monkeypatch.setattr(
        attention,
        "_active_flash_backend",
        FA4Backend(
            dense,
            varlen,
            _private_forward_should_not_run,
            _private_backward_should_not_run,
        ),
    )
    inputs = [torch.randn(2, 5, 1, 3) for _ in range(3)]
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
        ]
    ).view(2, 1, 1, 5)

    output = attention.unified_attention(*inputs, attn_mask=mask)

    assert seen["causal"] is False
    assert seen["q_shape"] == (7, 1, 3)
    assert seen["cu_seqlens_q"].tolist() == [0, 3, 7]
    assert seen["cu_seqlens_k"].tolist() == [0, 3, 7]
    assert seen["max_seqlen_q"] == 5
    assert seen["max_seqlen_k"] == 5
    assert "seqused_q" not in seen
    assert torch.count_nonzero(output[0, 3:]) == 0
    assert torch.count_nonzero(output[1, 4:]) == 0


def test_native_mode_bypasses_flash_backend(monkeypatch):
    def should_not_run(*_args, **_kwargs):
        raise AssertionError("native attention must bypass flash backend")

    monkeypatch.setattr(
        attention,
        "_active_flash_backend",
        FA4Backend(
            should_not_run,
            should_not_run,
            should_not_run,
            should_not_run,
        ),
    )
    inputs = [torch.randn(2, 4, 1, 3) for _ in range(3)]

    output = attention.unified_attention(*inputs, native_mode=True)

    assert output.shape == inputs[0].shape
    assert torch.isfinite(output).all()


def test_public_attention_uses_single_initialized_ops_policy():
    inputs = [torch.randn(2, 4, 1, 3) for _ in range(3)]
    ops.initialize_ops(
        OpsConfig(backend="sdpa", allow_fallback=False),
        device=torch.device("cpu"),
        parallel_state=object(),
    )
    try:
        output = ops.attention(*inputs)
        expected = attention.unified_attention(*inputs, native_mode=True)
        torch.testing.assert_close(output, expected)
        assert ops.get_resolved_attention_backend() == "sdpa"
    finally:
        ops.reset_ops()


def test_qk_cast_policy_preserves_default_and_gradient_flow(monkeypatch):
    seen = []

    def kernel(q, k, v, **kwargs):
        seen.append((q.dtype, k.dtype, v.dtype))
        return q + k + v

    monkeypatch.setattr(ops, "unified_attention", kernel)
    monkeypatch.setattr(ops, "_RESOLVED_BACKEND", "sdpa_math")
    q = torch.randn(1, 3, 2, 4, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, dtype=torch.bfloat16, requires_grad=True)
    ops.attention(q, k, v)
    output = ops.attention(q, k, v, call_policy=ops.AttentionCallPolicy(cast_qk_to_v=True))
    assert seen == [
        (torch.float32, torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16),
    ]
    assert torch.equal(output, q.to(v.dtype) + k.to(v.dtype) + v)
    output.sum().backward()
    for value in (q, k, v):
        assert torch.equal(value.grad, torch.ones_like(value))
