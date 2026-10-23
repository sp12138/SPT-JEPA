import pytest
import torch

from vforge.ops._attention import flash
from vforge.ops._attention.flash import (
    AttentionCallMetadata,
    FA3Backend,
    FA4Backend,
)


class _CustomOpWrapper:
    """Model torch.library CustomOpDef's generic public call signature."""

    def __init__(self, function):
        self._init_fn = function

    def __call__(self, *args, **kwargs):
        return self._init_fn(*args, **kwargs)


def test_fa3_backend_checkpoint_ops_use_private_kernel_dialect():
    seen = {}

    def public(*_args, **_kwargs):
        raise AssertionError("checkpoint path must use private ops")

    def forward(
        *args,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
    ):
        q, k, v = args[:3]
        seen["forward"] = {
            "cu_seqlens_q": args[7],
            "seqused_q": args[10],
            "window_size_left": window_size_left,
        }
        return q + k + v, torch.zeros(q.shape[:-1]), None, None

    def backward(
        *args,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        deterministic=False,
    ):
        dout = args[0]
        dq, dk, dv = args[12:15]
        seen["backward"] = {
            "cu_seqlens_q": args[6],
            "seqused_q": args[8],
            "window_size_left": window_size_left,
        }
        dq.copy_(dout)
        dk.copy_(2 * dout)
        dv.copy_(3 * dout)
        return torch.empty(0)

    backend = FA3Backend(
        public,
        public,
        _CustomOpWrapper(forward),
        _CustomOpWrapper(backward),
    )
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    metadata = AttentionCallMetadata(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=3,
        max_seqlen_k=3,
    )
    q = torch.randn(3, 1, 4)
    out, lse = backend.checkpoint_forward(
        q,
        q,
        q,
        is_causal=False,
        metadata=metadata,
    )
    dout = torch.ones_like(out)
    dq, dk, dv = backend.checkpoint_backward(
        q,
        q,
        q,
        out,
        dout,
        lse,
        is_causal=False,
        metadata=metadata,
    )

    assert backend.backend_id == "flash_attention_3"
    assert seen["forward"]["cu_seqlens_q"] is cu_seqlens
    assert seen["forward"]["seqused_q"] is None
    assert seen["forward"]["window_size_left"] == -1
    assert seen["backward"]["cu_seqlens_q"] is cu_seqlens
    assert seen["backward"]["seqused_q"] is None
    assert seen["backward"]["window_size_left"] == -1
    assert torch.equal(dq, dout)
    assert torch.equal(dk, 2 * dout)
    assert torch.equal(dv, 3 * dout)


def test_fa4_backend_checkpoint_ops_use_private_kernel_dialect():
    seen = {}

    def public(*_args, **_kwargs):
        raise AssertionError("checkpoint path must use private ops")

    def forward(q, k, v, **kwargs):
        seen["forward"] = kwargs
        return q + k + v, torch.zeros(q.shape[:-1])

    def backward(q, k, v, out, dout, lse, **kwargs):
        seen["backward"] = kwargs
        return dout, 2 * dout, 3 * dout

    backend = FA4Backend(public, public, forward, backward, deterministic=True)
    q = torch.randn(2, 1, 4)
    out, lse = backend.checkpoint_forward(
        q,
        q,
        q,
        is_causal=True,
        metadata=None,
    )
    dout = torch.ones_like(out)
    dq, dk, dv = backend.checkpoint_backward(
        q,
        q,
        q,
        out,
        dout,
        lse,
        is_causal=True,
        metadata=None,
    )

    assert backend.backend_id == "flash_attention_4"
    assert seen["forward"]["return_lse"] is True
    assert seen["forward"]["window_size_left"] is None
    assert seen["backward"]["causal"] is True
    assert seen["backward"]["window_size_left"] is None
    assert seen["backward"]["deterministic"] is True
    assert torch.equal(dq, dout)
    assert torch.equal(dk, 2 * dout)
    assert torch.equal(dv, 3 * dout)


def test_fa4_public_paths_normalize_qk_to_value_dtype(monkeypatch):
    seen = []

    def public(q, k, v, **_kwargs):
        seen.append((q.dtype, k.dtype, v.dtype))
        return v

    monkeypatch.setattr(flash, "build_metadata", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(flash, "mask_mod_cute", lambda _sparsity: None)
    backend = FA4Backend(public, public, public, public, deterministic=True)
    q = torch.randn(2, 3, 1, 4, dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn(2, 3, 1, 4, dtype=torch.bfloat16)

    dense = backend.run_dense(q, k, v, is_causal=False)
    varlen = backend.run_varlen(
        q,
        k,
        v,
        metadata=AttentionCallMetadata(),
        is_causal=False,
    )
    teacher_forcing = backend.run_teacher_forcing(
        q,
        k,
        v,
        sparsity=object(),
        need_bwd=False,
    )

    assert seen == [(torch.bfloat16, torch.bfloat16, torch.bfloat16)] * 3
    assert dense is varlen is teacher_forcing is v


def test_fa4_backend_checkpoint_ops_use_directional_sparse_metadata():
    seen = {}
    forward_sparse = object()
    backward_sparse = object()
    mask_mod = object()

    def public(*_args, **_kwargs):
        raise AssertionError("checkpoint path must use private ops")

    def forward(q, k, v, **kwargs):
        seen["forward"] = kwargs
        return q + k + v, torch.zeros(q.shape[:-1])

    def backward(q, k, v, out, dout, lse, **kwargs):
        seen["backward"] = kwargs
        return dout, 2 * dout, 3 * dout

    backend = FA4Backend(public, public, forward, backward, deterministic=True)
    forward_metadata = AttentionCallMetadata(
        mask_mod=mask_mod,
        block_sparse_tensors=forward_sparse,
    )
    backward_metadata = AttentionCallMetadata(
        mask_mod=mask_mod,
        block_sparse_tensors=backward_sparse,
    )
    q = torch.randn(2, 1, 4)
    out, lse = backend.checkpoint_forward(
        q,
        q,
        q,
        is_causal=False,
        metadata=forward_metadata,
    )
    backend.checkpoint_backward(
        q,
        q,
        q,
        out,
        torch.ones_like(out),
        lse,
        is_causal=False,
        metadata=backward_metadata,
    )

    assert seen["forward"]["mask_mod"] is mask_mod
    assert seen["forward"]["block_sparse_tensors"] is forward_sparse
    assert seen["backward"]["mask_mod"] is mask_mod
    assert seen["backward"]["block_sparse_tensors"] is backward_sparse
    assert seen["backward"]["deterministic"] is True


def test_fa3_backend_rejects_sparse_checkpoint_metadata():
    def public(*_args, **_kwargs):
        raise AssertionError("sparse FA3 must fail before a kernel call")

    def forward(
        *_args,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
    ):
        raise AssertionError("sparse FA3 must fail before a kernel call")

    def backward(
        *_args,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        deterministic=False,
    ):
        raise AssertionError("sparse FA3 must fail before a kernel call")

    backend = FA3Backend(public, public, forward, backward)
    metadata = AttentionCallMetadata(
        mask_mod=object(),
        block_sparse_tensors=object(),
    )
    q = torch.randn(2, 1, 4)

    with pytest.raises(NotImplementedError, match="FlashAttention-3"):
        backend.checkpoint_forward(
            q,
            q,
            q,
            is_causal=False,
            metadata=metadata,
        )


def test_fa3_backend_supports_legacy_tuple_window_dialect():
    seen = {}

    def public(*_args, **_kwargs):
        raise AssertionError("checkpoint path must use private ops")

    def forward(*args, window_size=(-1, -1), softcap=0.0):
        q, k, v = args[:3]
        seen["forward_window"] = window_size
        return q + k + v, torch.zeros(q.shape[:-1]), None, None

    def backward(
        *args,
        window_size=(-1, -1),
        softcap=0.0,
        deterministic=False,
    ):
        dout = args[0]
        seen["backward_window"] = window_size
        return dout, 2 * dout, 3 * dout, torch.empty(0)

    backend = FA3Backend(public, public, forward, backward)
    q = torch.randn(2, 1, 4)
    out, lse = backend.checkpoint_forward(
        q,
        q,
        q,
        is_causal=False,
        metadata=None,
    )
    dout = torch.ones_like(out)
    dq, dk, dv = backend.checkpoint_backward(
        q,
        q,
        q,
        out,
        dout,
        lse,
        is_causal=False,
        metadata=None,
    )

    assert seen == {
        "forward_window": (-1, -1),
        "backward_window": (-1, -1),
    }
    assert torch.equal(dq, dout)
    assert torch.equal(dk, 2 * dout)
    assert torch.equal(dv, 3 * dout)
