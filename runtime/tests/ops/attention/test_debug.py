import io
from contextlib import redirect_stdout

import torch

from vforge.ops._attention import dispatch as attention
from vforge.ops._attention.flash import FA4Backend


def test_attention_debug_prints_rank0_details(monkeypatch):
    attention.set_attention_debug(False)

    def dense(q, k, v, *, causal=False):
        return q + k + v

    def varlen(*_args, **_kwargs):
        raise AssertionError("dense path should not use varlen")

    def private(*_args, **_kwargs):
        raise AssertionError("direct path should not use private ops")

    monkeypatch.setattr(
        attention,
        "_active_flash_backend",
        FA4Backend(dense, varlen, private, private),
    )
    inputs = [torch.randn(2, 4, 1, 3) for _ in range(3)]

    attention.set_attention_debug(True)
    try:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            attention.unified_attention(*inputs, is_causal=True)
        text = buffer.getvalue()
    finally:
        attention.set_attention_debug(False)

    assert "VFORGE_ATTENTION_DEBUG" in text
    assert "backend=flash_attention_4" in text
    assert "path=direct" in text
    assert "kind=dense" in text
    assert "causal=True" in text


def test_attention_debug_disabled_is_silent(monkeypatch):
    attention.set_attention_debug(False)

    def dense(q, k, v, *, causal=False):
        return q + k + v

    monkeypatch.setattr(
        attention,
        "_active_flash_backend",
        FA4Backend(dense, dense, dense, dense),
    )
    inputs = [torch.randn(1, 2, 1, 2) for _ in range(3)]
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        attention.unified_attention(*inputs)
    assert buffer.getvalue() == ""
