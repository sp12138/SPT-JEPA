"""Wan's own side of the tensor-parallel seam, and its parameter names.

Building a sharded linear is covered in tests/distributed. What is Wan's here is
reaching into a diffusers ``FeedForward``, splitting its head count, and the
names the whole block exposes. Names especially, because a pretrained Wan
checkpoint and the state-dict mapper both address parameters by name, so a
rename fails as a missing key long after the model was built.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vforge.models.wan.tensor_parallel as wan_tp
from vforge.models.wan.tensor_parallel import declare_feed_forward_split, local_heads


def test_head_count_is_split_and_refuses_an_uneven_split(monkeypatch) -> None:
    assert local_heads(12) == 12

    state = SimpleNamespace(tp_enabled=True, config=SimpleNamespace(tp_size=4))
    monkeypatch.setattr(wan_tp, "get_parallel_state", lambda: state)
    assert local_heads(12) == 3
    # Silently rounding would hand different ranks different head counts and
    # the projections would stop lining up with the reshape.
    with pytest.raises(ValueError, match="must be divisible by tp_size"):
        local_heads(6)


def test_the_feed_forward_declares_its_split_whatever_tp_size_is() -> None:
    """Same module, same shapes, same names -- plus the two declarations.

    Unconditional on purpose. Gating it on tp being enabled would make these the
    only projections in the model whose declaration depends on how many ranks are
    up, so anything walking the tree for splits would answer differently by
    topology.
    """
    from diffusers.models.attention import FeedForward
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    from vforge.distributed.tensor_parallel import TP_STYLE_ATTR

    with torch.device("meta"):
        ffn = FeedForward(16, inner_dim=32, activation_fn="gelu-approximate")
    up, down = ffn.net[0].proj, ffn.net[2]

    assert declare_feed_forward_split(ffn) is ffn
    assert isinstance(getattr(ffn.net[0].proj, TP_STYLE_ATTR), ColwiseParallel)
    assert isinstance(getattr(ffn.net[2], TP_STYLE_ATTR), RowwiseParallel)
    # The names a pretrained checkpoint addresses, and the shapes it carries.
    assert ffn.net[0].proj.weight.shape == up.weight.shape
    assert ffn.net[2].weight.shape == down.weight.shape


def test_block_parameters_are_unchanged_at_tp1() -> None:
    """A rename here surfaces as a missing key at checkpoint load, not here."""
    from vforge.models.wan.v21.transformer import WanTransformerBlock

    with torch.device("meta"):
        block = WanTransformerBlock(dim=16, ffn_dim=32, num_heads=4, cross_attn_norm=True)

    shapes = {name: tuple(p.shape) for name, p in block.named_parameters()}
    assert shapes == {
        "attn1.to_q.weight": (16, 16),
        "attn1.to_q.bias": (16,),
        "attn1.to_k.weight": (16, 16),
        "attn1.to_k.bias": (16,),
        "attn1.to_v.weight": (16, 16),
        "attn1.to_v.bias": (16,),
        "attn1.to_out.0.weight": (16, 16),
        "attn1.to_out.0.bias": (16,),
        "attn1.norm_q.weight": (16,),
        "attn1.norm_k.weight": (16,),
        "attn2.to_q.weight": (16, 16),
        "attn2.to_q.bias": (16,),
        "attn2.to_k.weight": (16, 16),
        "attn2.to_k.bias": (16,),
        "attn2.to_v.weight": (16, 16),
        "attn2.to_v.bias": (16,),
        "attn2.to_out.0.weight": (16, 16),
        "attn2.to_out.0.bias": (16,),
        "attn2.norm_q.weight": (16,),
        "attn2.norm_k.weight": (16,),
        "ffn.net.0.proj.weight": (32, 16),
        "ffn.net.0.proj.bias": (32,),
        "ffn.net.2.weight": (16, 32),
        "ffn.net.2.bias": (16,),
        "norm2.weight": (16,),
        "norm2.bias": (16,),
        "scale_shift_table": (1, 6, 16),
    }
    assert block.attn1.heads == 4


def test_fusing_projections_under_tp_is_refused(monkeypatch) -> None:
    """Fused forward would look right and the input gradient would be wrong."""
    from vforge.models.wan import attention as attention_module
    from vforge.models.wan.v21 import transformer as transformer_module

    with torch.device("meta"):
        attn = transformer_module.WanAttention(dim=16, heads=4, dim_head=4)

    monkeypatch.setattr(attention_module, "tensor_parallel_enabled", lambda: True)
    with pytest.raises(NotImplementedError, match="fused qkv projections"):
        attn.fuse_projections()
