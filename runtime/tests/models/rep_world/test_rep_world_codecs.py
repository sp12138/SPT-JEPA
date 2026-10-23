from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vforge.models.rep_world.codecs.codec import FrozenCodec
from vforge.models.rep_world.codecs.eupe import EUPEBackbone
from vforge.models.rep_world.codecs.vrae import (
    TemporalAttentionPool,
    VRAEDecoder,
    VRAEDecoderConfig,
)
from vforge.models.rep_world.config import codec_geometry
from vforge.models.rep_world.model import pack_states, unpack_states
from vforge.ops import OpsConfig, initialize_ops, reset_ops


@pytest.fixture(autouse=True)
def ops():
    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa_math", allow_fallback=False),
        device=torch.device("cpu"),
        parallel_state=None,
    )
    yield
    reset_ops()


def test_temporal_pool_is_only_along_time():
    torch.manual_seed(17)
    pool = TemporalAttentionPool(12, 4, num_heads=3)
    with torch.no_grad():
        pool.query.normal_()
        pool.time_bias.normal_()
    x = torch.randn(2, 8, 12, 2, 3)
    output = pool(x)
    assert output.shape == (2, 2, 12, 2, 3)
    expected = torch.empty_like(output)
    for b in range(2):
        for state in range(2):
            for y in range(2):
                for z in range(3):
                    tokens = x[b, state * 4 : (state + 1) * 4, :, y, z]
                    key = pool.key(pool.norm_k(tokens)).view(4, 3, 4).transpose(0, 1)
                    value = pool.value(tokens).view(4, 3, 4).transpose(0, 1)
                    weights = (pool.query[0] * key).sum(-1) / 2 + pool.time_bias
                    reduced = (weights.softmax(-1)[..., None] * value).sum(1).flatten()
                    expected[b, state, :, y, z] = pool.norm_out(pool.proj(reduced))
    torch.testing.assert_close(output, expected)
    changed = x.clone()
    changed[:, 4:, :, 0, 0] += 10
    torch.testing.assert_close(pool(changed)[:, 0], output[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(pool(changed)[:, :, :, 1], output[:, :, :, 1], atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["chunk_causal", "full"])
def test_vrae_decoder_meta_shape_and_prefix_semantics(mode):
    cfg = VRAEDecoderConfig(
        input_dim=8,
        hidden_size=24,
        depth=2,
        num_heads=4,
        image_size=(8, 8),
        patch_size=4,
        attention_mode=mode,
    )
    torch.manual_seed(9)
    decoder = VRAEDecoder(cfg).eval()
    nn.init.normal_(decoder.decoder_spatial_pos_embed, std=0.02)
    with torch.device("meta"):
        meta = VRAEDecoder(cfg)
    meta.load_state_dict(decoder.state_dict(), assign=True, strict=True)
    z = torch.randn(1, 3, 8, 2, 2)
    output = decoder(z)
    assert output.shape == (1, 12, 3, 8, 8)
    torch.testing.assert_close(meta(z), output, atol=0, rtol=0)
    future = z.clone()
    future[:, 2] += 100
    if mode == "chunk_causal":
        torch.testing.assert_close(decoder(future)[:, :8], output[:, :8], atol=0, rtol=0)
        cache, pieces = None, []
        for state in z.split(1, 1):
            rgb, cache = decoder(state, past_key_values=cache, return_key_values=True)
            pieces.append(rgb)
        assert len(cache) == cfg.depth
        assert cache[0][0].shape == (1, 12, 4, 6)
        torch.testing.assert_close(torch.cat(pieces, 1), output, atol=1e-6, rtol=1e-5)
    else:
        assert not torch.allclose(decoder(future)[:, :8], output[:, :8])
        _, cache = decoder(z[:, :1], return_key_values=True)
        with pytest.raises(ValueError, match="cannot stream"):
            decoder(z[:, 1:2], past_key_values=cache)


def test_eupe_nonaffine_k7_fusion_and_key_bias():
    torch.manual_seed(81)
    model = EUPEBackbone(width=24, depth=3, heads=3, patch=4, layers=(1, 2)).eval()
    for parameter in (model.cls_token, model.storage_tokens, model.mask_token):
        nn.init.normal_(parameter, std=0.02)
    assert len(list(model.norm.parameters())) == 0
    selected = []
    handles = [
        block.register_forward_hook(lambda _, _args, y: selected.append(y))
        for block in model.blocks
    ]
    frames = torch.rand(2, 3, 8, 8)
    result = model(frames)
    assert result.shape == (2, 4, 24)
    expected = sum(F.layer_norm(y, (24,), eps=1e-5)[:, 5:] for y in selected[1:])
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    for handle in handles:
        handle.remove()
    with torch.no_grad():
        model.blocks[0].attn.qkv.bias[24:48].add_(1e5)
    torch.testing.assert_close(model(frames), result, atol=0, rtol=0)


class FakeCodec(nn.Module):
    def __init__(self, geo, *, mode="full"):
        super().__init__()
        self.geo = geo
        self.dummy = nn.Parameter(torch.zeros(()))
        self.decoder = SimpleNamespace(config=SimpleNamespace(attention_mode=mode))
        self.inputs = []

    def encode(self, x):
        geo = self.geo
        index = torch.arange(geo.tokens_per_state * geo.channels, dtype=x.dtype)
        tokens = index.reshape(1, geo.tokens_per_state, geo.channels).expand(x.shape[0], -1, -1)
        if geo.family == "vrae":
            return tokens.reshape(x.shape[0], 1, geo.grid, geo.grid, geo.channels).permute(
                0, 1, 4, 2, 3
            )
        return tokens

    def decode(self, x):
        self.inputs.append(x.clone())
        if self.geo.family == "vrae":
            return x.new_full((x.shape[0], x.shape[1] * 4, 3, 256, 256), x.shape[1] / 10)
        return x.new_full((x.shape[0], 3, 16, 256, 256), 0.2)


@pytest.mark.parametrize(
    "family,variant",
    [("videorae", "3d"), ("videorae", "1d"), ("vrae", "eupe"), ("vrae", "vjepa2_1")],
)
def test_codec_plane_channel_order_statistics_and_decode(family, variant):
    geo = replace(codec_geometry(family, variant), channels=2)
    model = FakeCodec(geo)
    codec = FrozenCodec(model, geo, mean=torch.tensor([1.0, 2.0]), std=torch.tensor([2.0, 4.0]))
    assert not model.training and not model.dummy.requires_grad
    rgb = torch.zeros(2, 3, geo.rgb_frames * 2, 256, 256, requires_grad=True)
    packed = codec.encode(rgb)
    assert packed.shape == (2, geo.packed_channels, 2, 16, 16)
    assert not packed.requires_grad
    states = unpack_states(packed, geo)
    torch.testing.assert_close(pack_states(states), packed, atol=0, rtol=0)
    raw = codec._statistics(states, inverse=True)
    expected = (
        torch.arange(geo.tokens_per_state * 2).reshape(geo.planes, 16, 16, 2).permute(3, 0, 1, 2)
    )
    torch.testing.assert_close(raw[0, 0], expected.float(), atol=0, rtol=0)
    decoded = codec.decode(packed)
    assert decoded.shape == rgb.shape and not decoded.requires_grad
    decoder_input = model.inputs[0]
    if family == "videorae" and variant == "1d":
        torch.testing.assert_close(decoder_input[0], expected.flatten(1).T.float())
    with pytest.raises(ValueError, match="complete native states"):
        codec.encode(rgb[:, :, :-1])


def test_full_decoder_stream_recomputes_prefix_but_never_overwrites_history():
    geo = replace(codec_geometry("vrae", "vjepa2_1"), channels=2)
    model = FakeCodec(geo)
    codec = FrozenCodec(model, geo)
    states = torch.randn(1, 2, 3, 16, 16)
    stream = codec.stream()
    stream.prime(states[:, :, :1])
    output = stream.step(states[:, :, 1:])
    assert [x.shape[1] for x in model.inputs] == [1, 2, 3]
    torch.testing.assert_close(output[:, :, :4], torch.full_like(output[:, :, :4], -0.6))
    torch.testing.assert_close(output[:, :, 4:], torch.full_like(output[:, :, 4:], -0.4))
    other = codec.stream()
    assert other.history is None and other.cache is None
