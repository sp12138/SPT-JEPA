import math

import pytest
import torch
import torch.nn.functional as F

from vforge.models.rep_world.codecs.videorae import (
    Compressor3D,
    FrozenVJEPA2Attention,
    Tokenizer1D,
    VideoDecoder,
    VideoRAE,
    VideoRotary,
    VideoTransformer,
    video_frequencies,
)
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


def test_rotary_prefix_offsets_and_frequency_interleaving():
    table = video_frequencies((2, 2, 2), prefix=3, head_dim=64)
    assert table.shape == (11, 32) and table.dtype == torch.complex128
    indices = [(axis, i) for i in range(12) for axis in range(3) if i < (12, 10, 10)[axis]]
    for row, position in [(0, (0, 0, 0)), (2, (2, 2, 2)), (3, (3, 3, 3)), (10, (4, 4, 4))]:
        for column, (axis, i) in enumerate(indices):
            count = (12, 10, 10)[axis]
            omega = 10000.0 ** torch.linspace(0, 1, count, dtype=torch.float64)[i]
            angle = omega * math.pi / 2 * position[axis]
            expected = torch.polar(torch.ones_like(angle), angle)
            torch.testing.assert_close(table[row, column], expected, atol=0, rtol=0)


def test_rotary_meta_materialization_does_not_destroy_complex_phase():
    with torch.device("meta"):
        rotary = VideoRotary((2, 2, 2))
    assert not list(rotary.buffers()) and not rotary.state_dict()
    rotary.to_empty(device="cpu").to(dtype=torch.bfloat16)
    value = torch.randn(1, 8, 2, 64).bfloat16()
    expected = (
        torch.view_as_real(
            torch.view_as_complex(value.float().reshape(1, 8, 2, 32, 2))
            * video_frequencies((2, 2, 2))[:, None]
        )
        .flatten(-2)
        .bfloat16()
    )
    torch.testing.assert_close(rotary(value), expected, atol=0, rtol=0)
    assert rotary._tables[value.device].dtype == torch.complex128


def test_original_vjepa2_rotary_repeats_vector_not_elements():
    value = torch.randn(1, 3, 5, 20)
    position = torch.arange(5)
    omega = 1 / 10000 ** (torch.arange(10).float() / 10)
    frequency = position[:, None] * omega
    real, imag = value.unflatten(-1, (-1, 2)).unbind(-1)
    swapped = torch.stack((-imag, real), -1).flatten(-2)
    expected = value * frequency.cos().repeat(1, 2) + swapped * frequency.sin().repeat(1, 2)
    actual = FrozenVJEPA2Attention(60, 3).rotate_axis(value, position)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    corrected = value * frequency.cos().repeat_interleave(
        2, -1
    ) + swapped * frequency.sin().repeat_interleave(2, -1)
    assert not torch.allclose(actual, corrected)


def test_transformer_gate_and_layer_scaling_match_explicit_reference():
    torch.manual_seed(9)
    module = VideoTransformer(24, 2, 3)
    rotary = VideoRotary((1, 2, 2), head_dim=12)
    source = torch.randn(2, 4, 24)
    hidden = source
    for index, (attn, ffd) in enumerate(zip(module.attn_layer, module.ffd_layer, strict=True)):
        q, k, v, gate = attn.to_qkv(hidden).chunk(4, -1)
        q, k, v = (x.unflatten(-1, (2, 12)) for x in (q, k, v))
        q, k = rotary(attn.q_norm(q.contiguous())), rotary(attn.k_norm(k.contiguous()))
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        output = output.transpose(1, 2).flatten(-2) * gate.sigmoid()
        hidden = hidden + attn.out_proj(output)
        hidden = (hidden + ffd(hidden)) / math.sqrt(index + 1)
    torch.testing.assert_close(module(source, rotary), hidden, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize("variant", ["3d", "1d"])
def test_small_codec_compressor_decoder_and_published_meta_contract(variant):
    torch.manual_seed(13)
    if variant == "3d":
        compressor = Compressor3D(
            teacher_width=12, width=24, channels=4, depth=1, heads=2, grid=(2, 2, 2)
        )
        decoder = VideoDecoder(
            variant, channels=4, width=24, depth=1, heads=2, grid=(2, 2, 2), patch=(2, 4, 4)
        )
        latent = compressor(torch.randn(2, 16, 12))
        assert latent.shape == (2, 4, 2, 2, 2)
    else:
        compressor = Tokenizer1D(
            teacher_width=12, width=24, depth=1, heads=2, tokens=3, grid=(4, 2, 2)
        )
        decoder = VideoDecoder(
            variant,
            channels=24,
            width=24,
            depth=1,
            heads=2,
            grid=(2, 2, 2),
            patch=(2, 4, 4),
            tokens=3,
        )
        with torch.no_grad():
            compressor.mask_token.fill_(0.2)
            decoder.mask_token.fill_(-0.1)
        latent = compressor(torch.randn(2, 16, 12))
        assert latent.shape == (2, 3, 24)
    pixels = decoder(latent)
    assert pixels.shape == (2, 3, 4, 8, 8) and torch.isfinite(pixels).all()
    with torch.device("meta"):
        published = VideoRAE(variant)
    assert all(p.is_meta and not p.requires_grad for p in published.parameters())
    assert not published.training and not list(published.buffers())
    assert not any("feature_aligner" in key for key in published.state_dict())
    if variant == "1d":
        assert published.tokenizer_encoder.tokens == published.decoder.tokens == 512
        assert published.enc_proj.out_features == 32
        assert len(published.teacher_model.feature_layers) == 9
    else:
        assert published.teacher_model.feature_layers == (2, 5, 8, 11)
