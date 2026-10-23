"""Protect the Stage 0 masks and DMD length paths shared by A/V blocks."""

from dataclasses import replace

import pytest
import torch

import vforge.ops as ops
from vforge.models.ltx.attention import Attention, AttentionOps
from vforge.models.ltx.ops import dmd_transformer_ops
from vforge.models.ltx.transformer import (
    BasicAVTransformerBlock,
    TransformerConfig,
    TransformerOpsConfig,
)
from vforge.models.ltx.transformer_args import TransformerArgs
from vforge.ops import OpsConfig, initialize_ops, reset_ops, sdpa_attention
from vforge.tasks.ltx.train import _stage0_attention_sparsity


@pytest.fixture(autouse=True)
def attention_runtime():
    initialize_ops(
        OpsConfig(backend="sdpa_math", allow_fallback=False),
        device=torch.device("cpu"),
        parallel_state=None,
    )
    try:
        yield
    finally:
        reset_ops()


def _args(length):
    return TransformerArgs(
        x=torch.randn(1, length, 8),
        context=torch.randn(1, 2, 8),
        context_mask=None,
        timesteps=torch.randn(1, 1, 48),
        embedded_timestep=torch.zeros(1, 1, 8),
        positional_embeddings=None,
        cross_positional_embeddings=None,
        cross_scale_shift_timestep=torch.randn(1, 1, 32),
        cross_gate_timestep=torch.ones(1, 1, 8),
        cross_attn_perturbation_mask=torch.ones(1, 1, 1),
        enabled=True,
    )


@pytest.mark.parametrize("mode", ["stage0_padding", "dmd_lengths"])
def test_av_attention_paths_preserve_logical_outputs_and_gradients(mode, monkeypatch):
    torch.manual_seed(19)
    config = TransformerConfig(dim=8, heads=2, d_head=4, context_dim=8)
    block = BasicAVTransformerBlock(
        video=config,
        audio=config,
        ops=TransformerOpsConfig(attention_ops=AttentionOps(attention_function=sdpa_attention)),
    )
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.uniform_(-0.2, 0.2)
    video, audio = _args(3), _args(5)
    expected = block(video, audio)
    sum(value.x.square().sum() for value in expected).backward()
    expected_grads = {name: p.grad.clone() for name, p in block.named_parameters()}
    block.zero_grad(set_to_none=True)
    masks = None
    lengths = []
    if mode == "stage0_padding":
        # Large padding values make an accidentally dropped mask observable.
        video = replace(video, x=torch.cat((video.x, torch.full((1, 1, 8), 17.0)), dim=1))
        audio = replace(audio, x=torch.cat((audio.x, torch.full((1, 1, 8), -23.0)), dim=1))
        masks = _stage0_attention_sparsity(video_length=3, audio_length=5, multiple=2)
    else:
        # Isolate model-level dispatch; collective correctness is tested in distributed tests.
        def sequence_to_head(attention, tensor, length):
            lengths.append(length)
            return tensor.unflatten(-1, (attention.heads, attention.dim_head))[:, :length]

        monkeypatch.setattr(Attention, "_sequence_to_head", sequence_to_head)
        monkeypatch.setattr(Attention, "_head_to_sequence", lambda self, tensor: tensor.flatten(2))
        video = replace(video, global_sequence_length=3)
        audio = replace(audio, global_sequence_length=5)
    actual = block(video, audio, masks=masks)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value.x[:, : reference.x.shape[1]], reference.x)
    sum(value.x[:, :n].square().sum() for value, n in zip(actual, (3, 5), strict=True)).backward()
    for name, parameter in block.named_parameters():
        torch.testing.assert_close(parameter.grad, expected_grads[name], atol=1e-5, rtol=1e-4)
    if mode == "dmd_lengths":
        assert lengths == [3, 3, 3, 5, 5, 5, 3, 5, 5, 5, 3, 3]


@pytest.mark.parametrize(
    "causal,norm_dtype", [(False, torch.float32), (True, torch.float32), (True, None)]
)
def test_bf16_parameters_run_without_autocast(causal, norm_dtype, monkeypatch):
    import copy

    from vforge.models.ltx.causal import CausalLTXModel
    from vforge.models.ltx.modality import Modality
    from vforge.models.ltx.model import LTXModel

    kernel = ops.unified_attention
    calls = []

    def checked_attention(q, k, v, **kwargs):
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        calls.append(q.dtype)
        return kernel(q, k, v, **kwargs)

    monkeypatch.setattr(ops, "unified_attention", checked_attention)
    torch.manual_seed(31)
    model_type = CausalLTXModel if causal else LTXModel
    model = model_type(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=8,
        out_channels=8,
        num_layers=1,
        cross_attention_dim=16,
        audio_num_attention_heads=2,
        audio_attention_head_dim=8,
        audio_in_channels=8,
        audio_out_channels=8,
        audio_cross_attention_dim=16,
        cross_attention_adaln=True,
        ops=dmd_transformer_ops(norm_dtype=norm_dtype),
    ).to(dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.2, 0.2)
    reference = copy.deepcopy(model)
    norm_outputs = []
    for module in (model.norm_out, model.audio_norm_out, model.transformer_blocks[0].attn1.q_norm):
        module.register_forward_hook(
            lambda _module, _args, output: norm_outputs.append(output.dtype)
        )

    def modality(length, dimensions):
        return Modality(
            latent=torch.randn(1, length, 8, dtype=torch.bfloat16),
            sigma=torch.ones(1),
            timesteps=torch.ones(1, length),
            positions=torch.rand(1, dimensions, length, 2) * 7.3,
            context=torch.randn(1, 2, 16, dtype=torch.bfloat16),
        )

    expected, actual = [], []
    # Retain two forwards to exercise multi-block gradient accumulation.
    for length in (4, 5):
        video, audio = modality(length, 3), modality(3, 1)
        if causal:
            # TF packing preserves cache/noising precision; FSDP must leave
            # latent metadata and positions alone; each Linear casts its own input.
            video = replace(video, latent=video.latent.float())
            audio = replace(audio, latent=audio.latent.float())
        with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
            expected.extend(reference(video=video, audio=audio))
        with torch.autocast("cpu", enabled=False):
            actual.extend(model(video=video, audio=audio))
        assert video.positions.dtype == audio.positions.dtype == torch.float32
    assert calls
    assert set(norm_outputs) == {norm_dtype or torch.bfloat16}
    for output, baseline in zip(actual, expected, strict=True):
        assert output.dtype == torch.bfloat16
        torch.testing.assert_close(output, baseline, rtol=0, atol=0)
    sum(value.float().square().mean() for value in expected).backward()
    sum(value.float().square().mean() for value in actual).backward()
    for (name, parameter), (ref_name, baseline) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == ref_name
        torch.testing.assert_close(parameter.grad, baseline.grad, rtol=0, atol=0)


@pytest.mark.parametrize("native_norm", [False, True])
def test_connector_preserves_preprocessing_precision(native_norm):
    from vforge.models.ltx.attention import AttentionOps
    from vforge.models.ltx.embeddings_connector import Embeddings1DConnector

    torch.manual_seed(42)
    connector = Embeddings1DConnector(
        attention_head_dim=8,
        num_attention_heads=2,
        num_layers=1,
        num_learnable_registers=0,
        ops=AttentionOps(norm_dtype=None if native_norm else torch.float32),
    ).to(torch.bfloat16)
    features = torch.randn(1, 4, 16, dtype=torch.bfloat16)
    with torch.autocast("cpu", enabled=False):
        output, _ = connector(features)
    assert output.dtype == (torch.bfloat16 if native_norm else torch.float32)
    assert all(parameter.dtype == torch.bfloat16 for parameter in connector.parameters())
    assert dmd_transformer_ops().connector_attention_ops.norm_dtype is None
