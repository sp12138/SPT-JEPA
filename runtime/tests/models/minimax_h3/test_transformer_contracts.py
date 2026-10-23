"""Tiny CPU contracts for the native MiniMax-H3 transformer."""

from __future__ import annotations

import pytest
import torch
from diffusers.models.transformers import transformer_minimax_h3 as upstream

from vforge.models.minimax_h3.native.model import MiniMaxH3Model as NativeMiniMaxH3Model
from vforge.models.minimax_h3.native.transformer import (
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3VforgeAttnProcessor,
)
from vforge.models.minimax_h3.native.transformer import (
    MiniMaxH3Transformer3DModel as NativeMiniMaxH3Transformer3DModel,
)


@pytest.fixture(autouse=True)
def _initialize_sdpa_ops():
    from vforge.distributed.parallel_state import ParallelState
    from vforge.ops import initialize_ops, reset_ops
    from vforge.ops.config import OpsConfig

    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa", allow_fallback=False),
        device=torch.device("cpu"),
        parallel_state=ParallelState(),
    )
    yield
    reset_ops()


def _config() -> dict:
    return {
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "hidden_size": 24,
        "num_layers": 2,
        "num_refiner_layers": 2,
        "ffn_dim": 32,
        "in_channels": 4,
        "audio_in_channels": 6,
        "patch_size": (1, 2, 2),
        "text_dim": 8,
        "freq_dim": 8,
        "time_embed_hidden_dim": 24,
        "time_embed_dim": 16,
        "rope_freq_dim": 2,
    }


def _inputs() -> dict:
    generator = torch.Generator().manual_seed(0)
    num_text, num_audio, num_video = 4, 6, 8
    sequence_length = num_text + num_audio + num_video
    text_indices = torch.arange(num_text)
    audio_indices = torch.arange(num_text, num_text + num_audio)
    video_indices = torch.arange(num_text + num_audio, sequence_length)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = 1
    token_tags[audio_indices] = 2
    token_tags[video_indices] = 0
    timestep_indices = torch.zeros(sequence_length, dtype=torch.long)
    timestep_indices[audio_indices] = 1
    position_ids = torch.zeros(sequence_length, 3)
    position_ids[:, 0] = torch.arange(sequence_length)
    return {
        "hidden_states": torch.randn(2, num_video, 16, generator=generator),
        "audio_hidden_states": torch.randn(2, num_audio, 6, generator=generator),
        "encoder_hidden_states": torch.randn(2, num_text, 8, generator=generator),
        "timestep": torch.tensor([0.7, 0.3]),
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": position_ids,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
    }


def test_native_forward_matches_pinned_diffusers_at_sp1() -> None:
    torch.manual_seed(1)
    reference = upstream.MiniMaxH3Transformer3DModel(**_config()).eval()
    model = NativeMiniMaxH3Transformer3DModel(**_config()).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    inputs = _inputs()
    with torch.no_grad():
        expected = reference(**inputs)
        actual = model(**inputs)
    torch.testing.assert_close(actual.sample, expected.sample, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        actual.audio_sample,
        expected.audio_sample,
        atol=1e-5,
        rtol=1e-5,
    )


def test_h3_state_dict_names_remain_upstream_compatible() -> None:
    reference = upstream.MiniMaxH3Transformer3DModel(**_config())
    model = NativeMiniMaxH3Transformer3DModel(**_config())
    assert set(model.state_dict()) == set(reference.state_dict())


def test_native_routes_token_refiner_through_vforge_attention() -> None:
    model = NativeMiniMaxH3Transformer3DModel(**_config())

    assert all(
        isinstance(block.attn.processor, MiniMaxH3VforgeAttnProcessor)
        for block in model.token_refiner.refiner_blocks
    )


def test_h3_transformer_is_the_explicit_fsdp_root() -> None:
    from vforge.distributed.parallelize import (
        MixedPrecisionConfig,
        _resolve_fsdp_precision_contract,
    )

    transformer = NativeMiniMaxH3Transformer3DModel(**_config())
    model = NativeMiniMaxH3Model(transformer)

    assert model.fsdp_execution_roots() == (transformer,)
    contract = transformer.get_fsdp_precision_contract()
    assert contract.preserve_forward_input_paths[0] == "."
    assert contract.preserve_forward_input_paths[1] == "norm_out"
    # The contract cannot name preserved modules: `_keep_in_fp32_modules`
    # already names the fp32 leaves for storage dtype, and the resolver reads
    # that list. Two lists could disagree, and a parameter cast to fp32 that
    # FSDP then runs under mixed policy is exactly the disagreement that stays
    # silent.
    resolved = _resolve_fsdp_precision_contract(transformer, config=MixedPrecisionConfig())
    assert resolved.preserve_module_paths == (
        "proj_in",
        "audio_proj_in",
        "time_embedder",
        "rope",
        "proj_out",
        "audio_proj_out",
    )
    assert resolved.preserve_forward_input_paths == contract.preserve_forward_input_paths


def test_native_h3_forward_loss_and_gradients_match_upstream() -> None:
    torch.manual_seed(7)
    reference = upstream.MiniMaxH3Transformer3DModel(**_config())
    native = NativeMiniMaxH3Transformer3DModel(**_config())
    native.load_state_dict(reference.state_dict(), strict=True)
    inputs = _inputs()

    reference_output = reference(**inputs)
    reference_loss = (
        reference_output.sample.square().mean() + reference_output.audio_sample.square().mean()
    )
    reference_loss.backward()
    native_output = native(**inputs)
    native_loss = native_output.sample.square().mean() + native_output.audio_sample.square().mean()
    native_loss.backward()

    torch.testing.assert_close(native_output.sample, reference_output.sample)
    torch.testing.assert_close(native_output.audio_sample, reference_output.audio_sample)
    torch.testing.assert_close(native_loss, reference_loss)
    reference_parameters = dict(reference.named_parameters())
    native_parameters = dict(native.named_parameters())
    assert native_parameters.keys() == reference_parameters.keys()
    for name, parameter in native_parameters.items():
        torch.testing.assert_close(parameter.grad, reference_parameters[name].grad)


def test_h3_rope_rebuilds_non_persistent_buffer_after_meta_materialization() -> None:
    from vforge.distributed.parallelize import materialize_meta_module

    with torch.device("meta"):
        rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=2, rope_theta=10000.0)
    assert rope.inv_freq.is_meta
    materialize_meta_module(rope, torch.device("cpu"))
    torch.testing.assert_close(
        rope.inv_freq,
        torch.tensor([1.0, 0.01]),
    )


def test_h3_varlen_attention_does_not_also_pass_a_padding_mask() -> None:
    processor = MiniMaxH3VforgeAttnProcessor()
    processor._pad_attn_mask = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    processor._seq_real_len = 3
    assert processor._resolve_attention_mask(None) is None
    assert processor._resolve_attention_mask(torch.ones(1, 1, 1, 4)) is None
