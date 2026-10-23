"""Three-stream block contracts for native MMDiT."""

from __future__ import annotations

import pytest
import torch

from vforge.distributed.parallel_state import ParallelState
from vforge.models.gen_engine.dit.attention import JointAttention
from vforge.models.gen_engine.dit.block import GenEngineMMDiTBlock
from vforge.models.gen_engine.dit.conditioning import RotaryPositionEmbedding
from vforge.models.gen_engine.dit.config import GenEngineDiTConfig
from vforge.models.gen_engine.dit.outputs import MoEBlockStatistics
from vforge.ops import initialize_ops, reset_ops
from vforge.ops.config import OpsConfig


@pytest.fixture(autouse=True)
def _initialize_sdpa_ops():
    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa", allow_fallback=False),
        device=torch.device("cpu"),
        parallel_state=ParallelState(),
    )
    yield
    reset_ops()


def _config(*, architecture: str = "mmdit_dense") -> GenEngineDiTConfig:
    value = {
        "schema": "vforge.gen_engine.dit",
        "schema_version": 1,
        "architecture": architecture,
        "num_attention_heads": 1,
        "attention_head_dim": 8,
        "hidden_size": 8,
        "num_layers": 3,
        "num_refiner_layers": 0,
        "ffn_dim": 16,
        "in_channels": 1,
        "audio_in_channels": 3,
        "patch_size": [1, 1, 1],
        "text_dim": 4,
        "freq_dim": 4,
        "time_embed_hidden_dim": 8,
        "time_embed_dim": 4,
        "rope_freq_dim": 1,
    }
    if architecture == "mmdit_moe":
        value["moe"] = {
            "start_layer": 1,
            "expert_intermediate_size": 4,
            "main_experts": 3,
            "main_top_k": 2,
            "block_experts": 2,
            "block_top_k": 1,
            "shared_multiplier": 2,
            "shared_gate": "sigmoid",
        }
    return GenEngineDiTConfig.from_dict(value)


def _block_inputs():
    sequence_length = 6
    return {
        "hidden_states": torch.randn(1, sequence_length, 8),
        "time_embedding": torch.randn(1, 4),
        "timestep_indices": torch.zeros(sequence_length, dtype=torch.long),
        "stream_tags": torch.tensor([1, 1, 0, 0, 2, 2]),
        "valid_mask": torch.ones(sequence_length, dtype=torch.bool),
        "visual_categories": torch.tensor([-1, -1, 0, 2, -1, -1]),
        "rotary_emb": RotaryPositionEmbedding(1)(torch.randn(sequence_length, 3)),
    }


def test_dense_block_owns_independent_stream_parameters_and_runs_joint_attention() -> None:
    block = GenEngineMMDiTBlock(_config(), layer_index=0)

    output = block(**_block_inputs())

    assert output.shape == (1, 6, 8)
    assert torch.isfinite(output).all()
    projections = block.attention.projections
    assert (
        projections["text"].to_q.weight.data_ptr() != projections["visual"].to_q.weight.data_ptr()
    )
    assert (
        projections["audio"].to_out.weight.data_ptr()
        != projections["visual"].to_out.weight.data_ptr()
    )
    assert block.adaln_modulation.linear.out_features == 6 * 8 * 3
    assert block.adaln_modulation.num_streams == 3


def test_dense_block_can_override_only_the_visual_ffn_width() -> None:
    payload = _config().to_dict()
    payload["visual_ffn_dim"] = 24
    block = GenEngineMMDiTBlock(
        GenEngineDiTConfig.from_dict(payload),
        layer_index=0,
    )

    assert block.text_ffn.intermediate_size == 16
    assert block.audio_ffn.intermediate_size == 16
    assert block.visual_ffn is not None
    assert block.visual_ffn.intermediate_size == 24


def test_block_computes_fused_stream_modulation_once() -> None:
    block = GenEngineMMDiTBlock(_config(), layer_index=0)
    call_count = {"value": 0}

    def hook(_module, _args, _output) -> None:
        call_count["value"] += 1

    handle = block.adaln_modulation.register_forward_hook(hook)
    try:
        block(**_block_inputs())
    finally:
        handle.remove()

    assert call_count["value"] == 1


def test_moe_policy_keeps_first_visual_ffn_dense() -> None:
    config = _config(architecture="mmdit_moe")

    blocks = [GenEngineMMDiTBlock(config, layer_index=index) for index in range(3)]

    assert not blocks[0].is_moe
    assert blocks[0].visual_ffn is not None
    assert blocks[1].is_moe
    assert blocks[2].is_moe
    assert all(block.text_ffn is not None and block.audio_ffn is not None for block in blocks)


def test_text_queries_and_keys_explicitly_ignore_dit_rope() -> None:
    attention = JointAttention(8, 1, 8, qk_norm_eps=1e-5)
    hidden = torch.randn(1, 3, 8)
    tags = torch.ones(3, dtype=torch.long)
    valid = torch.ones(3, dtype=torch.bool)
    identity_rope = (torch.ones(3, 6), torch.zeros(3, 6))
    arbitrary_rope = (torch.randn(3, 6), torch.randn(3, 6))

    first = attention(hidden, tags, valid, identity_rope)
    second = attention(hidden, tags, valid, arbitrary_rope)

    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("architecture", ["mmdit_dense", "mmdit_moe"])
def test_empty_local_streams_keep_every_fsdp_parameter_in_the_graph(
    architecture: str,
) -> None:
    block = GenEngineMMDiTBlock(_config(architecture=architecture), layer_index=1)
    hidden = torch.randn(1, 2, 8, requires_grad=True)
    output = block(
        hidden_states=hidden,
        time_embedding=torch.randn(1, 4),
        timestep_indices=torch.zeros(2, dtype=torch.long),
        stream_tags=torch.ones(2, dtype=torch.long),
        valid_mask=torch.ones(2, dtype=torch.bool),
        visual_categories=torch.full((2,), -1),
        rotary_emb=(torch.ones(2, 6), torch.zeros(2, 6)),
    )

    output.sum().backward()

    assert block.attention.projections["visual"].to_q.weight.grad is not None
    assert block.audio_ffn.down.weight.grad is not None
    if architecture == "mmdit_moe":
        assert block.visual_moe.router.main_router.weight.grad is not None
        assert all(
            expert.down.weight.grad is not None for expert in block.visual_moe.main_bank.experts
        )
    else:
        assert block.visual_ffn.down.weight.grad is not None


def test_padding_tag_never_contributes_to_visual_routing_counts() -> None:
    block = GenEngineMMDiTBlock(_config(architecture="mmdit_moe"), layer_index=1)
    result = block(
        hidden_states=torch.randn(1, 2, 8),
        time_embedding=torch.randn(1, 4),
        timestep_indices=torch.zeros(2, dtype=torch.long),
        stream_tags=torch.tensor([1, -1]),
        valid_mask=torch.tensor([True, False]),
        visual_categories=torch.tensor([-1, -1]),
        rotary_emb=(torch.ones(2, 6), torch.zeros(2, 6)),
        return_statistics=True,
    )
    statistics = MoEBlockStatistics.from_tensors(1, tuple(result[1:]))

    assert statistics.main.counts.sum().item() == 0
    assert statistics.block.counts.sum().item() == 0
