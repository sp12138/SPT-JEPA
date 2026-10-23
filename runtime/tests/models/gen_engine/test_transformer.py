"""Focused CPU contracts for the generation-engine scratch transformer."""

from __future__ import annotations

import importlib.util
import json

import pytest
import torch

try:
    _dit_available = importlib.util.find_spec("vforge.models.gen_engine.dit.model") is not None
except ModuleNotFoundError as exc:
    if exc.name not in {"vforge.models.gen_engine.dit", "vforge.models.gen_engine.dit.model"}:
        raise
    _dit_available = False

if _dit_available:
    from vforge.models.gen_engine.dit import (
        GenEngineModel,
        GenEngineTransformer3DModel,
        initialize_scratch_transformer,
    )
else:
    GenEngineModel = GenEngineTransformer3DModel = initialize_scratch_transformer = None

pytestmark = pytest.mark.usefixtures("gen_engine_dit")


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
        "schema": "vforge.gen_engine.dit",
        "schema_version": 1,
        "architecture": "mmdit_dense",
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "hidden_size": 16,
        "num_layers": 1,
        "num_refiner_layers": 1,
        "ffn_dim": 24,
        "in_channels": 2,
        "audio_in_channels": 3,
        "patch_size": (1, 1, 1),
        "text_dim": 5,
        "freq_dim": 8,
        "time_embed_hidden_dim": 16,
        "time_embed_dim": 8,
        "rope_freq_dim": 1,
        "max_reference_index": 4,
    }


def _moe_config(**overrides) -> dict:
    config = {
        **_config(),
        "architecture": "mmdit_moe",
        "num_layers": 2,
        "moe": {
            "start_layer": 1,
            "expert_intermediate_size": 8,
            "main_experts": 4,
            "main_top_k": 2,
            "block_experts": 3,
            "block_top_k": 2,
            "shared_multiplier": 2,
            "shared_gate": "sigmoid",
        },
    }
    config.update(overrides)
    return config


def _inputs(num_video: int = 2, num_audio: int = 2) -> dict:
    generator = torch.Generator().manual_seed(9)
    num_text = 2
    sequence_length = num_text + num_video + num_audio
    text_indices = torch.arange(num_text)
    video_indices = torch.arange(num_text, num_text + num_video)
    audio_indices = torch.arange(num_text + num_video, sequence_length)
    token_tags = torch.cat(
        (
            torch.ones(num_text, dtype=torch.long),
            torch.zeros(num_video, dtype=torch.long),
            torch.full((num_audio,), 2, dtype=torch.long),
        )
    )
    position_ids = torch.zeros(sequence_length, 3)
    position_ids[:, 0] = torch.arange(sequence_length)
    audio_plane_ids = torch.zeros(sequence_length, dtype=torch.long)
    if num_audio:
        audio_plane_ids[audio_indices] = torch.arange(num_audio, dtype=torch.long) % 2 + 1
    return {
        "hidden_states": torch.randn(1, num_video, 2, generator=generator),
        "audio_hidden_states": torch.randn(1, num_audio, 3, generator=generator),
        "encoder_hidden_states": torch.randn(1, num_text, 5, generator=generator),
        "timestep": torch.tensor([0.5]),
        "timestep_indices": torch.zeros(sequence_length, dtype=torch.long),
        "token_tags": token_tags,
        "position_ids": position_ids,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
        "audio_plane_ids": audio_plane_ids,
        "reference_modality_ids": torch.zeros(sequence_length, dtype=torch.long),
        "reference_index_ids": torch.zeros(sequence_length, dtype=torch.long),
    }


def test_zero_identity_ids_leave_text_and_target_rows_unchanged() -> None:
    model = GenEngineTransformer3DModel(**_config())
    with torch.no_grad():
        model.reference_modality_embedding.weight[2].fill_(2)
        model.reference_index_embedding.weight[1].fill_(3)
    packed = torch.randn(1, 4, model.config.hidden_size)
    modality_ids = torch.tensor([0, 0, 2, 0])
    index_ids = torch.tensor([0, 0, 1, 0])

    actual = model._add_reference_identity(packed, modality_ids, index_ids)

    assert torch.equal(actual[:, (0, 1, 3)], packed[:, (0, 1, 3)])
    torch.testing.assert_close(actual[:, 2], packed[:, 2] + 5)
    assert torch.count_nonzero(model.reference_modality_embedding.weight[0]) == 0
    assert torch.count_nonzero(model.reference_index_embedding.weight[0]) == 0


def test_audio_plane_embedding_distinguishes_left_and_right_rows() -> None:
    model = GenEngineTransformer3DModel(**_config())
    with torch.no_grad():
        model.audio_plane_embedding.weight[1].fill_(1.0)
        model.audio_plane_embedding.weight[2].fill_(2.0)
    packed = torch.zeros(1, 4, model.config.hidden_size)

    actual = model._add_audio_plane_identity(
        packed,
        torch.tensor([1, 1, 2, 2]),
    )

    torch.testing.assert_close(actual[:, :2], torch.ones_like(actual[:, :2]))
    torch.testing.assert_close(actual[:, 2:], torch.full_like(actual[:, 2:], 2.0))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reference_modality_ids", 4, "reference_modality_ids"),
        ("reference_index_ids", 5, "reference_index_ids"),
        ("audio_plane_ids", 3, "audio_plane_ids"),
    ],
)
def test_reference_identity_ids_validate_ranges(
    field: str,
    value: int,
    message: str,
) -> None:
    model = GenEngineTransformer3DModel(**_config())
    inputs = _inputs()
    inputs[field][-1] = value

    with pytest.raises(ValueError, match=message):
        model(**inputs)


def test_reference_identity_ids_require_full_packed_shape() -> None:
    model = GenEngineTransformer3DModel(**_config())
    inputs = _inputs()
    inputs["reference_index_ids"] = inputs["reference_index_ids"][:-1]

    with pytest.raises(ValueError, match="seq_len"):
        model(**inputs)


def test_audio_plane_ids_are_full_sequence_structural_metadata() -> None:
    model = GenEngineTransformer3DModel(**_config())
    inputs = _inputs()
    inputs["audio_plane_ids"][0] = 1
    with pytest.raises(ValueError, match="text and visual rows"):
        model(**inputs)

    inputs = _inputs()
    inputs["audio_plane_ids"][inputs["audio_indices"][0]] = 0
    with pytest.raises(ValueError, match="every audio row"):
        model(**inputs)


@pytest.mark.parametrize(
    ("modality_ids", "index_ids", "message"),
    [
        ([0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 0, 0], "both be zero"),
        ([1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], "text rows"),
        ([0, 0, 3, 0, 0, 0], [0, 0, 1, 0, 0, 0], "visual rows"),
        ([0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0], "audio rows"),
    ],
)
def test_reference_identity_ids_match_stream_semantics(
    modality_ids: list[int],
    index_ids: list[int],
    message: str,
) -> None:
    model = GenEngineTransformer3DModel(**_config())
    inputs = _inputs()
    inputs["reference_modality_ids"] = torch.tensor(modality_ids)
    inputs["reference_index_ids"] = torch.tensor(index_ids)

    with pytest.raises(ValueError, match=message):
        model(**inputs)


def test_native_state_config_and_temporal_position_scale() -> None:
    model = GenEngineTransformer3DModel(
        **_config(),
        position_time_scale=2.5,
    )
    state = model.state_dict()
    positions = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])

    scaled = model._position_ids_for_rope(positions)

    assert state["reference_modality_embedding.weight"].shape == (4, 16)
    assert state["reference_index_embedding.weight"].shape == (5, 16)
    assert state["audio_plane_embedding.weight"].shape == (3, 16)
    assert model.config.max_reference_index == 4
    assert model.config.position_time_scale == 2.5
    assert model.config.scratch_init_scheme == "mmdit_stable_v1"
    assert model.config.reference_embedding_init_std == 0.02
    assert model.config.audio_plane_embedding_init_std == 0.02
    torch.testing.assert_close(scaled[:, 0], positions[:, 0] * 2.5)
    assert torch.equal(scaled[:, 1:], positions[:, 1:])
    assert torch.equal(positions, torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan")])
def test_temporal_position_scale_must_be_positive(scale: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        GenEngineTransformer3DModel(**_config(), position_time_scale=scale)


def test_rope_width_must_fit_each_attention_head() -> None:
    with pytest.raises(ValueError, match="RoPE width"):
        GenEngineTransformer3DModel(
            **{
                **_config(),
                "attention_head_dim": 4,
                "hidden_size": 8,
                "rope_freq_dim": 1,
            }
        )


@pytest.mark.parametrize(("num_video", "num_audio"), [(0, 2), (2, 0), (0, 0)])
def test_forward_supports_empty_visual_and_audio_streams(
    num_video: int,
    num_audio: int,
) -> None:
    model = GenEngineTransformer3DModel(**_config()).eval()

    with torch.no_grad():
        output = model(**_inputs(num_video=num_video, num_audio=num_audio))

    assert output.sample.shape == (1, num_video, 2)
    assert output.audio_sample.shape == (1, num_audio, 3)
    assert torch.isfinite(output.sample).all()
    assert torch.isfinite(output.audio_sample).all()


def test_meta_wrapper_materializes_and_initializes_deterministically(
    tmp_path,
) -> None:
    from vforge.distributed.parallelize import materialize_meta_module

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({**_config(), "patch_size": [1, 1, 1], "_class_name": "ignored"}),
        encoding="utf-8",
    )
    model = GenEngineModel.build_empty(
        transformer_config_path=config_path,
        parameter_dtype=torch.float32,
    )
    assert all(parameter.is_meta for parameter in model.parameters())

    materialize_meta_module(model, torch.device("cpu"))
    model.initialize_parameters()
    model.initialize_scratch_weights(seed=37)
    first = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    model.initialize_scratch_weights(seed=37)

    assert first.keys() == dict(model.named_parameters()).keys()
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), name
        assert torch.equal(parameter, first[name]), name
    assert torch.count_nonzero(model.transformer.reference_modality_embedding.weight[0]) == 0
    assert torch.count_nonzero(model.transformer.reference_index_embedding.weight[0]) == 0
    assert model.transformer.reference_modality_embedding.weight[1:].std() < 0.1
    assert model.transformer.reference_index_embedding.weight[1:].std() < 0.1
    assert torch.count_nonzero(model.transformer.audio_plane_embedding.weight[0]) == 0
    assert model.transformer.audio_plane_embedding.weight[1:].std() < 0.1
    assert model.fsdp_execution_roots() == (model.transformer,)
    assert model.transformer.get_fsdp_precision_contract().preserve_forward_input_paths[0] == "."


def test_stable_initializer_applies_role_specific_parameter_contracts() -> None:
    config = {
        **_config(),
        "num_attention_heads": 8,
        "hidden_size": 64,
        "num_layers": 1,
        "num_refiner_layers": 2,
        "ffn_dim": 128,
        "text_dim": 64,
        "time_embed_hidden_dim": 64,
        "time_embed_dim": 32,
    }
    model = GenEngineTransformer3DModel(**config)

    initialize_scratch_transformer(model, seed=41)

    regular = model.transformer_blocks[0].attention.projections["visual"].to_q.weight
    expected_regular_variance = 1.0 / regular.shape[1]
    assert regular.abs().max() <= (3.0 / regular.shape[1]) ** 0.5
    assert regular.var(unbiased=False).item() == pytest.approx(
        expected_regular_variance,
        rel=0.08,
    )

    refiner_output = model.token_refiner.blocks[0].attention.to_out.weight
    expected_refiner_variance = 1.0 / (refiner_output.shape[1] * 2 * config["num_refiner_layers"])
    assert refiner_output.var(unbiased=False).item() == pytest.approx(
        expected_refiner_variance,
        rel=0.08,
    )

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            assert torch.count_nonzero(module.bias) == 0, name
        if isinstance(module, torch.nn.RMSNorm):
            assert torch.equal(module.weight, torch.ones_like(module.weight)), name
    assert torch.count_nonzero(model.transformer_blocks[0].adaln_modulation.linear.weight) == 0
    assert torch.count_nonzero(model.visual_norm_out.linear.weight) == 0
    assert torch.count_nonzero(model.audio_norm_out.linear.weight) == 0
    assert torch.count_nonzero(model.proj_out.weight) == 0
    assert torch.count_nonzero(model.audio_proj_out.weight) == 0


def test_stable_initializer_is_seeded_and_keeps_experts_distinct() -> None:
    model = GenEngineTransformer3DModel(**_moe_config())
    initialize_scratch_transformer(model, seed=17)
    first_q = (
        model.transformer_blocks[0].attention.projections["visual"].to_q.weight.detach().clone()
    )
    visual_moe = model.transformer_blocks[1].visual_moe
    first_expert = visual_moe.main_bank.experts[0].gate_up.weight.detach().clone()
    second_expert = visual_moe.main_bank.experts[1].gate_up.weight.detach().clone()

    initialize_scratch_transformer(model, seed=18)

    assert not torch.equal(
        model.transformer_blocks[0].attention.projections["visual"].to_q.weight,
        first_q,
    )
    assert not torch.equal(first_expert, second_expert)
    assert torch.count_nonzero(visual_moe.router.main_routing_bias) == 0
    assert torch.count_nonzero(visual_moe.router.block_routing_bias) == 0
    assert torch.count_nonzero(visual_moe.shared.shared_router_gate.weight) == 0
    assert torch.count_nonzero(model.reference_modality_embedding.weight[0]) == 0
    assert torch.count_nonzero(model.reference_index_embedding.weight[0]) == 0
    assert torch.count_nonzero(model.audio_plane_embedding.weight[0]) == 0


def test_stable_initializer_fails_closed_for_unknown_parameter_owner() -> None:
    model = GenEngineTransformer3DModel(**_config())
    model.register_parameter("unclassified", torch.nn.Parameter(torch.empty(1)))

    with pytest.raises(TypeError, match="no stable initialization policy"):
        initialize_scratch_transformer(model)


def test_stable_moe_blocks_start_as_exact_identity() -> None:
    model = GenEngineTransformer3DModel(**_moe_config(num_refiner_layers=0))
    initialize_scratch_transformer(model, seed=23)
    hidden_states = torch.randn(1, 6, model.config.hidden_size)
    block_inputs = {
        "time_embedding": torch.randn(2, model.config.time_embed_dim),
        "timestep_indices": torch.tensor([0, 0, 1, 1, 0, 1]),
        "stream_tags": torch.tensor([1, 1, 0, 0, 2, 2]),
        "valid_mask": torch.ones(6, dtype=torch.bool),
        "visual_categories": torch.tensor([-1, -1, 2, 0, -1, -1]),
        "rotary_emb": (
            torch.ones(6, 6 * model.config.rope_freq_dim),
            torch.zeros(6, 6 * model.config.rope_freq_dim),
        ),
    }

    for block in model.transformer_blocks:
        output = block(hidden_states=hidden_states, **block_inputs)
        assert torch.equal(output, hidden_states)


@pytest.mark.parametrize("architecture", ["mmdit_dense", "mmdit_moe"])
def test_stable_transformer_starts_with_exact_zero_velocity(architecture: str) -> None:
    config = _config() if architecture == "mmdit_dense" else _moe_config()
    model = GenEngineTransformer3DModel(**config).eval()
    initialize_scratch_transformer(model, seed=29)

    with torch.no_grad():
        output = model(**_inputs())

    assert torch.count_nonzero(output.sample) == 0
    assert torch.count_nonzero(output.audio_sample) == 0


def _has_nonzero_gradient(parameter: torch.Tensor) -> bool:
    return parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))


def test_zero_velocity_initialization_bootstraps_gradients_in_expected_stages() -> None:
    model = GenEngineTransformer3DModel(**_moe_config()).train()
    initialize_scratch_transformer(model, seed=31)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    inputs = _inputs()
    losses: list[float] = []

    def backward() -> None:
        optimizer.zero_grad(set_to_none=True)
        output = model(**inputs)
        loss = (output.sample - 1.0).square().mean()
        loss = loss + (output.audio_sample + 0.5).square().mean()
        losses.append(float(loss.detach()))
        loss.backward()

    first_block = model.transformer_blocks[0]
    moe_block = model.transformer_blocks[1]

    backward()
    assert _has_nonzero_gradient(model.proj_out.weight)
    assert _has_nonzero_gradient(model.audio_proj_out.weight)
    assert not _has_nonzero_gradient(model.visual_norm_out.linear.weight)
    assert not _has_nonzero_gradient(first_block.adaln_modulation.linear.weight)
    assert not _has_nonzero_gradient(first_block.attention.projections["visual"].to_q.weight)
    optimizer.step()

    backward()
    assert _has_nonzero_gradient(model.visual_norm_out.linear.weight)
    assert _has_nonzero_gradient(model.audio_norm_out.linear.weight)
    assert _has_nonzero_gradient(first_block.adaln_modulation.linear.weight)
    assert _has_nonzero_gradient(moe_block.adaln_modulation.linear.weight)
    assert _has_nonzero_gradient(model.proj_in.weight)
    assert not _has_nonzero_gradient(first_block.attention.projections["visual"].to_q.weight)
    optimizer.step()

    backward()
    assert _has_nonzero_gradient(first_block.attention.projections["visual"].to_q.weight)
    assert _has_nonzero_gradient(moe_block.attention.projections["visual"].to_q.weight)
    assert _has_nonzero_gradient(moe_block.visual_moe.router.main_router.weight)
    assert _has_nonzero_gradient(moe_block.visual_moe.router.block_router.weight)
    assert any(
        _has_nonzero_gradient(expert.down.weight)
        for expert in moe_block.visual_moe.main_bank.experts
    )
    optimizer.step()

    backward()
    assert _has_nonzero_gradient(first_block.attention.projections["text"].to_q.weight)
    assert _has_nonzero_gradient(first_block.text_ffn.down.weight)
    structurally_dead = (
        moe_block.attention.projections["text"].to_q.weight,
        moe_block.attention.projections["text"].q_norm.weight,
        moe_block.attention.projections["text"].to_out.weight,
        moe_block.ffn_norms["text"].weight,
        moe_block.text_ffn.gate_up.weight,
        moe_block.text_ffn.down.weight,
    )
    assert all(not _has_nonzero_gradient(parameter) for parameter in structurally_dead)
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]
