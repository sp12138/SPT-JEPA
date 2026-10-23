"""Numerical and schema contracts for native MMDiT conditioning."""

from __future__ import annotations

import math

import pytest
import torch

from vforge.models.gen_engine.dit import GenEngineDiTConfig
from vforge.models.gen_engine.dit.conditioning import (
    AdaLNModulation,
    RotaryPositionEmbedding,
    apply_rotary_embedding,
    timestep_embedding,
)


def test_timestep_embedding_has_expected_zero_time_layout() -> None:
    embedding = timestep_embedding(
        torch.tensor([0.0, 1.0]),
        8,
        flip_sin_to_cos=True,
        downscale_freq_shift=0.0,
    )

    torch.testing.assert_close(embedding[0, :4], torch.ones(4))
    torch.testing.assert_close(embedding[0, 4:], torch.zeros(4))
    assert embedding.dtype == torch.float32
    assert embedding.shape == (2, 8)


def test_three_axis_rope_is_identity_at_the_origin() -> None:
    rope = RotaryPositionEmbedding(frequency_dim=1, theta=10000.0)
    cos, sin = rope(torch.zeros(3, 3))
    hidden = torch.randn(2, 3, 1, 8)

    actual = apply_rotary_embedding(hidden, cos, sin)

    torch.testing.assert_close(actual, hidden)
    assert cos.shape == sin.shape == (3, 6)


def test_fused_adaln_projects_all_streams_in_one_forward() -> None:
    modulation = AdaLNModulation(time_dim=4, hidden_size=8, num_streams=3)
    time_embedding = torch.randn(2, 4)

    projected = modulation(time_embedding)
    visual = modulation.gather_stream(projected, 0, torch.tensor([1, 0]))
    text = modulation.gather_stream(projected, 1, torch.tensor([0]))

    assert projected.shape == (2, 3, 48)
    assert all(part.shape == (2, 8) for part in visual)
    assert all(part.shape == (1, 8) for part in text)
    weight = modulation.linear.weight
    assert not torch.equal(weight[:48], weight[48:96])


def test_native_config_rejects_removed_ep_field_and_latent_moe() -> None:
    base = {
        "schema": "vforge.gen_engine.dit",
        "schema_version": 1,
        "architecture": "mmdit_moe",
        "num_attention_heads": 1,
        "attention_head_dim": 8,
        "hidden_size": 8,
        "num_layers": 2,
    }
    with pytest.raises(TypeError, match="unknown visual MoE config fields.*expert_parallel_size"):
        GenEngineDiTConfig.from_dict(
            {
                **base,
                "moe": {
                    "start_layer": 1,
                    "expert_parallel_size": 1,
                },
            }
        )
    with pytest.raises(NotImplementedError, match="latent_moe_dim"):
        GenEngineDiTConfig.from_dict(
            {
                **base,
                "moe": {
                    "start_layer": 1,
                    "latent_moe_dim": 256,
                },
            }
        )


def test_native_config_requires_architecture_and_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="architecture"):
        GenEngineDiTConfig.from_dict({"hidden_size": 8})
    with pytest.raises(TypeError, match="unknown native"):
        GenEngineDiTConfig.from_dict(
            {
                "schema": "vforge.gen_engine.dit",
                "schema_version": 1,
                "architecture": "mmdit_dense",
                "unexpected": True,
            }
        )


def test_native_config_rejects_wrong_schema_and_illegal_top_k() -> None:
    with pytest.raises(ValueError, match="schema='vforge.gen_engine.dit'"):
        GenEngineDiTConfig.from_dict(
            {
                "schema": "legacy.h3",
                "schema_version": 1,
                "architecture": "mmdit_dense",
            }
        )
    with pytest.raises(ValueError, match="main_top_k"):
        GenEngineDiTConfig.from_dict(
            {
                "schema": "vforge.gen_engine.dit",
                "schema_version": 1,
                "architecture": "mmdit_moe",
                "num_layers": 2,
                "moe": {
                    "start_layer": 1,
                    "main_experts": 2,
                    "main_top_k": 3,
                },
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("router_logit_init_std", 0.0),
        ("router_logit_init_std", float("nan")),
        ("main_routed_scaling_factor", 0.0),
        ("main_routed_scaling_factor", float("inf")),
        ("block_routed_scaling_factor", -1.0),
    ],
)
def test_native_config_requires_positive_finite_moe_initialization_scales(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        GenEngineDiTConfig.from_dict(
            {
                "schema": "vforge.gen_engine.dit",
                "schema_version": 1,
                "architecture": "mmdit_moe",
                "num_layers": 2,
                "moe": {
                    "start_layer": 1,
                    field: value,
                },
            }
        )


def test_native_config_pins_stable_scratch_initialization_scheme() -> None:
    config = GenEngineDiTConfig.from_dict(
        {
            "schema": "vforge.gen_engine.dit",
            "schema_version": 1,
            "architecture": "mmdit_moe",
            "num_layers": 2,
            "moe": {"start_layer": 1},
        }
    )

    assert config.scratch_init_scheme == "mmdit_stable_v1"
    assert config.moe is not None
    assert config.moe.router_logit_init_std == 0.1
    assert config.moe.main_routed_scaling_factor == 1.0
    assert config.moe.block_routed_scaling_factor == 1.0
    with pytest.raises(ValueError, match="scratch_init_scheme"):
        GenEngineDiTConfig.from_dict(
            {
                "schema": "vforge.gen_engine.dit",
                "schema_version": 1,
                "architecture": "mmdit_dense",
                "scratch_init_scheme": "pytorch_default",
            }
        )


def test_production_moe_parameter_estimate_matches_design_document() -> None:
    estimate = GenEngineDiTConfig.from_dict(
        {
            "schema": "vforge.gen_engine.dit",
            "schema_version": 1,
            "architecture": "mmdit_moe",
            "moe": {},
        }
    ).estimate_core_parameters()

    assert math.isclose(estimate.total / 1e9, 255.88, abs_tol=0.01)
    assert math.isclose(estimate.active / 1e9, 38.37, abs_tol=0.01)
