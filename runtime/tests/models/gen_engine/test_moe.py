"""Focused contracts for native dual-bank Visual MoE."""

from __future__ import annotations

import math

import pytest
import torch

from vforge.models.gen_engine.dit import (
    GenEngineTransformer3DModel,
    initialize_scratch_transformer,
)
from vforge.models.gen_engine.dit.config import VisualMoEConfig
from vforge.models.gen_engine.dit.moe import (
    DualBankRouter,
    ExpertBank,
    GatedSharedExpert,
    balance_alpha,
)


def _moe_config(**overrides) -> VisualMoEConfig:
    values = {
        "start_layer": 1,
        "expert_intermediate_size": 4,
        "main_experts": 3,
        "main_top_k": 2,
        "block_experts": 2,
        "block_top_k": 1,
        "shared_multiplier": 2,
        "shared_gate": "sigmoid",
        "balance_alpha": 0.1,
        "balance_decay": 0.2,
        "balance_epsilon": 0.0,
        "balance_warmup_ratio": 0.1,
    }
    values.update(overrides)
    return VisualMoEConfig(**values)


def test_dual_bank_bias_only_changes_selection_and_each_bank_normalizes_itself() -> None:
    router = DualBankRouter(4, _moe_config())
    with torch.no_grad():
        router.main_router.weight.zero_()
        router.block_router.weight.zero_()
        router.main_routing_bias.copy_(torch.tensor([0.0, 2.0, 1.0]))
        router.block_routing_bias.copy_(torch.tensor([1.0, 0.0]))
    hidden = torch.randn(2, 4)
    categories = torch.tensor([0, 2])

    decision = router(hidden, categories)

    assert {int(value) for value in decision.main_indices[0]} == {1, 2}
    assert torch.equal(decision.block_indices, torch.zeros((2, 1), dtype=torch.long))
    torch.testing.assert_close(decision.main_weights, torch.full((2, 2), 0.5))
    torch.testing.assert_close(decision.block_weights, torch.ones(2, 1))
    torch.testing.assert_close(decision.main_weights.sum(dim=-1), torch.ones(2))
    torch.testing.assert_close(decision.block_weights.sum(dim=-1), torch.ones(2))
    assert decision.main_statistics.counts.sum().item() == 4
    assert decision.block_statistics.counts.sum().item() == 2
    assert decision.main_statistics.category_counts[0].sum().item() == 2
    assert decision.main_statistics.category_counts[2].sum().item() == 2
    assert router.main_router.weight.dtype == torch.float32
    assert router.main_routing_bias.dtype == torch.float32


def test_main_only_router_normalizes_over_the_main_bank() -> None:
    router = DualBankRouter(
        4,
        _moe_config(
            block_experts=0,
            block_top_k=0,
        ),
    )
    decision = router(torch.randn(3, 4), torch.tensor([0, 1, 2]))

    torch.testing.assert_close(
        decision.main_weights.sum(dim=-1),
        torch.ones(3),
    )
    assert decision.block_indices.shape == (3, 0)
    assert decision.block_statistics.counts.numel() == 0


def test_reference_dispatcher_matches_explicit_expert_sum() -> None:
    torch.manual_seed(4)
    bank = ExpertBank(hidden_size=4, intermediate_size=3, num_experts=3)
    hidden = torch.randn(4, 4)
    indices = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 1]])
    weights = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.5, 0.5]])

    actual = bank(hidden, indices, weights)
    expected = torch.zeros_like(hidden)
    for token in range(hidden.shape[0]):
        for slot in range(indices.shape[1]):
            expert = bank.experts[int(indices[token, slot])]
            expected[token] += weights[token, slot] * expert(hidden[token : token + 1])[0]

    torch.testing.assert_close(actual, expected)


def test_rms_error_bias_update_uses_decay_and_independent_banks() -> None:
    router = DualBankRouter(4, _moe_config())
    with torch.no_grad():
        router.main_routing_bias.copy_(torch.tensor([0.5, -0.5, 1.0]))
        router.block_routing_bias.copy_(torch.tensor([0.5, -0.5]))

    router.update_biases(
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor([1.0, 1.0]),
        alpha=0.1,
    )

    error = torch.tensor([1.0, 0.0, -1.0])
    normalized = error / error.square().mean().sqrt()
    expected_main = 0.8 * torch.tensor([0.5, -0.5, 1.0]) + 0.1 * normalized
    torch.testing.assert_close(router.main_routing_bias, expected_main)
    torch.testing.assert_close(
        router.block_routing_bias,
        0.8 * torch.tensor([0.5, -0.5]),
    )


def test_balance_alpha_warms_up_by_completed_optimizer_steps() -> None:
    config = _moe_config(balance_alpha=4e-5, balance_warmup_ratio=0.1)

    assert balance_alpha(config, completed_step=0, total_steps=100) == 0.0
    assert balance_alpha(config, completed_step=5, total_steps=100) == 2e-5
    assert balance_alpha(config, completed_step=10, total_steps=100) == 4e-5
    assert balance_alpha(config, completed_step=50, total_steps=100) == 4e-5


def test_shared_expert_modes_cover_disabled_always_on_and_sigmoid() -> None:
    hidden = torch.randn(3, 4)
    disabled = GatedSharedExpert(
        4,
        _moe_config(shared_multiplier=0, shared_gate="disabled"),
    )
    always = GatedSharedExpert(4, _moe_config(shared_gate="always_on"))
    sigmoid = GatedSharedExpert(4, _moe_config(shared_gate="sigmoid"))
    with torch.no_grad():
        sigmoid.shared_router_gate.weight.zero_()

    disabled_output, disabled_stats = disabled(hidden)
    always_output, always_stats = always(hidden)
    sigmoid_output, sigmoid_stats = sigmoid(hidden)

    assert torch.count_nonzero(disabled_output) == 0
    assert disabled_stats.value_sum.item() == 0
    torch.testing.assert_close(always_output, always.expert(hidden))
    assert always_stats.value_sum.item() == 3
    torch.testing.assert_close(sigmoid_output, 0.5 * sigmoid.expert(hidden))
    assert sigmoid_stats.value_sum.item() == 1.5


def test_per_bank_routed_scaling_applies_after_normalization() -> None:
    router = DualBankRouter(
        4,
        _moe_config(
            main_routed_scaling_factor=2.0,
            block_routed_scaling_factor=3.0,
        ),
    )
    with torch.no_grad():
        router.main_router.weight.zero_()
        router.block_router.weight.zero_()
        router.main_routing_bias.copy_(torch.tensor([0.0, 2.0, 1.0]))
        router.block_routing_bias.copy_(torch.tensor([1.0, 0.0]))
    hidden = torch.randn(2, 4)

    decision = router(hidden, torch.tensor([0, 2]))

    torch.testing.assert_close(decision.main_weights.sum(dim=-1), torch.full((2,), 2.0))
    torch.testing.assert_close(decision.block_weights.sum(dim=-1), torch.full((2,), 3.0))
    expected_entropy = 2 * math.log(2)
    assert decision.main_statistics.entropy_sum.item() == pytest.approx(expected_entropy)
    assert decision.block_statistics.entropy_sum.item() == 0.0
    bank = ExpertBank(hidden_size=4, intermediate_size=3, num_experts=3)
    scaled = bank(hidden, decision.main_indices, decision.main_weights)
    baseline = bank(hidden, decision.main_indices, decision.main_weights / 2.0)
    torch.testing.assert_close(scaled, 2.0 * baseline)


def _initialized_scale_probe() -> GenEngineTransformer3DModel:
    model = GenEngineTransformer3DModel(
        schema="vforge.gen_engine.dit",
        schema_version=1,
        architecture="mmdit_moe",
        num_attention_heads=32,
        attention_head_dim=8,
        hidden_size=256,
        num_layers=2,
        num_refiner_layers=0,
        ffn_dim=128,
        in_channels=1,
        audio_in_channels=3,
        patch_size=(1, 1, 1),
        text_dim=32,
        freq_dim=16,
        time_embed_hidden_dim=64,
        time_embed_dim=32,
        rope_freq_dim=1,
        moe={
            "start_layer": 1,
            "expert_intermediate_size": 16,
            "main_experts": 40,
            "main_top_k": 20,
            "block_experts": 16,
            "block_top_k": 2,
            "shared_multiplier": 2,
            "shared_gate": "sigmoid",
        },
    )
    initialize_scratch_transformer(model, seed=101)
    return model


def test_stable_router_initialization_is_balanced_and_unsaturated() -> None:
    model = _initialized_scale_probe()
    visual_moe = model.transformer_blocks[1].visual_moe
    generator = torch.Generator().manual_seed(103)
    hidden = torch.randn(4096, model.config.hidden_size, generator=generator)
    categories = torch.full((hidden.shape[0],), 2)

    main_logits = visual_moe.router.main_router(hidden)
    block_logits = visual_moe.router.block_router(hidden)
    decision = visual_moe.router(hidden, categories)

    assert main_logits.square().mean().sqrt().item() == pytest.approx(0.1, rel=0.08)
    assert block_logits.square().mean().sqrt().item() == pytest.approx(0.1, rel=0.08)
    assert not torch.any(torch.sigmoid(main_logits) <= 0.01)
    assert not torch.any(torch.sigmoid(main_logits) >= 0.99)
    main_entropy = decision.main_statistics.entropy_sum / hidden.shape[0]
    block_entropy = decision.block_statistics.entropy_sum / hidden.shape[0]
    assert main_entropy.item() / math.log(model.config.moe.main_top_k) > 0.99
    assert block_entropy.item() / math.log(model.config.moe.block_top_k) > 0.99
    main_counts = decision.main_statistics.counts
    block_counts = decision.block_statistics.counts
    assert main_counts.std(unbiased=False) / main_counts.mean() < 0.15
    assert block_counts.std(unbiased=False) / block_counts.mean() < 0.10


def test_stable_visual_moe_output_scale_matches_dense_ffn_order() -> None:
    model = _initialized_scale_probe()
    dense = model.transformer_blocks[0].visual_ffn
    visual_moe = model.transformer_blocks[1].visual_moe
    generator = torch.Generator().manual_seed(107)
    hidden = torch.randn(1, 4096, model.config.hidden_size, generator=generator)
    categories = torch.full((hidden.shape[1],), 2)

    with torch.no_grad():
        dense_output = dense(hidden)
        moe_output, statistics = visual_moe(hidden, categories)

    ratio = moe_output.square().mean().sqrt() / dense_output.square().mean().sqrt()
    assert 0.80 < ratio.item() < 1.05
    assert statistics.outputs.element_count.item() == moe_output.numel()
    assert statistics.outputs.main_square_sum.item() > 0
    assert statistics.outputs.block_square_sum.item() > 0
    assert statistics.outputs.shared_square_sum.item() > 0
