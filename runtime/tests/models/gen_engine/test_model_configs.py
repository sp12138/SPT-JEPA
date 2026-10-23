"""Contracts for the version-controlled Gen Engine DiT model catalog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vforge.models.gen_engine.dit import GenEngineDiTConfig

CONFIG_DIR = (
    Path(__file__).resolve().parents[3] / "vforge" / "models" / "gen_engine" / "dit" / "configs"
)


def _config_path(name: str) -> Path:
    return CONFIG_DIR / f"{name.removesuffix('.json')}.json"


def _load_config(name: str) -> GenEngineDiTConfig:
    payload = json.loads(_config_path(name).read_text(encoding="utf-8"))
    return GenEngineDiTConfig.from_dict(payload, require_architecture=True)


@pytest.mark.parametrize(
    ("name", "architecture", "total", "active"),
    [
        (
            "mmdit-dense-isoflop-4.1b",
            "mmdit_dense",
            4_145_283_072,
            4_145_283_072,
        ),
        (
            "mmdit-dense-leverage-10.4b",
            "mmdit_dense",
            10_362_028_032,
            10_362_028_032,
        ),
        (
            "mmdit-moe-32b-a4.1b",
            "mmdit_moe",
            32_116_448_256,
            4_144_634_880,
        ),
    ],
)
def test_model_config_parameter_budgets(
    name: str,
    architecture: str,
    total: int,
    active: int,
) -> None:
    path = _config_path(name)
    config = _load_config(name)
    estimate = config.estimate_core_parameters()

    assert path.name == f"{name}.json"
    assert config.architecture == architecture
    assert estimate.total == total
    assert estimate.active == active


def test_model_catalog_matches_packaged_json_files() -> None:
    assert {path.name for path in CONFIG_DIR.glob("*.json")} == {
        "mmdit-dense-isoflop-4.1b.json",
        "mmdit-dense-leverage-10.4b.json",
        "mmdit-moe-32b-a4.1b.json",
    }


def test_v0_moe_and_dense_baseline_geometry_is_locked() -> None:
    moe = _load_config("mmdit-moe-32b-a4.1b")
    dense_isoflop = _load_config("mmdit-dense-isoflop-4.1b")
    dense_leverage = _load_config("mmdit-dense-leverage-10.4b")

    for field in (
        "num_attention_heads",
        "attention_head_dim",
        "hidden_size",
        "num_layers",
        "num_refiner_layers",
        "ffn_dim",
        "refiner_ffn_dim",
        "in_channels",
        "audio_in_channels",
        "patch_size",
        "text_dim",
        "freq_dim",
        "time_embed_hidden_dim",
        "time_embed_dim",
        "rope_freq_dim",
    ):
        assert getattr(moe, field) == getattr(dense_isoflop, field), field

    assert moe.moe is not None
    assert moe.moe.start_layer == 2
    assert (moe.moe.main_experts, moe.moe.main_top_k) == (160, 8)
    assert (moe.moe.block_experts, moe.moe.block_top_k) == (0, 0)
    assert moe.moe.shared_multiplier == 1
    assert moe.moe.expert_intermediate_size == 1536
    assert moe.visual_ffn_dim == 4096
    assert dense_isoflop.visual_ffn_dim == 13184
    assert dense_leverage.visual_ffn_dim is not None
    assert moe.num_layers == dense_isoflop.num_layers == 28
    assert dense_leverage.num_layers == 32

    moe_estimate = moe.estimate_core_parameters()
    dense_isoflop_total = dense_isoflop.estimate_core_parameters().total
    dense_leverage_total = dense_leverage.estimate_core_parameters().total
    assert moe_estimate.active / dense_isoflop_total == pytest.approx(1.0, abs=0.001)
    assert dense_leverage_total / dense_isoflop_total == pytest.approx(2.5, abs=0.003)
    assert dense_leverage.hidden_size / dense_isoflop.hidden_size == 1.5
    assert dense_leverage.ffn_dim / dense_isoflop.ffn_dim == 1.4375
    assert dense_leverage.visual_ffn_dim / dense_isoflop.visual_ffn_dim == pytest.approx(
        1.4375, rel=0.001
    )
    assert dense_leverage.visual_ffn_dim / dense_leverage.ffn_dim == pytest.approx(
        dense_isoflop.visual_ffn_dim / dense_isoflop.ffn_dim,
        rel=0.001,
    )

    visual_total = (
        moe_estimate.attention // 3
        + moe_estimate.adaln // 3
        + moe_estimate.visual_ffn_total
        + moe_estimate.routers
    )
    visual_active = (
        moe_estimate.attention // 3
        + moe_estimate.adaln // 3
        + moe_estimate.visual_ffn_active
        + moe_estimate.routers
    )
    assert visual_total == 30_134_639_616
    assert visual_active == 2_162_826_240
