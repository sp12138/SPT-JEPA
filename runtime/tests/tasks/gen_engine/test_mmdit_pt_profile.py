"""Contracts for the full MMDiT MoE model and explicit raw-tensor profile."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from vforge.tasks.common import compose_standalone_config
from vforge.tasks.gen_engine.shared import _latent_codec_config_path

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "examples/configs/gen-engine/mmdit-moe-32b-a4.1b-pt4096.json"
CATALOG = ROOT / "vforge/models/gen_engine/dit/configs/mmdit-moe-32b-a4.1b.json"


def test_profile_only_changes_text_projection(gen_engine_dit):
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert catalog["text_dim"] == 5120
    assert profile == {**catalog, "text_dim": 4096}


def test_profile_builds_full_model_on_meta(gen_engine_dit):
    model = gen_engine_dit.GenEngineModel.build_empty(transformer_config_path=PROFILE)
    assert all(parameter.is_meta for parameter in model.parameters())
    assert sum(parameter.numel() for parameter in model.parameters()) == 32_187_187_072
    assert model.config.num_layers == 28
    assert model.config.hidden_size == 1536
    assert model.config.moe.main_experts == 160
    assert model.config.moe.main_top_k == 8


def test_profile_composes_raw_tensor_contract_without_guessing_codec():
    config = compose_standalone_config(
        "train",
        [
            "experiment=gen_engine_mmdit_moe32b",
            "checkpoint.enabled=false",
            "model.extras.artifact_cache_dir=/tmp/example-mmgen-cache",
        ],
    )
    extras = OmegaConf.to_container(config.model.extras, resolve=True)
    assert config.model.name == "gen_engine"
    assert extras["transformer_config_path"] == str(PROFILE.relative_to(ROOT))
    assert extras["tensor_format"] == "tensor"
    assert extras["audio_layout"] == "ATC"
    assert extras["require_digests"] is False
    assert extras["pin_artifacts"] is True
    assert extras["artifact_cache_dir"] == "/tmp/example-mmgen-cache"
    for modality in ("video", "audio"):
        assert extras[f"{modality}_latent_codec_config_path"] == ""
        with pytest.raises(ValueError, match="is required"):
            _latent_codec_config_path(extras, modality)


def test_profile_accepts_explicit_codec_files_without_dynamic_hydra_keys(tmp_path):
    video_path = tmp_path / "video.json"
    audio_path = tmp_path / "audio.json"
    config = compose_standalone_config(
        "train",
        [
            "experiment=gen_engine_mmdit_moe32b",
            "checkpoint.enabled=false",
            f"model.extras.video_latent_codec_config_path={video_path}",
            f"model.extras.audio_latent_codec_config_path={audio_path}",
        ],
    )
    assert config.model.extras.video_latent_codec_config_path == str(video_path)
    assert config.model.extras.audio_latent_codec_config_path == str(audio_path)


def test_base_profile_preserves_existing_audio_layout_and_digest_policy():
    config = compose_standalone_config(
        "train", ["experiment=gen_engine", "checkpoint.enabled=false"]
    )
    assert config.model.extras.audio_layout == "ACT"
    assert config.model.extras.tensor_format == "bundle"
    assert config.model.extras.require_digests is True
    assert config.model.extras.pin_artifacts is False
