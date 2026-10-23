"""The encode cache key is a compatibility contract, so freeze it per version.

``encoding_fingerprint`` decides when an existing cache is stale. Adding a key
invalidates every cache a version has ever written; dropping one lets a stale
cache be served as fresh. Both are silent at runtime, which is why the key sets
are spelled out here rather than derived.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

COMMON_KEYS = {
    "pretrained_path",
    "model_revision",
    "manifest_digest",
    "sample_ids_digest",
    "sample_count",
    "bundle_schema",
    "bundle_version",
    "max_frames",
    "max_sequence_length",
    "resolution",
    "fps",
    "condition_streams",
}

EXPECTED_KEYS = {
    "wan21": COMMON_KEYS | {"use_vae_condition", "use_clip_condition", "use_dino_condition"},
    "wan22_a14b": COMMON_KEYS | {"cache_flavor"},
    "wan22_ti2v_5b": COMMON_KEYS | {"cache_flavor", "vae_pretrained_path"},
}

ENCODE_MODULES = {
    "wan21": "vforge.tasks.wan.v21.encode",
    "wan22_a14b": "vforge.tasks.wan.v22.a14b.encode",
    "wan22_ti2v_5b": "vforge.tasks.wan.v22.ti2v_5b.encode",
}

EXPERIMENTS = {
    "wan21": "wan21_i2v_14b",
    "wan22_a14b": "wan22_a14b",
    "wan22_ti2v_5b": "wan22_ti2v_5b",
}


class _StubEncoder:
    def encode_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("the fingerprint must not depend on encoding a sample")


def _write_manifest(tmp_path: Path) -> str:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            [
                {"sample_id": "a", "video": "a.mp4", "prompt": "a"},
                {"sample_id": "b", "video": "b.mp4", "prompt": "b"},
            ]
        )
    )
    return str(path)


def _config(model: str, tmp_path: Path, *, latent_cache_mode: str) -> Any:
    from vforge.tasks.common import build_and_snapshot, compose_standalone_config

    manifest = _write_manifest(tmp_path)
    cfg = compose_standalone_config(
        "prepare",
        overrides=[
            f"experiment={EXPERIMENTS[model]}",
            f"runtime.output_dir={tmp_path}",
            f"latent_cache.mode={latent_cache_mode}",
            f"data.train_manifest={manifest}",
            f"data.cache_dir={tmp_path / 'cache'}",
            "model.pretrained_path=/nonexistent",
        ],
    )
    return build_and_snapshot(cfg, world_size=1, write_snapshot=False).config


def _stub_encoder_assets(model: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the fingerprint path off the VAE, the text encoder and the tokenizer."""
    import importlib

    module = importlib.import_module(ENCODE_MODULES[model])
    monkeypatch.setattr(
        module,
        "_build_condition_encoder",
        lambda **_kwargs: _StubEncoder(),
        raising=True,
    )
    monkeypatch.setattr(
        "vforge.tasks.wan.encode.build_raw_collate_fn",
        lambda **_kwargs: lambda samples: {},
        raising=True,
    )


def _build_components(model: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from vforge.tasks.registry import resolve_builder

    _stub_encoder_assets(model, monkeypatch)
    return resolve_builder(model, "encode")(_config(model, tmp_path, latent_cache_mode="online"))


@pytest.mark.parametrize("model", sorted(EXPECTED_KEYS))
def test_encoding_fingerprint_keys_are_frozen_per_version(
    model: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    components = _build_components(model, tmp_path, monkeypatch)

    assert set(components.encoding_fingerprint) == EXPECTED_KEYS[model]


@pytest.mark.parametrize("model", sorted(EXPECTED_KEYS))
def test_encode_components_describe_the_whole_manifest(
    model: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    components = _build_components(model, tmp_path, monkeypatch)

    assert components.sample_ids == ("a", "b")
    assert components.encoding_fingerprint["sample_count"] == 2
    assert components.codec is not None


@pytest.mark.parametrize("model", sorted(EXPECTED_KEYS))
def test_encode_rejects_offline_latent_cache_mode(
    model: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Offline mode means the cache already exists; there is nothing to write."""
    from vforge.tasks.registry import resolve_builder

    _stub_encoder_assets(model, monkeypatch)
    cfg = _config(model, tmp_path, latent_cache_mode="offline")

    with pytest.raises(ValueError, match="latent_cache.mode=online"):
        resolve_builder(model, "encode")(cfg)
