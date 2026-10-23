"""CPU tests for latent bundle IO and cache manifests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_latent_bundle_roundtrip(tmp_path: Path) -> None:
    from vforge.data.cache import load_latent_bundle, save_latent_bundle
    from vforge.tasks.wan.v21.shared import (
        WAN21_ENCODED_BATCH_CODEC,
        project_to_encoded_batch,
    )

    bundle = {
        "prompt": torch.randn(1, 4, 8),
        "frame_latents": torch.randn(1, 16, 2, 4, 4),
        "image_condition_latent_vae": None,
        "image_condition_latent_clip": None,
        "image_condition_latent_dino": None,
        "condition_latents_normal": torch.randn(1, 20, 2, 4, 4),
    }
    path = tmp_path / "sample.pt"
    save_latent_bundle(project_to_encoded_batch(bundle), str(path))
    loaded = WAN21_ENCODED_BATCH_CODEC.verify(str(path))
    assert torch.equal(loaded["prompt"], bundle["prompt"])
    assert loaded["image_condition_latent_vae"] is None
    again = load_latent_bundle(str(path))
    assert set(project_to_encoded_batch(again)) == set(loaded)


def test_legacy_bundle_defaults_missing_optional_image_conditions(
    tmp_path: Path,
) -> None:
    from vforge.data.cache import load_latent_bundle, save_latent_bundle
    from vforge.tasks.wan.v21.shared import WAN21_ENCODED_BATCH_CODEC

    path = tmp_path / "legacy.pt"
    save_latent_bundle(
        {
            "prompt": torch.randn(1, 4, 8),
            "frame_latents": torch.randn(1, 16, 2, 4, 4),
            "image_condition_latent_vae": None,
            "condition_latents_normal": torch.randn(1, 20, 2, 4, 4),
        },
        str(path),
    )
    assert "image_condition_latent_dino" not in load_latent_bundle(str(path))
    loaded = WAN21_ENCODED_BATCH_CODEC.verify(str(path))
    assert loaded["image_condition_latent_clip"] is None
    assert loaded["image_condition_latent_dino"] is None


def test_encoded_collate_accepts_legacy_and_batched_bundles() -> None:
    from vforge.tasks.wan.v21.shared import _collate_encoded_bundles

    legacy = {
        "prompt": torch.ones(2, 4),
        "frame_latents": torch.ones(3, 2, 2, 2),
        "image_condition_latent_vae": None,
        "image_condition_latent_clip": None,
        "image_condition_latent_dino": None,
        "condition_latents_normal": torch.ones(5, 2, 2, 2),
        "sample_id": "legacy",
    }
    batched = {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
        for key, value in legacy.items()
    }
    batched["sample_id"] = "batched"
    result = _collate_encoded_bundles([legacy, batched])
    assert result["prompt"].shape == (2, 2, 4)
    assert result["frame_latents"].shape == (2, 3, 2, 2, 2)
    assert result["condition_latents_normal"].shape == (2, 5, 2, 2, 2)


def test_cache_manifest_roundtrip(tmp_path: Path) -> None:
    from vforge.data.cache import (
        CACHE_SCHEMA_VERSION,
        CacheManifest,
        EncodedBatchCodec,
        read_cache_manifest,
        write_cache_manifest,
    )

    codec = EncodedBatchCodec(name="test", required_keys=("value",))
    manifest = CacheManifest(
        schema_version=CACHE_SCHEMA_VERSION,
        sample_ids=["a", "b"],
        completed=2,
        encoding_fingerprint={"k": 1},
        codec_name=codec.name,
        codec_schema_version=codec.schema_version,
    )
    write_cache_manifest(str(tmp_path), manifest)
    loaded = read_cache_manifest(str(tmp_path), codec=codec)
    assert loaded is not None
    assert loaded.sample_ids == ["a", "b"]
    assert loaded.completed == 2
    assert loaded.to_dict()["sample_ids_digest"]
    assert loaded.bucket_rule == "sha1-prefix-2"


def test_cache_manifest_rejects_tampered_sample_ids(tmp_path: Path) -> None:
    from vforge.data.cache import (
        CACHE_SCHEMA_VERSION,
        CacheManifest,
        EncodedBatchCodec,
        read_cache_manifest,
        write_cache_manifest,
    )

    codec = EncodedBatchCodec(name="test", required_keys=("value",))
    write_cache_manifest(
        str(tmp_path),
        CacheManifest(
            schema_version=CACHE_SCHEMA_VERSION,
            sample_ids=["a", "b"],
            completed=2,
            encoding_fingerprint={"k": 1},
            codec_name=codec.name,
            codec_schema_version=codec.schema_version,
        ),
    )
    path = tmp_path / "manifest.json"
    payload = json.loads(path.read_text())
    payload["sample_ids"] = ["a", "c"]
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="sample_ids_digest"):
        read_cache_manifest(str(tmp_path), codec=codec)


def test_online_prepare_offline_canonical_batch_byte_parity(tmp_path: Path) -> None:
    from vforge.data.dataset import EncodedCacheDataset, RawSample, RawSampleDataset
    from vforge.offline_encoder import EncodeComponents, run_encode
    from vforge.tasks.wan.v21.shared import (
        ENCODED_BATCH_KEYS,
        WAN21_ENCODED_BATCH_CODEC,
        project_to_encoded_batch,
    )

    online = project_to_encoded_batch(
        {
            "prompt": torch.arange(8, dtype=torch.bfloat16).reshape(1, 2, 4),
            "frame_latents": torch.arange(16, dtype=torch.bfloat16).reshape(1, 2, 2, 2, 2),
            "image_condition_latent_vae": None,
            "image_condition_latent_clip": None,
            "image_condition_latent_dino": None,
            "condition_latents_normal": torch.ones(1, 1, 2, 2, 2, dtype=torch.bfloat16),
            "online_only": torch.tensor(1),
        }
    )
    dataset = RawSampleDataset([RawSample(sample_id="same", video="unused.mp4", prompt="prompt")])
    manifest = run_encode(
        EncodeComponents(
            dataset=dataset,
            sample_ids=("same",),
            encode_condition=lambda _sample: online,
            cache_dir=str(tmp_path),
            encoding_fingerprint={"encoder": "fake-stable"},
            codec=WAN21_ENCODED_BATCH_CODEC,
            skip_existing=False,
        ),
        device=torch.device("cpu"),
    )
    offline = EncodedCacheDataset(
        str(tmp_path),
        manifest.sample_ids,
        WAN21_ENCODED_BATCH_CODEC,
    )[0]
    for key in ENCODED_BATCH_KEYS:
        if online[key] is None:
            assert offline[key] is None
        else:
            assert torch.equal(offline[key], online[key])
