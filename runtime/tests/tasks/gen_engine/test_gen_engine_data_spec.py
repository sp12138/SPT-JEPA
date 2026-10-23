"""CPU JSONL/bundle integration for task-built single and multi-source specs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from vforge.data.cache import save_latent_bundle
from vforge.data.loader import build_resumable_loader
from vforge.data.multi_source import ALGORITHM, MultiDataset
from vforge.data.spec import DataSpec
from vforge.distributed.parallel_state import ParallelConfig, ParallelState
from vforge.hydra_config import config_from_dict
from vforge.tasks.gen_engine.data import build_data_spec
from vforge.tasks.gen_engine.dataset import MMGenDataset
from vforge.tasks.gen_engine.train import _build_train_dataloader
from vforge.tasks.requests import DataSourceSpec, TrainRequest
from vforge.tasks.translate import train_request


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(directory: Path, name: str, count: int) -> Path:
    directory.mkdir()
    records = []
    for index in range(count):
        embedding = directory / f"mme-{index}.pt"
        latent = directory / f"video-{index}.pt"
        save_latent_bundle({"embedding": torch.full((2, 2), float(index))}, str(embedding))
        save_latent_bundle({"latent": torch.full((1, 2, 1, 1), float(index))}, str(latent))
        records.append(
            {
                "schema": "mmgen.sample",
                "schema_version": 1,
                "sample_id": f"{name}:{index}",
                "task": "t2v",
                "caption": "CPU source routing fixture",
                "mme": {
                    "embedding_path": embedding.name,
                    "embedding_digest": _digest(embedding),
                    "shape": [2, 2],
                    "dtype": "float32",
                    "fingerprint": "cpu-data-spec-v1",
                },
                "references": [],
                "target": {
                    "video": {
                        "latent_path": latent.name,
                        "latent_digest": _digest(latent),
                        "latent_shape": [1, 2, 1, 1],
                        "latent_dtype": "float32",
                        "effective_fps": 1.0,
                    },
                },
            }
        )
    manifest = directory / "samples.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return manifest


@pytest.fixture
def source_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TrainRequest:
    monkeypatch.setenv("VFORGE_SCRATCH_DIR", str(tmp_path / "scratch"))
    first = _manifest(tmp_path / "a", "A", 2)
    second = _manifest(tmp_path / "b", "B", 3)
    return TrainRequest(
        sources=(
            DataSourceSpec(name="A", manifest=str(first), weight=0.11),
            DataSourceSpec(name="B", manifest=str(second), weight=0.89),
        ),
        max_steps=137,
        mix_seed=7,
        sampler_seed=42,
        pin_memory=False,
    )


def _loader(spec: DataSpec, request: TrainRequest):
    return build_resumable_loader(
        spec,
        parallel_state=ParallelState(ParallelConfig()),
        batch_size=request.batch_size,
        num_workers=0,
        drop_last=True,
        seed=request.sampler_seed,
    )


def _model_extras(tmp_path: Path) -> dict[str, Any]:
    transformer = tmp_path / "transformer.json"
    transformer.write_text(
        json.dumps(
            {
                "schema": "vforge.gen_engine.dit",
                "schema_version": 1,
                "architecture": "mmdit_dense",
                "num_attention_heads": 1,
                "attention_head_dim": 8,
                "hidden_size": 8,
                "num_layers": 1,
                "num_refiner_layers": 1,
                "ffn_dim": 16,
                "in_channels": 1,
                "audio_in_channels": 1,
                "patch_size": [1, 1, 1],
                "text_dim": 2,
                "freq_dim": 4,
                "time_embed_hidden_dim": 8,
                "time_embed_dim": 4,
                "rope_freq_dim": 1,
                "max_reference_index": 4,
            }
        ),
        encoding="utf-8",
    )
    normalization = {"type": "channelwise_mean_std", "mean": [0.0], "std": [1.0]}
    mapping = {"type": "stride", "input_stride": 1}
    video_codec = tmp_path / "video_codec.json"
    video_codec.write_text(
        json.dumps(
            {
                "schema": "vforge.video_latent_codec",
                "schema_version": 1,
                "latent_channels": 1,
                "spatial_compression": 1,
                "normalization": normalization,
                "temporal": {"input_rate_unit": "frames_per_second", "mapping": mapping},
            }
        ),
        encoding="utf-8",
    )
    audio_codec = tmp_path / "audio_codec.json"
    audio_codec.write_text(
        json.dumps(
            {
                "schema": "vforge.audio_latent_codec",
                "schema_version": 1,
                "latent_channels": 1,
                "planes": 2,
                "normalization": normalization,
                "temporal": {"input_rate_unit": "samples_per_second", "mapping": mapping},
            }
        ),
        encoding="utf-8",
    )
    return {
        "transformer_config_path": str(transformer),
        "video_latent_codec_config_path": str(video_codec),
        "audio_latent_codec_config_path": str(audio_codec),
    }


def test_multi_spec_reads_jsonl_bundles_and_replays_tail(source_request: TrainRequest) -> None:
    sequences = []
    for _ in range(5):
        spec, fingerprint = build_data_spec(source_request, collate_fn=list, total_samples=137)
        assert isinstance(spec.dataset, MultiDataset)
        assert len(spec.dataset) == 5  # Physical rows are independent of the 137-sample plan.
        assert [source.weight for source in spec.dataset.sources] == [11, 89]
        routed = spec.dataset[("B", 2)]
        assert routed["sample_id"] == "B:2"
        torch.testing.assert_close(
            routed["target"]["video"]["latent"], torch.full((1, 2, 1, 1), 2.0)
        )
        assert routed["mme_embedding"].device.type == "cpu"
        sequence = [batch[0]["sample_id"] for batch in _loader(spec, source_request)]
        assert Counter(sample.split(":")[0] for sample in sequence) == {"A": 15, "B": 122}
        for source_id, size in (("A", 2), ("B", 3)):
            local = [sample for sample in sequence if sample.startswith(source_id)]
            for start in range(0, len(local) - size + 1, size):
                assert set(local[start : start + size]) == {f"{source_id}:{i}" for i in range(size)}
        assert fingerprint["sampling"]["algorithm"] == ALGORITHM
        assert fingerprint["sampling"]["total_samples"] == 137
        sequences.append(sequence)
    assert all(sequence == sequences[0] for sequence in sequences[1:])


def test_single_manifest_preserves_dataset_and_fingerprint(source_request: TrainRequest) -> None:
    single = replace(source_request, train_manifest=source_request.sources[0].manifest, sources=())
    spec, fingerprint = build_data_spec(single, collate_fn=list)
    assert isinstance(spec.dataset, MMGenDataset)
    assert spec.sampler_factory is None
    assert fingerprint == {
        "source_kind": "mmgen_precomputed",
        "manifest": str(spec.dataset.manifest_path),
        "manifest_digest": spec.dataset.content_digest(exact=True),
        "sample_count": 2,
        "target_modalities": [],
        "require_digests": True,
    }
    assert len(list(_loader(spec, single))) == 2


def test_manifest_revision_changes_source_version(source_request: TrainRequest) -> None:
    _, before = build_data_spec(source_request, collate_fn=list, total_samples=137)
    path = Path(source_request.sources[0].manifest)
    path.write_text(
        path.read_text(encoding="utf-8").replace("fixture", "revision"), encoding="utf-8"
    )
    _, after = build_data_spec(source_request, collate_fn=list, total_samples=137)
    assert before["dataset"]["sources"][0]["version"] != after["dataset"]["sources"][0]["version"]
    assert before["dataset"]["sources"][1] == after["dataset"]["sources"][1]


@pytest.mark.parametrize("require_digests", ["false", 1])
def test_helper_rejects_non_boolean_digest_policy(
    source_request: TrainRequest, require_digests: Any
) -> None:
    changed = replace(source_request, extras={"require_digests": require_digests})
    with pytest.raises(ValueError, match="require_digests must be true or false"):
        build_data_spec(changed, collate_fn=list, total_samples=137)


@pytest.mark.parametrize("weight", [-1.0, float("inf"), float("nan")])
def test_rejects_invalid_weights(source_request: TrainRequest, weight: float) -> None:
    changed = replace(source_request, sources=(replace(source_request.sources[0], weight=weight),))
    with pytest.raises(ValueError, match="weight must be finite and nonnegative"):
        build_data_spec(changed, collate_fn=list, total_samples=137)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": ""}, "nonempty name"),
        ({"manifest": ""}, "requires a JSONL manifest"),
        ({"cache_dir": "/unused/cache"}, "do not support cache_dir"),
        ({"index_files": ("unused.jsonl",)}, "do not support index_files"),
        ({"min_latents": 1}, "do not support min_latents"),
        ({"weight": 0.0}, "at least one source weight"),
    ],
)
def test_rejects_unsupported_source_fields(
    source_request: TrainRequest, overrides: dict[str, Any], message: str
) -> None:
    changed = replace(source_request, sources=(replace(source_request.sources[0], **overrides),))
    with pytest.raises(ValueError, match=message):
        build_data_spec(changed, collate_fn=list, total_samples=137)


def test_rejects_excessive_exact_ratio(source_request: TrainRequest) -> None:
    changed = replace(
        source_request,
        sources=(
            replace(source_request.sources[0], weight=0.0000001),
            replace(source_request.sources[1], weight=1),
        ),
    )
    with pytest.raises(ValueError, match="MAX_BLOCK_SIZE"):
        build_data_spec(changed, collate_fn=list, total_samples=137)


def test_rejects_ambiguous_missing_or_unbounded_inputs(source_request: TrainRequest) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_data_spec(replace(source_request, train_manifest="unused.jsonl"), collate_fn=list)
    with pytest.raises(ValueError, match="train_manifest or data.sources"):
        build_data_spec(replace(source_request, sources=()), collate_fn=list)
    with pytest.raises(ValueError, match="positive total_samples"):
        build_data_spec(source_request, collate_fn=list)
    with pytest.raises(ValueError, match="shuffle=true"):
        build_data_spec(replace(source_request, shuffle=False), collate_fn=list, total_samples=137)


def test_existing_config_translation_supports_jsonl_sources(source_request: TrainRequest) -> None:
    config = config_from_dict(
        {
            "data": {
                "sources": [
                    {"name": source.name, "manifest": source.manifest, "weight": source.weight}
                    for source in source_request.sources
                ],
                "mix_seed": 7,
                "sampler_seed": 42,
                "pin_memory": False,
            },
            "runtime": {"max_steps": 137},
        }
    )
    parsed = train_request(config)
    spec, _ = build_data_spec(parsed, collate_fn=list, total_samples=parsed.max_steps)
    assert isinstance(spec.dataset, MultiDataset)
    assert [source.weight for source in spec.dataset.sources] == [11, 89]
    assert len(list(_loader(spec, parsed))) == 137


def test_train_builder_packs_multi_source_samples(
    source_request: TrainRequest, tmp_path: Path
) -> None:
    source_request = replace(source_request, extras=_model_extras(tmp_path))
    loader, fingerprint = _build_train_dataloader(source_request)
    batches = list(loader)
    assert len(batches) == source_request.max_steps
    assert Counter(batch["sample_id"][0].split(":")[0] for batch in batches) == {"A": 15, "B": 122}
    assert all(batch["target_video_rows"].shape == (1, 2, 1) for batch in batches)
    assert all(batch["mme_embedding"].shape == (1, 2, 2) for batch in batches)
    assert fingerprint["sampling"]["total_samples"] == 137
    assert fingerprint["sampling"]["shuffle_seed"] == source_request.sampler_seed
    assert fingerprint["packing"]["patch_size"] == [1, 1, 1]
    assert fingerprint["loader"]["batch_size"] == 1
    with pytest.raises(ValueError, match="max_steps > 0"):
        _build_train_dataloader(replace(source_request, max_steps=0))
