"""CPU contract tests for precomputed ``mmgen.sample`` manifests."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import pytest
import torch

from vforge.data.cache import save_latent_bundle
from vforge.tasks.gen_engine.dataset import (
    MMGEN_AUDIO_BUNDLE_CODEC,
    MMGEN_MME_BUNDLE_CODEC,
    MMGEN_VISUAL_BUNDLE_CODEC,
    MMGenDataset,
)


def _save_bundle(path: Path, field: str, tensor: torch.Tensor) -> str:
    save_latent_bundle({field: tensor}, str(path))
    return str(path)


def _mme(path: str, *, shape: list[int] | None = None) -> dict[str, Any]:
    return {
        "embedding_path": path,
        "shape": shape or [3, 4],
        "dtype": "float32",
        "fingerprint": "mme-test-v1",
    }


def _video(path: str, *, shape: list[int] | None = None) -> dict[str, Any]:
    return {
        "latent_path": path,
        "latent_shape": shape or [4, 3, 2, 2],
        "latent_dtype": "float32",
        "effective_fps": 8.0,
    }


def _audio(path: str, *, shape: list[int] | None = None) -> dict[str, Any]:
    return {
        "latent_path": path,
        "latent_shape": shape or [2, 4, 5],
        "latent_dtype": "float32",
        "effective_fps": 40.0,
    }


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _t2v(mme_path: str, target_path: str, **overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": "mmgen.sample",
        "schema_version": 1,
        "sample_id": "sample-1",
        "task": "t2v",
        "caption": "A test clip.",
        "mme": _mme(mme_path),
        "references": [],
        "target": {"video": _video(target_path)},
    }
    record.update(overrides)
    return record


def _valid_t2v_files(tmp_path: Path) -> tuple[str, str]:
    mme_path = _save_bundle(tmp_path / "mme.pt", "embedding", torch.randn(3, 4))
    target_path = _save_bundle(tmp_path / "target.pt", "latent", torch.randn(4, 3, 2, 2))
    return mme_path, target_path


def test_codecs_define_portable_bundle_field_names() -> None:
    assert MMGEN_MME_BUNDLE_CODEC.required_keys == ("embedding",)
    assert MMGEN_VISUAL_BUNDLE_CODEC.required_keys == ("latent",)
    assert MMGEN_AUDIO_BUNDLE_CODEC.required_keys == ("latent",)


def test_t2v_loads_precomputed_mme_and_visual_target(tmp_path: Path) -> None:
    mme_path, target_path = _valid_t2v_files(tmp_path)
    path = _write_manifest(tmp_path / "samples.jsonl", [_t2v(mme_path, target_path)])

    dataset = MMGenDataset(path)
    sample = dataset[0]

    assert dataset.sample_ids == ["sample-1"]
    assert sample["sample_id"] == "sample-1"
    assert sample["caption"] == "A test clip."
    assert sample["references"] == []
    assert sample["mme_embedding"].shape == (3, 4)
    assert sample["target"]["video"]["latent"].shape == (4, 3, 2, 2)
    assert sample["target"]["video"]["effective_fps"] == 8.0


def test_nested_paths_resolve_relative_to_the_manifest(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "nested" / "cache"
    bundle_dir.mkdir(parents=True)
    _save_bundle(bundle_dir / "mme.pt", "embedding", torch.randn(3, 4))
    _save_bundle(bundle_dir / "target.pt", "latent", torch.randn(4, 3, 2, 2))
    record = _t2v("nested/cache/mme.pt", "nested/cache/target.pt")
    record["target"]["video"]["source_path"] = "nested/media/target.mp4"
    path = _write_manifest(tmp_path / "samples.jsonl", [record])

    dataset = MMGenDataset(path)
    normalized = dataset.records[0]

    assert normalized["mme"]["embedding_path"] == str(bundle_dir / "mme.pt")
    assert normalized["target"]["video"]["latent_path"] == str(bundle_dir / "target.pt")
    assert normalized["target"]["video"]["source_path"] == str(
        tmp_path / "nested" / "media" / "target.mp4"
    )
    assert dataset[0]["target"]["video"]["latent"].shape == (4, 3, 2, 2)


def test_r2v_strict_order_and_independent_reference_numbers(tmp_path: Path) -> None:
    paths: dict[str, str] = {}
    paths["mme"] = _save_bundle(tmp_path / "mme.pt", "embedding", torch.randn(3, 4))
    paths["image"] = _save_bundle(
        tmp_path / "image.pt",
        "latent",
        torch.randn(4, 1, 2, 2),
    )
    paths["pair_video"] = _save_bundle(
        tmp_path / "pair-video.pt",
        "latent",
        torch.randn(4, 3, 2, 2),
    )
    paths["pair_audio"] = _save_bundle(
        tmp_path / "pair-audio.pt",
        "latent",
        torch.randn(2, 4, 5),
    )
    paths["video"] = _save_bundle(
        tmp_path / "video.pt",
        "latent",
        torch.randn(4, 3, 2, 2),
    )
    paths["audio"] = _save_bundle(
        tmp_path / "audio.pt",
        "latent",
        torch.randn(2, 4, 5),
    )
    paths["target"] = _save_bundle(
        tmp_path / "target.pt",
        "latent",
        torch.randn(4, 3, 2, 2),
    )
    references = [
        {
            "ref_id": "image",
            "type": "image",
            "image": {
                "latent_path": paths["image"],
                "latent_shape": [4, 1, 2, 2],
                "latent_dtype": "float32",
            },
        },
        {
            "ref_id": "pair",
            "type": "video_audio",
            "video": _video(paths["pair_video"]),
            "audio": _audio(paths["pair_audio"]),
        },
        {
            "ref_id": "video",
            "type": "video",
            "video": _video(paths["video"]),
        },
        {
            "ref_id": "audio",
            "type": "audio",
            "audio": _audio(paths["audio"]),
        },
    ]
    record = _t2v(
        paths["mme"],
        paths["target"],
        task="r2v-video_edit",
        references=references,
    )
    sample = MMGenDataset(_write_manifest(tmp_path / "samples.jsonl", [record]))[0]

    assert [reference["reference_numbers"] for reference in sample["references"]] == [
        {"image": 1},
        {"video": 1, "audio": 1},
        {"video": 2},
        {"audio": 2},
    ]
    assert sample["references"][0]["image"]["latent"].shape == (4, 1, 2, 2)
    assert sample["references"][1]["audio"]["latent"].shape == (2, 4, 5)


@pytest.mark.parametrize(
    ("task", "references", "message"),
    [
        (
            "t2v",
            [{"ref_id": "a", "type": "audio", "audio": {"latent_path": "a.pt"}}],
            "t2v samples must not have references",
        ),
        ("r2v", [], "r2v samples require at least one reference"),
    ],
)
def test_task_controls_reference_presence(
    tmp_path: Path,
    task: str,
    references: list[dict[str, Any]],
    message: str,
) -> None:
    mme_path, target_path = _valid_t2v_files(tmp_path)
    record = _t2v(mme_path, target_path, task=task, references=references)

    with pytest.raises(ValueError, match=message):
        MMGenDataset(_write_manifest(tmp_path / "samples.jsonl", [record]))[0]


def test_reference_order_is_trusted_but_media_types_are_validated(tmp_path: Path) -> None:
    mme_path, target_path = _valid_t2v_files(tmp_path)
    audio = _save_bundle(tmp_path / "audio.pt", "latent", torch.randn(2, 4, 5))
    image = _save_bundle(tmp_path / "image.pt", "latent", torch.randn(4, 1, 2, 2))
    references = [
        {"ref_id": "a", "type": "audio", "audio": _audio(audio)},
        {
            "ref_id": "i",
            "type": "image",
            "image": {
                "latent_path": image,
                "latent_shape": [4, 1, 2, 2],
            },
        },
    ]
    record = _t2v(mme_path, target_path, task="r2v", references=references)

    sample = MMGenDataset(_write_manifest(tmp_path / "producer-order.jsonl", [record]))[0]
    assert [reference["type"] for reference in sample["references"]] == ["audio", "image"]
    assert [reference["reference_numbers"] for reference in sample["references"]] == [
        {"audio": 1},
        {"image": 1},
    ]

    record["references"] = [
        {
            "ref_id": "bad",
            "type": "video_audio",
            "video": _video(target_path),
        }
    ]
    with pytest.raises(ValueError, match="requires media"):
        MMGenDataset(_write_manifest(tmp_path / "media.jsonl", [record]))[0]


def test_manifest_rejects_duplicate_sample_and_reference_ids(tmp_path: Path) -> None:
    mme_path, target_path = _valid_t2v_files(tmp_path)
    record = _t2v(mme_path, target_path)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        _sample_ids = MMGenDataset(
            _write_manifest(tmp_path / "samples.jsonl", [record, record])
        ).sample_ids

    audio = _save_bundle(tmp_path / "audio.pt", "latent", torch.randn(2, 4, 5))
    reference = {"ref_id": "same", "type": "audio", "audio": _audio(audio)}
    record = _t2v(
        mme_path,
        target_path,
        task="r2v",
        references=[reference, reference],
    )
    with pytest.raises(ValueError, match="unique within the sample"):
        MMGenDataset(_write_manifest(tmp_path / "refs.jsonl", [record]))[0]


@pytest.mark.parametrize("target_keys", [("video",), ("audio",), ("video", "audio")])
def test_target_accepts_every_supported_modality_combination(
    tmp_path: Path,
    target_keys: tuple[str, ...],
) -> None:
    mme = _save_bundle(tmp_path / "mme.pt", "embedding", torch.randn(3, 4))
    visual = _save_bundle(tmp_path / "visual.pt", "latent", torch.randn(4, 3, 2, 2))
    audio = _save_bundle(tmp_path / "audio.pt", "latent", torch.randn(2, 4, 5))
    target: dict[str, Any] = {}
    if "video" in target_keys:
        target["video"] = _video(visual)
    if "audio" in target_keys:
        target["audio"] = _audio(audio)
    record = _t2v(mme, visual, target=target)

    sample = MMGenDataset(_write_manifest(tmp_path / "samples.jsonl", [record]))[0]

    assert tuple(sample["target"]) == target_keys


def test_target_selection_only_requires_and_loads_selected_latents(tmp_path: Path) -> None:
    mme, visual = _valid_t2v_files(tmp_path)
    record = _t2v(mme, visual)
    record["target"]["audio"] = {"source_path": "media/unused.wav"}
    path = _write_manifest(tmp_path / "samples.jsonl", [record])

    sample = MMGenDataset(path, target_modalities=("video",))[0]

    assert set(sample["target"]) == {"video"}


def test_target_and_target_spec_are_exclusive_and_one_is_required(tmp_path: Path) -> None:
    mme, visual = _valid_t2v_files(tmp_path)
    record = _t2v(mme, visual)
    record["target_spec"] = {
        "modalities": ["video"],
        "duration_seconds": 2,
        "height": 64,
        "width": 96,
        "fps": 24,
    }
    with pytest.raises(ValueError, match="exactly one"):
        MMGenDataset(_write_manifest(tmp_path / "both.jsonl", [record]))[0]

    record.pop("target")
    record.pop("target_spec")
    with pytest.raises(ValueError, match="exactly one"):
        MMGenDataset(_write_manifest(tmp_path / "neither.jsonl", [record]))[0]


@pytest.mark.parametrize(
    "target_spec",
    [
        {
            "modalities": ["video"],
            "duration_seconds": 2.0,
            "height": 64,
            "width": 96,
            "fps": 24,
        },
        {
            "modalities": ["audio"],
            "duration_seconds": 2.0,
            "sample_rate": 48_000,
        },
        {
            "modalities": ["video", "audio"],
            "duration_seconds": 2.0,
            "video": {"height": 64, "width": 96, "fps": 24},
            "audio": {"sample_rate": 48_000},
        },
    ],
)
def test_target_spec_supports_video_audio_and_joint_outputs(
    tmp_path: Path,
    target_spec: dict[str, Any],
) -> None:
    mme, visual = _valid_t2v_files(tmp_path)
    record = _t2v(mme, visual, target_spec=target_spec)
    record.pop("target")

    sample = MMGenDataset(_write_manifest(tmp_path / "samples.jsonl", [record]))[0]

    assert sample["target_spec"]["modalities"] == target_spec["modalities"]
    assert "target" not in sample


def test_target_modalities_select_an_inference_subset(tmp_path: Path) -> None:
    mme, visual = _valid_t2v_files(tmp_path)
    record = _t2v(
        mme,
        visual,
        target_spec={
            "modalities": ["video", "audio"],
            "duration_seconds": 2.0,
            "video": {"height": 64, "width": 96, "fps": 24},
            "audio": {"sample_rate": 48_000},
        },
    )
    record.pop("target")

    sample = MMGenDataset(
        _write_manifest(tmp_path / "samples.jsonl", [record]),
        target_modalities=("audio",),
    )[0]

    assert sample["target_spec"]["modalities"] == ["audio"]


@pytest.mark.parametrize(
    ("target_spec", "message"),
    [
        (
            {"modalities": ["video"], "duration_seconds": 2, "width": 64, "fps": 24},
            "height",
        ),
        (
            {"modalities": ["audio"], "duration_seconds": 2},
            "sample_rate",
        ),
        (
            {"modalities": ["audio"], "sample_rate": 48_000},
            "duration_seconds or num_samples",
        ),
    ],
)
def test_target_spec_requires_geometry_duration_and_sample_rate(
    tmp_path: Path,
    target_spec: dict[str, Any],
    message: str,
) -> None:
    mme, visual = _valid_t2v_files(tmp_path)
    record = _t2v(mme, visual, target_spec=target_spec)
    record.pop("target")

    with pytest.raises(ValueError, match=message):
        MMGenDataset(_write_manifest(tmp_path / "samples.jsonl", [record]))[0]


def test_bundle_io_is_lazy_and_runtime_paths_are_mandatory(tmp_path: Path) -> None:
    missing_mme = tmp_path / "missing-mme.pt"
    missing_target = tmp_path / "missing-target.pt"
    record = _t2v(str(missing_mme), str(missing_target))
    dataset = MMGenDataset(_write_manifest(tmp_path / "samples.jsonl", [record]))

    with pytest.raises(ValueError, match="failed to load bundle"):
        dataset[0]

    record["mme"] = {"shape": [3, 4], "dtype": "float32"}
    with pytest.raises(ValueError, match="embedding_path is required"):
        MMGenDataset(_write_manifest(tmp_path / "no-mme.jsonl", [record]))[0]


def test_getitem_validates_declared_shape_dtype_and_audio_contract(tmp_path: Path) -> None:
    mme, visual = _valid_t2v_files(tmp_path)
    record = _t2v(mme, visual)
    record["mme"]["shape"] = [99, 4]
    dataset = MMGenDataset(_write_manifest(tmp_path / "shape.jsonl", [record]))
    with pytest.raises(ValueError, match="does not match tensor shape"):
        dataset[0]

    record["mme"]["shape"] = [3, 4]
    record["mme"]["dtype"] = "float16"
    dataset = MMGenDataset(_write_manifest(tmp_path / "dtype.jsonl", [record]))
    with pytest.raises(ValueError, match="does not match tensor dtype"):
        dataset[0]

    bad_audio = _save_bundle(tmp_path / "bad-audio.pt", "latent", torch.randn(1, 4, 5))
    record["mme"]["dtype"] = "float32"
    record["target"] = {
        "audio": {
            "latent_path": bad_audio,
            "latent_shape": [2, 4, 5],
            "latent_dtype": "float32",
            "effective_fps": 40.0,
        }
    }
    dataset = MMGenDataset(_write_manifest(tmp_path / "audio.jsonl", [record]))
    with pytest.raises(ValueError, match="leading A dimension must be 2"):
        dataset[0]


def test_temporal_latents_require_finite_positive_effective_fps(tmp_path: Path) -> None:
    mme, visual = _valid_t2v_files(tmp_path)
    record = _t2v(mme, visual)
    del record["target"]["video"]["effective_fps"]
    with pytest.raises(ValueError, match="effective_fps is required"):
        MMGenDataset(_write_manifest(tmp_path / "missing.jsonl", [record]))[0]

    record["target"]["video"]["effective_fps"] = float("inf")
    with pytest.raises(ValueError, match="finite positive"):
        MMGenDataset(_write_manifest(tmp_path / "invalid.jsonl", [record]))[0]


def test_declared_digests_are_verified_lazily_and_can_be_required(tmp_path: Path) -> None:
    mme, visual = _valid_t2v_files(tmp_path)
    record = _t2v(mme, visual)
    record["mme"]["embedding_digest"] = "sha256:" + "0" * 64
    dataset = MMGenDataset(_write_manifest(tmp_path / "mismatch.jsonl", [record]))
    with pytest.raises(ValueError, match="digest mismatch"):
        dataset[0]

    record["mme"].pop("embedding_digest")
    with pytest.raises(ValueError, match="embedding_digest is required"):
        MMGenDataset(
            _write_manifest(tmp_path / "required.jsonl", [record]),
            require_digests=True,
        )[0]

    record["mme"]["embedding_digest"] = (
        "sha256:" + hashlib.sha256(Path(mme).read_bytes()).hexdigest()
    )
    record["target"]["video"]["latent_digest"] = (
        "sha256:" + hashlib.sha256(Path(visual).read_bytes()).hexdigest()
    )
    sample = MMGenDataset(
        _write_manifest(tmp_path / "valid.jsonl", [record]),
        require_digests=True,
    )[0]
    assert sample["target"]["video"]["latent"].shape == (4, 3, 2, 2)


def test_content_digest_hashes_exact_manifest_without_loading_bundles(
    tmp_path: Path,
) -> None:
    missing_mme = str(tmp_path / "missing-mme.pt")
    missing_target = str(tmp_path / "missing-target.pt")
    first = _t2v(missing_mme, missing_target)
    path = _write_manifest(tmp_path / "samples.jsonl", [first])
    dataset = MMGenDataset(path)
    exact_digest = dataset.content_digest(exact=True)
    assert dataset.content_digest() == exact_digest

    path.write_text(json.dumps(first, indent=None) + "\n\n", encoding="utf-8")
    reformatted = MMGenDataset(path)
    assert reformatted.content_digest(exact=True) != exact_digest

    first["mme"]["fingerprint"] = "mme-test-v2"
    changed = MMGenDataset(_write_manifest(path, [first]))
    assert changed.content_digest() != exact_digest


def test_lazy_validation_defers_record_parsing_until_access(tmp_path: Path) -> None:
    mme, visual = _valid_t2v_files(tmp_path)
    record = _t2v(mme, visual)
    record["schema_version"] = 2
    path = _write_manifest(tmp_path / "lazy.jsonl", [record])

    dataset = MMGenDataset(path)

    assert len(dataset) == 1
    assert dataset.content_digest() == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="schema_version must be 1"):
        dataset[0]


def test_lazy_dataset_reopens_manifest_after_pickling(tmp_path: Path) -> None:
    mme, visual = _valid_t2v_files(tmp_path)
    dataset = MMGenDataset(_write_manifest(tmp_path / "pickle.jsonl", [_t2v(mme, visual)]))
    assert dataset[0]["sample_id"] == "sample-1"

    restored = pickle.loads(pickle.dumps(dataset))

    assert restored[0]["sample_id"] == "sample-1"


def test_lazy_validation_enforces_model_reference_capacity_on_access(tmp_path: Path) -> None:
    mme, visual = _valid_t2v_files(tmp_path)
    audio = _save_bundle(tmp_path / "audio.pt", "latent", torch.randn(2, 4, 5))
    record = _t2v(
        mme,
        visual,
        task="r2v",
        references=[
            {"ref_id": "audio-1", "type": "audio", "audio": _audio(audio)},
            {"ref_id": "audio-2", "type": "audio", "audio": _audio(audio)},
        ],
    )
    dataset = MMGenDataset(
        _write_manifest(tmp_path / "capacity.jsonl", [record]),
        max_reference_index=1,
    )

    with pytest.raises(ValueError, match="exceeds max_reference_index=1"):
        dataset[0]


def test_manifest_must_be_jsonl_schema_v1(tmp_path: Path) -> None:
    mme, visual = _valid_t2v_files(tmp_path)
    record = _t2v(mme, visual)
    path = tmp_path / "samples.json"
    path.write_text(json.dumps([record]), encoding="utf-8")
    with pytest.raises(ValueError, match="must be JSONL"):
        MMGenDataset(path)

    record["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version must be 1"):
        MMGenDataset(_write_manifest(tmp_path / "version.jsonl", [record]))[0]
