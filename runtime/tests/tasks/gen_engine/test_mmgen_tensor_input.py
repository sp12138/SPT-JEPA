"""Contracts for direct tensor delivery and pinned MMGen artifact identity."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import pytest
import torch

from vforge.data.cache import save_latent_bundle
from vforge.tasks.gen_engine.dataset import MMGenDataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path, record: dict[str, Any]) -> Path:
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


def _save(path: Path, tensor: torch.Tensor, field: str, *, bundle: bool = False) -> None:
    if bundle:
        save_latent_bundle({field: tensor}, str(path))
    else:
        torch.save(tensor, path)


def _fixture(
    tmp_path: Path, *, bundle: bool = False
) -> tuple[Path, dict[str, Any], dict[str, torch.Tensor]]:
    tensors = {
        "mme": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "video": torch.arange(48, dtype=torch.float32).reshape(4, 3, 2, 2),
        "audio": torch.arange(30, dtype=torch.float32).reshape(2, 5, 3),
    }
    for name, tensor in tensors.items():
        _save(
            tmp_path / f"{name}.pt",
            tensor,
            "embedding" if name == "mme" else "latent",
            bundle=bundle,
        )
    record = {
        "schema": "mmgen.sample",
        "schema_version": 1,
        "sample_id": "direct-tensor-1",
        "task": "t2v",
        "caption": "Direct tensor input contract.",
        "mme": {
            "embedding_path": "mme.pt",
            "shape": [3, 4],
            "dtype": "float32",
            "fingerprint": "direct-tensor-v1",
        },
        "references": [],
        "target": {
            "video": {
                "latent_path": "video.pt",
                "latent_shape": [4, 3, 2, 2],
                "latent_dtype": "float32",
                "effective_fps": 30.0,
            },
            "audio": {
                "latent_path": "audio.pt",
                "latent_shape": [2, 5, 3],
                "latent_dtype": "float32",
                "effective_fps": 32000.0,
            },
        },
    }
    return _manifest(tmp_path / "samples.jsonl", record), record, tensors


@pytest.mark.parametrize("tensor_format", ["tensor", "auto"])
def test_direct_tensor_preserves_values_and_uses_safe_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tensor_format: str
) -> None:
    path, _, tensors = _fixture(tmp_path)
    original_load = torch.load
    calls: list[Any] = []

    def checked_load(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    sample = MMGenDataset.from_manifest(path, tensor_format=tensor_format)[0]

    assert torch.equal(sample["mme_embedding"], tensors["mme"])
    for modality in ("video", "audio"):
        assert torch.equal(sample["target"][modality]["latent"], tensors[modality])
    assert calls and all(value is True for value in calls)


@pytest.mark.parametrize("tensor_format", ["bundle", "auto"])
def test_bundle_delivery_remains_supported(tmp_path: Path, tensor_format: str) -> None:
    path, _, tensors = _fixture(tmp_path, bundle=True)
    sample = MMGenDataset(path, tensor_format=tensor_format)[0]
    assert torch.equal(sample["mme_embedding"], tensors["mme"])
    assert torch.equal(sample["target"]["video"]["latent"], tensors["video"])


def test_auto_accepts_mixed_raw_and_bundle_artifacts(tmp_path: Path) -> None:
    path, _, tensors = _fixture(tmp_path)
    _save(tmp_path / "video.pt", tensors["video"], "latent", bundle=True)
    sample = MMGenDataset(path, tensor_format="auto")[0]
    assert torch.equal(sample["mme_embedding"], tensors["mme"])
    assert torch.equal(sample["target"]["video"]["latent"], tensors["video"])


@pytest.mark.parametrize(("bundle", "tensor_format"), [(False, "bundle"), (True, "tensor")])
def test_explicit_format_rejects_the_other_container(
    tmp_path: Path, bundle: bool, tensor_format: str
) -> None:
    path, _, _ = _fixture(tmp_path, bundle=bundle)
    with pytest.raises(ValueError):
        MMGenDataset(path, tensor_format=tensor_format)[0]


def _write_unpickle_marker(path: str) -> torch.Tensor:
    Path(path).write_text("unsafe unpickle executed", encoding="utf-8")
    return torch.zeros(3, 4)


class _UnsafePayload:
    def __init__(self, marker: Path) -> None:
        self.marker = str(marker)

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return _write_unpickle_marker, (self.marker,)


@pytest.mark.parametrize("tensor_format", ["tensor", "auto"])
def test_unsafe_pickle_is_rejected_without_an_unrestricted_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tensor_format: str
) -> None:
    path, _, _ = _fixture(tmp_path)
    marker = tmp_path / "unpickle-marker"
    torch.save(_UnsafePayload(marker), tmp_path / "mme.pt")
    original_load = torch.load
    calls: list[Any] = []

    def checked_load(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    with pytest.raises((ValueError, pickle.UnpicklingError)):
        MMGenDataset(path, tensor_format=tensor_format)[0]
    assert not marker.exists()
    assert calls and all(value is True for value in calls)


@pytest.mark.parametrize("payload", [{"embedding": torch.zeros(3, 4)}, [1, 2, 3], 1])
def test_auto_rejects_objects_that_are_neither_tensor_nor_bundle(
    tmp_path: Path, payload: Any
) -> None:
    path, _, _ = _fixture(tmp_path)
    torch.save(payload, tmp_path / "mme.pt")
    with pytest.raises(ValueError):
        MMGenDataset(path, tensor_format="auto")[0]


def test_auto_rejects_unknown_bundle_version(tmp_path: Path) -> None:
    path, _, _ = _fixture(tmp_path)
    torch.save(
        {"version": 999, "fields": {"embedding": {"kind": "tensor", "tensor": torch.zeros(3, 4)}}},
        tmp_path / "mme.pt",
    )
    with pytest.raises(ValueError, match="version"):
        MMGenDataset(path, tensor_format="auto")[0]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shape", [4, 3], "does not match tensor shape"),
        ("dtype", "bfloat16", "does not match tensor dtype"),
    ],
)
def test_tensor_delivery_checks_declared_geometry_and_dtype(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    path, record, _ = _fixture(tmp_path)
    record["mme"][field] = value
    _manifest(path, record)
    with pytest.raises(ValueError, match=message):
        MMGenDataset(path, tensor_format="tensor")[0]


def test_atc_normalizes_target_and_reference_after_source_validation(tmp_path: Path) -> None:
    path, record, tensors = _fixture(tmp_path)
    record["task"] = "r2v"
    record["references"] = [
        {"ref_id": "sound-1", "type": "audio", "audio": dict(record["target"]["audio"])},
    ]
    _manifest(path, record)
    dataset = MMGenDataset(path, tensor_format="tensor", audio_layout="ATC")
    original_records = dataset.records
    sample = dataset[0]
    expected = tensors["audio"].transpose(1, 2).contiguous()

    for audio in (sample["target"]["audio"], sample["references"][0]["audio"]):
        assert torch.equal(audio["latent"], expected)
        assert torch.equal(audio["latent"].transpose(1, 2), tensors["audio"])
        assert audio["latent_shape"] == [2, 3, 5]
        assert audio["effective_fps"] == 32000.0
        assert audio["latent"].is_contiguous()
    assert dataset.records == original_records
    assert dataset.records[0]["target"]["audio"]["latent_shape"] == [2, 5, 3]
    assert dataset.records[0]["references"][0]["audio"]["latent_shape"] == [2, 5, 3]


def test_atc_does_not_validate_against_an_already_transposed_declaration(tmp_path: Path) -> None:
    path, record, _ = _fixture(tmp_path)
    record["target"]["audio"]["latent_shape"] = [2, 3, 5]
    _manifest(path, record)
    with pytest.raises(ValueError, match="does not match tensor shape"):
        MMGenDataset(path, tensor_format="tensor", audio_layout="ATC")[0]


def test_pinned_identity_uses_artifact_bytes_and_excludes_the_cache_location(
    tmp_path: Path,
) -> None:
    path, _, _ = _fixture(tmp_path)
    first = MMGenDataset(
        path, tensor_format="tensor", pin_artifacts=True, artifact_cache_dir=tmp_path / "cache-a"
    )
    second = MMGenDataset(
        path, tensor_format="tensor", pin_artifacts=True, artifact_cache_dir=tmp_path / "cache-b"
    )
    assert first.input_contract() == second.input_contract()
    assert first.data_version() == second.data_version()
    contract = json.dumps(first.input_contract(), sort_keys=True)
    assert first.input_contract()["artifacts"]["artifact_count"] == 3
    assert str(tmp_path / "cache-a") not in contract
    assert first.content_digest() == _sha256(path)
    assert first.data_version() != first.content_digest()


@pytest.mark.parametrize("artifact", ["mme", "video", "audio"])
def test_pinned_artifact_mutation_rejects_access_and_changes_a_new_dataset_version(
    tmp_path: Path,
    artifact: str,
) -> None:
    path, _, tensors = _fixture(tmp_path)
    original = MMGenDataset(path, tensor_format="tensor", pin_artifacts=True)
    version = original.data_version()
    manifest_digest = original.content_digest()
    torch.save(tensors[artifact] + 1, tmp_path / f"{artifact}.pt")

    with pytest.raises(ValueError, match="digest|changed|mismatch"):
        original[0]
    revised = MMGenDataset(path, tensor_format="tensor", pin_artifacts=True)
    assert revised.content_digest() == manifest_digest
    assert revised.data_version() != version
    sample = revised[0]
    actual = sample["mme_embedding"] if artifact == "mme" else sample["target"][artifact]["latent"]
    assert torch.equal(actual, tensors[artifact] + 1)


def test_reference_artifact_is_pinned_even_when_it_is_not_a_target(tmp_path: Path) -> None:
    path, record, tensors = _fixture(tmp_path)
    torch.save(tensors["audio"], tmp_path / "reference-audio.pt")
    record["task"] = "r2v"
    reference = dict(record["target"]["audio"], latent_path="reference-audio.pt")
    record["references"] = [{"ref_id": "sound-1", "type": "audio", "audio": reference}]
    _manifest(path, record)
    dataset = MMGenDataset(path, tensor_format="tensor", pin_artifacts=True)
    version = dataset.data_version()
    torch.save(tensors["audio"] + 1, tmp_path / "reference-audio.pt")
    with pytest.raises(ValueError, match="digest|changed|mismatch"):
        dataset[0]
    assert MMGenDataset(path, tensor_format="tensor", pin_artifacts=True).data_version() != version


def test_pinned_manifest_rejects_an_in_place_revision(tmp_path: Path) -> None:
    path, record, _ = _fixture(tmp_path)
    dataset = MMGenDataset(path, tensor_format="tensor", pin_artifacts=True)
    record["caption"] = "An edited source record must not use the old content identity."
    _manifest(path, record)
    with pytest.raises(ValueError, match="manifest changed"):
        dataset[0]


def test_pickling_preserves_pinned_identity_and_runtime_audio_conversion(tmp_path: Path) -> None:
    path, _, _ = _fixture(tmp_path)
    dataset = MMGenDataset(path, tensor_format="tensor", audio_layout="ATC", pin_artifacts=True)
    baseline = dataset[0]
    restored = pickle.loads(pickle.dumps(dataset))

    assert restored.input_contract() == dataset.input_contract()
    assert restored.data_version() == dataset.data_version()
    assert torch.equal(
        restored[0]["target"]["audio"]["latent"], baseline["target"]["audio"]["latent"]
    )
    torch.save(torch.zeros(3, 4), tmp_path / "mme.pt")
    with pytest.raises(ValueError, match="digest|changed|mismatch"):
        restored[0]


def test_pin_checks_selected_targets_and_does_not_fetch_original_media(tmp_path: Path) -> None:
    path, record, _ = _fixture(tmp_path)
    record["target"]["video"]["source_path"] = "s3://unused-original-media/source.mp4"
    record["target"]["video"]["source_digest"] = "sha256:" + "0" * 64
    record["target"]["audio"]["latent_path"] = "unselected-missing-audio.pt"
    _manifest(path, record)
    dataset = MMGenDataset(
        path, tensor_format="tensor", pin_artifacts=True, target_modalities=("video",)
    )

    assert set(dataset[0]["target"]) == {"video"}
    contract = json.dumps(dataset.input_contract())
    assert "unused-original-media" not in contract
    assert "unselected-missing-audio" not in contract


def test_format_and_layout_are_part_of_the_resume_identity(tmp_path: Path) -> None:
    path, _, _ = _fixture(tmp_path)
    direct = MMGenDataset(path, tensor_format="tensor")
    automatic = MMGenDataset(path, tensor_format="auto")
    transposed = MMGenDataset(path, tensor_format="tensor", audio_layout="ATC")

    assert len({direct.data_version(), automatic.data_version(), transposed.data_version()}) == 3
    assert direct.input_contract() != automatic.input_contract()
    assert direct.input_contract() != transposed.input_contract()


def test_default_contract_remains_lazy_and_preserves_manifest_digest(tmp_path: Path) -> None:
    path, record, _ = _fixture(tmp_path)
    record["schema_version"] = 99
    _manifest(path, record)
    for name in ("mme", "video", "audio"):
        (tmp_path / f"{name}.pt").unlink()

    dataset = MMGenDataset(path)
    assert len(dataset) == 1
    assert dataset.input_contract() is None
    assert dataset.data_version() == dataset.content_digest() == _sha256(path)
    with pytest.raises(ValueError, match="schema_version"):
        dataset[0]


def test_raw_tensor_declared_digest_is_checked_even_without_pinning(tmp_path: Path) -> None:
    path, record, _ = _fixture(tmp_path)
    record["mme"]["embedding_digest"] = "sha256:" + "0" * 64
    _manifest(path, record)
    with pytest.raises(ValueError, match="digest mismatch"):
        MMGenDataset(path, tensor_format="tensor")[0]


def test_pinning_does_not_satisfy_a_required_upstream_digest(tmp_path: Path) -> None:
    path, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="digest is required"):
        dataset = MMGenDataset(
            path, tensor_format="tensor", pin_artifacts=True, require_digests=True
        )
        dataset[0]


@pytest.mark.parametrize(
    "options",
    [
        {"tensor_format": "guess"},
        {"audio_layout": "guess"},
    ],
)
def test_invalid_reading_options_are_rejected(tmp_path: Path, options: dict[str, Any]) -> None:
    path, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError):
        MMGenDataset(path, **options)
