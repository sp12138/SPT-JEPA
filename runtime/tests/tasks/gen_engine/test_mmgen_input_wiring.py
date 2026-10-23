"""Exercise file-format and frozen-input contracts through train/eval entry wiring."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from vforge.data.cache import save_latent_bundle
from vforge.tasks.gen_engine.data import build_data_spec
from vforge.tasks.gen_engine.eval import _open_mmgen_dataset
from vforge.tasks.requests import DataSourceSpec, EvalRequest, TrainRequest
from vforge.utils import s3_staging

INPUT_OPTIONS = {
    "require_digests": False,
    "tensor_format": "tensor",
    "audio_layout": "ATC",
    "pin_artifacts": True,
}


def _fixture(directory: Path, *, bundle=False):
    directory.mkdir()
    tensors = {
        "mme": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "image": torch.ones(4, 1, 2, 2),
        "ref_video": torch.ones(4, 2, 2, 2),
        "ref_audio": torch.arange(30, dtype=torch.float32).reshape(2, 5, 3),
        "video": torch.ones(4, 2, 2, 2),
        "audio": torch.arange(30, dtype=torch.float32).reshape(2, 5, 3),
    }
    for name, tensor in tensors.items():
        path = directory / f"{name}.pt"
        if bundle:
            save_latent_bundle({"embedding" if name == "mme" else "latent": tensor}, str(path))
        else:
            torch.save(tensor, path)

    def media(name):
        value = {
            "latent_path": f"{name}.pt",
            "latent_shape": list(tensors[name].shape),
            "latent_dtype": "float32",
            "source_path": f"s3://unused-provenance/{name}.raw",
        }
        if name != "image":
            value["effective_fps"] = 24.0
        return value

    record = {
        "schema": "mmgen.sample",
        "schema_version": 1,
        "sample_id": directory.name,
        "task": "r2v",
        "caption": "Wiring contract fixture.",
        "mme": {
            "embedding_path": "mme.pt",
            "shape": [3, 4],
            "dtype": "float32",
        },
        "references": [
            {"ref_id": "picture", "type": "image", "image": media("image")},
            {
                "ref_id": "clip",
                "type": "video_audio",
                "video": media("ref_video"),
                "audio": media("ref_audio"),
            },
        ],
        "target": {"video": media("video"), "audio": media("audio")},
    }
    manifest = directory / "samples.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return manifest, tensors


def test_single_train_wires_format_layout_cache_and_frozen_identity(tmp_path):
    manifest, tensors = _fixture(tmp_path / "source")
    request = TrainRequest(
        train_manifest=str(manifest),
        extras={**INPUT_OPTIONS, "artifact_cache_dir": str(tmp_path / "cache")},
    )
    spec, contract = build_data_spec(request, collate_fn=list)
    assert contract["input"] == spec.dataset.input_contract()
    assert contract["input"]["artifacts"]["artifact_count"] == 6
    assert contract["input"]["tensor_format"] == "tensor"
    assert contract["input"]["audio_layout"] == "ATC"
    assert spec.dataset._artifact_store.cache_dir == tmp_path / "cache"
    sample = spec.dataset[0]
    assert torch.equal(sample["target"]["video"]["latent"], tensors["video"])
    assert torch.equal(sample["target"]["audio"]["latent"], tensors["audio"].transpose(1, 2))
    assert torch.equal(
        sample["references"][1]["audio"]["latent"], tensors["ref_audio"].transpose(1, 2)
    )
    assert sample["target"]["audio"]["latent_shape"] == [2, 3, 5]


@pytest.mark.parametrize("artifact", ["mme", "image", "ref_video", "ref_audio", "video", "audio"])
def test_each_consumed_artifact_changes_train_fingerprint_without_editing_jsonl(tmp_path, artifact):
    manifest, tensors = _fixture(tmp_path / "source")
    request = TrainRequest(train_manifest=str(manifest), extras=INPUT_OPTIONS)
    original, before = build_data_spec(request, collate_fn=list)
    frozen_worker = pickle.loads(pickle.dumps(original.dataset))
    torch.save(tensors[artifact] + 1, manifest.parent / f"{artifact}.pt")
    _, after = build_data_spec(request, collate_fn=list)
    assert before["manifest_digest"] == after["manifest_digest"]
    assert before["input"] != after["input"]
    with pytest.raises(ValueError, match="changed during dataset session"):
        original.dataset[0]
    with pytest.raises(ValueError, match="changed during dataset session"):
        frozen_worker[0]


def test_multi_source_versions_include_file_bytes_and_format_options(tmp_path):
    first, tensors = _fixture(tmp_path / "first")
    second, _ = _fixture(tmp_path / "second")
    request = TrainRequest(
        sources=(
            DataSourceSpec(name="first", manifest=str(first), weight=11),
            DataSourceSpec(name="second", manifest=str(second), weight=89),
        ),
        extras=INPUT_OPTIONS,
    )
    _, before = build_data_spec(request, collate_fn=list, total_samples=192)
    torch.save(tensors["ref_audio"] + 1, first.parent / "ref_audio.pt")
    _, changed = build_data_spec(request, collate_fn=list, total_samples=192)
    assert before["dataset"]["sources"][0] != changed["dataset"]["sources"][0]
    assert before["dataset"]["sources"][1] == changed["dataset"]["sources"][1]
    _, new_layout = build_data_spec(
        replace(request, extras={**INPUT_OPTIONS, "audio_layout": "ACT"}),
        collate_fn=list,
        total_samples=192,
    )
    for previous, current in zip(
        changed["dataset"]["sources"], new_layout["dataset"]["sources"], strict=True
    ):
        assert previous["version"] != current["version"]


def test_selected_target_pinning_keeps_all_reference_modalities(tmp_path):
    manifest, _ = _fixture(tmp_path / "source")
    (manifest.parent / "audio.pt").unlink()
    request = TrainRequest(
        train_manifest=str(manifest),
        extras={**INPUT_OPTIONS, "target_modalities": ["video"]},
    )
    spec, contract = build_data_spec(request, collate_fn=list)
    assert contract["input"]["artifacts"]["artifact_count"] == 5
    assert set(spec.dataset[0]["target"]) == {"video"}
    assert "audio" in spec.dataset[0]["references"][1]


def test_pin_rejects_manifest_edit_even_when_file_size_is_unchanged(tmp_path, monkeypatch):
    manifest, _ = _fixture(tmp_path / "source")
    request = TrainRequest(train_manifest=str(manifest), extras=INPUT_OPTIONS)
    spec, _ = build_data_spec(request, collate_fn=list)
    worker = pickle.loads(pickle.dumps(spec.dataset))
    before = manifest.read_bytes()
    after = before.replace(b"Wiring", b"Edited")
    assert len(before) == len(after)
    manifest.write_bytes(after)
    for dataset in (spec.dataset, worker):
        # Some filesystems preserve metadata on same-size writes. Force this
        # case on every platform, so only verification of line bytes can pass.
        frozen_stat = dataset._pinned_manifest_stat
        monkeypatch.setattr(dataset, "_manifest_stat", lambda stat=frozen_stat: stat)
        with pytest.raises(ValueError, match="manifest changed"):
            dataset[0]


def test_default_bundle_fingerprint_and_shape_remain_compatible(tmp_path):
    manifest, tensors = _fixture(tmp_path / "source", bundle=True)
    request = TrainRequest(train_manifest=str(manifest), extras={"require_digests": False})
    spec, contract = build_data_spec(request, collate_fn=list)
    assert "input" not in contract
    assert spec.dataset.data_version() == spec.dataset.content_digest()
    assert torch.equal(spec.dataset[0]["target"]["audio"]["latent"], tensors["audio"])


def test_s3_byte_revision_changes_train_contract_and_frozen_workers_stay_offline(
    tmp_path, monkeypatch
):
    manifest, tensors = _fixture(tmp_path / "source")
    objects = {f"{name}.pt": (manifest.parent / f"{name}.pt").read_bytes() for name in tensors}
    record = json.loads(manifest.read_text())

    def remote_paths(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"latent_path", "embedding_path"}:
                    value[key] = "s3://fixture-bucket/" + item
                else:
                    remote_paths(item)
        elif isinstance(value, list):
            for item in value:
                remote_paths(item)

    remote_paths(record)
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    class Client:
        def head_object(self, *, Bucket, Key):
            assert Bucket == "fixture-bucket"
            content = objects[Key]
            return {"ContentLength": len(content), "ETag": hashlib.sha256(content).hexdigest()}

        def download_file(self, bucket, key, target, **kwargs):
            assert bucket == "fixture-bucket"
            Path(target).write_bytes(objects[key])

    monkeypatch.setattr(s3_staging, "_s3_client", lambda source: Client())
    monkeypatch.setattr(s3_staging, "_transfer_config", lambda *args, **kwargs: None)
    request = TrainRequest(
        train_manifest=str(manifest),
        extras={**INPUT_OPTIONS, "artifact_cache_dir": str(tmp_path / "cache")},
    )
    original, before = build_data_spec(request, collate_fn=list)
    worker = pickle.loads(pickle.dumps(original.dataset))
    revised_tensor = tensors["video"] + 1
    buffer = io.BytesIO()
    torch.save(revised_tensor, buffer)
    objects["video.pt"] = buffer.getvalue()
    revised, after = build_data_spec(request, collate_fn=list)
    assert before["manifest_digest"] == after["manifest_digest"]
    assert before["input"] != after["input"]

    def offline(source):
        raise AssertionError("worker resolution must use frozen local cache only")

    monkeypatch.setattr(s3_staging, "_s3_client", offline)
    assert torch.equal(worker[0]["target"]["video"]["latent"], tensors["video"])
    assert torch.equal(revised.dataset[0]["target"]["video"]["latent"], revised_tensor)


@pytest.mark.parametrize("mode", ["validation", "inference"])
def test_eval_uses_same_format_layout_and_frozen_file_contract(tmp_path, mode):
    manifest, tensors = _fixture(tmp_path / "source")
    if mode == "inference":
        record = json.loads(manifest.read_text())
        del record["target"]
        record["target_spec"] = {
            "modalities": ["video"],
            "duration_seconds": 1,
            "video": {"height": 16, "width": 16, "fps": 24},
        }
        manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
        (manifest.parent / "video.pt").unlink()
        (manifest.parent / "audio.pt").unlink()
    request = EvalRequest(data_source=mode, manifest=str(manifest), extras=INPUT_OPTIONS)
    indexed = _open_mmgen_dataset(request)
    dataset = indexed.dataset
    contract = dataset.input_contract()
    assert contract["artifacts"]["artifact_count"] == (6 if mode == "validation" else 4)
    sample = indexed[0]
    assert sample["__vforge_eval_index__"] == 0
    assert torch.equal(
        sample["references"][1]["audio"]["latent"], tensors["ref_audio"].transpose(1, 2)
    )
    torch.save(tensors["ref_audio"] + 1, manifest.parent / "ref_audio.pt")
    revised = _open_mmgen_dataset(request).dataset
    assert revised.data_version() != dataset.data_version()
    with pytest.raises(ValueError, match="changed during dataset session"):
        indexed[0]
