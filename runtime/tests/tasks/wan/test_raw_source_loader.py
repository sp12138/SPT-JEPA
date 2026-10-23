"""Raw ``data.sources`` entries must reach the loader, not just the schema.

The rest of the raw dataset tests stop at the dataset object. These check the
seam above it: that a ``kind`` set in a config survives translation and lands on
the mixed loader with a fingerprint resume can compare.
"""

from __future__ import annotations

import json

import pytest
import torch

from vforge.tasks.requests import DataSourceSpec, TrainRequest


def _index(tmp_path, name="index.json", count=2) -> str:
    entries = [
        {
            "id": f"{index:04d}",
            "uuid": "take-0",
            "file": {
                "video": {
                    "video_path": f"rgb-{index}.mp4",
                    "normal_path": f"normal-{index}.mp4",
                }
            },
        }
        for index in range(count)
    ]
    path = tmp_path / name
    path.write_text(json.dumps(entries), encoding="utf-8")
    return str(path)


def _stub_reader(monkeypatch):
    import vforge.data.self_forcing_dataset as self_forcing

    monkeypatch.setattr(
        self_forcing,
        "read_video_tensor",
        lambda _path, **_kwargs: torch.zeros(9, 3, 8, 8),
    )


def _spec(tmp_path, **overrides) -> DataSourceSpec:
    fields = {
        "name": "sf",
        "kind": "self_forcing",
        "index_files": (_index(tmp_path),),
        "frames": 9,
        "height": 16,
        "width": 16,
    }
    fields.update(overrides)
    return DataSourceSpec(**fields)


def _build(request: TrainRequest, monkeypatch):
    """Run the shared builder with the real loader stubbed out."""
    import vforge.tasks.wan.shared as shared

    captured: dict = {}

    def fake_mixed_loader(dataset, **kwargs):
        captured["dataset"] = dataset
        captured["kwargs"] = kwargs
        return object()

    def fake_loader(dataset, **kwargs):
        captured["plain_dataset"] = dataset
        return object()

    monkeypatch.setattr(shared, "build_mixed_resumable_loader", fake_mixed_loader)
    monkeypatch.setattr(shared, "build_resumable_loader", fake_loader)
    _, fingerprint = shared.build_train_dataloader(
        request,
        codec=object(),
        collate_encoded=None,
        make_raw_collate=lambda: "raw-collate",
        preprocess_fingerprint={"resolution": [16, 16]},
    )
    return captured, fingerprint


def _request(tmp_path, **overrides) -> TrainRequest:
    fields = {
        "latent_cache_mode": "online",
        "pretrained_path": "/tmp/pretrained",
        "max_steps": 10,
        "sources": (_spec(tmp_path),),
    }
    fields.update(overrides)
    return TrainRequest(**fields)


def test_a_raw_source_builds_a_mixed_loader_over_the_decoding_dataset(
    tmp_path, monkeypatch
) -> None:
    from vforge.data.mixed_dataset import MixedDataset

    _stub_reader(monkeypatch)
    captured, fingerprint = _build(_request(tmp_path), monkeypatch)

    assert isinstance(captured["dataset"], MixedDataset)
    assert captured["dataset"].sub_names == ["sf"]
    assert fingerprint["source_kind"] == "mixed_raw_video"
    assert fingerprint["sources"][0]["kind"] == "self_forcing"


def test_the_raw_path_uses_the_raw_collate_and_broadcasts(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch)
    captured, _ = _build(_request(tmp_path), monkeypatch)

    assert captured["kwargs"]["collate_fn"] == "raw-collate"
    assert captured["kwargs"]["broadcast_batches"] is True


def test_the_fingerprint_carries_the_loader_knobs_and_preprocessing(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch)
    _, fingerprint = _build(_request(tmp_path, batch_size=2, sampler_seed=7), monkeypatch)

    assert fingerprint["batch_size"] == 2
    assert fingerprint["sampler_seed"] == 7
    assert fingerprint["preprocess"] == {"resolution": [16, 16]}


def test_compose_without_mounted_index_files_still_builds(tmp_path, monkeypatch) -> None:
    """Hydra compose runs on machines that have the config but not the data."""
    _stub_reader(monkeypatch)
    spec = _spec(tmp_path, index_files=("/nonexistent/index.json",))
    _, fingerprint = _build(
        _request(tmp_path, max_steps=0, sources=(spec,)),
        monkeypatch,
    )

    assert fingerprint == {
        "source_kind": "mixed_raw_video",
        "empty": True,
        "mix_seed": 20260726,
        "steps_per_epoch": 1_000_000_000,
    }


def test_extras_reach_the_dataset_instance(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch)
    captured, _ = _build(
        _request(tmp_path, sources=(_spec(tmp_path, extras={"max_num": 1}),)),
        monkeypatch,
    )

    assert captured["dataset"].sub_lengths == [1]


def test_an_offline_run_still_refuses_a_raw_source(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch)

    with pytest.raises(ValueError, match="latent_cache.mode=online"):
        _build(_request(tmp_path, latent_cache_mode="offline"), monkeypatch)
