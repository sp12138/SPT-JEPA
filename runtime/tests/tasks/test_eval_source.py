from __future__ import annotations

import json

import pytest
from torch.utils.data import Dataset

from vforge.data.cache import EncodedBatchCodec
from vforge.tasks.eval_source import open_eval_dataset
from vforge.tasks.requests import DataSourceSpec, EvalRequest


def _codec() -> EncodedBatchCodec:
    return EncodedBatchCodec(name="eval-source-test", required_keys=("prompt",))


def test_sample_index_selects_before_eval_sharding(tmp_path) -> None:
    manifest = tmp_path / "requests.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"sample_id": sample_id, "image": f"/{sample_id}.png"})
            for sample_id in ("zero", "one", "two")
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = open_eval_dataset(
        EvalRequest(
            data_source="inference",
            manifest=str(manifest),
            sample_index=1,
            num_samples=1,
        ),
        codec=_codec(),
    )

    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == "one"
    assert dataset[0]["__vforge_eval_index__"] == 1
    assert dataset[0]["__vforge_eval_seed_offset__"] == 1


def test_manifest_can_preserve_an_external_dataset_seed_offset(tmp_path) -> None:
    manifest = tmp_path / "requests.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "mixed-source-sample",
                "image": "/sample.png",
                "eval_seed_offset": 114,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = open_eval_dataset(
        EvalRequest(data_source="inference", manifest=str(manifest)),
        codec=_codec(),
    )

    assert dataset[0]["__vforge_eval_index__"] == 0
    assert dataset[0]["__vforge_eval_seed_offset__"] == 114


def test_sample_index_rejects_an_out_of_range_index(tmp_path) -> None:
    manifest = tmp_path / "requests.json"
    manifest.write_text(
        json.dumps([{"sample_id": "only", "image": "/only.png"}]),
        encoding="utf-8",
    )

    with pytest.raises(IndexError, match="outside dataset length 1"):
        open_eval_dataset(
            EvalRequest(
                data_source="inference",
                manifest=str(manifest),
                sample_index=1,
            ),
            codec=_codec(),
        )


def test_flat_manifest_exposes_first_seen_source_groups(tmp_path) -> None:
    manifest = tmp_path / "requests.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"sample_id": "a0", "image": "/a0.png", "source": "a"},
                {"sample_id": "b0", "image": "/b0.png", "meta": {"source": "b"}},
                {"sample_id": "a1", "image": "/a1.png", "source": "a"},
                {"sample_id": "unknown", "image": "/unknown.png"},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = open_eval_dataset(
        EvalRequest(data_source="inference", manifest=str(manifest)),
        codec=_codec(),
    )

    assert dataset.eval_source_groups == ((0, 2), (1,), (3,))


class _RawEvalDataset(Dataset):
    def __init__(self, prefix: str, size: int) -> None:
        self.prefix = prefix
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        return {"sample_id": f"{self.prefix}-{index}", "video_frames": object()}


def test_raw_eval_sources_open_without_a_manifest(monkeypatch) -> None:
    import vforge.tasks.eval_source as eval_source

    opened: list[tuple[str, str]] = []

    def open_source(source, *, mode, index):
        opened.append((source["name"], mode))
        return _RawEvalDataset(source["name"], index + 2)

    monkeypatch.setattr(eval_source, "open_raw_dataset", open_source)
    sources = tuple(
        DataSourceSpec(
            name=name,
            kind="gameplay",
            index_files=(f"/{name}.pkl",),
            frames=5,
            height=4,
            width=6,
        )
        for name in ("first", "second")
    )
    dataset = open_eval_dataset(
        EvalRequest(data_source="validation", sources=sources),
        codec=_codec(),
    )

    assert opened == [("first", "val"), ("second", "val")]
    assert len(dataset) == 5
    assert dataset.eval_source_groups == ((0, 1), (2, 3, 4))
    assert dataset[2]["sample_id"] == "second-0"
    assert dataset[2]["__vforge_eval_index__"] == 2
    assert dataset[2]["__vforge_eval_seed_offset__"] == 2


def test_manifest_and_raw_eval_sources_are_mutually_exclusive(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    source = DataSourceSpec(
        kind="gameplay",
        index_files=("/items.pkl",),
        frames=5,
        height=4,
        width=6,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        open_eval_dataset(
            EvalRequest(
                data_source="validation",
                manifest=str(manifest),
                sources=(source,),
            ),
            codec=_codec(),
        )


def test_model_owned_encoded_eval_source_opener() -> None:
    source = DataSourceSpec(
        name="taichu-indexed",
        cache_dir="/cache",
        min_latents=99,
        max_latents=99,
    )
    opened = []

    def open_encoded(item, codec):
        opened.append((item, codec.name))
        return _RawEvalDataset("cached", 2)

    dataset = open_eval_dataset(
        EvalRequest(data_source="cache", sources=(source,)),
        codec=_codec(),
        open_encoded_source=open_encoded,
    )

    assert opened == [(source, "eval-source-test")]
    assert len(dataset) == 2
    assert dataset[1]["sample_id"] == "cached-1"


def test_encoded_eval_source_requires_a_model_owned_opener() -> None:
    source = DataSourceSpec(cache_dir="/cache")

    with pytest.raises(NotImplementedError, match="does not support"):
        open_eval_dataset(
            EvalRequest(data_source="cache", sources=(source,)),
            codec=_codec(),
        )
