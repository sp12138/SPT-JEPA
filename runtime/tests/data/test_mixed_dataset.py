"""Mixed-dataset routing, per-sub RR sampling, both openers, and the loader."""

from __future__ import annotations

import json
from collections import Counter
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from vforge.data.cache import EncodedBatchCodec, digest_sample_ids, save_latent_bundle
from vforge.data.mixed_dataset import (
    MixedDataset,
    MixedSampler,
    build_mixed_resumable_loader,
    open_mixed_cache_dataset,
    open_mixed_raw_dataset,
)


class _Items(Dataset):
    def __init__(self, length: int, *, tag: str = "") -> None:
        self._length = length
        self._tag = tag

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, int | str]:
        return {"tag": self._tag, "local": int(index)}


def test_compound_index_roundtrips() -> None:
    encoded = MixedDataset.encode_index(3, 42)
    assert MixedDataset.decode_index(encoded) == (3, 42)
    assert encoded == 3 * 10_000_000_000 + 42


def test_mixed_dataset_routes_getitem_by_compound_index() -> None:
    mixed = MixedDataset(
        [_Items(4, tag="a"), _Items(5, tag="b")],
        weights=[1.0, 2.0],
        names=["a", "b"],
        mix_seed=7,
        steps_per_epoch=16,
    )
    assert mixed[MixedDataset.encode_index(1, 3)] == {"tag": "b", "local": 3}
    assert mixed.resume_contract() == mixed.resume_contract()
    assert mixed.resume_contract()["sub_names"] == ["a", "b"]
    assert mixed.resume_contract()["sub_lengths"] == [4, 5]
    assert mixed.resume_contract()["mix_seed"] == 7


def test_pick_sub_id_is_a_pure_function_of_mix_seed_and_step() -> None:
    left = MixedDataset(
        [_Items(4), _Items(4), _Items(4)],
        weights=[0.2, 0.75, 0.05],
        mix_seed=20260726,
        steps_per_epoch=100,
    )
    right = MixedDataset(
        [_Items(4), _Items(4), _Items(4)],
        weights=[0.2, 0.75, 0.05],
        mix_seed=20260726,
        steps_per_epoch=100,
    )
    picks = [left.pick_sub_id(step) for step in range(256)]
    assert picks == [right.pick_sub_id(step) for step in range(256)]
    counts = Counter(picks)
    total = sum(counts.values())
    assert abs(counts[0] / total - 0.2) < 0.08
    assert abs(counts[1] / total - 0.75) < 0.08
    assert abs(counts[2] / total - 0.05) < 0.08


def test_mixed_sampler_keeps_batches_homogeneous_and_dp_shards_disjoint() -> None:
    mixed = MixedDataset(
        [_Items(8, tag="a"), _Items(6, tag="b")],
        weights=[1.0, 1.0],
        mix_seed=11,
        steps_per_epoch=12,
    )
    rank0 = MixedSampler(mixed, dp_rank=0, dp_world_size=2, batch_size=2, shuffle=False, seed=0)
    rank1 = MixedSampler(mixed, dp_rank=1, dp_world_size=2, batch_size=2, shuffle=False, seed=0)
    first0 = list(islice(iter(rank0), 2))
    first1 = list(islice(iter(rank1), 2))
    decoded0 = [MixedDataset.decode_index(index) for index in first0]
    decoded1 = [MixedDataset.decode_index(index) for index in first1]
    sub_ids = {sub_id for sub_id, _ in decoded0 + decoded1}
    assert len(sub_ids) == 1
    assert {local for _, local in decoded0}.isdisjoint({local for _, local in decoded1})
    assert rank0.resume_contract() == rank1.resume_contract()


def test_mixed_sampler_reshuffles_only_after_a_full_sub_epoch() -> None:
    mixed = MixedDataset(
        [_Items(8)],
        weights=[1.0],
        mix_seed=3,
        steps_per_epoch=8,
    )
    sampler = MixedSampler(mixed, dp_rank=0, dp_world_size=1, batch_size=2, shuffle=True, seed=9)
    first_epoch = list(islice(iter(sampler), 8))
    locals_seen = [MixedDataset.decode_index(index)[1] for index in first_epoch]
    assert len(locals_seen) == len(set(locals_seen)) == 8

    ordered = MixedSampler(mixed, dp_rank=0, dp_world_size=1, batch_size=2, shuffle=False, seed=9)
    assert [MixedDataset.decode_index(index)[1] for index in islice(iter(ordered), 8)] == list(
        range(8)
    )


_CODEC = EncodedBatchCodec(
    name="mix-test",
    required_keys=("prompt", "frame_latents"),
)


def _write_indexed(cache_dir: Path, count: int, *, fill: float) -> None:
    for index in range(count):
        path = cache_dir / "train" / f"{index // 1000:04d}" / f"{index:08d}.pt"
        save_latent_bundle(
            {
                "prompt": torch.ones(1, 2, 2),
                "frame_latents": torch.full((1, 2, 3, 2, 2), fill),
            },
            str(path),
        )


def test_open_mixed_cache_injects_bounds_and_fingerprints_each_source(
    tmp_path: Path,
) -> None:
    first = tmp_path / "seedance"
    second = tmp_path / "gameplay"
    _write_indexed(first, 2, fill=0.1)
    _write_indexed(second, 3, fill=0.2)

    mixed, fingerprint = open_mixed_cache_dataset(
        [
            {
                "name": "seedance",
                "cache_dir": str(first),
                "weight": 0.2,
                "min_latents": 21,
                "max_latents": 38,
            },
            {
                "name": "gameplay",
                "cache_dir": str(second),
                "weight": 0.8,
                "min_latents": 21,
                "max_latents": 38,
            },
        ],
        codec=_CODEC,
        mix_seed=20260726,
        steps_per_epoch=64,
    )

    assert isinstance(mixed, MixedDataset)
    assert mixed.sub_names == ["seedance", "gameplay"]
    assert mixed.sub_lengths == [2, 3]
    sample = mixed[MixedDataset.encode_index(0, 1)]
    assert sample["min_latents"] == 21
    assert sample["max_latents"] == 38
    assert torch.equal(sample["frame_latents"], torch.full((1, 2, 3, 2, 2), 0.1))

    assert fingerprint["source_kind"] == "mixed_encoded_cache"
    assert fingerprint["mix_seed"] == 20260726
    assert fingerprint["steps_per_epoch"] == 64
    assert fingerprint["sources"][0]["cache_dir"] == str(first)
    assert fingerprint["sources"][0]["sample_count"] == 2
    assert fingerprint["sources"][1]["sample_count"] == 3
    assert fingerprint["sources"][0]["sample_ids_digest"] == digest_sample_ids(
        ["00000000", "00000001"]
    )
    assert fingerprint["sources"][0]["weight"] == 0.2
    assert fingerprint["sources"][1]["min_latents"] == 21


def _write_index(tmp_path, name, entries) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(entries), encoding="utf-8")
    return str(path)


def _sf_entry(index: int, uuid="take-0"):
    return {
        "id": f"{index:04d}",
        "uuid": uuid,
        "file": {
            "video": {
                "video_path": f"rgb-{index}.mp4",
                "normal_path": f"normal-{index}.mp4",
            }
        },
    }


def _stub_reader(monkeypatch, count=25):
    import vforge.data.self_forcing_dataset as self_forcing

    monkeypatch.setattr(
        self_forcing,
        "read_video_tensor",
        lambda _path, **_kwargs: torch.zeros(count, 3, 8, 8),
    )


def _source(tmp_path, name, entries, **overrides):
    source = {
        "name": name,
        "kind": "self_forcing",
        "index_files": (_write_index(tmp_path, f"{name}.json", entries),),
        "frames": 9,
        "height": 16,
        "width": 16,
    }
    source.update(overrides)
    return source


def test_the_mixture_carries_each_source_name_and_weight(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch)
    mixed, fingerprint = open_mixed_raw_dataset(
        [
            _source(tmp_path, "a", [_sf_entry(0)], weight=3.0),
            _source(tmp_path, "b", [_sf_entry(1), _sf_entry(2)], weight=1.0),
        ],
        mode="val",
    )

    assert mixed.sub_names == ["a", "b"]
    assert mixed.sub_lengths == [1, 2]
    assert mixed.sub_probs == [pytest.approx(0.75), pytest.approx(0.25)]
    assert fingerprint["source_kind"] == "mixed_raw_video"
    assert [source["name"] for source in fingerprint["sources"]] == ["a", "b"]


def test_a_single_source_still_goes_through_the_mixture(tmp_path, monkeypatch) -> None:
    """One code path for one source and for many keeps resume state uniform."""
    _stub_reader(monkeypatch)
    mixed, _ = open_mixed_raw_dataset([_source(tmp_path, "a", [_sf_entry(0)])], mode="val")

    assert isinstance(mixed, MixedDataset)
    assert mixed.num_sub_datasets == 1


def test_latent_bounds_are_stamped_onto_every_sample(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch)
    mixed, _ = open_mixed_raw_dataset(
        [_source(tmp_path, "a", [_sf_entry(0)], min_latents=2, max_latents=6)],
        mode="val",
    )
    sample = mixed[0]

    assert (sample["min_latents"], sample["max_latents"]) == (2, 6)


def test_the_fingerprint_notices_an_edited_index_file(tmp_path, monkeypatch) -> None:
    """Index paths are stable while their contents are not, so hash identity."""
    _stub_reader(monkeypatch)
    source = _source(tmp_path, "a", [_sf_entry(0)])
    _, before = open_mixed_raw_dataset([source], mode="val")

    _write_index(tmp_path, "a.json", [_sf_entry(0), _sf_entry(1)])
    _, after = open_mixed_raw_dataset([source], mode="val")

    assert before["sources"][0]["sample_ids_digest"] != after["sources"][0]["sample_ids_digest"]
    assert before["sources"][0]["index_files"] != after["sources"][0]["index_files"]


def test_the_fingerprint_notices_a_changed_decode_geometry(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch)
    entries = [_sf_entry(0)]
    _, before = open_mixed_raw_dataset([_source(tmp_path, "a", entries)], mode="val")
    _, after = open_mixed_raw_dataset(
        [_source(tmp_path, "a", entries, height=32, extras={"max_num": 1})],
        mode="val",
    )

    assert before["sources"][0]["height"] != after["sources"][0]["height"]
    assert after["sources"][0]["extras"] == {"max_num": 1}


def _leader_state(*, dp_size: int = 1, dp_rank: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(dp_size=dp_size),
        dp_rank=dp_rank,
        batch_group=None,
        is_batch_leader=True,
    )


def test_mixed_loader_installs_mixed_sampler_not_distributed() -> None:
    mixed = MixedDataset(
        [list(range(8)), list(range(4))],
        [0.5, 0.5],
        steps_per_epoch=3,
    )
    loader = build_mixed_resumable_loader(
        mixed,
        parallel_state=_leader_state(),
        batch_size=2,
        num_workers=0,
        drop_last=True,
        seed=0,
        shuffle=True,
    )

    assert isinstance(loader.sampler, MixedSampler)
    assert not isinstance(loader.sampler, DistributedSampler)
    assert loader.sampler.batch_size == 2
    assert loader.sampler.dp_world_size == 1
    assert len(loader) == 3


def test_mixed_loader_rejects_a_plain_dataset() -> None:
    with pytest.raises(TypeError, match="MixedDataset"):
        build_mixed_resumable_loader(
            list(range(4)),
            parallel_state=_leader_state(),
            batch_size=1,
            num_workers=0,
            drop_last=True,
            seed=0,
        )


def test_mixed_follower_does_not_construct_stateful_loader(monkeypatch) -> None:
    import vforge.data.mixed_dataset as mixed_dataset

    class _UnexpectedLoader:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("a follower must not construct a StatefulDataLoader")

    monkeypatch.setattr(mixed_dataset, "StatefulDataLoader", _UnexpectedLoader)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: 2)
    mixed = MixedDataset([list(range(4))], [1.0], steps_per_epoch=1)
    loader = build_mixed_resumable_loader(
        mixed,
        parallel_state=SimpleNamespace(
            config=SimpleNamespace(dp_size=1),
            dp_rank=0,
            batch_group=object(),
            is_batch_leader=False,
        ),
        batch_size=1,
        num_workers=0,
        drop_last=True,
        seed=0,
    )

    from vforge.data.loader import LeaderBroadcastDataLoader

    assert isinstance(loader, LeaderBroadcastDataLoader)
    assert loader.sampler is None
