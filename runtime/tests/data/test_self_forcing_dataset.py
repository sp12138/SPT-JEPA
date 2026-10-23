"""CPU tests for the self-forcing datasets: windowing, ODE pairs, chunk joins."""

from __future__ import annotations

import json

import pytest
import torch


def _write_index(tmp_path, entries) -> str:
    path = tmp_path / "index.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return str(path)


def _entry(index: int, *, uuid="take-0", frames=None, **video_overrides):
    video = {
        "video_path": f"rgb-{index}.mp4",
        "normal_path": f"normal-{index}.mp4",
    }
    if frames is not None:
        video["num_frames"] = frames
    video.update(video_overrides)
    return {
        "id": f"{index:04d}",
        "uuid": uuid,
        "source": "sf",
        "file": {"video": video, "text_caption": [{"caption": f"caption {index}"}]},
    }


def _stub_reader(monkeypatch, module_name, *, lengths=None, default=40):
    """Decode each path to a clip whose frame values are its frame numbers."""
    import importlib

    module = importlib.import_module(module_name)
    lengths = lengths or {}

    def _read(path, *, max_frames=None):
        count = lengths.get(str(path), default)
        ramp = torch.arange(count, dtype=torch.float32).reshape(count, 1, 1, 1)
        return ramp.expand(count, 3, 8, 8).contiguous() / 255.0

    monkeypatch.setattr(module, "read_video_tensor", _read)


def _dataset(tmp_path, entries, **kwargs):
    from vforge.data.self_forcing_dataset import SelfForcingDataset

    kwargs.setdefault("frames", 21)
    kwargs.setdefault("height", 16)
    kwargs.setdefault("width", 16)
    return SelfForcingDataset(index_files=[_write_index(tmp_path, entries)], **kwargs)


def _source_frame_numbers(clip: torch.Tensor) -> list[int]:
    """Recover which stub source frames a clip kept, from their ramp values."""
    return [round(float(frame.max()) * 255) for frame in clip]


def test_a_clip_is_subsampled_and_resized_to_the_configured_geometry(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset")
    sample = _dataset(tmp_path, [_entry(0)], mode="val")[0]

    assert sample["video_frames"].shape == (21, 3, 16, 16)
    assert sample["normal_condition"].shape == (21, 3, 16, 16)
    assert torch.equal(sample["image_condition"], sample["video_frames"][0:1])


def test_subsampling_spans_the_stride_aligned_clip(tmp_path, monkeypatch) -> None:
    """40 source frames align down to 37, and the 21 kept ones span all of it."""
    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset")
    sample = _dataset(tmp_path, [_entry(0)], mode="val")[0]

    kept = _source_frame_numbers(sample["video_frames"])
    assert kept[0] == 0
    assert kept[-1] == 36
    assert kept == sorted(kept)


def test_a_short_clip_aligns_down_instead_of_padding_up(tmp_path, monkeypatch) -> None:
    """Taichu only ever shortens: 7 frames align to 5 and stay there."""
    _stub_reader(
        monkeypatch,
        "vforge.data.self_forcing_dataset",
        lengths={"rgb-0.mp4": 7, "normal-0.mp4": 7},
    )
    sample = _dataset(tmp_path, [_entry(0)], mode="val")[0]

    assert sample["video_frames"].shape[0] == 5
    assert _source_frame_numbers(sample["video_frames"]) == [0, 1, 2, 3, 4]


def test_a_clip_below_the_vae_stride_is_an_error(tmp_path, monkeypatch) -> None:
    _stub_reader(
        monkeypatch,
        "vforge.data.self_forcing_dataset",
        lengths={"rgb-0.mp4": 4, "normal-0.mp4": 4},
    )
    dataset = _dataset(tmp_path, [_entry(0)], mode="val")

    with pytest.raises(ValueError, match="too short for the 4n\\+1 VAE stride"):
        dataset[0]


def test_head_sampling_keeps_the_leading_frames(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset")
    sample = _dataset(tmp_path, [_entry(0)], mode="val", extras={"frame_sample_mode": "head"})[0]

    assert _source_frame_numbers(sample["video_frames"]) == list(range(21))


def test_an_unknown_frame_sample_mode_is_rejected(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset")

    with pytest.raises(ValueError, match="frame_sample_mode"):
        _dataset(tmp_path, [_entry(0)], mode="val", extras={"frame_sample_mode": "middle"})


def test_train_and_val_keep_the_same_frames(tmp_path, monkeypatch) -> None:
    """Subsampling replaced the random window, so the mode no longer shifts it."""
    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset")
    train = _dataset(tmp_path, [_entry(0)], mode="train")
    val = _dataset(tmp_path, [_entry(0)], mode="val")

    torch.manual_seed(0)
    assert torch.equal(train[0]["video_frames"], val[0]["video_frames"])
    assert torch.equal(train[0]["video_frames"], train[0]["video_frames"])


def test_the_normal_stream_comes_back_unit_length(tmp_path, monkeypatch) -> None:
    """Interpolation shortens normals; the resize renormalizes them."""
    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset")
    sample = _dataset(tmp_path, [_entry(0)], mode="val")[0]

    norms = torch.linalg.norm(sample["normal_condition"] * 2.0 - 1.0, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # The RGB stream is left alone, so the two streams are no longer equal.
    assert not torch.allclose(sample["normal_condition"], sample["video_frames"])


def test_the_prompt_is_the_indexed_caption(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset")
    sample = _dataset(tmp_path, [_entry(0)], mode="val")[0]

    assert sample["prompt"] == "caption 0"


def test_an_entry_without_a_caption_still_loads(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset")
    entry = _entry(0)
    entry["file"].pop("text_caption")

    assert _dataset(tmp_path, [entry], mode="val")[0]["prompt"] == ""


def test_ode_pairs_ride_along_when_the_index_names_them(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset")
    ode_path = tmp_path / "ode.pt"
    torch.save(torch.arange(6.0).reshape(1, 2, 3), ode_path)
    sample = _dataset(tmp_path, [_entry(0, ode_path=str(ode_path))], mode="val")[0]

    # Index 0 drops the batch axis the solver dump carries.
    assert sample["ode_pairs"].shape == (2, 3)


def test_entries_missing_a_paired_normal_are_dropped(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset")
    entry = _entry(1)
    entry["file"]["video"].pop("normal_path")

    with pytest.raises(ValueError, match="0 usable records"):
        _dataset(tmp_path, [entry], mode="val")


def test_a_failed_read_reuses_the_last_good_clip(tmp_path, monkeypatch) -> None:
    import vforge.data.self_forcing_dataset as self_forcing

    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset")
    dataset = _dataset(tmp_path, [_entry(0), _entry(1)], mode="val")
    good = dataset[0]

    def _boom(*_args, **_kwargs):
        raise OSError("truncated stream")

    monkeypatch.setattr(self_forcing, "read_video_tensor", _boom)

    assert torch.equal(dataset[1]["video_frames"], good["video_frames"])


def test_a_failed_normal_read_keeps_the_video_that_decoded(tmp_path, monkeypatch) -> None:
    """The two stream caches are separate, so one bad file costs only itself."""
    import vforge.data.self_forcing_dataset as self_forcing

    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset")
    dataset = _dataset(tmp_path, [_entry(0), _entry(1)], mode="val")
    first = dataset[0]

    real_read = self_forcing.read_video_tensor

    def _fail_normals(path, **kwargs):
        if "normal" in str(path):
            raise OSError("truncated stream")
        return real_read(path, **kwargs)

    monkeypatch.setattr(self_forcing, "read_video_tensor", _fail_normals)
    second = dataset[1]

    # Identity, not value: the stub decodes every path alike, so only object
    # identity distinguishes a fresh decode from the cached stand-in.
    assert second["video_frames"] is not first["video_frames"]
    assert second["normal_condition"] is first["normal_condition"]


def test_the_first_read_has_nothing_to_fall_back_on(tmp_path, monkeypatch) -> None:
    import vforge.data.self_forcing_dataset as self_forcing

    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset")
    dataset = _dataset(tmp_path, [_entry(0)], mode="val")
    monkeypatch.setattr(
        self_forcing,
        "read_video_tensor",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("truncated stream")),
    )

    with pytest.raises(OSError):
        dataset[0]


def _long_dataset(tmp_path, entries, **kwargs):
    from vforge.data.self_forcing_dataset import SelfForcingLongVideoDataset

    kwargs.setdefault("frames", 9)
    kwargs.setdefault("height", 16)
    kwargs.setdefault("width", 16)
    return SelfForcingLongVideoDataset(index_files=[_write_index(tmp_path, entries)], **kwargs)


def test_long_video_joins_chunks_and_drops_the_overlap_frame(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset", default=9)
    sample = _long_dataset(tmp_path, [_entry(0), _entry(1), _entry(2)])[0]

    # Three 9-frame chunks joined at a shared boundary frame: 9 + 8 + 8.
    assert sample["video_frames"].shape == (25, 3, 16, 16)
    assert sample["num_chunks"] == 3


def test_long_video_groups_by_uuid(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset", default=9)
    dataset = _long_dataset(
        tmp_path,
        [_entry(0, uuid="a"), _entry(1, uuid="b"), _entry(2, uuid="a")],
    )

    assert sorted(dataset.sample_ids) == ["a", "b"]
    assert {record.uuid: len(record.chunks) for record in dataset.records} == {"a": 2, "b": 1}


def test_long_video_orders_chunks_by_id_not_index_order(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset", default=9)
    dataset = _long_dataset(tmp_path, [_entry(2), _entry(0), _entry(1)])

    assert [chunk.chunk_id for chunk in dataset.records[0].chunks] == ["0000", "0001", "0002"]


def test_long_video_honors_a_per_chunk_frame_count(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset", default=20)
    sample = _long_dataset(tmp_path, [_entry(0, frames=13), _entry(1, frames=5)])[0]

    assert sample["video_frames"].shape[0] == 13 + 4


def test_long_video_skips_entries_without_a_uuid(tmp_path, monkeypatch) -> None:
    _stub_reader(monkeypatch, "vforge.data.self_forcing_dataset", default=9)
    orphan = _entry(3)
    orphan.pop("uuid")
    dataset = _long_dataset(tmp_path, [_entry(0), orphan])

    assert dataset.sample_ids == ["take-0"]
