"""CPU tests for the taichu raw-dataset substrate: helpers and the base class."""

from __future__ import annotations

import json
import pickle

import pytest
import torch


def test_align_4np1_rounds_down_to_the_vae_stride() -> None:
    from vforge.data.raw_dataset import align_4np1

    assert [align_4np1(n) for n in (0, 4, 5, 6, 8, 9, 149, 150)] == [0, 0, 5, 5, 5, 9, 149, 149]


def test_pad_or_trim_repeats_the_last_frame_rather_than_wrapping() -> None:
    """A wrapped pad would teach the model that the clip loops back to frame 0."""
    from vforge.data.raw_dataset import pad_or_trim

    data = torch.arange(3, dtype=torch.float32).reshape(3, 1)

    assert pad_or_trim(data, 5).flatten().tolist() == [0, 1, 2, 2, 2]
    assert pad_or_trim(data, 2).flatten().tolist() == [0, 1]
    assert pad_or_trim(data, 3) is data


def test_resize_center_crop_covers_the_target_without_distorting() -> None:
    from vforge.data.raw_dataset import resize_center_crop

    # 4:3 source, 16:9 target: covering crops the height, never letterboxes.
    frames = torch.rand(2, 3, 60, 80)
    out = resize_center_crop(frames, 36, 64)

    assert out.shape == (2, 3, 36, 64)


def test_resize_center_crop_keeps_a_centered_stripe_centered() -> None:
    """Cover-then-crop is symmetric, so a centered feature must stay centered."""
    from vforge.data.raw_dataset import resize_center_crop

    frames = torch.zeros(1, 3, 64, 64)
    frames[:, :, 30:34, :] = 1.0
    out = resize_center_crop(frames, 32, 32)

    rows = out[0, 0].mean(dim=1)
    assert int(rows.argmax()) in {15, 16}


def test_unit_normalize_normal_restores_unit_length_in_place() -> None:
    from vforge.data.raw_dataset import unit_normalize_normal

    # 0.75 encodes 0.5 per axis, whose norm is 0.87 -- not a unit vector.
    frames = torch.full((1, 3, 2, 2), 0.75)
    out = unit_normalize_normal(frames)

    decoded = out * 2.0 - 1.0
    assert torch.allclose(torch.linalg.norm(decoded, dim=1), torch.ones(1, 2, 2), atol=1e-5)
    assert out is frames


def test_unit_normalize_normal_leaves_neutral_grey_untouched() -> None:
    """0.5 decodes to the zero vector; scaling it up would invent a direction."""
    from vforge.data.raw_dataset import unit_normalize_normal

    out = unit_normalize_normal(torch.full((1, 3, 2, 2), 0.5))

    assert torch.allclose(out, torch.full((1, 3, 2, 2), 0.5), atol=1e-5)


def test_prompt_clean_unescapes_twice_and_collapses_whitespace() -> None:
    from vforge.data.raw_dataset import prompt_clean

    assert prompt_clean("a &amp;quot;b&amp;quot;\n\n  c ") == 'a "b" c'


def test_select_ref_frame_indices_is_strictly_increasing_and_in_range() -> None:
    from vforge.data.raw_dataset import select_ref_frame_indices

    for _ in range(50):
        picks = select_ref_frame_indices(29, 4)
        assert len(picks) == 4
        assert picks == sorted(set(picks))
        assert all(0 <= index < 29 for index in picks)


def test_select_ref_frame_indices_clamps_picks_to_the_clip_length() -> None:
    from vforge.data.raw_dataset import select_ref_frame_indices

    assert select_ref_frame_indices(2, 8) == [0, 1]


def test_load_index_records_reads_pickle_and_json(tmp_path) -> None:
    from vforge.data.raw_dataset import load_index_records

    pkl_path = tmp_path / "index.pkl"
    pkl_path.write_bytes(pickle.dumps([{"a": 1}]))
    json_path = tmp_path / "index.json"
    json_path.write_text(json.dumps([{"b": 2}]), encoding="utf-8")

    assert load_index_records(pkl_path) == [{"a": 1}]
    assert load_index_records(json_path) == [{"b": 2}]


def test_remap_path_resolves_logical_taichu_mounts() -> None:
    from vforge.data.raw_dataset import remap_path

    replacements = (("/efs", "/gwm-efs"), ("/threed-code", "/gwm-code"))

    assert remap_path("/efs/shared/items.pkl", replacements) == ("/gwm-efs/shared/items.pkl")
    assert remap_path("/threed-code/models/vae", replacements) == ("/gwm-code/models/vae")
    assert remap_path("/tmp/local.pkl", replacements) == "/tmp/local.pkl"


def test_load_index_records_rejects_an_unknown_extension(tmp_path) -> None:
    from vforge.data.raw_dataset import load_index_records

    path = tmp_path / "index.yaml"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported index format"):
        load_index_records(path)


def _build_dataset(tmp_path, entries, **kwargs):
    """Build a minimal RawVideoDataset subclass over a JSON index."""
    import dataclasses
    from typing import Any

    from vforge.data.raw_dataset import RawVideoDataset

    @dataclasses.dataclass
    class _Record:
        id: str
        value: int
        source: str = ""

    class _Dataset(RawVideoDataset):
        def parse_record(self, raw: Any):
            if raw.get("skip"):
                return None
            return _Record(id=str(raw["id"]), value=int(raw["value"]))

        def build_sample(self, record) -> dict[str, Any]:
            return {"video_frames": torch.full((5, 3, 2, 2), float(record.value))}

    path = tmp_path / "index.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return _Dataset(index_files=[str(path)], frames=5, height=2, width=2, **kwargs)


def test_base_drops_unusable_entries_and_tags_the_source(tmp_path) -> None:
    dataset = _build_dataset(
        tmp_path,
        [{"id": "a", "value": 1}, {"id": "b", "value": 2, "skip": True}],
        name="corpus",
    )

    assert dataset.sample_ids == ["a"]
    assert dataset.records[0].source == "corpus"


def test_base_stamps_the_sample_id_and_text_width(tmp_path) -> None:
    dataset = _build_dataset(
        tmp_path,
        [{"id": "a", "value": 1}],
        max_sequence_length=226,
    )
    sample = dataset[0]

    assert sample["sample_id"] == "a"
    assert sample["max_sequence_length"] == 226


def test_base_truncates_to_max_num(tmp_path) -> None:
    dataset = _build_dataset(
        tmp_path,
        [{"id": str(index), "value": index} for index in range(10)],
        extras={"max_num": 3},
    )

    assert len(dataset) == 3


def test_base_rejects_an_index_that_yields_nothing(tmp_path) -> None:
    with pytest.raises(ValueError, match="0 usable records"):
        _build_dataset(tmp_path, [{"id": "a", "value": 1, "skip": True}])


def test_base_extracts_reference_frames_when_asked(tmp_path) -> None:
    """The knob lives on the base so it means the same thing for every kind."""
    dataset = _build_dataset(
        tmp_path,
        [{"id": "a", "value": 1}],
        extras={"extract_reference_frame": True, "ref_num_frames": 2},
    )

    assert dataset[0]["reference_frames"].shape == (2, 3, 2, 2)


def test_base_leaves_reference_frames_out_by_default(tmp_path) -> None:
    assert "reference_frames" not in _build_dataset(tmp_path, [{"id": "a", "value": 1}])[0]


def test_base_rejects_a_bad_mode(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported mode"):
        _build_dataset(tmp_path, [{"id": "a", "value": 1}], mode="test")
