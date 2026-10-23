"""CPU tests for GameplayDataset: windowing, cameras, and the normal split."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch


def _write_index(tmp_path, entries) -> str:
    path = tmp_path / "items.pkl"
    path.write_bytes(pickle.dumps(entries))
    return str(path)


def _stub_decoder(monkeypatch, *, height=72, width=128, total=400, fills=None):
    """Return uint8 clips whose value encodes which file was read."""
    import vforge.data.gameplay_dataset as gameplay

    fills = fills or {}

    def _read(path, start, end_inclusive):
        first = max(0, min(int(start), total - 1))
        last = max(first, min(int(end_inclusive), total - 1))
        value = fills.get(str(path), 200)
        return torch.full((last - first + 1, 3, height, width), value, dtype=torch.uint8)

    monkeypatch.setattr(gameplay, "read_video_window", _read)


def _entry(**overrides):
    entry = {
        "source_id": "src",
        "chunk_id": "c0",
        "video_path": "rgb.mp4",
        "text": "a &amp;quot;clip&amp;quot;",
        "interval": [10, 200],
        "fps": 30.0,
    }
    entry.update(overrides)
    return entry


def _dataset(tmp_path, entries, **kwargs):
    from vforge.data.gameplay_dataset import GameplayDataset

    kwargs.setdefault("frames", 21)
    kwargs.setdefault("height", 32)
    kwargs.setdefault("width", 64)
    return GameplayDataset(index_files=[_write_index(tmp_path, entries)], **kwargs)


def test_window_is_head_truncated_and_aligned_to_the_vae_stride(tmp_path, monkeypatch) -> None:
    """The caption describes the head of the interval, so the tail is what goes."""
    _stub_decoder(monkeypatch)
    dataset = _dataset(tmp_path, [_entry()])

    assert dataset.plan_window(dataset.records[0]) == (10, 30, 21)
    assert dataset[0]["video_frames"].shape == (21, 3, 32, 64)


def test_a_short_interval_shrinks_to_the_nearest_valid_length(tmp_path, monkeypatch) -> None:
    _stub_decoder(monkeypatch)
    dataset = _dataset(tmp_path, [_entry(interval=[0, 6])])

    assert dataset.plan_window(dataset.records[0]) == (0, 4, 5)


def test_an_interval_too_short_to_align_is_rejected(tmp_path, monkeypatch) -> None:
    _stub_decoder(monkeypatch)
    dataset = _dataset(tmp_path, [_entry(interval=[0, 2])])

    with pytest.raises(ValueError, match="4n\\+1"):
        dataset[0]


def test_a_missing_interval_reads_from_the_top(tmp_path, monkeypatch) -> None:
    _stub_decoder(monkeypatch)
    dataset = _dataset(tmp_path, [_entry(interval=None)])

    assert dataset.plan_window(dataset.records[0]) == (0, 20, 21)
    assert dataset.records[0].id.endswith(":full")


def test_entries_without_a_video_or_caption_are_dropped(tmp_path, monkeypatch) -> None:
    _stub_decoder(monkeypatch)
    dataset = _dataset(
        tmp_path,
        [_entry(), _entry(video_path=None), _entry(text=None), _entry(interval=[9, 2])],
    )

    assert len(dataset) == 1


def test_the_prompt_is_cleaned_and_the_first_frame_is_the_image_condition(
    tmp_path, monkeypatch
) -> None:
    _stub_decoder(monkeypatch)
    sample = _dataset(tmp_path, [_entry()])[0]

    assert sample["prompt"] == 'a "clip"'
    assert torch.equal(sample["image_condition"], sample["video_frames"][0:1])


def test_a_missing_normal_becomes_neutral_grey(tmp_path, monkeypatch) -> None:
    """Grey is the rotation-invariant zero vector, so it survives augmentation."""
    _stub_decoder(monkeypatch)
    sample = _dataset(tmp_path, [_entry()])[0]

    assert torch.allclose(sample["normal_condition"], torch.tensor(0.5, dtype=torch.bfloat16))


def test_an_unreadable_video_falls_back_to_black_rather_than_killing_the_run(
    tmp_path, monkeypatch
) -> None:
    import vforge.data.gameplay_dataset as gameplay

    def _boom(*_args, **_kwargs):
        raise OSError("moov atom not found")

    monkeypatch.setattr(gameplay, "read_video_window", _boom)
    sample = _dataset(tmp_path, [_entry()])[0]

    assert sample["video_frames"].shape == (21, 3, 32, 64)
    assert float(sample["video_frames"].abs().max()) == 0.0


def test_intrinsics_follow_the_cover_and_crop_and_recenter(tmp_path, monkeypatch) -> None:
    _stub_decoder(monkeypatch)
    intrinsics = np.tile(np.eye(3, dtype=np.float32), (300, 1, 1))
    intrinsics[:, 0, 0] = 1000.0
    intrinsics[:, 1, 1] = 1000.0
    intrinsics[:, 0, 2] = 640.0
    intrinsics[:, 1, 2] = 360.0
    poses = np.tile(np.eye(4, dtype=np.float32), (300, 1, 1))

    sample = _dataset(tmp_path, [_entry(Ks=intrinsics, c2ws=poses)])[0]

    # Capture was 720x1280, target is 32x64: cover scale is max(32/720, 64/1280).
    scale = max(32 / 720, 64 / 1280)
    assert sample["Ks"].shape == (21, 3, 3)
    assert sample["Ks"][0, 0, 0] == pytest.approx(1000.0 * scale)
    assert sample["Ks"][0, 0, 2] == pytest.approx(32.0)
    assert sample["Ks"][0, 1, 2] == pytest.approx(16.0)
    assert sample["c2ws"].shape == (21, 4, 4)


def test_short_camera_tracks_are_padded_to_the_window(tmp_path, monkeypatch) -> None:
    _stub_decoder(monkeypatch)
    intrinsics = np.tile(np.eye(3, dtype=np.float32), (4, 1, 1))

    sample = _dataset(tmp_path, [_entry(Ks=intrinsics)])[0]

    assert sample["Ks"].shape == (21, 3, 3)


def test_the_mask_splits_the_normal_into_foreground_and_background(tmp_path, monkeypatch) -> None:
    _stub_decoder(
        monkeypatch,
        fills={"rgb.mp4": 200, "normal.mp4": 255, "mask.mp4": 255},
    )
    sample = _dataset(
        tmp_path,
        [_entry(normal_path="normal.mp4", mask_path="mask.mp4")],
        extras={"use_partial_normal_aug": True},
    )[0]

    # A fully-on mask puts everything in the foreground and greys the background.
    assert torch.allclose(sample["normal_fg_condition"], sample["normal_condition"], atol=1e-2)
    assert torch.allclose(
        sample["normal_bg_condition"], torch.tensor(0.5, dtype=torch.bfloat16), atol=1e-2
    )


def test_sample_carries_lightweight_sources_that_redecode_the_same_media(
    tmp_path,
    monkeypatch,
) -> None:
    from vforge.data.gameplay_dataset import (
        GAMEPLAY_MEDIA_SOURCES_KEY,
        decode_gameplay_media_source,
    )

    _stub_decoder(
        monkeypatch,
        fills={"rgb.mp4": 200, "normal.mp4": 255, "mask.mp4": 255},
    )
    sample = _dataset(
        tmp_path,
        [_entry(normal_path="normal.mp4", mask_path="mask.mp4")],
        extras={"use_partial_normal_aug": True},
    )[0]
    sources = sample[GAMEPLAY_MEDIA_SOURCES_KEY]

    assert set(sources) == {"gt", "normal", "normal_fg", "normal_bg"}
    assert not any(isinstance(value, torch.Tensor) for value in sources.values())
    assert torch.allclose(
        decode_gameplay_media_source(sources["gt"]),
        sample["video_frames"].float(),
        atol=1e-2,
    )
    assert torch.allclose(
        decode_gameplay_media_source(sources["normal_fg"]),
        sample["normal_fg_condition"].float(),
        atol=1e-2,
    )


def test_the_normal_split_is_absent_unless_the_mask_is_enabled(tmp_path, monkeypatch) -> None:
    _stub_decoder(monkeypatch, fills={"rgb.mp4": 200, "normal.mp4": 255, "mask.mp4": 255})
    sample = _dataset(tmp_path, [_entry(normal_path="normal.mp4", mask_path="mask.mp4")])[0]

    assert "normal_fg_condition" not in sample


def test_rotation_augmentation_turns_the_normal_and_leaves_the_video_alone(
    tmp_path, monkeypatch
) -> None:
    _stub_decoder(monkeypatch, fills={"rgb.mp4": 200, "normal.mp4": 20})
    entries = [_entry(normal_path="normal.mp4")]
    plain = _dataset(tmp_path, entries)[0]
    rotated = _dataset(
        tmp_path,
        entries,
        extras={"normal_rot_aug_enabled": True, "normal_rot_aug_full_so3": True},
    )[0]

    assert torch.equal(plain["video_frames"], rotated["video_frames"])
    assert not torch.allclose(plain["normal_condition"], rotated["normal_condition"], atol=1e-2)


def test_rotation_augmentation_is_off_outside_training(tmp_path, monkeypatch) -> None:
    """Val runs compare against a fixed target, so augmentation would be noise."""
    _stub_decoder(monkeypatch, fills={"rgb.mp4": 200, "normal.mp4": 20})
    dataset = _dataset(
        tmp_path,
        [_entry(normal_path="normal.mp4")],
        mode="val",
        extras={"normal_rot_aug_enabled": True},
    )

    assert dataset.normal_augmentor is None


def test_the_source_whitelist_can_exclude_this_source(tmp_path, monkeypatch) -> None:
    _stub_decoder(monkeypatch, fills={"rgb.mp4": 200, "normal.mp4": 20})
    entries = [_entry(normal_path="normal.mp4")]
    plain = _dataset(tmp_path, entries)[0]
    filtered = _dataset(
        tmp_path,
        entries,
        name="gameplay20s",
        extras={
            "normal_rot_aug_enabled": True,
            "normal_rot_aug_sources": ["seedance"],
        },
    )[0]

    assert torch.allclose(plain["normal_condition"], filtered["normal_condition"], atol=1e-2)


# ---------------------------------------------------------------------------
# defer_spatial: the worker decodes and nothing else
#
# What the deferred stage then does with these samples is pinned against
# taichu's implementation in test_preprocess_parity.py. What is checked
# here is the other half of the contract -- that the dataset emits exactly the
# keys, dtypes, and shapes that stage requires.
# ---------------------------------------------------------------------------


def _deferred(tmp_path, entries, **kwargs):
    extras = dict(kwargs.pop("extras", {}))
    extras["defer_spatial"] = True
    return _dataset(tmp_path, entries, extras=extras, **kwargs)


def test_deferred_samples_carry_raw_uint8_frames_and_the_target_geometry(
    tmp_path, monkeypatch
) -> None:
    from vforge.data.preprocess import (
        DEFERRED_SPATIAL_KEY,
        TARGET_HEIGHT_KEY,
        TARGET_WIDTH_KEY,
    )

    _stub_decoder(monkeypatch)
    sample = _deferred(tmp_path, [_entry()])[0]

    # Source resolution, not the target: resizing is what was deferred.
    assert sample["video_frames"].dtype == torch.uint8
    assert sample["video_frames"].shape == (21, 3, 72, 128)
    assert sample[DEFERRED_SPATIAL_KEY] is True
    assert (sample[TARGET_HEIGHT_KEY], sample[TARGET_WIDTH_KEY]) == (32, 64)


def test_a_deferred_missing_normal_travels_as_a_marker_not_a_grey_clip(
    tmp_path, monkeypatch
) -> None:
    """0.5 has no exact uint8 spelling, so the flag beats sending pixels."""
    from vforge.data.preprocess import NORMAL_MISSING_KEY

    _stub_decoder(monkeypatch)
    sample = _deferred(tmp_path, [_entry()])[0]

    assert sample[NORMAL_MISSING_KEY] is True
    assert sample["normal_condition"].dtype == torch.uint8
    assert sample["normal_condition"].numel() == 0


def test_a_deferred_mask_is_carried_raw_and_split_downstream(tmp_path, monkeypatch) -> None:
    from vforge.data.preprocess import RAW_NORMAL_MASK_KEY

    _stub_decoder(monkeypatch, fills={"rgb.mp4": 200, "normal.mp4": 255, "mask.mp4": 255})
    sample = _deferred(
        tmp_path,
        [_entry(normal_path="normal.mp4", mask_path="mask.mp4")],
        extras={"use_partial_normal_aug": True},
    )[0]

    assert "normal_fg_condition" not in sample
    assert sample[RAW_NORMAL_MASK_KEY].dtype == torch.uint8
    assert sample[RAW_NORMAL_MASK_KEY].shape == (21, 3, 72, 128)


def test_a_deferred_rotation_is_drawn_but_left_for_the_device(tmp_path, monkeypatch) -> None:
    from vforge.data.preprocess import NORMAL_ROTATION_KEY

    _stub_decoder(monkeypatch, fills={"rgb.mp4": 200, "normal.mp4": 20})
    sample = _deferred(
        tmp_path,
        [_entry(normal_path="normal.mp4")],
        extras={"normal_rot_aug_enabled": True, "normal_rot_aug_full_so3": True},
    )[0]

    rotation = sample[NORMAL_ROTATION_KEY]
    assert rotation.shape == (3, 3)
    assert torch.allclose(rotation @ rotation.T, torch.eye(3), atol=1e-5)
    # Still the untouched decode: the rotation has not been applied to it.
    assert sample["normal_condition"].dtype == torch.uint8
    assert sample["normal_condition"].unique().tolist() == [20]


def test_both_modes_consume_the_rng_identically(tmp_path, monkeypatch) -> None:
    """SP peers draw per-sample rotations from the global RNG and must stay in
    lockstep, so deferring the rotation may postpone applying it but not drawing
    it."""
    _stub_decoder(monkeypatch, fills={"rgb.mp4": 200, "normal.mp4": 20})
    entries = [_entry(normal_path="normal.mp4")]
    extras = {"normal_rot_aug_enabled": True, "normal_rot_aug_prob": 0.5}

    torch.manual_seed(0)
    _dataset(tmp_path, entries, extras=dict(extras))[0]
    after_plain = torch.randint(0, 2**31 - 1, (8,))

    torch.manual_seed(0)
    _deferred(tmp_path, entries, extras=dict(extras))[0]
    after_deferred = torch.randint(0, 2**31 - 1, (8,))

    assert torch.equal(after_plain, after_deferred)


def test_a_deferred_unreadable_video_falls_back_to_black_uint8(tmp_path, monkeypatch) -> None:
    import vforge.data.gameplay_dataset as gameplay

    def _boom(*_args, **_kwargs):
        raise OSError("moov atom not found")

    monkeypatch.setattr(gameplay, "read_video_window", _boom)
    sample = _deferred(tmp_path, [_entry()])[0]

    # Already at the target size, so the deferred resize is a no-op on it.
    assert sample["video_frames"].dtype == torch.uint8
    assert sample["video_frames"].shape == (21, 3, 32, 64)
    assert int(sample["video_frames"].max()) == 0


def test_reference_frames_are_refused_under_defer_spatial(tmp_path, monkeypatch) -> None:
    """The deferred stage has no contract for the key, so they would reach the
    encoder unresized while everything beside them was scaled."""
    _stub_decoder(monkeypatch)

    with pytest.raises(ValueError, match="extract_reference_frame"):
        _deferred(tmp_path, [_entry()], extras={"extract_reference_frame": True})


def test_the_deferred_pixels_land_where_the_in_worker_pixels_do(tmp_path, monkeypatch) -> None:
    """Equal within tolerance, not bit for bit.

    The two paths differ by design in three places -- fp32 is held to the end of
    the deferred stage where the worker drops to bfloat16 after the resize, the
    normal is rotated before rather than after the mask blend, and on a GPU the
    antialias kernel is a different one. This asserts they still describe the
    same picture.
    """
    import vforge.data.gameplay_dataset as gameplay
    from vforge.data.preprocess import (
        NORMAL_MISSING_KEY,
        TARGET_HEIGHT_KEY,
        TARGET_WIDTH_KEY,
        preprocess_deferred_batch,
    )

    def _read(path, start, end_inclusive):
        # A gradient rather than a constant, so the resize has something to do.
        count = max(0, min(int(end_inclusive), 399)) - max(0, min(int(start), 399)) + 1
        ramp = torch.arange(72 * 128, dtype=torch.uint8).reshape(1, 1, 72, 128)
        offset = 7 if "normal" in str(path) else 0
        return (ramp + offset).expand(count, 3, 72, 128).contiguous()

    monkeypatch.setattr(gameplay, "read_video_window", _read)
    entries = [_entry(normal_path="normal.mp4")]
    plain = _dataset(tmp_path, entries)[0]
    sample = _deferred(tmp_path, entries)[0]

    batch = {
        "video_frames": sample["video_frames"].unsqueeze(0),
        "normal_condition": sample["normal_condition"].unsqueeze(0),
        TARGET_HEIGHT_KEY: sample[TARGET_HEIGHT_KEY],
        TARGET_WIDTH_KEY: sample[TARGET_WIDTH_KEY],
        NORMAL_MISSING_KEY: sample[NORMAL_MISSING_KEY],
        "__prepare_deferred_spatial__": True,
    }
    assert preprocess_deferred_batch(batch, device=torch.device("cpu"))

    torch.testing.assert_close(
        batch["video_frames"][0].float(), plain["video_frames"].float(), atol=4e-3, rtol=0
    )
    torch.testing.assert_close(
        batch["normal_condition"][0].float(), plain["normal_condition"].float(), atol=4e-3, rtol=0
    )
