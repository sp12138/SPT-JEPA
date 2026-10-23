"""CPU tests for Wan raw-media collation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


class _StubTokenizer:
    model_max_length = 4

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, prompts, **kwargs):
        self.calls.append(kwargs)
        count = len(prompts)
        width = int(kwargs.get("max_length", 4))
        return SimpleNamespace(
            input_ids=torch.ones(count, width, dtype=torch.long),
            attention_mask=torch.ones(count, width, dtype=torch.long),
        )


def _stub_clips(monkeypatch, fills: dict[str, float]) -> None:
    """Decode each path to a constant clip so content identifies its source."""
    import vforge.data.collate as collate

    monkeypatch.setattr(
        collate,
        "read_video_tensor",
        lambda path, *, max_frames=None: torch.full((2, 3, 4, 4), fills[str(path)]),
    )


def test_raw_collate_leaves_the_grid_condition_absent_when_unused(monkeypatch) -> None:
    from vforge.data.collate import collate_raw_wan_batch

    _stub_clips(monkeypatch, {"a.mp4": 0.1, "a-normal.mp4": 0.2})

    batch = collate_raw_wan_batch(
        [{"sample_id": "a", "video": "a.mp4", "normal": "a-normal.mp4", "prompt": ""}],
        tokenizer=_StubTokenizer(),
    )

    assert batch["grid_condition"] is None
    assert batch["normal_condition"].unique().tolist() == [pytest.approx(0.2)]


def test_raw_collate_reads_a_declared_grid_condition(monkeypatch) -> None:
    from vforge.data.collate import collate_raw_wan_batch

    _stub_clips(monkeypatch, {"a.mp4": 0.1, "a-normal.mp4": 0.2, "a-grid.mp4": 0.3})

    batch = collate_raw_wan_batch(
        [
            {
                "sample_id": "a",
                "video": "a.mp4",
                "normal": "a-normal.mp4",
                "grid": "a-grid.mp4",
                "prompt": "",
            }
        ],
        tokenizer=_StubTokenizer(),
    )

    assert batch["grid_condition"].shape == (1, 2, 3, 4, 4)
    assert batch["grid_condition"].unique().tolist() == [pytest.approx(0.3)]


def test_raw_collate_rejects_a_grid_condition_only_some_samples_declare(
    monkeypatch,
) -> None:
    """Half a batch conditioned on a substitute would be an invisible bug."""
    from vforge.data.collate import collate_raw_wan_batch

    _stub_clips(monkeypatch, {"a.mp4": 0.1, "b.mp4": 0.1, "a-grid.mp4": 0.3})

    with pytest.raises(ValueError, match="grid"):
        collate_raw_wan_batch(
            [
                {"sample_id": "a", "video": "a.mp4", "grid": "a-grid.mp4", "prompt": ""},
                {"sample_id": "b", "video": "b.mp4", "prompt": ""},
            ],
            tokenizer=_StubTokenizer(),
        )


def test_the_configured_text_width_beats_the_tokenizer_declaration() -> None:
    """Otherwise the embedding width is a property of the tokenizer files, not
    of the run, and a cache written at 512 would silently be read at another."""
    from vforge.data.collate import tokenize_prompts

    tokenizer = _StubTokenizer()
    text_inputs, _ = tokenize_prompts(tokenizer, ["a"], max_sequence_length=226)

    assert tokenizer.calls[0]["max_length"] == 226
    assert text_inputs.shape == (1, 226)


def test_raw_collate_accepts_pre_decoded_clips() -> None:
    """The taichu datasets decode in __getitem__, so they arrive as tensors."""
    from vforge.data.collate import collate_raw_wan_batch

    batch = collate_raw_wan_batch(
        [
            {
                "sample_id": "a",
                "video_frames": torch.full((5, 3, 4, 4), 0.1),
                "normal_condition": torch.full((5, 3, 4, 4), 0.2),
                "prompt": "hello",
            }
        ],
        tokenizer=_StubTokenizer(),
    )

    assert batch["video_frames"].shape == (1, 5, 3, 4, 4)
    assert batch["normal_condition"].unique().tolist() == [pytest.approx(0.2)]
    assert batch["image_condition"].shape == (1, 1, 3, 4, 4)


def test_raw_collate_requires_pixels_or_a_path() -> None:
    from vforge.data.collate import collate_raw_wan_batch

    with pytest.raises(ValueError, match="video_frames"):
        collate_raw_wan_batch([{"sample_id": "a", "prompt": ""}], tokenizer=_StubTokenizer())


def test_raw_collate_stacks_the_optional_condition_tensors() -> None:
    from vforge.data.collate import collate_raw_wan_batch

    def _sample(sample_id: str) -> dict:
        return {
            "sample_id": sample_id,
            "video_frames": torch.zeros(5, 3, 4, 4),
            "prompt": "",
            "Ks": torch.eye(3).expand(5, 3, 3).contiguous(),
            "c2ws": torch.eye(4).expand(5, 4, 4).contiguous(),
            "reference_frames": torch.zeros(2, 3, 4, 4),
            "normal_fg_condition": torch.zeros(5, 3, 4, 4),
            "normal_bg_condition": torch.zeros(5, 3, 4, 4),
        }

    batch = collate_raw_wan_batch([_sample("a"), _sample("b")], tokenizer=_StubTokenizer())

    assert batch["Ks"].shape == (2, 5, 3, 3)
    assert batch["c2ws"].shape == (2, 5, 4, 4)
    assert batch["reference_frames"].shape == (2, 2, 3, 4, 4)
    assert batch["normal_fg_condition"].shape == (2, 5, 3, 4, 4)


def test_raw_collate_rejects_a_condition_tensor_only_some_samples_carry() -> None:
    from vforge.data.collate import collate_raw_wan_batch

    with pytest.raises(ValueError, match="Ks"):
        collate_raw_wan_batch(
            [
                {
                    "sample_id": "a",
                    "video_frames": torch.zeros(5, 3, 4, 4),
                    "prompt": "",
                    "Ks": torch.eye(3).expand(5, 3, 3).contiguous(),
                },
                {"sample_id": "b", "video_frames": torch.zeros(5, 3, 4, 4), "prompt": ""},
            ],
            tokenizer=_StubTokenizer(),
        )


def test_raw_collate_leaves_optional_streams_out_when_no_sample_has_them() -> None:
    from vforge.data.collate import collate_raw_wan_batch

    batch = collate_raw_wan_batch(
        [{"sample_id": "a", "video_frames": torch.zeros(5, 3, 4, 4), "prompt": ""}],
        tokenizer=_StubTokenizer(),
    )

    assert "Ks" not in batch
    assert "reference_frames" not in batch
    assert "audio" not in batch


def test_a_stamped_text_width_overrides_the_collate_default() -> None:
    """A mixed run has one closure but several sources; the source knows better."""
    from vforge.data.collate import collate_raw_wan_batch

    tokenizer = _StubTokenizer()
    collate_raw_wan_batch(
        [
            {
                "sample_id": "a",
                "video_frames": torch.zeros(5, 3, 4, 4),
                "prompt": "",
                "max_sequence_length": 226,
            }
        ],
        tokenizer=tokenizer,
        max_sequence_length=512,
    )

    assert tokenizer.calls[0]["max_length"] == 226


def test_a_batch_that_disagrees_on_text_width_is_an_error() -> None:
    from vforge.data.collate import collate_raw_wan_batch

    def _sample(sample_id: str, width: int) -> dict:
        return {
            "sample_id": sample_id,
            "video_frames": torch.zeros(5, 3, 4, 4),
            "prompt": "",
            "max_sequence_length": width,
        }

    with pytest.raises(ValueError, match="max_sequence_length"):
        collate_raw_wan_batch(
            [_sample("a", 226), _sample("b", 512)],
            tokenizer=_StubTokenizer(),
        )


def test_raw_collate_forwards_latent_bounds_and_audio() -> None:
    from vforge.data.collate import collate_raw_wan_batch

    batch = collate_raw_wan_batch(
        [
            {
                "sample_id": "a",
                "video_frames": torch.zeros(5, 3, 4, 4),
                "prompt": "",
                "min_latents": 2,
                "max_latents": 6,
                # Waveforms differ in length, so the encoder wants a list.
                "audio": torch.zeros(1600),
            }
        ],
        tokenizer=_StubTokenizer(),
    )

    assert (batch["min_latents"], batch["max_latents"]) == (2, 6)
    assert isinstance(batch["audio"], list)


def test_inference_collate_uses_image_and_normal_geometry(monkeypatch) -> None:
    import vforge.data.collate as collate
    from vforge.data.collate import collate_raw_wan_inference_batch

    monkeypatch.setattr(
        collate,
        "read_video_tensor",
        lambda _path, *, max_frames=None: torch.full((5, 3, 4, 4), 0.2),
    )
    monkeypatch.setattr(
        collate,
        "read_image_tensor",
        lambda _path: torch.full((3, 4, 4), 0.5),
    )

    batch = collate_raw_wan_inference_batch(
        [{"sample_id": "a", "image": "first.png", "normal": "normal.mp4"}],
        tokenizer=_StubTokenizer(),
        required_streams=("normal",),
    )

    assert batch["image_condition"].shape == (1, 1, 3, 4, 4)
    assert batch["normal_condition"].shape == (1, 5, 3, 4, 4)
    assert (batch["num_frames"], batch["height"], batch["width"]) == (5, 4, 4)
    assert "video_frames" not in batch


def test_inference_collate_requires_equal_multi_ref_counts_per_batch(monkeypatch) -> None:
    import vforge.data.collate as collate
    from vforge.data.collate import collate_raw_wan_inference_batch

    monkeypatch.setattr(
        collate,
        "read_image_tensor",
        lambda _path: torch.zeros(3, 4, 4),
    )
    monkeypatch.setattr(
        collate,
        "read_reference_frames",
        lambda sample: torch.zeros(len(sample["ref"]), 3, 4, 4),
    )

    with pytest.raises(ValueError, match="same ref count"):
        collate_raw_wan_inference_batch(
            [
                {"sample_id": "a", "image": "a.png", "num_frames": 5, "ref": ["a-ref.png"]},
                {
                    "sample_id": "b",
                    "image": "b.png",
                    "num_frames": 5,
                    "ref": ["b-ref-0.png", "b-ref-1.png"],
                },
            ],
            tokenizer=_StubTokenizer(),
        )


def test_inference_collate_requires_num_frames_without_temporal_condition(
    monkeypatch,
) -> None:
    import vforge.data.collate as collate
    from vforge.data.collate import collate_raw_wan_inference_batch

    monkeypatch.setattr(
        collate,
        "read_image_tensor",
        lambda _path: torch.zeros(3, 4, 4),
    )

    with pytest.raises(ValueError, match="require num_frames"):
        collate_raw_wan_inference_batch(
            [{"sample_id": "a", "image": "first.png"}],
            tokenizer=_StubTokenizer(),
        )


def test_inference_collate_rejects_target_video_and_missing_required_stream(
    monkeypatch,
) -> None:
    import vforge.data.collate as collate
    from vforge.data.collate import collate_raw_wan_inference_batch

    monkeypatch.setattr(
        collate,
        "read_image_tensor",
        lambda _path: torch.zeros(3, 4, 4),
    )
    with pytest.raises(ValueError, match="condition-only"):
        collate_raw_wan_inference_batch(
            [{"image": "first.png", "video": "target.mp4"}],
            tokenizer=_StubTokenizer(),
        )
    with pytest.raises(ValueError, match="normal"):
        collate_raw_wan_inference_batch(
            [{"image": "first.png", "num_frames": 5}],
            tokenizer=_StubTokenizer(),
            required_streams=("normal",),
        )


def test_inference_collate_rejects_unaligned_temporal_geometry(monkeypatch) -> None:
    import vforge.data.collate as collate
    from vforge.data.collate import collate_raw_wan_inference_batch

    monkeypatch.setattr(
        collate,
        "read_image_tensor",
        lambda _path: torch.zeros(3, 4, 4),
    )

    with pytest.raises(ValueError, match=r"num_frames - 1"):
        collate_raw_wan_inference_batch(
            [{"image": "first.png", "num_frames": 6}],
            tokenizer=_StubTokenizer(),
        )


# ---------------------------------------------------------------------------
# defer_spatial transport
# ---------------------------------------------------------------------------


def _deferred_sample(
    *,
    sample_id: str = "a",
    normal: bool = True,
    rotation: torch.Tensor | None = None,
    mask: bool = False,
    target: tuple[int, int] = (2, 2),
) -> dict:
    from vforge.data.preprocess import (
        DEFERRED_SPATIAL_KEY,
        NORMAL_MISSING_KEY,
        NORMAL_ROTATION_KEY,
        RAW_NORMAL_MASK_KEY,
        TARGET_HEIGHT_KEY,
        TARGET_WIDTH_KEY,
    )

    frames = torch.full((2, 3, 4, 4), 7, dtype=torch.uint8)
    sample = {
        "sample_id": sample_id,
        "prompt": "",
        "video_frames": frames,
        "image_condition": frames[0:1],
        "normal_condition": (
            torch.full((2, 3, 4, 4), 9, dtype=torch.uint8)
            if normal
            else torch.empty(0, dtype=torch.uint8)
        ),
        DEFERRED_SPATIAL_KEY: True,
        TARGET_HEIGHT_KEY: target[0],
        TARGET_WIDTH_KEY: target[1],
        NORMAL_MISSING_KEY: not normal,
    }
    if rotation is not None:
        sample[NORMAL_ROTATION_KEY] = rotation
    if mask:
        sample[RAW_NORMAL_MASK_KEY] = torch.full((2, 3, 4, 4), 255, dtype=torch.uint8)
    return sample


def test_deferred_markers_survive_the_collate_and_the_tensors_stack() -> None:
    from vforge.data.collate import collate_raw_wan_batch
    from vforge.data.preprocess import (
        DEFERRED_SPATIAL_KEY,
        NORMAL_ROTATION_KEY,
        RAW_NORMAL_MASK_KEY,
        TARGET_HEIGHT_KEY,
    )

    batch = collate_raw_wan_batch(
        [
            _deferred_sample(sample_id="a", rotation=torch.eye(3), mask=True),
            _deferred_sample(sample_id="b", rotation=torch.eye(3), mask=True),
        ],
        tokenizer=_StubTokenizer(),
    )

    assert batch[DEFERRED_SPATIAL_KEY] is True
    assert batch[TARGET_HEIGHT_KEY] == 2
    assert batch["video_frames"].dtype == torch.uint8
    assert batch[NORMAL_ROTATION_KEY].shape == (2, 3, 3)
    assert batch[RAW_NORMAL_MASK_KEY].shape == (2, 2, 3, 4, 4)


def test_a_deferred_missing_normal_stays_empty_instead_of_a_zeros_clip() -> None:
    """A zeros fill here would move a second raw-resolution payload saying nothing."""
    from vforge.data.collate import collate_raw_wan_batch

    batch = collate_raw_wan_batch(
        [_deferred_sample(sample_id="a", normal=False)],
        tokenizer=_StubTokenizer(),
    )

    assert batch["normal_condition"].shape == (1, 0)
    assert batch["normal_condition"].dtype == torch.uint8


def test_an_unrotated_deferred_sample_is_filled_with_the_identity() -> None:
    """``normal_rot_aug_prob < 1`` leaves some samples unrotated, and identity
    reproduces them exactly rather than crashing the stack."""
    from vforge.data.collate import collate_raw_wan_batch
    from vforge.data.preprocess import NORMAL_ROTATION_KEY

    turn = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    batch = collate_raw_wan_batch(
        [_deferred_sample(sample_id="a", rotation=turn), _deferred_sample(sample_id="b")],
        tokenizer=_StubTokenizer(),
    )

    assert torch.equal(batch[NORMAL_ROTATION_KEY][0], turn)
    assert torch.equal(batch[NORMAL_ROTATION_KEY][1], torch.eye(3))


def test_deferred_and_processed_samples_cannot_share_a_batch(monkeypatch) -> None:
    from vforge.data.collate import collate_raw_wan_batch

    _stub_clips(monkeypatch, {"a.mp4": 0.1})
    with pytest.raises(ValueError, match="mixes deferred"):
        collate_raw_wan_batch(
            [_deferred_sample(), {"sample_id": "b", "video": "a.mp4", "prompt": ""}],
            tokenizer=_StubTokenizer(),
        )


def test_deferred_samples_must_agree_on_whether_a_normal_exists() -> None:
    """The deferred stage resolves it once per batch, so half a column is fatal."""
    from vforge.data.collate import collate_raw_wan_batch

    with pytest.raises(ValueError, match="with and without a normal"):
        collate_raw_wan_batch(
            [_deferred_sample(sample_id="a"), _deferred_sample(sample_id="b", normal=False)],
            tokenizer=_StubTokenizer(),
        )


def test_deferred_samples_must_agree_on_the_target_geometry() -> None:
    from vforge.data.collate import collate_raw_wan_batch

    with pytest.raises(ValueError, match="target height"):
        collate_raw_wan_batch(
            [
                _deferred_sample(sample_id="a", target=(2, 2)),
                _deferred_sample(sample_id="b", target=(4, 2)),
            ],
            tokenizer=_StubTokenizer(),
        )


def test_a_partial_mask_column_is_rejected() -> None:
    from vforge.data.collate import collate_raw_wan_batch

    with pytest.raises(ValueError, match="only some deferred samples carry a normal mask"):
        collate_raw_wan_batch(
            [
                _deferred_sample(sample_id="a", mask=True),
                _deferred_sample(sample_id="b", mask=False),
            ],
            tokenizer=_StubTokenizer(),
        )


def test_a_collated_deferred_batch_survives_the_broadcast_and_leaves_no_markers() -> None:
    """The leader broadcasts before it encodes, so the markers have to make it
    through the tensor/metadata split intact."""
    from vforge.data.collate import collate_raw_wan_batch
    from vforge.data.loader import _rebuild_batch, _split_batch
    from vforge.data.preprocess import MARKER_KEYS, preprocess_deferred_batch

    batch = collate_raw_wan_batch(
        [
            _deferred_sample(sample_id="a", rotation=torch.eye(3), mask=True),
            _deferred_sample(sample_id="b", rotation=torch.eye(3), mask=True),
        ],
        tokenizer=_StubTokenizer(),
    )
    meta, tensors = _split_batch(batch)
    received = _rebuild_batch(meta, tensors)

    assert preprocess_deferred_batch(received, device=torch.device("cpu"))
    assert not [key for key in MARKER_KEYS if key in received]
    assert received["video_frames"].shape == (2, 2, 3, 2, 2)
    assert received["video_frames"].dtype == torch.bfloat16
    assert received["image_condition"].shape == (2, 1, 3, 2, 2)
    assert received["normal_fg_condition"].shape == (2, 2, 3, 2, 2)
