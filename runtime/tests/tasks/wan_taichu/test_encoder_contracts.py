from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


def test_audio_raw_collate_aligns_tensor_waveform_to_taichu_duration(monkeypatch) -> None:
    from vforge.tasks.wan_taichu.batch import build_raw_collate_fn

    def build_base_collate(*, extras, pretrained_path):
        del extras, pretrained_path

        def collate(samples):
            return {
                "sample_id": [sample["sample_id"] for sample in samples],
                "video_frames": torch.zeros(len(samples), 121, 3, 2, 2),
            }

        return collate

    monkeypatch.setattr(
        "vforge.tasks.wan_taichu.batch._build_base_raw_collate_fn",
        build_base_collate,
    )
    collate = build_raw_collate_fn(
        extras={"audio_sample_rate": 16000, "audio_video_rate": 25},
        pretrained_path="/unused",
        streams=(),
        use_audio=True,
    )
    short = torch.arange(10, dtype=torch.float32)
    long = torch.arange(80000, dtype=torch.float32)
    batch = collate(
        [
            {"sample_id": "short", "audio": short},
            {"sample_id": "long", "audio": long},
        ]
    )

    # Taichu uses the elapsed time between frame 0 and frame T-1.
    assert batch["audio"].shape == (2, 76800)
    assert torch.equal(batch["audio"][0, :10], short)
    assert torch.count_nonzero(batch["audio"][0, 10:] != short[-1]) == 0
    assert torch.equal(batch["audio"][1], long[:76800])


def test_raw_collate_preserves_and_aligns_inline_camera_fields(monkeypatch) -> None:
    from vforge.tasks.wan_taichu.batch import build_raw_collate_fn

    def build_base_collate(*, extras, pretrained_path):
        del extras, pretrained_path

        def collate(samples):
            return {
                "sample_id": [sample["sample_id"] for sample in samples],
                "video_frames": torch.zeros(len(samples), 3, 3, 4, 5),
                "grid_condition": None,
            }

        return collate

    monkeypatch.setattr(
        "vforge.tasks.wan_taichu.batch._build_base_raw_collate_fn",
        build_base_collate,
    )
    collate = build_raw_collate_fn(
        extras={},
        pretrained_path="/unused",
        streams=("grid", "normal"),
    )
    c2ws_short = torch.eye(4).repeat(2, 1, 1)
    intrinsics_long = torch.eye(3).repeat(4, 1, 1)
    batch = collate(
        [
            {
                "sample_id": "first",
                "c2ws": c2ws_short.tolist(),
                "Ks": intrinsics_long.tolist(),
            },
            {
                "sample_id": "second",
                "c2ws": torch.eye(4).repeat(3, 1, 1),
                "Ks": torch.eye(3).repeat(3, 1, 1),
            },
        ]
    )

    assert batch["c2ws"].shape == (2, 3, 4, 4)
    assert batch["Ks"].shape == (2, 3, 3, 3)
    assert torch.equal(batch["c2ws"][0, 1], batch["c2ws"][0, 2])
    assert torch.equal(batch["Ks"][0], intrinsics_long[:3])


def test_raw_collate_prefers_existing_grid_video_over_camera(monkeypatch) -> None:
    from vforge.tasks.wan_taichu.batch import build_raw_collate_fn

    grid = torch.ones(1, 3, 3, 4, 5)

    def build_base_collate(*, extras, pretrained_path):
        del extras, pretrained_path
        return lambda samples: {
            "video_frames": torch.zeros(len(samples), 3, 3, 4, 5),
            "grid_condition": grid,
        }

    monkeypatch.setattr(
        "vforge.tasks.wan_taichu.batch._build_base_raw_collate_fn",
        build_base_collate,
    )
    collate = build_raw_collate_fn(
        extras={},
        pretrained_path="/unused",
        streams=("grid",),
    )

    batch = collate([{"sample_id": "grid-video"}])

    assert batch["grid_condition"] is grid
    assert "c2ws" not in batch
    assert "Ks" not in batch


def test_raw_collate_allows_missing_camera_for_black_grid_fallback(monkeypatch) -> None:
    from vforge.tasks.wan_taichu.batch import build_raw_collate_fn

    monkeypatch.setattr(
        "vforge.tasks.wan_taichu.batch._build_base_raw_collate_fn",
        lambda **_: (
            lambda samples: {
                "video_frames": torch.zeros(len(samples), 3, 3, 4, 5),
                "grid_condition": None,
            }
        ),
    )
    collate = build_raw_collate_fn(
        extras={},
        pretrained_path="/unused",
        streams=("grid",),
    )

    batch = collate([{"sample_id": "no-camera"}])

    assert batch["grid_condition"] is None
    assert "c2ws" not in batch
    assert "Ks" not in batch


def test_inference_collate_allows_missing_camera_for_black_grid_fallback(
    monkeypatch,
) -> None:
    from vforge.tasks.wan_taichu.encoder import build_inference_collate_fn

    monkeypatch.setattr(
        "vforge.data.collate.collate_raw_wan_inference_batch",
        lambda samples, **_: {
            "num_frames": 3,
            "grid_condition": None,
            "video_frames": torch.zeros(len(samples), 3, 3, 4, 5),
        },
    )
    collate = build_inference_collate_fn(
        tokenizer=object(),
        extras={},
        streams=("grid",),
        allow_target_video=True,
    )

    batch = collate([{"sample_id": "no-camera"}])

    assert batch["grid_condition"] is None
    assert "c2ws" not in batch
    assert "Ks" not in batch


def test_inference_collate_accepts_decoded_gameplay_tensors(monkeypatch) -> None:
    from vforge.tasks.wan_taichu.encoder import build_inference_collate_fn

    def collate_decoded(samples, **_):
        videos = torch.stack([sample["video_frames"] for sample in samples])
        normals = torch.stack([sample["normal_condition"] for sample in samples])
        return {
            "sample_id": [sample["sample_id"] for sample in samples],
            "video_frames": videos,
            "normal_condition": normals,
            "grid_condition": None,
        }

    monkeypatch.setattr(
        "vforge.tasks.wan_taichu.batch.collate_raw_samples",
        collate_decoded,
    )
    collate = build_inference_collate_fn(
        tokenizer=object(),
        extras={},
        streams=("rgb", "grid", "normal"),
        allow_target_video=True,
    )
    sample = {
        "sample_id": "pkl-sample",
        "video_frames": torch.zeros(5, 3, 4, 6),
        "normal_condition": torch.full((5, 3, 4, 6), 0.5),
        "c2ws": torch.eye(4).repeat(5, 1, 1),
        "Ks": torch.eye(3).repeat(5, 1, 1),
    }

    batch = collate([sample])

    assert batch["video_frames"].shape == (1, 5, 3, 4, 6)
    assert batch["num_frames"] == 5
    assert batch["height"] == 4
    assert batch["width"] == 6
    assert batch["c2ws"].shape == (1, 5, 4, 4)
    assert batch["Ks"].shape == (1, 5, 3, 3)


def test_raw_collate_rejects_unpaired_camera_fields(monkeypatch) -> None:
    from vforge.tasks.wan_taichu.batch import build_raw_collate_fn

    monkeypatch.setattr(
        "vforge.tasks.wan_taichu.batch._build_base_raw_collate_fn",
        lambda **_: (
            lambda samples: {
                "video_frames": torch.zeros(len(samples), 2, 3, 4, 5),
                "grid_condition": None,
            }
        ),
    )
    collate = build_raw_collate_fn(
        extras={},
        pretrained_path="/unused",
        streams=("grid",),
    )

    with pytest.raises(ValueError, match="missing 'Ks'"):
        collate(
            [
                {
                    "sample_id": "missing-intrinsics",
                    "c2ws": torch.eye(4).repeat(2, 1, 1),
                }
            ]
        )


def test_condition_encoder_renders_one_grid_per_camera_sample(monkeypatch) -> None:
    from vforge.models.wan_taichu.encoder import WanTaichuConditionEncoder

    calls = []

    def render(c2ws, intrinsics, height, width, *, device, random_rot_deg):
        calls.append((c2ws.clone(), intrinsics.clone(), height, width, device, random_rot_deg))
        return torch.full((c2ws.shape[0], 3, height, width), float(len(calls)))

    monkeypatch.setattr("vforge.models.wan_taichu.encoder.camera_to_grid", render)
    encoder = SimpleNamespace(grid_random_rot_deg=30.0)
    frames = torch.zeros(2, 3, 3, 4, 5)
    batch = {
        "c2ws": torch.eye(4).repeat(2, 3, 1, 1),
        "Ks": torch.eye(3).repeat(2, 3, 1, 1),
    }

    result = WanTaichuConditionEncoder._render_grid_from_camera(
        encoder,
        batch,
        frames=frames,
    )

    assert result.shape == (2, 3, 3, 4, 5)
    assert torch.count_nonzero(result[0] != 1.0) == 0
    assert torch.count_nonzero(result[1] != 2.0) == 0
    assert [call[5] for call in calls] == [30.0, 30.0]


def test_condition_encoder_uses_camera_fallback_for_grid_pixels() -> None:
    from vforge.models.wan_taichu.encoder import WanTaichuConditionEncoder

    rendered = torch.full((1, 2, 3, 4, 5), 0.75)
    encoder = SimpleNamespace(
        raw_stream_keys={"grid": "grid_condition"},
        _render_grid_from_camera=lambda batch, frames: rendered,
        scale_pixels=WanTaichuConditionEncoder.scale_pixels,
    )

    pixels = WanTaichuConditionEncoder._condition_pixels(
        encoder,
        "grid",
        {"c2ws": torch.eye(4).repeat(1, 2, 1, 1)},
        frames=torch.zeros(1, 2, 3, 4, 5),
    )

    assert torch.equal(pixels, torch.full_like(rendered, 0.5))


def test_condition_encoder_prefers_existing_grid_pixels() -> None:
    from vforge.models.wan_taichu.encoder import WanTaichuConditionEncoder

    raw = torch.full((1, 2, 3, 4, 5), 0.25)
    encoder = SimpleNamespace(
        raw_stream_keys={"grid": "grid_condition"},
        _render_grid_from_camera=lambda *args, **kwargs: pytest.fail(
            "camera renderer must not run when grid_condition exists"
        ),
        scale_pixels=WanTaichuConditionEncoder.scale_pixels,
    )

    pixels = WanTaichuConditionEncoder._condition_pixels(
        encoder,
        "grid",
        {"grid_condition": raw},
        frames=torch.zeros(1, 2, 3, 4, 5),
    )

    assert torch.equal(pixels, torch.full_like(raw, -0.5))


def test_condition_encoder_caches_constant_fallback_latents() -> None:
    from vforge.models.wan_taichu.encoder import WanTaichuConditionEncoder

    encoded_inputs = []

    def encode_image(pixels):
        encoded_inputs.append(pixels.clone())
        return pixels[:, :1]

    encoder = SimpleNamespace(
        vae=nn.Linear(1, 1),
        training_dtype=torch.float32,
        mask_channels=0,
        _constant_condition_cache={},
        scale_pixels=WanTaichuConditionEncoder.scale_pixels,
        encode_image=encode_image,
    )
    kwargs = {
        "pixel_value": 0.5,
        "batch_size": 1,
        "num_frames": 2,
        "height": 3,
        "width": 4,
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }

    first = WanTaichuConditionEncoder.encode_constant_condition(encoder, **kwargs)
    second = WanTaichuConditionEncoder.encode_constant_condition(encoder, **kwargs)

    assert len(encoded_inputs) == 1
    assert first.data_ptr() == second.data_ptr()
    assert torch.count_nonzero(first) == 0


def test_wan21_shaped_encoder_emits_v3_fields_per_stream() -> None:
    from vforge.models.wan_taichu.encoder import WanTaichuConditionEncoder
    from vforge.tasks.wan_taichu.batch import build_encoded_batch_codec
    from vforge.tasks.wan_taichu.variants import require_train_variant

    frames = torch.zeros(1, 2, 3, 4, 5)
    grid = torch.full_like(frames, 0.25)
    normal = torch.full_like(frames, 0.75)
    encoder = SimpleNamespace(
        vae=nn.Linear(1, 1),
        training_dtype=torch.float32,
        streams=("grid", "normal"),
        mask_channels=4,
        use_vae_condition=False,
        use_clip_condition=False,
        use_dino_condition=False,
        use_audio_condition=False,
        scale_pixels=WanTaichuConditionEncoder.scale_pixels,
        encode_prompt=lambda text_inputs, text_mask, batch_size: torch.zeros(batch_size, 2, 8),
        encode_image=lambda pixels: torch.zeros(
            pixels.shape[0], 16, pixels.shape[2], pixels.shape[3], pixels.shape[4]
        ),
        _condition_pixels=lambda name, batch, frames: WanTaichuConditionEncoder.scale_pixels(
            batch[f"{name}_condition"]
        ),
    )
    batch = {
        "video_frames": frames,
        "image_condition": frames[:, :1],
        "grid_condition": grid,
        "normal_condition": normal,
        "text_inputs": torch.zeros(1, 2, dtype=torch.long),
        "text_mask": torch.ones(1, 2, dtype=torch.long),
    }

    result = WanTaichuConditionEncoder._encode_wan21_batch(encoder, batch)
    codec = build_encoded_batch_codec(
        require_train_variant("wan22_a14b"),
        ("grid", "normal"),
    )
    canonical = codec.canonicalize(result)

    assert canonical["condition_latents_grid"].shape == (1, 20, 2, 4, 5)
    assert canonical["condition_latents_normal"].shape == (1, 20, 2, 4, 5)
