"""Wan Taichu v3 encoded-batch schema."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from vforge.data.cache import EncodedBatchCodec
from vforge.data.preprocess import (
    DEFERRED_SPATIAL_KEY,
    NORMAL_MISSING_KEY,
    NORMAL_ROTATION_KEY,
    RAW_NORMAL_MASK_KEY,
    TARGET_HEIGHT_KEY,
    TARGET_WIDTH_KEY,
)
from vforge.data.video_io import read_video_tensor
from vforge.tasks.wan_taichu.variants import WanTaichuVariant

COMMON_FIELDS = ("prompt", "frame_latents")
IMAGE_FIELDS = (
    "image_condition_latent_vae",
    "image_condition_latent_clip",
    "image_condition_latent_dino",
)
EXTRA_LATENT_FIELDS = (
    "first_frame_latents",
    "ref_frame_latents",
    "normal_fg_body",
    "normal_bg_body",
)
LATENT_BOUND_FIELDS = ("min_latents", "max_latents")
_RGB_KEYS = ("video_frames", "video")
_NORMAL_KEYS = ("normal_condition", "normal")
_GRID_KEYS = ("grid_condition", "grid")
_OPTIONAL_RAW_TENSOR_STREAMS = (
    "Ks",
    "c2ws",
    "reference_frames",
    "normal_fg_condition",
    "normal_bg_condition",
)


def stream_field(name: str) -> str:
    return f"condition_latents_{name}"


def fallback_stream_field(name: str) -> str:
    return f"condition_fallback_latents_{name}"


def build_encoded_batch_codec(
    variant: WanTaichuVariant,
    streams: Sequence[str],
) -> EncodedBatchCodec:
    """Return the strict per-stream v3 cache schema for one train request."""
    resolved = variant.resolve_condition_streams(streams)
    optional = (
        *IMAGE_FIELDS,
        *EXTRA_LATENT_FIELDS,
        *(fallback_stream_field(name) for name in resolved),
        "audio_features",
        "audio_features_l",
    )
    stream_id = "-".join(resolved)
    return EncodedBatchCodec(
        name=f"wan_taichu_{variant.name}_{stream_id}",
        required_keys=(*COMMON_FIELDS, *(stream_field(name) for name in resolved)),
        optional_keys=tuple(dict.fromkeys(optional)),
    )


def canonicalize_encoded_batch(
    bundle: Mapping[str, Any],
    *,
    variant: WanTaichuVariant,
    streams: Sequence[str],
) -> dict[str, Any]:
    """Validate and project a v3 per-stream cache into the model batch."""
    resolved = variant.resolve_condition_streams(streams)
    out = dict(bundle)
    _require_tensor(out, "prompt", rank=3)
    frames = _require_tensor(out, "frame_latents", rank=5)
    if int(frames.shape[1]) != variant.z_dim:
        raise ValueError(
            f"{variant.name} frame_latents must have z_dim={variant.z_dim}; "
            f"got shape={tuple(frames.shape)}"
        )

    block_channels = variant.mask_channels + variant.z_dim

    canonical: dict[str, Any] = {
        "sample_id": out.get("sample_id"),
        "prompt": out["prompt"],
        "frame_latents": frames,
    }
    bounds = read_latent_bounds(out, context="Wan Taichu encoded batch")
    for field, value in zip(LATENT_BOUND_FIELDS, bounds, strict=True):
        if value is not None:
            canonical[field] = value
    geometry = tuple(frames.shape[2:])
    batch_size = int(frames.shape[0])
    for name in resolved:
        field = stream_field(name)
        value = _require_tensor(out, field, rank=5)
        _validate_latent(
            value,
            field=field,
            channels=block_channels,
            batch_size=batch_size,
            geometry=geometry,
        )
        canonical[field] = value

    for field in (*IMAGE_FIELDS, *EXTRA_LATENT_FIELDS, "audio_features", "audio_features_l"):
        if out.get(field) is not None:
            canonical[field] = out[field]
    for name in resolved:
        field = fallback_stream_field(name)
        value = out.get(field)
        if value is None:
            continue
        _validate_latent(
            value,
            field=field,
            channels=block_channels,
            batch_size=batch_size,
            geometry=geometry,
        )
        canonical[field] = value

    first = canonical.get("first_frame_latents")
    if variant.first_frame_inpaint:
        if not isinstance(first, torch.Tensor):
            raise ValueError(f"{variant.name} requires first_frame_latents")
        _validate_latent(
            first,
            field="first_frame_latents",
            channels=variant.z_dim,
            batch_size=batch_size,
            geometry=(1, *geometry[1:]),
        )
    ref = canonical.get("ref_frame_latents")
    if ref is not None:
        if not variant.ref_token:
            raise ValueError(f"{variant.name} does not accept ref_frame_latents")
        if not isinstance(ref, torch.Tensor):
            raise ValueError("ref_frame_latents must be a tensor")
        if ref.ndim != 5:
            raise ValueError(f"ref_frame_latents must be [B,C,K,H,W]; got shape={tuple(ref.shape)}")
        expected = (batch_size, variant.z_dim)
        actual = (
            int(ref.shape[0]),
            int(ref.shape[1]),
        )
        if actual != expected or any(int(size) < 1 for size in ref.shape[2:]):
            raise ValueError(
                "ref_frame_latents must have positive temporal/spatial dimensions and "
                f"batch/channel geometry={expected}; got {tuple(ref.shape)}"
            )
    audio_fields = ("audio_features", "audio_features_l")
    present_audio = [field for field in audio_fields if canonical.get(field) is not None]
    if present_audio and not variant.audio:
        raise ValueError(f"{variant.name} does not accept {present_audio}")
    if variant.audio and not isinstance(canonical.get("audio_features"), torch.Tensor):
        raise ValueError(f"{variant.name} requires audio_features")
    for field in present_audio:
        value = canonical[field]
        if not isinstance(value, torch.Tensor) or value.ndim != 4:
            raise ValueError(f"{field} must be [B,L,D,F]; got {type(value).__name__}")
        if int(value.shape[0]) != batch_size:
            raise ValueError(f"{field} batch={value.shape[0]} must match frame batch={batch_size}")
        if (
            int(value.shape[1]) != variant.audio_num_layers
            or int(value.shape[2]) != variant.audio_dim
        ):
            raise ValueError(
                f"{field} must use [L={variant.audio_num_layers},D={variant.audio_dim}]; "
                f"got shape={tuple(value.shape)}"
            )
        if int(value.shape[3]) < 1:
            raise ValueError(f"{field} must contain at least one audio frame")
    return canonical


def canonicalize_training_bundle(
    codec: EncodedBatchCodec,
    bundle: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the cache schema without discarding task-owned runtime metadata.

    Source length bounds choose the training shape for this step; they are not
    latent-cache fields. ``metadata`` lets the online path reattach them after
    the frozen encoders have projected a raw batch into the strict cache shape.
    """
    canonical = codec.canonicalize(dict(bundle))
    runtime = bundle if metadata is None else metadata
    if "sample_id" in runtime:
        canonical["sample_id"] = runtime.get("sample_id")
    bounds = read_latent_bounds(runtime, context="Wan Taichu training batch")
    for field, value in zip(LATENT_BOUND_FIELDS, bounds, strict=True):
        if value is not None:
            canonical[field] = value
    return canonical


def collate_encoded_bundles(
    codec: EncodedBatchCodec,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collate offline encoded samples using one variant's strict codec."""
    if not samples:
        raise ValueError("cannot collate an empty Wan Taichu encoded batch")
    batch: dict[str, Any] = {}
    for key in codec.keys:
        values = [sample.get(key) for sample in samples]
        if all(value is None for value in values):
            batch[key] = None
            continue
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise ValueError(f"encoded cache field {key!r} mixes tensors and missing values")
        if key in {"prompt", "image_condition_latent_clip", "image_condition_latent_dino"}:
            expected_rank = 3
        elif key in {"audio_features", "audio_features_l"}:
            expected_rank = 4
        else:
            expected_rank = 5
        normalized = []
        for value in values:
            if value.ndim == expected_rank - 1:
                value = value.unsqueeze(0)
            if value.ndim != expected_rank:
                raise ValueError(
                    f"encoded cache field {key!r} has rank {value.ndim}; "
                    f"expected {expected_rank - 1} (per-sample) or {expected_rank}"
                )
            normalized.append(value)
        batch[key] = torch.cat(normalized, dim=0)
    batch["sample_id"] = [str(sample["sample_id"]) for sample in samples]
    batch.update(collate_latent_bounds(samples))
    return batch


def collate_latent_bounds(samples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Require one source-owned length interval across a Wan Taichu batch."""
    per_sample = [read_latent_bounds(sample, context="Wan Taichu sample") for sample in samples]
    if not per_sample:
        return {}
    if any(bounds != per_sample[0] for bounds in per_sample[1:]):
        raise ValueError(
            "Wan Taichu batch mixes latent bounds: "
            f"{[dict(zip(LATENT_BOUND_FIELDS, bounds, strict=True)) for bounds in per_sample]}"
        )
    return {
        field: value
        for field, value in zip(LATENT_BOUND_FIELDS, per_sample[0], strict=True)
        if value is not None
    }


def build_raw_collate_fn(
    *,
    extras: Mapping[str, Any],
    pretrained_path: str,
    streams: Sequence[str],
    use_audio: bool = False,
):
    """Build the raw-online collate before Wan Taichu frozen encoding."""
    base_collate = _build_base_raw_collate_fn(
        extras=extras,
        pretrained_path=pretrained_path,
    )
    sample_rate = int(extras.get("audio_sample_rate", 16000))
    video_rate = int(extras.get("audio_video_rate", 25))
    needs_camera = "grid" in streams

    def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch = base_collate(samples)
        video_frames = int(batch["video_frames"].shape[1])
        if needs_camera and batch.get("grid_condition") is None:
            has_camera = any(
                sample.get("c2ws") is not None or sample.get("Ks") is not None for sample in samples
            )
            if has_camera:
                batch["c2ws"] = _stack_camera_field(
                    samples,
                    key="c2ws",
                    frame_count=video_frames,
                    matrix_shape=(4, 4),
                )
                batch["Ks"] = _stack_camera_field(
                    samples,
                    key="Ks",
                    frame_count=video_frames,
                    matrix_shape=(3, 3),
                )
        if use_audio:
            batch["audio"] = _collate_audio(
                samples,
                target_samples=int(max(video_frames - 1, 0) / video_rate * sample_rate),
                sample_rate=sample_rate,
            )
        return batch

    return collate


def _build_base_raw_collate_fn(*, extras: Mapping[str, Any], pretrained_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_path,
        subfolder="tokenizer",
    )
    max_frames = extras.get("max_frames")
    max_frames_i = int(max_frames) if max_frames is not None else None
    max_sequence_length = int(extras.get("max_sequence_length", 512))

    def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_raw_samples(
            samples,
            tokenizer=tokenizer,
            max_frames=max_frames_i,
            max_sequence_length=max_sequence_length,
        )

    return collate


def collate_raw_samples(
    samples: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_frames: int | None = None,
    max_sequence_length: int = 512,
) -> dict[str, Any]:
    """Collate decoded or path-based raw samples for online encoding."""
    if not samples:
        raise ValueError("cannot collate an empty Wan Taichu raw batch")
    deferred = _deferred_markers(samples)
    videos = [_sample_clip(sample, *_RGB_KEYS, max_frames=max_frames) for sample in samples]
    if any(video is None for video in videos):
        raise ValueError("every Wan Taichu training sample needs 'video_frames' or 'video'")
    resolved_videos = [video for video in videos if video is not None]
    temporal = min(int(video.shape[0]) for video in resolved_videos)
    resolved_videos = [video[:temporal] for video in resolved_videos]
    video_frames = torch.stack(resolved_videos, dim=0)

    if deferred is not None and deferred[NORMAL_MISSING_KEY]:
        normal_condition = torch.empty((len(samples), 0), dtype=torch.uint8)
    else:
        normal_condition = _stack_condition_videos(
            samples,
            resolved_videos,
            keys=_NORMAL_KEYS,
            fill_missing=deferred is None,
        )
        if normal_condition is None:
            normal_condition = torch.zeros_like(video_frames)
    grid_condition = _stack_condition_videos(
        samples,
        resolved_videos,
        keys=_GRID_KEYS,
        fill_missing=False,
    )

    prompts = [str(sample.get("prompt") or "") for sample in samples]
    text_inputs, text_mask = _tokenize_prompts(
        tokenizer,
        prompts,
        max_sequence_length=_batch_sequence_length(samples, max_sequence_length),
    )
    batch: dict[str, Any] = {
        "sample_id": [sample.get("sample_id") for sample in samples],
        "video_frames": video_frames,
        "normal_condition": normal_condition,
        "grid_condition": grid_condition,
        "image_condition": video_frames[:, 0:1].contiguous(),
        "text_inputs": text_inputs,
        "text_mask": text_mask,
        "prompt": prompts,
        **collate_latent_bounds(samples),
    }
    for field in _OPTIONAL_RAW_TENSOR_STREAMS:
        stacked = _stack_optional_tensors(samples, field)
        if stacked is not None:
            batch[field] = stacked
    if deferred is not None:
        batch.update(deferred)
    return batch


def _tokenize_prompts(
    tokenizer: Any,
    prompts: list[str],
    *,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return encoded.input_ids.long(), encoded.attention_mask.long()


def _sample_clip(
    sample: Mapping[str, Any],
    tensor_key: str,
    path_key: str,
    *,
    max_frames: int | None,
) -> torch.Tensor | None:
    clip = sample.get(tensor_key)
    if isinstance(clip, torch.Tensor):
        return clip if max_frames is None else clip[:max_frames]
    path = sample.get(path_key)
    if not path:
        return None
    return read_video_tensor(str(path), max_frames=max_frames)


def _declares_stream(sample: Mapping[str, Any], keys: tuple[str, str]) -> bool:
    tensor_key, path_key = keys
    return isinstance(sample.get(tensor_key), torch.Tensor) or bool(sample.get(path_key))


def _stack_condition_videos(
    samples: list[dict[str, Any]],
    videos: list[torch.Tensor],
    *,
    keys: tuple[str, str],
    fill_missing: bool,
) -> torch.Tensor | None:
    declared = [_declares_stream(sample, keys) for sample in samples]
    if not any(declared):
        return None
    if not all(declared) and not fill_missing:
        raise ValueError(f"only some samples declare a {keys[0]!r} stream")
    temporal = int(videos[0].shape[0])
    clips: list[torch.Tensor] = []
    for sample, video in zip(samples, videos, strict=True):
        clip = _sample_clip(sample, *keys, max_frames=temporal)
        if clip is None:
            clips.append(torch.zeros_like(video))
        else:
            if int(clip.shape[0]) < temporal:
                raise ValueError(
                    f"{keys[0]} shorter than rgb for sample {sample.get('sample_id')!r}"
                )
            clips.append(clip[:temporal].to(video.dtype))
    return torch.stack(clips, dim=0)


def _stack_optional_tensors(
    samples: list[dict[str, Any]],
    field: str,
) -> torch.Tensor | None:
    present = [isinstance(sample.get(field), torch.Tensor) for sample in samples]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"only some samples carry {field!r}")
    return torch.stack([sample[field] for sample in samples], dim=0)


def _batch_sequence_length(samples: list[dict[str, Any]], fallback: int) -> int:
    values = {
        int(sample["max_sequence_length"])
        for sample in samples
        if sample.get("max_sequence_length") is not None
    }
    if not values:
        return fallback
    if len(values) != 1:
        raise ValueError(f"batch mixes max_sequence_length values {sorted(values)}")
    return values.pop()


def _deferred_markers(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    flags = [bool(sample.get(DEFERRED_SPATIAL_KEY)) for sample in samples]
    if not any(flags):
        return None
    if not all(flags):
        raise ValueError("a batch mixes deferred and already-processed samples")
    markers: dict[str, Any] = {DEFERRED_SPATIAL_KEY: True}
    for field in (TARGET_HEIGHT_KEY, TARGET_WIDTH_KEY):
        values = {int(sample[field]) for sample in samples}
        if len(values) != 1:
            raise ValueError(f"deferred samples disagree on {field}: {sorted(values)}")
        markers[field] = values.pop()
    missing = {bool(sample[NORMAL_MISSING_KEY]) for sample in samples}
    if len(missing) != 1:
        raise ValueError("a deferred batch mixes samples with and without normal")
    markers[NORMAL_MISSING_KEY] = missing.pop()

    rotations = [sample.get(NORMAL_ROTATION_KEY) for sample in samples]
    if any(rotation is not None for rotation in rotations):
        identity = torch.eye(3)
        markers[NORMAL_ROTATION_KEY] = torch.stack(
            [identity if rotation is None else rotation for rotation in rotations]
        )
    masks = [sample.get(RAW_NORMAL_MASK_KEY) for sample in samples]
    if any(mask is not None for mask in masks):
        if not all(mask is not None for mask in masks):
            raise ValueError("only some deferred samples carry a normal mask")
        markers[RAW_NORMAL_MASK_KEY] = torch.stack(masks)
    return markers


def _stack_camera_field(
    samples: list[dict[str, Any]],
    *,
    key: str,
    frame_count: int,
    matrix_shape: tuple[int, int],
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for sample in samples:
        source = sample.get(key)
        if source is None:
            raise ValueError(
                "Wan Taichu grid conditioning requires either a grid video for every "
                f"sample or paired inline c2ws/Ks; sample {sample.get('sample_id')!r} "
                f"is missing {key!r}"
            )
        value = torch.as_tensor(source, dtype=torch.float32)
        if value.ndim != 3 or value.shape[1:] != matrix_shape:
            raise ValueError(
                f"Wan Taichu camera field {key!r} must be "
                f"[T,{matrix_shape[0]},{matrix_shape[1]}], got {tuple(value.shape)}"
            )
        if value.shape[0] < 1:
            raise ValueError(f"Wan Taichu camera field {key!r} is empty")
        if value.shape[0] < frame_count:
            value = torch.cat(
                [value, value[-1:].expand(frame_count - value.shape[0], *matrix_shape)],
                dim=0,
            )
        values.append(value[:frame_count].contiguous())
    return torch.stack(values, dim=0)


def _collate_audio(
    samples: list[dict[str, Any]],
    *,
    target_samples: int,
    sample_rate: int,
) -> torch.Tensor:
    waveforms = []
    for sample in samples:
        source = sample.get("audio")
        if source is None:
            source = sample.get("video")
        waveform = (
            source.detach().cpu().float()
            if isinstance(source, torch.Tensor)
            else _read_audio_waveform(str(source), sample_rate=sample_rate)
        ).flatten()
        if waveform.numel() < target_samples:
            value = waveform[-1] if waveform.numel() else waveform.new_tensor(0.0)
            waveform = torch.cat([waveform, value.expand(target_samples - waveform.numel())])
        waveforms.append(waveform[:target_samples])
    return torch.stack(waveforms, dim=0)


def _read_audio_waveform(path: str, *, sample_rate: int) -> torch.Tensor:
    import av

    chunks: list[torch.Tensor] = []
    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError(f"Wan Taichu audio sample has no audio stream: {path}")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                chunks.append(torch.from_numpy(converted.to_ndarray()).float().flatten())
        for converted in resampler.resample(None):
            chunks.append(torch.from_numpy(converted.to_ndarray()).float().flatten())
    if not chunks:
        raise ValueError(f"Wan Taichu audio decoder produced no samples: {path}")
    return torch.cat(chunks)


def read_latent_bounds(
    bundle: Mapping[str, Any],
    *,
    context: str,
) -> tuple[int | None, int | None]:
    minimum, maximum = (
        _optional_scalar_int(bundle.get(field), field=field) for field in LATENT_BOUND_FIELDS
    )
    if minimum is not None and minimum < 1:
        raise ValueError(f"{context} min_latents must be positive; got {minimum}")
    if maximum is not None and maximum < 1:
        raise ValueError(f"{context} max_latents must be positive; got {maximum}")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{context} min_latents={minimum} must be <= max_latents={maximum}")
    return minimum, maximum


def _optional_scalar_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _scalar_int(value, field=field)


def _scalar_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer, not bool")
    if isinstance(value, torch.Tensor):
        flat = value.reshape(-1)
        if flat.numel() < 1:
            raise ValueError(f"{field} must not be empty")
        values = [int(item) for item in flat.tolist()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError(f"{field} must not be empty")
        values = [int(item) for item in value]
    else:
        return int(value)
    if any(item != values[0] for item in values[1:]):
        raise ValueError(f"{field} must be homogeneous within one batch; got {values}")
    return values[0]


def _require_tensor(bundle: Mapping[str, Any], field: str, *, rank: int) -> torch.Tensor:
    value = bundle.get(field)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Wan Taichu encoded bundle requires tensor field {field!r}")
    if value.ndim != rank:
        raise ValueError(f"{field} must have rank {rank}; got shape={tuple(value.shape)}")
    return value


def _validate_latent(
    value: torch.Tensor,
    *,
    field: str,
    channels: int,
    batch_size: int,
    geometry: tuple[int, ...],
) -> None:
    if value.ndim != 5:
        raise ValueError(f"{field} must be [B,C,T,H,W]; got shape={tuple(value.shape)}")
    expected = (batch_size, channels, *geometry)
    if tuple(value.shape) != expected:
        raise ValueError(f"{field} must have shape={expected}; got {tuple(value.shape)}")


__all__ = [
    "COMMON_FIELDS",
    "build_encoded_batch_codec",
    "build_raw_collate_fn",
    "canonicalize_encoded_batch",
    "canonicalize_training_bundle",
    "collate_encoded_bundles",
    "collate_latent_bounds",
    "collate_raw_samples",
    "fallback_stream_field",
    "read_latent_bounds",
    "stream_field",
]
