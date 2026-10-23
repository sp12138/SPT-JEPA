"""Wan raw-media collation: sample paths or clips in, encoder-ready batch out.

Every function here reads the Wan condition contract -- an RGB clip, optional
``normal`` and ``grid`` clips aligned to it, a first-frame image condition, and
a tokenized prompt -- which is why none of it belongs in the content-agnostic
root of ``vforge/data``. The decoders themselves are shared and are imported
from :mod:`vforge.data.video_io`.

Samples arrive in one of two shapes. ``RawSampleDataset`` yields paths and this
module decodes them; the taichu datasets in this package decode inside
``__getitem__`` and yield clips. Everything after the decode -- temporal
alignment, the image condition, tokenization, batch assembly -- is the same
either way, so both go through one collate rather than two that must be kept
in step.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from vforge.data.preprocess import (
    DEFERRED_SPATIAL_KEY,
    NORMAL_MISSING_KEY,
    NORMAL_ROTATION_KEY,
    RAW_NORMAL_MASK_KEY,
    TARGET_HEIGHT_KEY,
    TARGET_WIDTH_KEY,
)
from vforge.data.video_io import read_image_tensor, read_reference_frames, read_video_tensor

# Streams a sample may deliver either way: the decoded key it uses when the
# dataset already has pixels, and the path key it uses when it does not.
_RGB_KEYS = ("video_frames", "video")
_NORMAL_KEYS = ("normal_condition", "normal")
_GRID_KEYS = ("grid_condition", "grid")

# Per-sample tensors the encoder reads verbatim. Absent from every sample or
# present on all of them -- a partial column means half the batch would be
# conditioned on nothing, which is not something to paper over.
_OPTIONAL_TENSOR_STREAMS = (
    "Ks",
    "c2ws",
    "reference_frames",
    "normal_fg_condition",
    "normal_bg_condition",
)


def tokenize_prompts(
    tokenizer: Any,
    prompts: list[str],
    *,
    max_sequence_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(text_inputs, text_mask)`` matching Wan encoder expectations.

    The configured length wins outright. Letting the tokenizer's declared
    ``model_max_length`` override it, as this used to, means the text embedding
    width is a property of the tokenizer files rather than of the run config --
    so a checkpoint whose tokenizer reports a different limit would silently
    produce embeddings that do not line up with an existing latent cache.
    """
    encoded = tokenizer(
        list(prompts),
        padding="max_length",
        max_length=int(max_sequence_length),
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return encoded.input_ids.long(), encoded.attention_mask.long()


def _declares_stream(sample: dict[str, Any], tensor_key: str, path_key: str) -> bool:
    return isinstance(sample.get(tensor_key), torch.Tensor) or bool(sample.get(path_key))


def _sample_clip(
    sample: dict[str, Any],
    tensor_key: str,
    path_key: str,
    *,
    max_frames: int | None,
) -> torch.Tensor | None:
    """Return one stream's clip, decoding the path only when there is no tensor."""
    clip = sample.get(tensor_key)
    if isinstance(clip, torch.Tensor):
        return clip if max_frames is None else clip[:max_frames]
    path = sample.get(path_key)
    if not path:
        return None
    return read_video_tensor(str(path), max_frames=max_frames)


def _stack_condition_videos(
    samples: list[dict[str, Any]],
    videos: list[torch.Tensor],
    *,
    keys: tuple[str, str],
    fill_missing: bool,
) -> torch.Tensor | None:
    """Collect one aligned condition clip per sample, or ``None`` if unused.

    A condition declared by some samples but not others would silently train
    part of the batch against a substitute, so a partial column is an error
    unless the caller accepts a blank fill.
    """
    tensor_key, path_key = keys
    t_min = videos[0].shape[0]
    declared = [_declares_stream(sample, tensor_key, path_key) for sample in samples]
    if not any(declared):
        return None
    if not all(declared) and not fill_missing:
        raise ValueError(f"only some samples in the batch declare a {tensor_key!r} stream")

    clips: list[torch.Tensor] = []
    for sample, video in zip(samples, videos, strict=True):
        clip = _sample_clip(sample, tensor_key, path_key, max_frames=t_min)
        if clip is None:
            clips.append(torch.zeros_like(video))
            continue
        if clip.shape[0] < t_min:
            raise ValueError(f"{tensor_key} shorter than rgb for sample {sample.get('sample_id')}")
        clips.append(clip[:t_min].to(video.dtype))
    return torch.stack(clips, dim=0)


def _stack_optional_tensors(samples: list[dict[str, Any]], key: str) -> torch.Tensor | None:
    present = [isinstance(sample.get(key), torch.Tensor) for sample in samples]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"only some samples in the batch carry {key!r}")
    return torch.stack([sample[key] for sample in samples], dim=0)


def _deferred_markers(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Fold the deferred-spatial transport keys into batch-wide ones.

    Returns ``None`` for an ordinary batch. The scalar markers describe a
    decision the deferred stage makes once for the whole batch, so disagreement
    within one is an error rather than something to reconcile here.
    """
    flags = [bool(sample.get(DEFERRED_SPATIAL_KEY)) for sample in samples]
    if not any(flags):
        return None
    if not all(flags):
        raise ValueError(
            "a batch mixes deferred and already-processed samples; their pixels "
            "have different dtypes and resolutions and cannot be stacked"
        )

    markers: dict[str, Any] = {DEFERRED_SPATIAL_KEY: True}
    for key, label in ((TARGET_HEIGHT_KEY, "height"), (TARGET_WIDTH_KEY, "width")):
        values = {int(sample[key]) for sample in samples}
        if len(values) != 1:
            raise ValueError(
                f"deferred samples disagree on target {label} ({sorted(values)}); "
                "one batch is resized to one geometry"
            )
        markers[key] = values.pop()

    missing = {bool(sample[NORMAL_MISSING_KEY]) for sample in samples}
    if len(missing) != 1:
        raise ValueError(
            "a deferred batch mixes samples with and without a normal stream, "
            "but the deferred stage resolves that once per batch; give the "
            "source a normal for every entry or for none of them"
        )
    markers[NORMAL_MISSING_KEY] = missing.pop()

    rotations = [sample.get(NORMAL_ROTATION_KEY) for sample in samples]
    if any(rotation is not None for rotation in rotations):
        # A per-sample coin under normal_rot_aug_prob < 1 leaves some samples
        # unrotated. Identity stands in for them: einsum against it reproduces
        # the input exactly and its bias term is zero, so a filled row is
        # indistinguishable from having skipped the rotation.
        identity = torch.eye(3)
        markers[NORMAL_ROTATION_KEY] = torch.stack(
            [identity if rotation is None else rotation for rotation in rotations],
            dim=0,
        )

    masks = [sample.get(RAW_NORMAL_MASK_KEY) for sample in samples]
    if any(mask is not None for mask in masks):
        if not all(mask is not None for mask in masks):
            raise ValueError(
                "only some deferred samples carry a normal mask; a partial "
                "column would split half the batch against a substitute"
            )
        markers[RAW_NORMAL_MASK_KEY] = torch.stack(masks, dim=0)
    return markers


def _batch_sequence_length(samples: list[dict[str, Any]], fallback: int) -> int:
    """Resolve the tokenizer width, preferring what the datasets stamped on.

    A mixed run has one collate closure but several sources, so the closure's
    configured value cannot speak for all of them. Disagreement within a batch
    is fatal rather than resolved by a max: padding one source's captions wider
    than it was cached at is the bug this is meant to catch.
    """
    stamped = {
        int(sample["max_sequence_length"])
        for sample in samples
        if sample.get("max_sequence_length") is not None
    }
    if not stamped:
        return int(fallback)
    if len(stamped) > 1:
        raise ValueError(
            f"batch mixes max_sequence_length values {sorted(stamped)}; "
            "sources sharing a loader must agree on the text width"
        )
    return stamped.pop()


def collate_raw_wan_batch(
    samples: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_frames: int | None = None,
    max_sequence_length: int = 512,
) -> dict[str, Any]:
    """Collate raw samples -- decoded or path-based -- into an encoder batch."""
    deferred = _deferred_markers(samples)
    videos = [_sample_clip(sample, *_RGB_KEYS, max_frames=max_frames) for sample in samples]
    if any(video is None for video in videos):
        raise ValueError("every Wan training sample needs 'video_frames' or 'video'")
    # Align temporal length to the shortest clip in the micro-batch. Every
    # source emits 4n+1 frames and the minimum of 4n+1 values is itself 4n+1,
    # so this cannot land on a length the causal VAE would reject.
    t_min = min(video.shape[0] for video in videos)
    videos = [video[:t_min] for video in videos]
    video_frames = torch.stack(videos, dim=0)

    if deferred is not None and deferred[NORMAL_MISSING_KEY]:
        # The deferred stage builds the neutral normal itself, at full precision
        # from the marker. Carrying a zeros clip instead would move a second
        # full-size uint8 payload that says nothing.
        normal_condition = torch.empty((len(samples), 0), dtype=torch.uint8)
    else:
        normal_condition = _stack_condition_videos(
            samples,
            videos,
            keys=_NORMAL_KEYS,
            # A deferred sample that reaches here declares a normal, so a blank
            # fill would only mask a source that disagrees with its own marker.
            fill_missing=deferred is None,
        )
        if normal_condition is None:
            normal_condition = torch.zeros_like(video_frames)
    grid_condition = _stack_condition_videos(
        samples,
        videos,
        keys=_GRID_KEYS,
        fill_missing=False,
    )
    image_condition = video_frames[:, 0:1].contiguous()

    prompts = [str(sample.get("prompt") or "") for sample in samples]
    text_inputs, text_mask = tokenize_prompts(
        tokenizer,
        prompts,
        max_sequence_length=_batch_sequence_length(samples, max_sequence_length),
    )
    batch: dict[str, Any] = {
        "sample_id": [sample.get("sample_id") for sample in samples],
        "video_frames": video_frames,
        "normal_condition": normal_condition,
        "grid_condition": grid_condition,
        "image_condition": image_condition,
        "text_inputs": text_inputs,
        "text_mask": text_mask,
        "prompt": prompts,
    }
    for key in _OPTIONAL_TENSOR_STREAMS:
        stacked = _stack_optional_tensors(samples, key)
        if stacked is not None:
            batch[key] = stacked
    for key in ("min_latents", "max_latents"):
        value = samples[0].get(key)
        if value is not None:
            batch[key] = int(value)
    if any(sample.get("audio") is not None for sample in samples):
        # The encoder runs waveforms through its processor one at a time
        # because they have different lengths, so this must stay a list.
        batch["audio"] = [sample.get("audio") for sample in samples]
    if deferred is not None:
        batch.update(deferred)
    return batch


def collate_raw_wan_inference_batch(
    samples: list[dict[str, Any]],
    *,
    tokenizer: Any,
    max_frames: int | None = None,
    max_sequence_length: int = 512,
    default_num_frames: int | None = None,
    required_streams: tuple[str, ...] = (),
    temporal_scale_factor: int = 4,
    allow_target_video: bool = False,
) -> dict[str, Any]:
    """Collate condition-only Wan requests without a target RGB video."""
    if not samples:
        raise ValueError("Wan inference collation requires at least one sample")

    images: list[torch.Tensor] = []
    normals: list[torch.Tensor] = []
    normal_foregrounds: list[torch.Tensor | None] = []
    normal_backgrounds: list[torch.Tensor | None] = []
    grids: list[torch.Tensor | None] = []
    references: list[torch.Tensor | None] = []
    geometries: list[tuple[int, int, int]] = []
    for sample in samples:
        if sample.get("video") and not allow_target_video:
            raise ValueError(
                "evaluation.data_source=inference is condition-only and must not contain 'video'"
            )
        for stream in required_streams:
            if stream != "rgb" and not sample.get(stream):
                raise ValueError(f"Wan inference requires a {stream!r} condition video")
        target = (
            read_video_tensor(str(sample["video"]), max_frames=max_frames)
            if sample.get("video")
            else None
        )
        normal = (
            read_video_tensor(str(sample["normal"]), max_frames=max_frames)
            if sample.get("normal")
            else None
        )
        normal_fg = (
            read_video_tensor(str(sample["normal_fg"]), max_frames=max_frames)
            if sample.get("normal_fg")
            else None
        )
        normal_bg = (
            read_video_tensor(str(sample["normal_bg"]), max_frames=max_frames)
            if sample.get("normal_bg")
            else None
        )
        grid = (
            read_video_tensor(str(sample["grid"]), max_frames=max_frames)
            if sample.get("grid")
            else None
        )
        temporal = [
            clip for clip in (target, normal, normal_fg, normal_bg, grid) if clip is not None
        ]
        if temporal:
            shape = tuple(temporal[0].shape)
            if any(tuple(clip.shape) != shape for clip in temporal[1:]):
                raise ValueError(
                    f"Wan condition videos have mismatched geometry for "
                    f"{sample.get('sample_id')}: {[tuple(clip.shape) for clip in temporal]}"
                )
            num_frames, _, height, width = shape
            declared_geometry = {
                "num_frames": num_frames,
                "height": height,
                "width": width,
            }
            for field, actual in declared_geometry.items():
                declared = sample.get(field)
                if declared is not None and int(declared) != actual:
                    raise ValueError(
                        f"Wan {field}={declared} does not match condition geometry "
                        f"{actual} for {sample.get('sample_id')}"
                    )
        else:
            requested = sample.get("num_frames", default_num_frames)
            if requested is None:
                raise ValueError(
                    "Wan inference requests without a temporal condition require num_frames"
                )
            num_frames = int(requested)
            height = int(sample.get("height") or 0)
            width = int(sample.get("width") or 0)
        if num_frames < 1 or (num_frames - 1) % temporal_scale_factor:
            raise ValueError(
                "Wan num_frames must satisfy "
                f"(num_frames - 1) % {temporal_scale_factor} == 0, got {num_frames}"
            )

        if sample.get("image"):
            image = read_image_tensor(str(sample["image"]))
        elif target is not None:
            image = target[0]
        else:
            raise ValueError("Wan inference requests require 'image'")
        if height <= 0 or width <= 0:
            height, width = int(image.shape[-2]), int(image.shape[-1])
        if tuple(image.shape[-2:]) != (height, width):
            raise ValueError(
                f"Wan image geometry {tuple(image.shape[-2:])} does not match "
                f"target {(height, width)} for {sample.get('sample_id')}"
            )
        reference = read_reference_frames(sample)
        if reference is not None and tuple(reference.shape[-2:]) != (height, width):
            raise ValueError(
                f"Wan reference geometry {tuple(reference.shape[-2:])} does not match "
                f"target {(height, width)} for {sample.get('sample_id')}"
            )
        if normal is None:
            normal = image.new_zeros((num_frames, *image.shape))
        images.append(image)
        normals.append(normal)
        normal_foregrounds.append(normal_fg)
        normal_backgrounds.append(normal_bg)
        grids.append(grid)
        references.append(reference)
        geometries.append((num_frames, height, width))

    if len(set(geometries)) != 1:
        raise ValueError(f"Wan inference batch requires shared geometry, got {geometries}")
    if any(grid is None for grid in grids) and not all(grid is None for grid in grids):
        raise ValueError("only some Wan inference samples declare a 'grid' video")
    if any(reference is None for reference in references) and not all(
        reference is None for reference in references
    ):
        raise ValueError("only some Wan inference samples declare a 'ref'/'reference'")
    if references[0] is not None:
        counts = {int(reference.shape[0]) for reference in references if reference is not None}
        if len(counts) != 1:
            raise ValueError(
                f"Wan samples in one batch must declare the same ref count, got {counts}"
            )
    for name, clips in (
        ("normal_fg", normal_foregrounds),
        ("normal_bg", normal_backgrounds),
    ):
        if any(clip is None for clip in clips) and not all(clip is None for clip in clips):
            raise ValueError(f"only some Wan inference samples declare a {name!r} video")
    prompts = [str(sample.get("prompt") or sample.get("text") or "") for sample in samples]
    text_inputs, text_mask = tokenize_prompts(
        tokenizer,
        prompts,
        max_sequence_length=max_sequence_length,
    )
    num_frames, height, width = geometries[0]
    return {
        "sample_id": [sample.get("sample_id") for sample in samples],
        "image_condition": torch.stack(images, dim=0).unsqueeze(1),
        "reference_condition": (
            torch.stack([reference for reference in references if reference is not None], dim=0)
            if references[0] is not None
            else None
        ),
        "normal_condition": torch.stack(normals, dim=0),
        "normal_fg_condition": (
            torch.stack([clip for clip in normal_foregrounds if clip is not None], dim=0)
            if normal_foregrounds[0] is not None
            else None
        ),
        "normal_bg_condition": (
            torch.stack([clip for clip in normal_backgrounds if clip is not None], dim=0)
            if normal_backgrounds[0] is not None
            else None
        ),
        "grid_condition": (
            torch.stack([grid for grid in grids if grid is not None], dim=0)
            if grids[0] is not None
            else None
        ),
        "text_inputs": text_inputs,
        "text_mask": text_mask,
        "prompt": prompts,
        "num_frames": num_frames,
        "height": height,
        "width": width,
    }


def build_raw_collate_fn(*, extras: Mapping[str, Any], pretrained_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_path,
        subfolder="tokenizer",
    )
    max_frames = extras.get("max_frames")
    max_frames_i = int(max_frames) if max_frames is not None else None
    max_sequence_length = int(extras.get("max_sequence_length", 512))

    def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_raw_wan_batch(
            samples,
            tokenizer=tokenizer,
            max_frames=max_frames_i,
            max_sequence_length=max_sequence_length,
        )

    return _collate


def build_inference_collate_fn(
    *,
    extras: Mapping[str, Any],
    pretrained_path: str,
    allow_target_video: bool,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_path,
        subfolder="tokenizer",
    )
    max_frames = extras.get("max_frames")
    max_frames_i = int(max_frames) if max_frames is not None else None
    configured_frames = extras.get("num_frames")
    default_num_frames = int(configured_frames) if configured_frames is not None else None
    max_sequence_length = int(extras.get("max_sequence_length", 512))
    configured_streams = extras.get("condition_streams", ("normal",))
    required_streams = (
        (str(configured_streams),)
        if isinstance(configured_streams, str)
        else tuple(str(stream) for stream in configured_streams)
    )
    temporal_scale_factor = int(extras.get("temporal_scale_factor", 4))

    def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_raw_wan_inference_batch(
            samples,
            tokenizer=tokenizer,
            max_frames=max_frames_i,
            max_sequence_length=max_sequence_length,
            default_num_frames=default_num_frames,
            required_streams=required_streams,
            temporal_scale_factor=temporal_scale_factor,
            allow_target_video=allow_target_video,
        )

    return _collate


__all__ = [
    "build_inference_collate_fn",
    "build_raw_collate_fn",
    "collate_raw_wan_batch",
    "collate_raw_wan_inference_batch",
    "tokenize_prompts",
]
