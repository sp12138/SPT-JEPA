"""Wan 2.2 TI2V pieces every verb needs: mode, codec, encoder, model."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch
import torch.nn as nn

from vforge.data.cache import EncodedBatchCodec
from vforge.models.wan.model import WanChannelLayout
from vforge.tasks.wan.shared import (
    WanConditionSpec,
    model_config_overrides,
    parameter_storage_dtype,
    parse_condition_streams,
    read_condition_streams,
    validate_shared,
)
from vforge.utils.training import resolve_dtype

logger = logging.getLogger(__name__)

#: Every stream this version knows how to encode, in the order a config that
#: asks for all of them would concatenate them.
ALL_STREAMS: tuple[str, ...] = ("rgb", "grid", "normal")
DEFAULT_CONDITION_STREAMS: tuple[str, ...] = ("normal",)

# Raw batch key each non-rgb stream is encoded from.  ``rgb`` derives from the
# target video's reference frame.
STREAM_PIXEL_KEYS = MappingProxyType(
    {
        "normal": "normal_condition",
        "grid": "grid_condition",
    }
)

PARTIAL_NORMAL_SOURCES: tuple[str, ...] = ("normal_fg_body", "normal_bg_body")


def condition_streams(value: Any) -> tuple[str, ...]:
    return parse_condition_streams(value, known=ALL_STREAMS)


def condition_spec(streams: Any) -> WanConditionSpec:
    """Resolve a configured stream list into the model-facing TI2V contract."""
    return WanConditionSpec(streams=condition_streams(streams), mask_channels=0)


def stream_field(name: str) -> str:
    return f"condition_latents_{name}"


def stream_fields(streams: Any) -> dict[str, str]:
    """Encoded field backing each stream, in declared order.

    This is what the encoder and the sampler receive: model code never learns
    the naming rule, only which field to read for which stream.
    """
    return {name: stream_field(name) for name in condition_streams(streams)}


def stream_drop_prob_key(name: str) -> str:
    return f"{name}_drop_prob"


def encoded_batch_keys(
    streams: Any = DEFAULT_CONDITION_STREAMS,
    *,
    partial_normal_aug: bool = False,
) -> tuple[str, ...]:
    resolved = condition_streams(streams)
    if partial_normal_aug and "normal" not in resolved:
        raise ValueError(
            "partial_normal_aug augments the 'normal' stream, but "
            f"condition_streams={list(resolved)} does not carry it"
        )
    keys = ["prompt", "frame_latents", *(stream_field(name) for name in resolved)]
    if partial_normal_aug:
        keys.extend(PARTIAL_NORMAL_SOURCES)
    keys.append("first_frame_latents")
    return tuple(keys)


# Cached bundles carry latents from a z_dim=48 VAE, so a Wan2.1 cache is not a
# valid input here even though both use the same bundle file format.
CACHE_FLAVOR = "wan22-ti2v-normal-inpaint"

# Rank of each field once batched. Caches are written per sample and carry
# ``rank - 1``; the declared rank is the only way to tell a missing batch dim
# from a genuinely malformed tensor. Every latent block is [B, C, T, H, W].
_PROMPT_RANK = 3
_LATENT_RANK = 5


def _condition_streams(extras: Mapping[str, Any]) -> tuple[str, ...]:
    return read_condition_streams(
        extras,
        known=ALL_STREAMS,
        default=DEFAULT_CONDITION_STREAMS,
    )


def _condition_spec(extras: Mapping[str, Any]) -> WanConditionSpec:
    return WanConditionSpec(streams=_condition_streams(extras), mask_channels=0)


def _stream_fields(extras: Mapping[str, Any]) -> dict[str, str]:
    return {name: stream_field(name) for name in _condition_streams(extras)}


def _resolve_model_config(
    extras: Mapping[str, Any],
    pretrained_path: str,
) -> tuple[dict[str, Any], WanChannelLayout]:
    """Read the published config once and widen it for the configured streams.

    Every verb goes through here exactly once per checkpoint: the transformer
    config, the latent width the stream arithmetic needs, and the layout the
    weight-loading hooks need all come out of the same read.
    """
    from vforge.models.wan.model import published_z_dim
    from vforge.models.wan.v22.ti2v_5b.ct.model import (
        read_published_config,
        widen_transformer_config,
    )

    if not pretrained_path:
        raise ValueError("model.pretrained_path is required")
    published = read_published_config(pretrained_path)
    in_channels = _condition_spec(extras).input_channels(z_dim=published_z_dim(published))
    return widen_transformer_config(
        published,
        in_channels=in_channels,
        **model_config_overrides(extras),
    )


def _partial_normal_aug(extras: Mapping[str, Any]) -> bool:
    return bool(extras.get("partial_normal_aug", False))


def _build_codec(keys: tuple[str, ...]) -> EncodedBatchCodec:
    """A codec over exactly ``keys``, all of them required.

    Unlike Wan2.1 nothing here is optional: a TI2V bundle either carries the
    reference frame and every configured stream or it is the wrong cache.
    Legacy manifests are accepted because the first wan22 caches predate the
    codec identity field.
    """
    return EncodedBatchCodec(
        name="wan22_ti2v_5b",
        required_keys=keys,
        accept_legacy_manifest=True,
    )


def _codec(extras: Mapping[str, Any]) -> EncodedBatchCodec:
    """What the training step reads out of a bundle."""
    return _build_codec(
        encoded_batch_keys(
            _condition_streams(extras),
            partial_normal_aug=_partial_normal_aug(extras),
        )
    )


def _encoder_codec(extras: Mapping[str, Any]) -> EncodedBatchCodec:
    """What this encoder can produce, which encode and eval are bounded by.

    The partial-normal sources are foreground/background normal clips that only
    the Taichu dataset supplies, so they stay out of the set even when training
    is configured to consume them; otherwise encode would write them as ``None``
    and training would fail on read instead of at config time.
    """
    return _build_codec(encoded_batch_keys(_condition_streams(extras)))


def _collate_encoded_bundles(
    codec: EncodedBatchCodec,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Batch cached bundles, promoting the per-sample rank the cache stores."""
    batch: dict[str, Any] = {}
    for key in codec.keys:
        values = [sample.get(key) for sample in samples]
        expected_rank = _PROMPT_RANK if key == "prompt" else _LATENT_RANK
        if all(value is None for value in values):
            batch[key] = None
        elif all(isinstance(value, torch.Tensor) for value in values):
            normalized = []
            for value in values:
                if value.ndim == expected_rank - 1:
                    value = value.unsqueeze(0)
                if value.ndim != expected_rank:
                    raise ValueError(
                        f"encoded cache field {key!r} has rank {value.ndim}; "
                        f"expected {expected_rank - 1} (legacy) or {expected_rank}"
                    )
                normalized.append(value)
            batch[key] = torch.cat(normalized, dim=0)
        else:
            raise ValueError(f"encoded cache field {key!r} mixes tensors and None values")
    batch["sample_id"] = [str(sample["sample_id"]) for sample in samples]
    for key in ("min_latents", "max_latents"):
        values = [sample.get(key) for sample in samples]
        present = [value for value in values if value is not None]
        if not present:
            continue
        as_ints = [int(value) for value in present]
        if len(present) != len(values) or len(set(as_ints)) != 1:
            raise ValueError(
                f"encoded cache field {key!r} is not homogeneous in the batch; "
                "MixedSampler must keep every batch on one source"
            )
        batch[key] = as_ints[0]
    return batch


def _stream_drop_probs(extras: Mapping[str, Any]) -> dict[str, float]:
    """Per-stream CFG dropout probability, keyed ``<stream>_drop_prob``."""
    return {
        name: float(extras.get(stream_drop_prob_key(name)) or 0.0)
        for name in _condition_streams(extras)
    }


def _build_condition_encoder(
    *,
    extras: Mapping[str, Any],
    pretrained_path: str,
    precision: str,
    require_assets: bool,
) -> Any:
    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor
    from transformers import AutoTokenizer, UMT5EncoderModel

    from vforge.models.wan.v22.ti2v_5b.ct.encoder import Wan22Ti2v5bConditionEncoder

    if require_assets and not pretrained_path:
        raise ValueError("model.pretrained_path is required for online encode/prepare")
    vae_path = str(extras.get("vae_pretrained_path") or pretrained_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).eval()
    text_encoder = UMT5EncoderModel.from_pretrained(
        pretrained_path,
        subfolder="text_encoder",
    ).eval()

    from vforge.distributed.runtime import get_runtime

    runtime = get_runtime()
    if runtime is not None:
        vae.to(runtime.device)
        text_encoder.to(runtime.device)
    video_processor = VideoProcessor(vae_scale_factor=2 ** len(vae.temperal_downsample))
    return Wan22Ti2v5bConditionEncoder(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        video_processor=video_processor,
        training_dtype=resolve_dtype(precision),
        stream_fields=_stream_fields(extras),
        raw_stream_keys=STREAM_PIXEL_KEYS,
        stream_drop_probs=_stream_drop_probs(extras),
        text_drop_prob=float(extras.get("text_drop_prob", 0.0)),
    )


def _require_meta_init(init_device: str) -> None:
    if init_device != "meta":
        raise ValueError(
            "standalone Wan22 requires model.init_device=meta so FSDP2 can "
            "shard before allocating the full transformer"
        )


def _build_wan22_model(
    *,
    config: Mapping[str, Any],
    init_device: str,
) -> nn.Module:
    from vforge.models.wan.v22.ti2v_5b.ct.model import Wan22Ti2v5bModel

    _require_meta_init(init_device)
    return Wan22Ti2v5bModel.build_empty(
        config,
        parameter_dtype=parameter_storage_dtype(),
    )


def _validate(
    *,
    extras: Mapping[str, Any],
    latent_cache_mode: str,
    usp_size: int,
) -> None:
    validate_shared(latent_cache_mode=latent_cache_mode, usp_size=usp_size)
    streams = _condition_streams(extras)
    # A drop probability on an absent stream is a config left over from another
    # stream list, and silently ignoring it would mean training without the CFG
    # dropout the config asked for.
    stray = [
        stream_drop_prob_key(name)
        for name in ALL_STREAMS
        if name not in streams and float(extras.get(stream_drop_prob_key(name)) or 0.0) > 0.0
    ]
    if stray:
        raise ValueError(
            f"condition_streams={list(streams)} does not carry {stray}"
            "; drop the key or add the stream"
        )
    if _partial_normal_aug(extras):
        # The alternate bodies come from foreground/background normal clips the
        # raw batch has no field for, so only a cache written with them can
        # serve the augmentation.
        if latent_cache_mode != "offline":
            raise ValueError(
                "partial_normal_aug reads normal_fg_body / normal_bg_body from "
                "an externally written cache, so it needs "
                "latent_cache.mode=offline; this encoder has no "
                "foreground/background normal source to produce them from"
            )
        if "normal" not in streams:
            raise ValueError(
                "partial_normal_aug augments the 'normal' stream, but "
                f"condition_streams={list(streams)} does not carry it"
            )
