"""Wan 2.1 pieces every verb needs: model, condition encoder, collation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch
import torch.nn as nn

from vforge.data.cache import EncodedBatchCodec
from vforge.tasks.wan.shared import (
    WanConditionSpec,
    model_config_overrides,
    parameter_storage_dtype,
    parse_condition_streams,
    read_condition_streams,
)
from vforge.utils.training import resolve_dtype

logger = logging.getLogger(__name__)

ALL_STREAMS: tuple[str, ...] = ("rgb", "normal")
DEFAULT_CONDITION_STREAMS: tuple[str, ...] = ("normal",)
RAW_STREAM_KEYS = MappingProxyType({"normal": "normal_condition"})


def condition_streams(value: Any) -> tuple[str, ...]:
    return parse_condition_streams(value, known=ALL_STREAMS)


def condition_spec(streams: Any) -> WanConditionSpec:
    return WanConditionSpec(streams=condition_streams(streams), mask_channels=4)


def _condition_streams(extras: Mapping[str, Any]) -> tuple[str, ...]:
    return read_condition_streams(
        extras,
        known=ALL_STREAMS,
        default=DEFAULT_CONDITION_STREAMS,
    )


def _condition_spec(extras: Mapping[str, Any]) -> WanConditionSpec:
    return WanConditionSpec(streams=_condition_streams(extras), mask_channels=4)


ENCODED_BATCH_KEYS = (
    "prompt",
    "frame_latents",
    "image_condition_latent_vae",
    "image_condition_latent_clip",
    "image_condition_latent_dino",
    "condition_latents_normal",
)
OPTIONAL_ENCODED_BATCH_KEYS = frozenset(
    {
        "image_condition_latent_clip",
        "image_condition_latent_dino",
    }
)
WAN21_ENCODED_BATCH_CODEC = EncodedBatchCodec(
    name="wan21",
    required_keys=tuple(
        key for key in ENCODED_BATCH_KEYS if key not in OPTIONAL_ENCODED_BATCH_KEYS
    ),
    optional_keys=tuple(key for key in ENCODED_BATCH_KEYS if key in OPTIONAL_ENCODED_BATCH_KEYS),
    accept_legacy_manifest=True,
)


def project_to_encoded_batch(bundle: dict[str, Any]) -> dict[str, Any]:
    """Project an encoder bundle onto the canonical Wan21 training schema."""
    return WAN21_ENCODED_BATCH_CODEC.canonicalize(bundle)


def _collate_encoded_bundles(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    expected_ranks = {
        "prompt": 3,
        "frame_latents": 5,
        "image_condition_latent_vae": 5,
        "image_condition_latent_clip": 3,
        "image_condition_latent_dino": 3,
        "condition_latents_normal": 5,
    }
    for key in ENCODED_BATCH_KEYS:
        values = [sample.get(key) for sample in samples]
        if all(value is None for value in values):
            batch[key] = None
        elif all(isinstance(value, torch.Tensor) for value in values):
            normalized = []
            expected_rank = expected_ranks[key]
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
    return batch


def _build_condition_encoder(
    *,
    extras: Mapping[str, Any],
    pretrained_path: str,
    precision: str,
    require_assets: bool,
    condition: WanConditionSpec | None = None,
    raw_stream_keys: Mapping[str, str] = RAW_STREAM_KEYS,
) -> Any:
    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        AutoTokenizer,
        CLIPImageProcessor,
        CLIPVisionModel,
        UMT5EncoderModel,
    )

    from vforge.models.wan.v21.encoder import Wan21ConditionEncoder

    spec = condition or _condition_spec(extras)
    pretrained = pretrained_path
    if require_assets and not pretrained:
        raise ValueError("model.pretrained_path is required for online encode/prepare")
    vae_path = str(extras.get("vae_pretrained_path") or pretrained)
    tokenizer = AutoTokenizer.from_pretrained(pretrained, subfolder="tokenizer")
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).eval()
    text_encoder = UMT5EncoderModel.from_pretrained(
        pretrained,
        subfolder="text_encoder",
    ).eval()
    use_clip = bool(extras.get("use_clip_condition", False))
    image_encoder = None
    image_processor = None
    if use_clip:
        clip_root = str(extras.get("clip_pretrained_path") or pretrained)
        image_encoder = CLIPVisionModel.from_pretrained(
            clip_root,
            subfolder="image_encoder",
            torch_dtype=torch.float32,
        ).eval()
        image_processor = CLIPImageProcessor.from_pretrained(
            clip_root,
            subfolder="image_processor",
        )
    use_dino = bool(extras.get("use_dino_condition", False))
    dino_encoder = None
    dino_processor = None
    if use_dino:
        dino_root = extras.get("dino_pretrained_path")
        if not dino_root:
            raise ValueError("use_dino_condition=true requires model.extras.dino_pretrained_path")
        dino_encoder = AutoModel.from_pretrained(
            str(dino_root),
            torch_dtype=torch.float32,
        ).eval()
        dino_processor = AutoImageProcessor.from_pretrained(str(dino_root))
    from vforge.distributed.runtime import get_runtime

    runtime = get_runtime()
    if runtime is not None:
        vae.to(runtime.device)
        text_encoder.to(runtime.device)
        if image_encoder is not None:
            image_encoder.to(runtime.device)
        if dino_encoder is not None:
            dino_encoder.to(runtime.device)
    scale = 2 ** len(vae.temperal_downsample)
    video_processor = VideoProcessor(vae_scale_factor=scale)
    return Wan21ConditionEncoder(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        video_processor=video_processor,
        training_dtype=resolve_dtype(precision),
        use_vae_condition=bool(extras.get("use_vae_condition", True)),
        use_clip_condition=use_clip,
        image_encoder=image_encoder,
        image_processor=image_processor,
        use_dino_condition=use_dino,
        dino_encoder=dino_encoder,
        dino_processor=dino_processor,
        streams=spec.streams,
        mask_channels=spec.mask_channels,
        raw_stream_keys=raw_stream_keys,
        rgb_first_frame_drop_prob=float(extras.get("rgb_first_frame_drop_prob", 0.0)),
        rgb_cond_drop_prob=float(extras.get("rgb_cond_drop_prob", 0.0)),
        normal_drop_prob=float(extras.get("normal_drop_prob", 0.0)),
        text_drop_prob=float(extras.get("text_drop_prob", 0.0)),
    )


def _build_wan21_model(
    *,
    extras: Mapping[str, Any],
    init_device: str,
    pretrained_path: str,
) -> nn.Module:
    from vforge.models.wan.v21.model import Wan21Model

    if not pretrained_path:
        raise ValueError("model.pretrained_path is required")
    if init_device != "meta":
        raise ValueError(
            "standalone Wan21 requires model.init_device=meta so FSDP2 can "
            "shard before allocating 14B parameters"
        )
    model = Wan21Model.build_empty_from_pretrained(
        pretrained_path,
        parameter_dtype=parameter_storage_dtype(),
        **model_config_overrides(extras),
    )
    return model
