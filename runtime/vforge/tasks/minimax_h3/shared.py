"""MiniMax-H3 pieces shared by train, encode, and evaluation tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from vforge.data.cache import EncodedBatchCodec
from vforge.training.checkpoint.mapping import StateDictMapper

H3_ENCODED_BATCH_KEYS = (
    "prompt_embeds",
    "condition_video_rows",
    "target_video_rows",
    "target_audio_rows",
    "audio_loss_mask",
    "token_tags",
    "position_ids",
    "video_indices",
    "audio_indices",
    "text_indices",
    "num_condition_video_rows",
    "num_condition_audio_rows",
    "num_latent_frames",
    "latent_height",
    "latent_width",
    "num_audio_latents",
)

H3_ENCODED_BATCH_CODEC = EncodedBatchCodec(
    name="minimax_h3_fl2va",
    required_keys=H3_ENCODED_BATCH_KEYS,
    schema_version=2,
    variable_shape_keys=frozenset(H3_ENCODED_BATCH_KEYS),
)


def _torch_dtype(precision: str) -> torch.dtype:
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _validate_shared(
    *,
    extras: Mapping[str, Any],
    latent_cache_mode: str,
    usp_size: int,
) -> None:
    variant = str(extras.get("variant", "fl2va")).lower()
    if variant != "fl2va":
        raise ValueError(f"MiniMax-H3 first release supports variant=fl2va, got {variant!r}")
    if latent_cache_mode not in {"online", "offline"}:
        raise ValueError(f"unsupported latent_cache.mode={latent_cache_mode!r}")
    if usp_size < 1:
        raise ValueError("parallel.usp_size must be >= 1")


def _build_h3_model(
    *,
    extras: Mapping[str, Any],
    init_device: str,
    pretrained_path: str,
) -> nn.Module:
    if not pretrained_path:
        raise ValueError("model.pretrained_path is required")
    if init_device != "meta":
        raise ValueError(
            "MiniMax-H3 requires model.init_device=meta so FSDP2 shards before "
            "allocating the 33B transformer"
        )
    from vforge.models.minimax_h3.native.model import MiniMaxH3Model

    config_overrides = {
        key.removeprefix("h3_"): value for key, value in extras.items() if key.startswith("h3_")
    }
    return MiniMaxH3Model.build_empty_from_pretrained(
        pretrained_path,
        parameter_dtype=torch.float32,
        **config_overrides,
    )


def _h3_state_dict_mapper() -> StateDictMapper:
    return StateDictMapper.strip_prefix("transformer.")


def _collate_encoded_bundles(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {}
    canonical = [H3_ENCODED_BATCH_CODEC.canonicalize(sample) for sample in samples]
    structural = {
        "token_tags",
        "position_ids",
        "video_indices",
        "audio_indices",
        "text_indices",
        "num_condition_video_rows",
        "num_condition_audio_rows",
        "num_latent_frames",
        "latent_height",
        "latent_width",
        "num_audio_latents",
    }
    batch: dict[str, Any] = {}
    for key in H3_ENCODED_BATCH_KEYS:
        values = [sample[key] for sample in canonical]
        if key in structural:
            first = values[0]
            for value in values[1:]:
                if isinstance(first, torch.Tensor):
                    if not torch.equal(first, value):
                        raise ValueError(
                            f"H3 batch requires one shared packed layout; {key} differs"
                        )
                elif first != value:
                    raise ValueError(f"H3 batch requires one shared packed layout; {key} differs")
            batch[key] = first
            continue
        tensors = []
        for value in values:
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"H3 encoded field {key!r} must be a tensor")
            tensors.append(value.unsqueeze(0) if value.ndim == 2 else value)
        batch[key] = torch.cat(tensors, dim=0)
    batch["sample_id"] = [str(sample["sample_id"]) for sample in samples]
    return batch


__all__ = [
    "H3_ENCODED_BATCH_CODEC",
    "H3_ENCODED_BATCH_KEYS",
]
