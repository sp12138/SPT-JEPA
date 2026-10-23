"""Wan 2.2 A14B pieces every verb needs: codec, encoder, expert pair.

A14B consumes the same encoded bundle as Wan2.1 -- one pre-assembled
``condition_latents_normal`` carrying every stream's mask-plus-body block -- so
the bundle keys, the collation and the condition encoder all come from the 2.1
plugin. Only the cache identity, the model construction and the expert-specific
validation live here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from vforge.data.cache import EncodedBatchCodec
from vforge.models.wan.model import WanChannelLayout
from vforge.models.wan.v22.a14b.model import (
    ALL_EXPERTS,
    DEFAULT_BOUNDARY_RATIO,
    DEFAULT_NUM_TRAIN_TIMESTEPS,
    HIGH_NOISE_EXPERT,
    LOW_NOISE_EXPERT,
)
from vforge.tasks.wan.shared import (
    WanConditionSpec,
    model_config_overrides,
    parameter_storage_dtype,
    parse_condition_streams,
    read_condition_streams,
    validate_shared,
)
from vforge.tasks.wan.v21.shared import (
    ENCODED_BATCH_KEYS,
    OPTIONAL_ENCODED_BATCH_KEYS,
)
from vforge.tasks.wan.v21.shared import (
    _build_condition_encoder as _build_wan21_condition_encoder,
)

logger = logging.getLogger(__name__)

MASK_CHANNELS = 4
ALL_STREAMS: tuple[str, ...] = ("rgb", "normal")
DEFAULT_CONDITION_STREAMS: tuple[str, ...] = ("rgb",)
TRAIN_EXPERTS = ("high", "low", "both")


def condition_streams(value: Any) -> tuple[str, ...]:
    return parse_condition_streams(value, known=ALL_STREAMS)


def condition_spec(streams: Any) -> WanConditionSpec:
    """Resolve a configured stream list into A14B's masked contract."""
    return WanConditionSpec(
        streams=condition_streams(streams),
        mask_channels=MASK_CHANNELS,
    )


# Bundles hold Wan2.2 A14B latents (z_dim=16, one masked block per stream), so a
# cache prepared for a different stream list is the wrong input even though
# the field names match Wan2.1 exactly.
CACHE_FLAVOR = "wan22-a14b-normal-mask"

#: A14B has no CLIP / DINO / VAE image branch: the reference frame reaches the
#: model as the ``rgb`` condition stream instead. These extras exist on the 2.1
#: encoder and must stay off.
_IMAGE_BRANCH_KEYS = (
    "use_vae_condition",
    "use_clip_condition",
    "use_dino_condition",
)

WAN22_A14B_ENCODED_BATCH_CODEC = EncodedBatchCodec(
    name="wan22_a14b",
    required_keys=tuple(
        key for key in ENCODED_BATCH_KEYS if key not in OPTIONAL_ENCODED_BATCH_KEYS
    ),
    optional_keys=tuple(key for key in ENCODED_BATCH_KEYS if key in OPTIONAL_ENCODED_BATCH_KEYS),
    accept_legacy_manifest=True,
)


def project_to_encoded_batch(bundle: dict[str, Any]) -> dict[str, Any]:
    """Project an encoder bundle onto the canonical A14B training schema."""
    return WAN22_A14B_ENCODED_BATCH_CODEC.canonicalize(bundle)


def _condition_streams(extras: Mapping[str, Any]) -> tuple[str, ...]:
    return read_condition_streams(
        extras,
        known=ALL_STREAMS,
        default=DEFAULT_CONDITION_STREAMS,
    )


def _condition_spec(extras: Mapping[str, Any]) -> WanConditionSpec:
    return WanConditionSpec(
        streams=_condition_streams(extras),
        mask_channels=MASK_CHANNELS,
    )


def _resolve_expert_configs(
    extras: Mapping[str, Any],
    pretrained_path: str,
    *,
    expert_names: tuple[str, ...] = ALL_EXPERTS,
) -> tuple[dict[str, dict[str, Any]], dict[str, WanChannelLayout]]:
    """Read both experts' published configs once and widen them together.

    Every verb goes through here exactly once per checkpoint. Both experts
    denoise the same latent, so a single z_dim decides the width of both, and
    the layouts the weight-loading hooks need fall out of the same read.
    """
    from vforge.models.wan.model import published_z_dim
    from vforge.models.wan.v22.a14b.model import (
        read_published_configs,
        widen_expert_config,
    )

    if not pretrained_path:
        raise ValueError("model.pretrained_path is required")
    published = read_published_configs(pretrained_path, expert_names=expert_names)
    z_dim = published_z_dim(next(iter(published.values())))
    in_channels = _condition_spec(extras).input_channels(z_dim=z_dim)
    overrides = model_config_overrides(extras)
    configs: dict[str, dict[str, Any]] = {}
    layouts: dict[str, WanChannelLayout] = {}
    for name, config in published.items():
        configs[name], layouts[name] = widen_expert_config(
            config,
            in_channels=in_channels,
            **overrides,
        )
    return configs, layouts


def _boundary_ratio(extras: Mapping[str, Any]) -> float:
    return float(extras.get("boundary_ratio", DEFAULT_BOUNDARY_RATIO))


def _num_train_timesteps(extras: Mapping[str, Any]) -> int:
    return int(extras.get("num_train_timesteps", DEFAULT_NUM_TRAIN_TIMESTEPS))


def _train_expert(extras: Mapping[str, Any]) -> str:
    legacy = {"train_high_noise_expert", "train_low_noise_expert"} & set(extras)
    if legacy:
        raise ValueError(
            f"deprecated Wan22 A14B expert flags {sorted(legacy)} are not supported; "
            "use model.extras.train_expert=high|low|both"
        )
    value = str(extras.get("train_expert", "both"))
    if value not in TRAIN_EXPERTS:
        raise ValueError(f"model.extras.train_expert must be one of {TRAIN_EXPERTS}; got {value!r}")
    return value


def _training_expert_names(extras: Mapping[str, Any]) -> tuple[str, ...]:
    expert = _train_expert(extras)
    if expert == "high":
        return (HIGH_NOISE_EXPERT,)
    if expert == "low":
        return (LOW_NOISE_EXPERT,)
    return ALL_EXPERTS


def _build_condition_encoder(
    *,
    extras: Mapping[str, Any],
    pretrained_path: str,
    precision: str,
    require_assets: bool,
) -> Any:
    """The Wan2.1 encoder with A14B's image branch defaults applied."""
    return _build_wan21_condition_encoder(
        extras={**{key: False for key in _IMAGE_BRANCH_KEYS}, **extras},
        pretrained_path=pretrained_path,
        precision=precision,
        require_assets=require_assets,
        condition=_condition_spec(extras),
    )


def _build_a14b_model(
    *,
    expert_configs: Mapping[str, Mapping[str, Any]],
    extras: Mapping[str, Any],
    init_device: str,
    expert_names: tuple[str, ...] = ALL_EXPERTS,
) -> nn.Module:
    from vforge.models.wan.v22.a14b.model import Wan22A14bModel

    if init_device != "meta":
        raise ValueError(
            "standalone Wan22 A14B requires model.init_device=meta so FSDP2 can "
            "shard before allocating expert parameters"
        )
    model = Wan22A14bModel.build_empty(
        expert_configs,
        parameter_dtype=parameter_storage_dtype(),
        boundary_ratio=_boundary_ratio(extras),
        num_train_timesteps=_num_train_timesteps(extras),
        expert_names=expert_names,
    )
    return model


def _validate(
    *,
    extras: Mapping[str, Any],
    latent_cache_mode: str,
    usp_size: int,
) -> None:
    validate_shared(latent_cache_mode=latent_cache_mode, usp_size=usp_size)
    _condition_streams(extras)
    boundary = _boundary_ratio(extras)
    if not 0.0 < boundary < 1.0:
        raise ValueError(f"model.extras.boundary_ratio must be in (0, 1); got {boundary}")
    _train_expert(extras)
    enabled = [key for key in _IMAGE_BRANCH_KEYS if bool(extras.get(key, False))]
    if enabled:
        raise ValueError(
            f"Wan22 A14B has no image conditioning branch, so {enabled} cannot "
            "be honoured; the reference frame is carried by the rgb condition "
            "stream instead"
        )


__all__ = [
    "CACHE_FLAVOR",
    "WAN22_A14B_ENCODED_BATCH_CODEC",
    "project_to_encoded_batch",
]
