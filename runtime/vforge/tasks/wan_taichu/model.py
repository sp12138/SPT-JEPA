"""Wan Taichu variant-to-model assembly shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from vforge.models.wan_taichu.model import (
    WanTaichuChannelLayout,
    published_z_dim,
    read_published_transformer_config,
)
from vforge.tasks.wan.shared import model_config_overrides, parameter_storage_dtype
from vforge.tasks.wan_taichu.variants import WanTaichuVariant, require_train_variant

_A14B_EXPERT_SUBFOLDERS = {"wan": "transformer", "wan_2": "transformer_2"}


def build_empty_model(
    *,
    variant: str,
    transformer_configs: Mapping[str, Mapping[str, Any]],
    parameter_dtype=None,
    frame_per_block: int = 3,
    first_frame_own_block: bool = True,
    local_attn_size: int = -1,
    sink_size: int = 0,
    use_fsdp_precision_contract: bool = True,
    upcast_shared_text_context: bool = True,
) -> nn.Module:
    """Pass one resolved variant's structural parameters to the shared model."""
    spec = require_train_variant(variant)
    expected = tuple(spec.experts)
    actual = set(transformer_configs)
    if len(transformer_configs) != len(expected) or actual != set(expected):
        raise ValueError(
            f"{variant} transformer configs must have keys {list(expected)}; "
            f"got {list(transformer_configs)}"
        )

    from vforge.models.wan_taichu.model import WanTaichuModel

    return WanTaichuModel.build_empty(
        variant=spec.name,
        architecture=spec.architecture,
        attention=spec.attention,
        transformer_configs=transformer_configs,
        expert_names=spec.experts,
        parameter_dtype=parameter_dtype,
        boundary_ratio=spec.boundary_ratio,
        frame_per_block=frame_per_block,
        first_frame_own_block=first_frame_own_block,
        local_attn_size=local_attn_size,
        sink_size=sink_size,
        use_fsdp_precision_contract=use_fsdp_precision_contract,
        upcast_shared_text_context=upcast_shared_text_context,
    )


def build_model(
    *,
    spec: WanTaichuVariant,
    streams: tuple[str, ...],
    extras: Mapping[str, Any],
    pretrained_path: str,
    init_device: str,
) -> tuple[nn.Module, dict[str, WanTaichuChannelLayout]]:
    """Build a model from its published config and return weight-loading layouts."""
    if not pretrained_path:
        raise ValueError("model.pretrained_path is required")
    if init_device != "meta":
        raise ValueError("Wan Taichu requires model.init_device=meta")
    overrides = model_config_overrides(extras)
    requested_in = spec.input_channels(streams)
    configs: dict[str, dict[str, Any]] = {}
    layouts: dict[str, WanTaichuChannelLayout] = {}

    if spec.architecture == "wan22_a14b":
        for name in spec.experts:
            published = read_published_transformer_config(
                pretrained_path,
                subfolder=_A14B_EXPERT_SUBFOLDERS[name],
            )
            _validate_published_z_dim(spec, published)
            configs[name], layouts[name] = _widen_a14b_config(
                published,
                in_channels=requested_in,
                **overrides,
            )
    elif spec.architecture == "wan22_ti2v_5b":
        published = read_published_transformer_config(pretrained_path)
        _validate_published_z_dim(spec, published)
        configs[""], layouts[""] = _widen_ti2v_config(
            published,
            in_channels=requested_in,
            **overrides,
        )
        configs[""].update(
            use_ref_token=spec.ref_token,
            in_channels_ref=spec.z_dim,
        )
    else:
        published = read_published_transformer_config(pretrained_path)
        _validate_published_z_dim(spec, published)
        declared_in = int(published["in_channels"])
        if requested_in < declared_in:
            raise ValueError(
                f"{spec.name} requested in_channels={requested_in} is narrower than "
                f"the published width {declared_in}"
            )
        config = dict(published)
        config.update(overrides)
        config["in_channels"] = requested_in
        config["out_channels"] = spec.z_dim
        for key in ("use_vae_condition", "use_clip_condition", "use_dino_condition"):
            if key in extras:
                config[key] = bool(extras[key])
        if spec.audio:
            config.update(
                use_audio_condition=True,
                audio_dim=int(extras["audio_dim"]),
                audio_num_layers=int(extras["audio_num_layers"]),
                num_audio_token=int(extras["num_audio_token"]),
                audio_inject_layers_s=list(extras["audio_inject_layers_s"]),
                audio_inject_layers_l=list(extras["audio_inject_layers_l"]),
                audio_window_s=int(extras["audio_window_s"]),
                audio_window_l=int(extras["audio_window_l"]),
                audio_zero_init=bool(extras.get("audio_zero_init", False)),
            )
        configs[""] = config
        layouts[""] = WanTaichuChannelLayout(
            z_dim=spec.z_dim,
            pretrained_in_channels=declared_in,
            in_channels=requested_in,
        )

    model = build_empty_model(
        variant=spec.name,
        transformer_configs=configs,
        parameter_dtype=parameter_storage_dtype(),
        frame_per_block=int(extras.get("frame_per_block", 3)),
        first_frame_own_block=bool(extras.get("first_frame_own_block", True)),
        local_attn_size=int(extras.get("local_attn_size", -1)),
        sink_size=int(extras.get("sink_size", 0)),
    )
    return model, layouts


def _validate_published_z_dim(
    spec: WanTaichuVariant,
    config: Mapping[str, Any],
) -> None:
    actual = published_z_dim(config)
    if actual != spec.z_dim:
        raise ValueError(f"{spec.name} expects z_dim={spec.z_dim}; checkpoint declares {actual}")


def _widen_a14b_config(
    published: Mapping[str, Any],
    *,
    in_channels: int,
    **overrides: Any,
) -> tuple[dict[str, Any], WanTaichuChannelLayout]:
    config = dict(published)
    z_dim = published_z_dim(config)
    declared_in = int(config["in_channels"])
    if int(in_channels) < declared_in:
        raise ValueError(
            f"the requested patch_embedding width {in_channels} is narrower than "
            f"the published width {declared_in}"
        )
    config.update(
        in_channels=int(in_channels),
        out_channels=z_dim,
        use_vae_condition=False,
    )
    config.update(overrides)
    return config, WanTaichuChannelLayout(
        z_dim=z_dim,
        pretrained_in_channels=declared_in,
        in_channels=int(in_channels),
    )


def _widen_ti2v_config(
    published: Mapping[str, Any],
    *,
    in_channels: int,
    **overrides: Any,
) -> tuple[dict[str, Any], WanTaichuChannelLayout]:
    config = dict(published)
    z_dim = published_z_dim(config)
    declared_in = int(config["in_channels"])
    if declared_in not in {z_dim, int(in_channels)}:
        raise ValueError(
            "Wan Taichu TI2V expects published in_channels to equal z_dim or "
            f"the requested width {in_channels}; got {declared_in}"
        )
    config.update(
        in_channels=int(in_channels),
        out_channels=z_dim,
        use_vae_condition=False,
    )
    config.update(overrides)
    return config, WanTaichuChannelLayout(
        z_dim=z_dim,
        pretrained_in_channels=declared_in,
        in_channels=int(in_channels),
    )


__all__ = ["build_empty_model", "build_model"]
