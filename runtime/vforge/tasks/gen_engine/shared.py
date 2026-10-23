"""Shared scratch-model and packed-data contracts for the generation engine."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vforge.models.gen_engine.codec import LatentCodecConfig
from vforge.tasks.gen_engine.dataset import dataset_input_options
from vforge.tasks.requests import TrainRequest
from vforge.training.checkpoint.mapping import StateDictMapper

PACKING_VERSION = 6
LAYOUT_VERSION = 2

_MODEL_DIMENSION_KEYS = (
    "schema",
    "schema_version",
    "architecture",
    "scratch_init_scheme",
    "num_attention_heads",
    "attention_head_dim",
    "hidden_size",
    "num_layers",
    "num_refiner_layers",
    "ffn_dim",
    "refiner_ffn_dim",
    "in_channels",
    "audio_in_channels",
    "patch_size",
    "text_dim",
    "freq_dim",
    "time_embed_hidden_dim",
    "time_embed_dim",
    "rope_freq_dim",
    "rope_theta",
    "max_reference_index",
    "position_time_scale",
    "reference_embedding_init_std",
    "reference_identity_mode",
    "num_audio_planes",
    "audio_plane_embedding_init_std",
    "moe",
)

_DIT_SUBMODULE_MODULES = frozenset(
    {
        "vforge.models.gen_engine.dit",
        "vforge.models.gen_engine.dit.config",
        "vforge.models.gen_engine.dit.model",
    }
)


def _require_dit(module: str, attribute: str) -> Any:
    """Import from the private DiT submodule with one actionable failure."""
    try:
        imported = import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name not in _DIT_SUBMODULE_MODULES:
            raise
        raise ImportError(
            "Gen Engine requires its private DiT submodule. Run: "
            "git submodule update --init --checkout vforge/models/gen_engine/dit"
        ) from exc
    return getattr(imported, attribute)


@dataclass(frozen=True)
class GenEngineConfig:
    """One parsed model/data contract shared by train and evaluation assembly."""

    transformer: dict[str, Any]
    codec: LatentCodecConfig
    patch_size: tuple[int, int, int]
    extras: dict[str, Any]
    transformer_digest: str
    video_codec_digest: str
    audio_codec_digest: str

    @property
    def text_width(self) -> int:
        return int(self.transformer["text_dim"])

    @property
    def row_widths(self) -> tuple[int, int]:
        return _stream_row_widths(self.transformer, self.patch_size)


def validate_execution(request: TrainRequest, engine: str) -> None:
    """Reject unsupported execution before process groups or model allocation."""
    if engine != "fsdp2" or request.tp_size != 1 or request.cp_size != 1:
        raise ValueError(
            "gen_engine requires parallel.engine=fsdp2, parallel.tp_size=1 and "
            "parallel.cp_size=1; tensor/context parallelism is not implemented"
        )


def target_modalities(extras: Mapping[str, Any]) -> tuple[str, ...] | None:
    value = extras.get("target_modalities")
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("model.extras.target_modalities must be a video/audio list")
    result = tuple(value)
    if (
        not result
        or len(set(result)) != len(result)
        or any(item not in {"video", "audio"} for item in result)
    ):
        raise ValueError(
            "model.extras.target_modalities must be a non-empty unique video/audio list"
        )
    return result


def _transformer_config_path(extras: Mapping[str, Any]) -> Path:
    value = extras.get("transformer_config_path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model.extras.transformer_config_path is required")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"transformer config does not exist: {path}")
    return path


def _load_transformer_config(extras: Mapping[str, Any]) -> dict[str, Any]:
    path = _transformer_config_path(extras)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read transformer config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"transformer config {path} must contain a JSON object")
    # Normalize once here so every consumer -- validation, row widths, resume
    # fingerprints and the model builder -- sees the same resolved defaults.
    config_type = _require_dit("vforge.models.gen_engine.dit.config", "GenEngineDiTConfig")
    return config_type.from_dict(value, require_architecture=True).to_dict()


def _latent_codec_config_path(
    extras: Mapping[str, Any],
    modality: str,
) -> Path:
    key = f"{modality}_latent_codec_config_path"
    value = extras.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"model.extras.{key} is required")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{modality} latent codec config does not exist: {path}")
    return path


def _load_latent_codec_config(extras: Mapping[str, Any]) -> LatentCodecConfig:
    return LatentCodecConfig.from_files(
        _latent_codec_config_path(extras, "video"),
        _latent_codec_config_path(extras, "audio"),
    )


def _latent_codec_config_digest(
    extras: Mapping[str, Any],
    modality: str,
) -> str:
    return hashlib.sha256(_latent_codec_config_path(extras, modality).read_bytes()).hexdigest()


def _patch_size(
    extras: Mapping[str, Any],
    transformer_config: Mapping[str, Any] | None = None,
) -> tuple[int, int, int]:
    config = (
        transformer_config if transformer_config is not None else _load_transformer_config(extras)
    )
    configured = config.get("patch_size", (1, 2, 2))
    requested = extras.get("patch_size") or configured
    if not isinstance(requested, Sequence) or isinstance(requested, (str, bytes)):
        raise ValueError("model.extras.patch_size must contain three positive integers")
    patch_size = tuple(int(value) for value in requested)
    if len(patch_size) != 3 or any(value <= 0 for value in patch_size):
        raise ValueError(
            f"model.extras.patch_size must contain three positive integers, got {patch_size}"
        )
    config_patch = tuple(int(value) for value in configured)
    if config_patch != patch_size:
        raise ValueError(
            "model.extras.patch_size must match the scratch transformer config; "
            f"got {patch_size} and {config_patch}"
        )
    return patch_size


def _model_dimensions(
    extras: Mapping[str, Any],
    transformer_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = (
        transformer_config if transformer_config is not None else _load_transformer_config(extras)
    )
    dimensions = {key: config[key] for key in _MODEL_DIMENSION_KEYS if key in config}
    dimensions["patch_size"] = list(_patch_size(extras, config))
    return dimensions


def _stream_row_widths(
    transformer_config: Mapping[str, Any],
    patch_size: tuple[int, int, int],
) -> tuple[int, int]:
    try:
        video_channels = int(transformer_config["in_channels"])
        audio_channels = int(transformer_config["audio_in_channels"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "transformer config requires integer in_channels and audio_in_channels"
        ) from exc
    if video_channels <= 0 or audio_channels <= 0:
        raise ValueError("transformer stream channel dimensions must be positive")
    return video_channels * math.prod(patch_size), audio_channels


def _transformer_config_digest(extras: Mapping[str, Any]) -> str:
    return hashlib.sha256(_transformer_config_path(extras).read_bytes()).hexdigest()


def _validate_codec_against_transformer(
    codec: LatentCodecConfig,
    transformer_config: Mapping[str, Any],
) -> None:
    try:
        visual_channels = int(transformer_config["in_channels"])
        audio_channels = int(transformer_config["audio_in_channels"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "transformer config requires integer in_channels and audio_in_channels"
        ) from exc
    if codec.visual.latent_channels != visual_channels:
        raise ValueError(
            "video latent codec latent_channels must match transformer in_channels: "
            f"{codec.visual.latent_channels} != {visual_channels}"
        )
    if codec.audio.latent_channels != audio_channels:
        raise ValueError(
            "audio latent codec latent_channels must match transformer audio_in_channels: "
            f"{codec.audio.latent_channels} != {audio_channels}"
        )
    num_audio_planes = int(transformer_config.get("num_audio_planes", 2))
    if codec.audio.planes != num_audio_planes:
        raise ValueError(
            "audio latent codec planes must match transformer num_audio_planes: "
            f"{codec.audio.planes} != {num_audio_planes}"
        )


def _validate_shared(
    *,
    extras: Mapping[str, Any],
    pretrained_path: str,
    init_device: str,
    latent_cache_mode: str,
    batch_size: int,
) -> GenEngineConfig:
    if pretrained_path:
        raise ValueError("gen_engine trains from scratch; model.pretrained_path must be empty")
    if init_device != "meta":
        raise ValueError(
            "gen_engine requires model.init_device=meta so parameters are initialized "
            "after distributed materialization"
        )
    if latent_cache_mode != "offline":
        raise ValueError("gen_engine requires latent_cache.mode=offline")
    if batch_size != 1:
        raise ValueError("gen_engine packed layout requires data.batch_size=1")
    require_digests = extras.get("require_digests", True)
    if not isinstance(require_digests, bool):
        raise ValueError("model.extras.require_digests must be true or false")
    dataset_input_options(extras)
    transformer_config = _load_transformer_config(extras)
    latent_codec = _load_latent_codec_config(extras)
    num_audio_planes = int(transformer_config.get("num_audio_planes", 2))
    if num_audio_planes != 2:
        raise ValueError(
            "gen_engine data contract requires transformer num_audio_planes=2, "
            f"got {num_audio_planes}"
        )
    _validate_codec_against_transformer(latent_codec, transformer_config)
    patch_size = _patch_size(extras, transformer_config)
    if int(transformer_config.get("text_dim", 0)) < 1:
        raise ValueError("transformer config requires positive text_dim")
    resolved = dict(extras)
    resolved["target_modalities"] = target_modalities(extras)
    for name, default in (
        ("video_flow_shift", 12.0),
        ("audio_flow_shift", 3.0),
        ("video_loss_weight", 1.0),
        ("audio_loss_weight", 1.0),
        ("reference_timestep", 0.999),
        ("text_timestep", 1.0),
    ):
        value = float(extras.get(name, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"model.extras.{name} must be finite and non-negative")
        if name.endswith("flow_shift") and value == 0:
            raise ValueError(f"model.extras.{name} must be positive")
        if name.endswith("timestep") and value > 1:
            raise ValueError(f"model.extras.{name} must be in [0, 1]")
        resolved[name] = value
    allowed = {
        "transformer_config_path",
        "video_latent_codec_config_path",
        "audio_latent_codec_config_path",
        "patch_size",
        "require_digests",
        "tensor_format",
        "audio_layout",
        "artifact_cache_dir",
        "pin_artifacts",
        "target_modalities",
        "video_flow_shift",
        "audio_flow_shift",
        "video_loss_weight",
        "audio_loss_weight",
        "reference_timestep",
        "text_timestep",
    }
    unknown = set(extras) - allowed
    if unknown:
        raise ValueError(f"unknown gen_engine model extras: {sorted(unknown)}")
    return GenEngineConfig(
        transformer_config,
        latent_codec,
        patch_size,
        resolved,
        _transformer_config_digest(extras),
        _latent_codec_config_digest(extras, "video"),
        _latent_codec_config_digest(extras, "audio"),
    )


def _build_gen_engine_model(
    *,
    extras: Mapping[str, Any],
    init_device: str,
    pretrained_path: str,
    config: GenEngineConfig | None = None,
) -> nn.Module:
    if config is None:
        config = _validate_shared(
            extras=extras,
            pretrained_path=pretrained_path,
            init_device=init_device,
            latent_cache_mode="offline",
            batch_size=1,
        )
    model_type = _require_dit("vforge.models.gen_engine.dit.model", "GenEngineModel")
    return model_type.build_empty(config=config.transformer, parameter_dtype=torch.float32)


def _initialize_gen_engine_model(model: nn.Module, *, seed: int = 0) -> None:
    initialize = getattr(model, "initialize_parameters", None)
    if not callable(initialize):
        raise TypeError("GenEngineModel must define initialize_parameters()")
    initialize(seed=int(seed))


def _gen_engine_state_dict_mapper() -> StateDictMapper:
    return StateDictMapper.strip_prefix("transformer.")


__all__ = [
    "LAYOUT_VERSION",
    "PACKING_VERSION",
]
