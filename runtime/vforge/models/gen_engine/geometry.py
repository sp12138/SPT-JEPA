"""Resolve requested output sizes and VAE clocks before model execution."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .codec import AudioLatentCodec, LatentCodecConfig, VisualLatentCodec
from .layout import AudioGeometry, TargetGeometry, VisualGeometry

_MISSING = object()


def _value(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, _MISSING)
    return getattr(source, name, _MISSING)


def _config_value(
    config: Any,
    names: Sequence[str],
    *,
    sections: Sequence[str] = (),
    section_names: Sequence[str] = (),
) -> Any:
    for name in names:
        value = _value(config, name)
        if value is not _MISSING and value is not None:
            return value
    for section_name in sections:
        section = _value(config, section_name)
        if section is _MISSING or section is None:
            continue
        for name in section_names:
            value = _value(section, name)
            if value is not _MISSING and value is not None:
                return value
    return _MISSING


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return result


def _positive_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return result


def _required_config_value(
    config: Any,
    names: Sequence[str],
    *,
    label: str,
    sections: Sequence[str],
    section_names: Sequence[str],
) -> Any:
    value = _config_value(
        config,
        names,
        sections=sections,
        section_names=section_names,
    )
    if value is _MISSING:
        raise ValueError(f"generation config requires {label}")
    return value


def _visual_channels(config: Any) -> int:
    return _positive_integer(
        _required_config_value(
            config,
            ("visual_channels", "video_channels", "visual_in_channels", "in_channels"),
            label="visual_channels",
            sections=("visual", "video"),
            section_names=("channels", "in_channels", "latent_channels"),
        ),
        label="generation config visual_channels",
    )


def _audio_channels(config: Any) -> int:
    value = _config_value(
        config,
        ("audio_channels", "num_audio_channels"),
        sections=("audio",),
        section_names=("channels", "num_channels"),
    )
    if value is _MISSING:
        return 2
    return _positive_integer(
        value,
        label="generation config audio_channels",
    )


def _audio_features(config: Any) -> int:
    return _positive_integer(
        _required_config_value(
            config,
            ("audio_features", "audio_feature_channels", "audio_in_channels"),
            label="audio_features",
            sections=("audio",),
            section_names=("features", "feature_channels", "in_channels"),
        ),
        label="generation config audio_features",
    )


def _spec_value(spec: Mapping[str, Any], section: str, name: str) -> Any:
    nested = spec.get(section)
    if isinstance(nested, Mapping) and nested.get(name) is not None:
        return nested[name]
    return spec.get(name)


def _target_effective_fps(target: Mapping[str, Any], *, stream: str) -> float:
    nested_names = ("video", "visual") if stream == "video" else ("audio",)
    for name in nested_names:
        nested = target.get(name)
        if isinstance(nested, Mapping) and nested.get("effective_fps") is not None:
            return _positive_number(
                nested["effective_fps"],
                label=f"target.{stream}.effective_fps",
            )
    direct_names = (
        ("visual_effective_fps", "video_effective_fps")
        if stream == "video"
        else ("audio_effective_fps",)
    )
    for name in direct_names:
        if target.get(name) is not None:
            return _positive_number(
                target[name],
                label=f"target.{stream}.effective_fps",
            )
    raise ValueError(f"target {stream} metadata requires effective_fps")


def _duration_from_video(spec: Mapping[str, Any], *, fps: float) -> tuple[float, int]:
    num_frames_value = _spec_value(spec, "video", "num_frames")
    if num_frames_value is not None:
        num_frames = _positive_integer(num_frames_value, label="target_spec.video.num_frames")
        return num_frames / fps, num_frames
    duration = _positive_number(
        spec.get("duration_seconds"),
        label="target_spec.duration_seconds",
    )
    num_frames = max(1, math.ceil(duration * fps - 1e-12))
    return duration, num_frames


def _duration_from_audio(spec: Mapping[str, Any], *, sample_rate: int) -> float:
    num_samples_value = _spec_value(spec, "audio", "num_samples")
    if num_samples_value is not None:
        num_samples = _positive_integer(
            num_samples_value,
            label="target_spec.audio.num_samples",
        )
        return num_samples / sample_rate
    return _positive_number(
        spec.get("duration_seconds"),
        label="target_spec.duration_seconds",
    )


def _audio_input_length(spec: Mapping[str, Any], *, sample_rate: int) -> int:
    num_samples_value = _spec_value(spec, "audio", "num_samples")
    if num_samples_value is not None:
        return _positive_integer(
            num_samples_value,
            label="target_spec.audio.num_samples",
        )
    duration = _positive_number(
        spec.get("duration_seconds"),
        label="target_spec.duration_seconds",
    )
    return max(1, math.ceil(duration * sample_rate - 1e-12))


def _target_modalities(target_spec: Mapping[str, Any]) -> tuple[str, ...]:
    values = target_spec.get("modalities")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError("target_spec.modalities must be a non-empty video/audio sequence")
    modalities = tuple(str(value) for value in values)
    if len(set(modalities)) != len(modalities):
        raise ValueError("target_spec.modalities must not contain duplicates")
    if any(value not in {"video", "audio"} for value in modalities):
        raise ValueError("target_spec.modalities supports only video and audio")
    return modalities


def _explicit_latent_shape(
    target_spec: Mapping[str, Any],
    *,
    stream: str,
    rank: int,
) -> tuple[int, ...] | None:
    nested = target_spec.get(stream)
    value = nested.get("latent_shape") if isinstance(nested, Mapping) else None
    if value is None:
        value = target_spec.get(f"{stream}_latent_shape")
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"target_spec.{stream}.latent_shape must contain {rank} integers")
    shape = tuple(
        _positive_integer(size, label=f"target_spec.{stream}.latent_shape[{index}]")
        for index, size in enumerate(value)
    )
    if len(shape) != rank:
        raise ValueError(f"target_spec.{stream}.latent_shape must contain {rank} integers")
    return shape


def target_geometry_from_gen_config(
    target_spec: Mapping[str, Any],
    generation_config: Any,
    latent_codec: LatentCodecConfig | None = None,
) -> TargetGeometry:
    """Convert raw output geometry to latent geometry using generation config."""

    if not isinstance(target_spec, Mapping):
        raise TypeError("target_spec must be a mapping")
    modalities = _target_modalities(target_spec)
    visual = None
    audio = None

    if "video" in modalities:
        channels = (
            latent_codec.visual.latent_channels
            if latent_codec is not None
            else _visual_channels(generation_config)
        )
        explicit_shape = _explicit_latent_shape(
            target_spec,
            stream="video",
            rank=4,
        )
        if explicit_shape is not None:
            if explicit_shape[0] != channels:
                raise ValueError(
                    "target_spec.video.latent_shape channel count does not match "
                    f"generation config: {explicit_shape[0]} != {channels}"
                )
            effective_value = _config_value(
                target_spec,
                ("visual_effective_fps", "video_effective_fps"),
                sections=("video",),
                section_names=("effective_fps",),
            )
            if effective_value is _MISSING:
                effective_value = (
                    _spec_value(target_spec, "video", "fps")
                    if latent_codec is not None
                    else _required_config_value(
                        generation_config,
                        ("visual_effective_fps", "video_effective_fps"),
                        label="visual_effective_fps",
                        sections=("visual", "video"),
                        section_names=("effective_fps",),
                    )
                )
            visual = VisualGeometry(
                explicit_shape,
                _positive_number(
                    effective_value,
                    label="target_spec.video.effective_fps",
                ),
            )
        else:
            visual = _target_visual_geometry_from_pixels(
                target_spec,
                generation_config,
                channels=channels,
                codec=(latent_codec.visual if latent_codec is not None else None),
            )

    if "audio" in modalities:
        channels = (
            latent_codec.audio.planes
            if latent_codec is not None
            else _audio_channels(generation_config)
        )
        features = (
            latent_codec.audio.latent_channels
            if latent_codec is not None
            else _audio_features(generation_config)
        )
        explicit_shape = _explicit_latent_shape(
            target_spec,
            stream="audio",
            rank=3,
        )
        if explicit_shape is not None:
            expected_prefix = (channels, features)
            if explicit_shape[:2] != expected_prefix:
                raise ValueError(
                    "target_spec.audio.latent_shape prefix does not match generation "
                    f"config: {explicit_shape[:2]} != {expected_prefix}"
                )
            effective_value = _config_value(
                target_spec,
                ("audio_effective_fps",),
                sections=("audio",),
                section_names=("effective_fps",),
            )
            if effective_value is _MISSING:
                effective_value = (
                    _spec_value(target_spec, "audio", "sample_rate")
                    if latent_codec is not None
                    else _required_config_value(
                        generation_config,
                        ("audio_effective_fps",),
                        label="audio_effective_fps",
                        sections=("audio",),
                        section_names=("effective_fps",),
                    )
                )
            audio = AudioGeometry(
                explicit_shape,
                _positive_number(
                    effective_value,
                    label="target_spec.audio.effective_fps",
                ),
            )
        else:
            audio = _target_audio_geometry_from_duration(
                target_spec,
                generation_config,
                channels=channels,
                features=features,
                codec=(latent_codec.audio if latent_codec is not None else None),
            )

    return TargetGeometry(visual=visual, audio=audio)


def _target_visual_geometry_from_pixels(
    target_spec: Mapping[str, Any],
    generation_config: Any,
    *,
    channels: int,
    codec: VisualLatentCodec | None = None,
) -> VisualGeometry:
    height = _positive_integer(
        _spec_value(target_spec, "video", "height"),
        label="target_spec.video.height",
    )
    width = _positive_integer(
        _spec_value(target_spec, "video", "width"),
        label="target_spec.video.width",
    )
    fps = _positive_number(
        _spec_value(target_spec, "video", "fps"),
        label="target_spec.video.fps",
    )
    spatial_compression = (
        codec.spatial_compression
        if codec is not None
        else _positive_integer(
            _required_config_value(
                generation_config,
                (
                    "visual_spatial_compression",
                    "visual_spatial_compression_ratio",
                    "spatial_compression_ratio",
                ),
                label="visual_spatial_compression",
                sections=("visual", "video"),
                section_names=("spatial_compression", "spatial_compression_ratio"),
            ),
            label="generation config visual_spatial_compression",
        )
    )
    if height % spatial_compression or width % spatial_compression:
        raise ValueError(
            "target video pixel geometry must be divisible by visual spatial "
            f"compression={spatial_compression}, got height={height} width={width}"
        )

    duration, pixel_frames = _duration_from_video(target_spec, fps=fps)
    if codec is not None:
        del duration
        latent_frames = codec.temporal.latent_frames(pixel_frames)
        effective_fps = fps
    else:
        temporal_value = _config_value(
            generation_config,
            (
                "visual_temporal_compression",
                "visual_temporal_compression_ratio",
                "temporal_compression_ratio",
            ),
            sections=("visual", "video"),
            section_names=("temporal_compression", "temporal_compression_ratio"),
        )
        effective_value = _config_value(
            generation_config,
            ("visual_effective_fps", "video_effective_fps"),
            sections=("visual", "video"),
            section_names=("effective_fps",),
        )
        if temporal_value is _MISSING and effective_value is _MISSING:
            raise ValueError(
                "generation config requires visual_temporal_compression or visual_effective_fps"
            )
        temporal_compression = (
            None
            if temporal_value is _MISSING
            else _positive_integer(
                temporal_value,
                label="generation config visual_temporal_compression",
            )
        )
        configured_effective_fps = (
            None
            if effective_value is _MISSING
            else _positive_number(
                effective_value,
                label="generation config visual_effective_fps",
            )
        )
        if temporal_compression is not None:
            derived_effective_fps = fps / temporal_compression
            if configured_effective_fps is not None and not math.isclose(
                configured_effective_fps,
                derived_effective_fps,
                rel_tol=1e-6,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "visual_effective_fps disagrees with target fps / temporal "
                    f"compression: {configured_effective_fps} != {derived_effective_fps}"
                )
            latent_frames = math.ceil(pixel_frames / temporal_compression)
            effective_fps = configured_effective_fps or derived_effective_fps
        else:
            assert configured_effective_fps is not None
            latent_frames = max(
                1,
                math.ceil(duration * configured_effective_fps - 1e-12),
            )
            effective_fps = configured_effective_fps
    visual = VisualGeometry(
        (
            channels,
            latent_frames,
            height // spatial_compression,
            width // spatial_compression,
        ),
        effective_fps,
    )
    return visual


def _target_audio_geometry_from_duration(
    target_spec: Mapping[str, Any],
    generation_config: Any,
    *,
    channels: int,
    features: int,
    codec: AudioLatentCodec | None = None,
) -> AudioGeometry:
    sample_rate = _positive_integer(
        _spec_value(target_spec, "audio", "sample_rate"),
        label="target_spec.audio.sample_rate",
    )
    if codec is not None:
        effective_fps = float(sample_rate)
        latent_frames = codec.temporal.latent_frames(
            _audio_input_length(target_spec, sample_rate=sample_rate)
        )
    else:
        effective_fps = _positive_number(
            _required_config_value(
                generation_config,
                ("audio_effective_fps",),
                label="audio_effective_fps",
                sections=("audio",),
                section_names=("effective_fps",),
            ),
            label="generation config audio_effective_fps",
        )
        duration = _duration_from_audio(target_spec, sample_rate=sample_rate)
        latent_frames = max(1, math.ceil(duration * effective_fps - 1e-12))
    return AudioGeometry((channels, features, latent_frames), effective_fps)
