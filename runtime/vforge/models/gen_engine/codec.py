"""Typed VAE encoder-output and latent contracts for packing and decoding."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

VIDEO_LATENT_CODEC_SCHEMA = "vforge.video_latent_codec"
AUDIO_LATENT_CODEC_SCHEMA = "vforge.audio_latent_codec"
VIDEO_LATENT_CODEC_SCHEMA_VERSION = 1
AUDIO_LATENT_CODEC_SCHEMA_VERSION = 1


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return result


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if result < 0 or result != value:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
    return result


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _check_keys(
    value: Mapping[str, Any],
    *,
    label: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"{label} is missing required fields: {sorted(missing)}")
    unknown = set(value) - required - optional
    if unknown:
        raise ValueError(f"{label} has unsupported fields: {sorted(unknown)}")


def _number_tuple(value: Any, *, label: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain {length} numbers")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ValueError(f"{label}[{index}] must be finite")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}[{index}] must be finite") from exc
        if not math.isfinite(number):
            raise ValueError(f"{label}[{index}] must be finite")
        result.append(number)
    if len(result) != length:
        raise ValueError(f"{label} must contain {length} values, got {len(result)}")
    return tuple(result)


def _input_rate(value: float, *, label: str) -> float:
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return rate


@dataclass(frozen=True)
class DiagonalGaussianSampling:
    """Sample a latent from channel-concatenated ``[mean, logvar]`` moments.

    Noise is drawn from the current PyTorch RNG. The codec does not own a seed;
    resume and worker seeding are left to the training runtime.
    """

    logvar_min: float = -30.0
    logvar_max: float = 20.0

    def __post_init__(self) -> None:
        logvar_min = float(self.logvar_min)
        logvar_max = float(self.logvar_max)
        if not math.isfinite(logvar_min) or not math.isfinite(logvar_max):
            raise ValueError("posterior_sampling logvar bounds must be finite")
        if logvar_min >= logvar_max:
            raise ValueError(
                "posterior_sampling.logvar_min must be smaller than posterior_sampling.logvar_max"
            )
        object.__setattr__(self, "logvar_min", logvar_min)
        object.__setattr__(self, "logvar_max", logvar_max)

    def sample(
        self,
        moments: torch.Tensor,
        *,
        channel_dim: int,
        latent_channels: int,
    ) -> torch.Tensor:
        if not isinstance(moments, torch.Tensor):
            raise TypeError("posterior moments must be a torch.Tensor")
        channel_dim %= moments.ndim
        expected_channels = 2 * int(latent_channels)
        if moments.shape[channel_dim] != expected_channels:
            raise ValueError(
                "posterior moments channel dimension has size "
                f"{moments.shape[channel_dim]}, expected {expected_channels} "
                f"for {latent_channels} latent channels"
            )

        mean, logvar = moments.float().chunk(2, dim=channel_dim)
        std = torch.exp(0.5 * logvar.clamp(self.logvar_min, self.logvar_max))
        noise = torch.randn_like(mean)
        return mean + std * noise

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "diagonal_gaussian",
            "logvar_min": self.logvar_min,
            "logvar_max": self.logvar_max,
        }


def _parse_posterior_sampling(
    value: Any,
    *,
    label: str,
) -> DiagonalGaussianSampling:
    config = _mapping(value, label=label)
    _check_keys(
        config,
        label=label,
        required=frozenset({"type", "logvar_min", "logvar_max"}),
    )
    if config["type"] != "diagonal_gaussian":
        raise ValueError(f"{label}.type must be 'diagonal_gaussian', got {config['type']!r}")
    return DiagonalGaussianSampling(
        logvar_min=float(config["logvar_min"]),
        logvar_max=float(config["logvar_max"]),
    )


class TemporalMapping(Protocol):
    """Map VAE latent indices to positions on the input-media clock."""

    def input_indices(self, latent_frames: int) -> torch.Tensor: ...

    def align_input_length(self, input_length: int) -> int: ...

    def latent_frames(self, input_length: int) -> int: ...

    def to_dict(self) -> dict[str, Any]: ...

    def local_times(
        self,
        latent_frames: int,
        *,
        input_rate: float,
        patch_t: int = 1,
    ) -> torch.Tensor: ...


def _local_times(
    mapping: TemporalMapping,
    latent_frames: int,
    *,
    input_rate: float,
    patch_t: int,
) -> torch.Tensor:
    latent_frames = _positive_int(latent_frames, label="latent_frames")
    patch_t = _positive_int(patch_t, label="patch_t")
    if latent_frames % patch_t:
        raise ValueError(f"latent_frames={latent_frames} must be divisible by patch_t={patch_t}")
    rate = _input_rate(input_rate, label="effective_fps")
    indices = mapping.input_indices(latent_frames)
    if indices.shape != (latent_frames,):
        raise RuntimeError(
            "temporal mapping returned invalid input-index shape "
            f"{tuple(indices.shape)} for {latent_frames} latent frames"
        )
    return indices[::patch_t].to(torch.float32).div_(rate)


@dataclass(frozen=True)
class StrideTemporalMapping:
    """One latent sample for every fixed number of input-media samples."""

    input_stride: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_stride",
            _positive_int(self.input_stride, label="temporal.mapping.input_stride"),
        )

    def input_indices(self, latent_frames: int) -> torch.Tensor:
        return torch.arange(int(latent_frames), dtype=torch.float64) * self.input_stride

    def align_input_length(self, input_length: int) -> int:
        return _positive_int(input_length, label="input_length")

    def latent_frames(self, input_length: int) -> int:
        length = self.align_input_length(input_length)
        return math.ceil(length / self.input_stride)

    def local_times(
        self,
        latent_frames: int,
        *,
        input_rate: float,
        patch_t: int = 1,
    ) -> torch.Tensor:
        return _local_times(
            self,
            latent_frames,
            input_rate=input_rate,
            patch_t=patch_t,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": "stride", "input_stride": self.input_stride}


@dataclass(frozen=True)
class PeriodicOffsetsTemporalMapping:
    """A repeating non-uniform latent grid on the input-media clock."""

    input_period: int
    latent_offsets: tuple[int, ...]
    valid_length_remainder: int

    def __post_init__(self) -> None:
        period = _positive_int(
            self.input_period,
            label="temporal.mapping.input_period",
        )
        offsets = tuple(
            _nonnegative_int(
                value,
                label=f"temporal.mapping.latent_offsets[{index}]",
            )
            for index, value in enumerate(self.latent_offsets)
        )
        if not offsets:
            raise ValueError("temporal.mapping.latent_offsets must not be empty")
        if offsets[0] != 0:
            raise ValueError("temporal.mapping.latent_offsets must start at zero")
        if any(value < 0 or value >= period for value in offsets):
            raise ValueError("temporal.mapping.latent_offsets must lie within input_period")
        if any(left >= right for left, right in zip(offsets, offsets[1:], strict=False)):
            raise ValueError("temporal.mapping.latent_offsets must be strictly increasing")
        remainder = _nonnegative_int(
            self.valid_length_remainder,
            label="temporal.mapping.valid_length_remainder",
        )
        if not 0 <= remainder < period:
            raise ValueError(
                f"temporal.mapping.valid_length_remainder must be in [0, {period}), got {remainder}"
            )
        if remainder and not any(offset < remainder for offset in offsets):
            raise ValueError("temporal.mapping.valid_length_remainder must include a latent offset")
        object.__setattr__(self, "input_period", period)
        object.__setattr__(self, "latent_offsets", offsets)
        object.__setattr__(self, "valid_length_remainder", remainder)

    def input_indices(self, latent_frames: int) -> torch.Tensor:
        latent_frames = int(latent_frames)
        indices = torch.arange(latent_frames, dtype=torch.long)
        offsets = torch.tensor(self.latent_offsets, dtype=torch.long)
        groups = torch.div(indices, len(self.latent_offsets), rounding_mode="floor")
        return (
            groups * self.input_period
            + offsets.index_select(0, indices.remainder(len(self.latent_offsets)))
        ).to(torch.float64)

    def align_input_length(self, input_length: int) -> int:
        length = _positive_int(input_length, label="input_length")
        return length + (self.valid_length_remainder - length) % self.input_period

    def latent_frames(self, input_length: int) -> int:
        aligned = self.align_input_length(input_length)
        periods, remainder = divmod(aligned, self.input_period)
        return periods * len(self.latent_offsets) + sum(
            offset < remainder for offset in self.latent_offsets
        )

    def local_times(
        self,
        latent_frames: int,
        *,
        input_rate: float,
        patch_t: int = 1,
    ) -> torch.Tensor:
        return _local_times(
            self,
            latent_frames,
            input_rate=input_rate,
            patch_t=patch_t,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "periodic_offsets",
            "input_period": self.input_period,
            "latent_offsets": list(self.latent_offsets),
            "valid_length_remainder": self.valid_length_remainder,
        }


def parse_temporal_mapping(value: Any, *, label: str) -> TemporalMapping:
    config = _mapping(value, label=label)
    mapping_type = config.get("type")
    if mapping_type == "stride":
        _check_keys(
            config,
            label=label,
            required=frozenset({"type", "input_stride"}),
        )
        return StrideTemporalMapping(
            _positive_int(
                config["input_stride"],
                label=f"{label}.input_stride",
            )
        )
    if mapping_type == "periodic_offsets":
        _check_keys(
            config,
            label=label,
            required=frozenset(
                {
                    "type",
                    "input_period",
                    "latent_offsets",
                    "valid_length_remainder",
                }
            ),
        )
        offsets = config["latent_offsets"]
        if not isinstance(offsets, Sequence) or isinstance(offsets, (str, bytes)):
            raise ValueError(f"{label}.latent_offsets must be an integer list")
        return PeriodicOffsetsTemporalMapping(
            input_period=_positive_int(
                config["input_period"],
                label=f"{label}.input_period",
            ),
            latent_offsets=tuple(
                _nonnegative_int(offset, label=f"{label}.latent_offsets[{index}]")
                for index, offset in enumerate(offsets)
            ),
            valid_length_remainder=_nonnegative_int(
                config["valid_length_remainder"],
                label=f"{label}.valid_length_remainder",
            ),
        )
    raise ValueError(f"{label}.type must be 'stride' or 'periodic_offsets', got {mapping_type!r}")


@dataclass(frozen=True)
class LatentNormalization:
    """Channel-wise affine conversion between VAE space and model space."""

    mean: tuple[float, ...]
    std: tuple[float, ...]

    @property
    def channels(self) -> int:
        return len(self.mean)

    def __post_init__(self) -> None:
        if not self.mean:
            raise ValueError("latent normalization mean/std must not be empty")
        if len(self.mean) != len(self.std):
            raise ValueError("latent normalization mean/std lengths must match")
        mean = _number_tuple(
            self.mean,
            label="latent normalization mean",
            length=len(self.mean),
        )
        std = _number_tuple(
            self.std,
            label="latent normalization std",
            length=len(self.mean),
        )
        if not all(value > 0 for value in std):
            raise ValueError("latent normalization std values must be finite and positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    def _stats(
        self,
        latent: torch.Tensor,
        *,
        channel_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(latent, torch.Tensor):
            raise TypeError("latent must be a torch.Tensor")
        channel_dim %= latent.ndim
        if latent.shape[channel_dim] != self.channels:
            raise ValueError(
                f"latent channel dimension has size {latent.shape[channel_dim]}, "
                f"expected {self.channels}"
            )
        shape = [1] * latent.ndim
        shape[channel_dim] = self.channels
        mean = torch.tensor(self.mean, device=latent.device, dtype=torch.float32).view(shape)
        std = torch.tensor(self.std, device=latent.device, dtype=torch.float32).view(shape)
        return mean, std

    def normalize(self, latent: torch.Tensor, *, channel_dim: int) -> torch.Tensor:
        mean, std = self._stats(latent, channel_dim=channel_dim)
        return latent.float().sub(mean).div_(std)

    def denormalize(self, latent: torch.Tensor, *, channel_dim: int) -> torch.Tensor:
        mean, std = self._stats(latent, channel_dim=channel_dim)
        return latent.float().mul(std).add_(mean)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "channelwise_mean_std",
            "mean": list(self.mean),
            "std": list(self.std),
        }


def _parse_normalization(
    value: Any,
    *,
    label: str,
    channels: int,
) -> LatentNormalization:
    config = _mapping(value, label=label)
    _check_keys(
        config,
        label=label,
        required=frozenset({"type", "mean", "std"}),
    )
    if config["type"] != "channelwise_mean_std":
        raise ValueError(f"{label}.type must be 'channelwise_mean_std', got {config['type']!r}")
    return LatentNormalization(
        mean=_number_tuple(config["mean"], label=f"{label}.mean", length=channels),
        std=_number_tuple(config["std"], label=f"{label}.std", length=channels),
    )


@dataclass(frozen=True)
class VisualLatentCodec:
    latent_channels: int
    spatial_compression: int
    normalization: LatentNormalization
    temporal: TemporalMapping
    posterior_sampling: DiagonalGaussianSampling | None = None

    def __post_init__(self) -> None:
        channels = _positive_int(
            self.latent_channels,
            label="visual.latent_channels",
        )
        if self.normalization.channels != channels:
            raise ValueError(
                "visual normalization length must match latent_channels: "
                f"{self.normalization.channels} != {channels}"
            )
        object.__setattr__(self, "latent_channels", channels)
        object.__setattr__(
            self,
            "spatial_compression",
            _positive_int(
                self.spatial_compression,
                label="visual.spatial_compression",
            ),
        )

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return self.normalization.normalize(latent, channel_dim=0)

    @property
    def input_channels(self) -> int:
        return self.latent_channels * (2 if self.posterior_sampling is not None else 1)

    def to_model_space(self, encoder_output: torch.Tensor) -> torch.Tensor:
        latent = (
            self.posterior_sampling.sample(
                encoder_output,
                channel_dim=0,
                latent_channels=self.latent_channels,
            )
            if self.posterior_sampling is not None
            else encoder_output
        )
        return self.normalize(latent)

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        return self.normalization.denormalize(latent, channel_dim=0)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "latent_channels": self.latent_channels,
            "spatial_compression": self.spatial_compression,
            "normalization": self.normalization.to_dict(),
            "temporal": {
                "input_rate_unit": "frames_per_second",
                "mapping": self.temporal.to_dict(),
            },
        }
        if self.posterior_sampling is not None:
            result["posterior_sampling"] = self.posterior_sampling.to_dict()
        return result


@dataclass(frozen=True)
class AudioLatentCodec:
    latent_channels: int
    planes: int
    normalization: LatentNormalization
    temporal: TemporalMapping

    def __post_init__(self) -> None:
        channels = _positive_int(
            self.latent_channels,
            label="audio.latent_channels",
        )
        if self.normalization.channels != channels:
            raise ValueError(
                "audio normalization length must match latent_channels: "
                f"{self.normalization.channels} != {channels}"
            )
        object.__setattr__(self, "latent_channels", channels)
        object.__setattr__(
            self,
            "planes",
            _positive_int(self.planes, label="audio.planes"),
        )

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return self.normalization.normalize(latent, channel_dim=1)

    def to_model_space(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.normalize(encoder_output)

    def denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        return self.normalization.denormalize(latent, channel_dim=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "latent_channels": self.latent_channels,
            "planes": self.planes,
            "normalization": self.normalization.to_dict(),
            "temporal": {
                "input_rate_unit": "samples_per_second",
                "mapping": self.temporal.to_dict(),
            },
        }


def _parse_stream_temporal(
    value: Any,
    *,
    label: str,
    expected_unit: str,
) -> TemporalMapping:
    temporal = _mapping(value, label=label)
    _check_keys(
        temporal,
        label=label,
        required=frozenset({"input_rate_unit", "mapping"}),
    )
    unit = temporal["input_rate_unit"]
    if unit != expected_unit:
        raise ValueError(f"{label}.input_rate_unit must be {expected_unit!r}, got {unit!r}")
    return parse_temporal_mapping(temporal["mapping"], label=f"{label}.mapping")


@dataclass(frozen=True)
class LatentCodecConfig:
    visual: VisualLatentCodec
    audio: AudioLatentCodec

    @classmethod
    def from_split_dicts(
        cls,
        video_value: Mapping[str, Any],
        audio_value: Mapping[str, Any],
    ) -> LatentCodecConfig:
        """Compose independently versioned video and audio codec contracts."""

        _check_keys(
            video_value,
            label="video latent codec config",
            required=frozenset(
                {
                    "schema",
                    "schema_version",
                    "latent_channels",
                    "spatial_compression",
                    "normalization",
                    "temporal",
                }
            ),
            optional=frozenset({"posterior_sampling"}),
        )
        if video_value["schema"] != VIDEO_LATENT_CODEC_SCHEMA:
            raise ValueError(
                "video latent codec schema must be "
                f"{VIDEO_LATENT_CODEC_SCHEMA!r}, got {video_value['schema']!r}"
            )
        if video_value["schema_version"] != VIDEO_LATENT_CODEC_SCHEMA_VERSION:
            raise ValueError(
                "video latent codec schema_version must be "
                f"{VIDEO_LATENT_CODEC_SCHEMA_VERSION}, got {video_value['schema_version']!r}"
            )

        _check_keys(
            audio_value,
            label="audio latent codec config",
            required=frozenset(
                {
                    "schema",
                    "schema_version",
                    "latent_channels",
                    "planes",
                    "normalization",
                    "temporal",
                }
            ),
        )
        if audio_value["schema"] != AUDIO_LATENT_CODEC_SCHEMA:
            raise ValueError(
                "audio latent codec schema must be "
                f"{AUDIO_LATENT_CODEC_SCHEMA!r}, got {audio_value['schema']!r}"
            )
        if audio_value["schema_version"] != AUDIO_LATENT_CODEC_SCHEMA_VERSION:
            raise ValueError(
                "audio latent codec schema_version must be "
                f"{AUDIO_LATENT_CODEC_SCHEMA_VERSION}, got {audio_value['schema_version']!r}"
            )

        video_channels = _positive_int(
            video_value["latent_channels"],
            label="video latent codec latent_channels",
        )
        audio_channels = _positive_int(
            audio_value["latent_channels"],
            label="audio latent codec latent_channels",
        )
        return cls(
            visual=VisualLatentCodec(
                latent_channels=video_channels,
                spatial_compression=_positive_int(
                    video_value["spatial_compression"],
                    label="video latent codec spatial_compression",
                ),
                normalization=_parse_normalization(
                    video_value["normalization"],
                    label="video latent codec normalization",
                    channels=video_channels,
                ),
                temporal=_parse_stream_temporal(
                    video_value["temporal"],
                    label="video latent codec temporal",
                    expected_unit="frames_per_second",
                ),
                posterior_sampling=(
                    _parse_posterior_sampling(
                        video_value["posterior_sampling"],
                        label="video latent codec posterior_sampling",
                    )
                    if "posterior_sampling" in video_value
                    else None
                ),
            ),
            audio=AudioLatentCodec(
                latent_channels=audio_channels,
                planes=_positive_int(
                    audio_value["planes"],
                    label="audio latent codec planes",
                ),
                normalization=_parse_normalization(
                    audio_value["normalization"],
                    label="audio latent codec normalization",
                    channels=audio_channels,
                ),
                temporal=_parse_stream_temporal(
                    audio_value["temporal"],
                    label="audio latent codec temporal",
                    expected_unit="samples_per_second",
                ),
            ),
        )

    @classmethod
    def from_files(
        cls,
        video_path: str | Path,
        audio_path: str | Path,
    ) -> LatentCodecConfig:
        """Load and compose independent video/audio codec JSON files."""

        def load(path: str | Path, *, label: str) -> Mapping[str, Any]:
            source = Path(path).expanduser().resolve()
            if not source.is_file():
                raise ValueError(f"{label} does not exist: {source}")
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"failed to read {label} {source}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"{label} {source} must contain a JSON object")
            return payload

        return cls.from_split_dicts(
            load(video_path, label="video latent codec config"),
            load(audio_path, label="audio latent codec config"),
        )

    def video_config_dict(self) -> dict[str, Any]:
        return {
            "schema": VIDEO_LATENT_CODEC_SCHEMA,
            "schema_version": VIDEO_LATENT_CODEC_SCHEMA_VERSION,
            **self.visual.to_dict(),
        }

    def audio_config_dict(self) -> dict[str, Any]:
        return {
            "schema": AUDIO_LATENT_CODEC_SCHEMA,
            "schema_version": AUDIO_LATENT_CODEC_SCHEMA_VERSION,
            **self.audio.to_dict(),
        }


__all__ = [
    "AUDIO_LATENT_CODEC_SCHEMA",
    "AUDIO_LATENT_CODEC_SCHEMA_VERSION",
    "AudioLatentCodec",
    "DiagonalGaussianSampling",
    "LatentCodecConfig",
    "LatentNormalization",
    "PeriodicOffsetsTemporalMapping",
    "StrideTemporalMapping",
    "TemporalMapping",
    "VIDEO_LATENT_CODEC_SCHEMA",
    "VIDEO_LATENT_CODEC_SCHEMA_VERSION",
    "VisualLatentCodec",
    "parse_temporal_mapping",
]
