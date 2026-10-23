"""Single source of truth for Wan Taichu Native variants."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

WanTaichuArchitecture = Literal["wan21", "wan22_a14b", "wan22_ti2v_5b"]


@dataclass(frozen=True)
class WanTaichuVariant:
    """Model and conditioning capabilities fixed by one explicit variant."""

    name: str
    architecture: WanTaichuArchitecture
    attention: Literal["bidirectional", "block_causal_teacher"]
    z_dim: int
    mask_channels: int
    default_condition_streams: tuple[str, ...]
    allowed_condition_streams: tuple[str, ...]
    noise_shift: float
    experts: tuple[str, ...]
    first_frame_inpaint: bool = False
    ref_token: bool = False
    audio: bool = False
    audio_dim: int = 0
    audio_num_layers: int = 0
    num_audio_token: int = 0
    audio_inject_layers_s: tuple[int, ...] = ()
    audio_inject_layers_l: tuple[int, ...] = ()
    audio_window_s: int = 0
    audio_window_l: int = -1
    boundary_ratio: float | None = None
    frame_per_block: int | None = None
    first_frame_own_block: bool = True

    def __post_init__(self) -> None:
        if self.z_dim <= 0:
            raise ValueError(f"{self.name} z_dim must be positive")
        if self.mask_channels < 0:
            raise ValueError(f"{self.name} mask_channels must be non-negative")
        if not self.default_condition_streams:
            raise ValueError(f"{self.name} must declare a default condition stream")
        if not set(self.default_condition_streams) <= set(self.allowed_condition_streams):
            raise ValueError(f"{self.name} default streams must be allowed")
        if len(set(self.allowed_condition_streams)) != len(self.allowed_condition_streams):
            raise ValueError(f"{self.name} allowed streams must be unique")
        if self.experts not in (("",), ("wan", "wan_2")):
            raise ValueError(f"{self.name} has unsupported checkpoint experts {self.experts}")
        if self.attention == "block_causal_teacher":
            if self.frame_per_block is None or self.frame_per_block < 1:
                raise ValueError(f"{self.name} must declare a positive frame_per_block")
        elif self.frame_per_block is not None:
            raise ValueError(f"{self.name} declares frame_per_block without causal attention")
        if self.audio:
            if self.architecture != "wan21" or self.attention != "bidirectional":
                raise ValueError(f"{self.name} audio requires bidirectional Wan21")
            if not self.audio_inject_layers_s and not self.audio_inject_layers_l:
                raise ValueError(f"{self.name} audio requires speak or listen injection layers")
            if self.audio_dim < 1 or self.audio_num_layers < 1 or self.num_audio_token < 1:
                raise ValueError(f"{self.name} audio dimensions must be positive")
        elif self.audio_inject_layers_s or self.audio_inject_layers_l:
            raise ValueError(f"{self.name} declares audio injection layers without audio")

    @property
    def ordinary_bidirectional(self) -> bool:
        return self.attention == "bidirectional" and not self.ref_token and not self.audio

    @property
    def train_supported(self) -> bool:
        return self.attention in {"bidirectional", "block_causal_teacher"}

    def resolve_condition_streams(self, value: Any = None) -> tuple[str, ...]:
        if value is None:
            return self.default_condition_streams
        if isinstance(value, str):
            raise TypeError("condition_streams must be a sequence, not a string")
        streams = tuple(str(stream) for stream in value)
        if not streams:
            raise ValueError(f"{self.name} condition_streams must not be empty")
        if len(set(streams)) != len(streams):
            raise ValueError(f"{self.name} condition_streams must not contain duplicates")
        unknown = set(streams) - set(self.allowed_condition_streams)
        if unknown:
            raise ValueError(
                f"{self.name} does not support condition streams {sorted(unknown)}; "
                f"expected a subset of {list(self.allowed_condition_streams)}"
            )
        return streams

    def input_channels(self, condition_streams: Any = None) -> int:
        streams = self.resolve_condition_streams(condition_streams)
        return self.z_dim + len(streams) * (self.mask_channels + self.z_dim)


_VARIANTS = {
    "wan21_i2v_14b": WanTaichuVariant(
        name="wan21_i2v_14b",
        architecture="wan21",
        attention="bidirectional",
        z_dim=16,
        mask_channels=4,
        default_condition_streams=("normal",),
        allowed_condition_streams=("rgb", "normal"),
        noise_shift=1.0,
        experts=("",),
    ),
    "wan21_i2v_14b_audio": WanTaichuVariant(
        name="wan21_i2v_14b_audio",
        architecture="wan21",
        attention="bidirectional",
        z_dim=16,
        mask_channels=4,
        default_condition_streams=("rgb",),
        allowed_condition_streams=("rgb", "normal"),
        noise_shift=5.0,
        experts=("",),
        audio=True,
        audio_dim=1024,
        audio_num_layers=25,
        num_audio_token=4,
        audio_inject_layers_s=tuple(range(0, 40, 2)),
        audio_window_s=0,
        audio_window_l=-1,
    ),
    "wan22_a14b": WanTaichuVariant(
        name="wan22_a14b",
        architecture="wan22_a14b",
        attention="bidirectional",
        z_dim=16,
        mask_channels=4,
        default_condition_streams=("rgb",),
        allowed_condition_streams=("rgb", "grid", "normal"),
        noise_shift=3.0,
        experts=("wan", "wan_2"),
        boundary_ratio=0.9,
    ),
    "wan22_a14b_causal_teacher": WanTaichuVariant(
        name="wan22_a14b_causal_teacher",
        architecture="wan22_a14b",
        attention="block_causal_teacher",
        z_dim=16,
        mask_channels=4,
        default_condition_streams=("rgb", "grid", "normal"),
        allowed_condition_streams=("rgb", "grid", "normal"),
        noise_shift=3.0,
        experts=("wan", "wan_2"),
        boundary_ratio=0.9,
        frame_per_block=3,
        first_frame_own_block=False,
    ),
    "wan22_ti2v_5b": WanTaichuVariant(
        name="wan22_ti2v_5b",
        architecture="wan22_ti2v_5b",
        attention="bidirectional",
        z_dim=48,
        mask_channels=0,
        default_condition_streams=("normal",),
        allowed_condition_streams=("rgb", "grid", "normal"),
        noise_shift=5.0,
        experts=("",),
        first_frame_inpaint=True,
    ),
    "wan22_ti2v_5b_ref": WanTaichuVariant(
        name="wan22_ti2v_5b_ref",
        architecture="wan22_ti2v_5b",
        attention="bidirectional",
        z_dim=48,
        mask_channels=0,
        default_condition_streams=("grid", "normal"),
        allowed_condition_streams=("rgb", "grid", "normal"),
        noise_shift=5.0,
        experts=("",),
        first_frame_inpaint=True,
        ref_token=True,
    ),
    "wan22_ti2v_5b_causal_teacher": WanTaichuVariant(
        name="wan22_ti2v_5b_causal_teacher",
        architecture="wan22_ti2v_5b",
        attention="block_causal_teacher",
        z_dim=48,
        mask_channels=0,
        default_condition_streams=("normal",),
        allowed_condition_streams=("rgb", "grid", "normal"),
        noise_shift=5.0,
        experts=("",),
        first_frame_inpaint=True,
        frame_per_block=3,
        first_frame_own_block=True,
    ),
}

WAN_TAICHU_VARIANTS = MappingProxyType(_VARIANTS)


def variant_names() -> tuple[str, ...]:
    return tuple(WAN_TAICHU_VARIANTS)


def require_variant(name: str) -> WanTaichuVariant:
    try:
        return WAN_TAICHU_VARIANTS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown Wan Taichu variant {name!r}; expected one of {list(variant_names())}"
        ) from exc


def require_ordinary_bidirectional_variant(name: str) -> WanTaichuVariant:
    variant = require_variant(name)
    if not variant.ordinary_bidirectional:
        raise NotImplementedError(
            f"Wan Taichu variant {name!r} belongs to a later migration phase; "
            "this builder currently supports ordinary bidirectional variants only"
        )
    return variant


def require_train_variant(name: str) -> WanTaichuVariant:
    variant = require_variant(name)
    if not variant.train_supported:
        raise NotImplementedError(
            f"Wan Taichu variant {name!r} belongs to a later migration phase; "
            "this builder currently supports bidirectional and block-causal teacher variants"
        )
    return variant


__all__ = [
    "WAN_TAICHU_VARIANTS",
    "WanTaichuArchitecture",
    "WanTaichuVariant",
    "require_ordinary_bidirectional_variant",
    "require_train_variant",
    "require_variant",
    "variant_names",
]
