"""Model-only dimensions; task paths and training lifecycle belong to tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodecGeometry:
    family: str
    variant: str
    rgb_frames: int
    channels: int
    planes: int
    grid: int = 16
    spatially_ordered: bool = True

    @property
    def packed_channels(self) -> int:
        return self.channels * self.planes

    @property
    def tokens_per_state(self) -> int:
        return self.planes * self.grid**2

    @property
    def temporal_compression(self) -> int | None:
        # The 1D token layout has no physical time axis.
        return self.rgb_frames // self.planes if self.spatially_ordered else None


CODECS = {
    ("videorae", "3d"): CodecGeometry("videorae", "3d", 16, 64, 4),
    ("videorae", "1d"): CodecGeometry("videorae", "1d", 16, 32, 2, spatially_ordered=False),
    ("vrae", "eupe"): CodecGeometry("vrae", "eupe", 4, 768, 1),
    ("vrae", "vjepa2_1"): CodecGeometry("vrae", "vjepa2_1", 4, 1024, 1),
    ("pixel", "rgb4"): CodecGeometry("pixel", "rgb4", 4, 3072, 1),
    ("pixel", "rgb16"): CodecGeometry("pixel", "rgb16", 16, 3072, 4),
    ("wan22", "ti2v5b"): CodecGeometry("wan22", "ti2v5b", 4, 48, 1),
}


def codec_geometry(family: str, variant: str) -> CodecGeometry:
    try:
        return CODECS[family, variant]
    except KeyError as exc:
        raise ValueError(f"Unsupported codec {family}/{variant}; choose {tuple(CODECS)}") from exc


@dataclass(frozen=True)
class WorldConfig:
    channels: int
    state_planes: int
    grid: int = 16
    width: int = 768
    depth: int = 12
    heads: int = 12
    patch_size: int = 1
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    adaln_mode: str = "adaln_lora"
    adaln_lora_dim: int = 256
    condition_width: int = 768
    condition_dropout: float = 0.1
    cross_attention_every: int = 1
    cross_attention_position: str = "after_self"
    cross_attention_scope: str = "global"
    cross_attention_3d_rope: bool = False
    condition_mode: str = "cross_attention"
    pixel_depth: int = 0
    pixel_width: int = 16

    def __post_init__(self):
        if (
            min(
                self.channels,
                self.state_planes,
                self.grid,
                self.width,
                self.depth,
                self.heads,
                self.patch_size,
            )
            < 1
        ):
            raise ValueError("World dimensions must be positive")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("World head dimension must be even and divide width")
        if self.grid % self.patch_size:
            raise ValueError("World spatial patch must divide latent grid")
        if self.adaln_mode not in {"adaln_lora", "fully_shared", "per_block"}:
            raise ValueError("Unknown AdaLN mode")
        if self.condition_width < 0 or not 0 <= self.condition_dropout <= 1:
            raise ValueError("Invalid condition width/dropout")
        if self.cross_attention_position not in {"before_self", "after_self"}:
            raise ValueError("cross_attention_position must be before_self or after_self")
        if self.condition_mode not in {"cross_attention", "entry_concat"}:
            raise ValueError("condition_mode must be cross_attention or entry_concat")
        if self.condition_mode == "entry_concat" and (
            self.patch_size != 1 or not self.condition_width
        ):
            raise ValueError("Entry concat requires patch_size=1 and a conditioner")
        if self.cross_attention_scope not in {"global", "same_spatial"}:
            raise ValueError("cross_attention_scope must be global or same_spatial")
        if self.cross_attention_scope == "same_spatial" and (
            self.condition_mode != "cross_attention"
            or self.patch_size != 1
            or not self.condition_width
        ):
            raise ValueError(
                "Same-spatial CA requires cross_attention, patch_size=1 and a conditioner"
            )
        if self.cross_attention_3d_rope and (
            self.condition_mode != "cross_attention"
            or self.patch_size != 1
            or not self.condition_width
        ):
            raise ValueError("3D CA RoPE requires cross_attention, patch_size=1 and a conditioner")
        if self.pixel_depth < 0 or self.pixel_width < 1 or self.pixel_width % 4:
            raise ValueError("Invalid PixelDiT depth/width")
        if self.pixel_depth and (self.patch_size != 1 or self.channels != 3072 or self.grid != 16):
            raise ValueError("PixelDiT requires RGB4-style losslessly packed RGB/patch16 geometry")
        if self.cross_attention_every < 1 or self.mlp_ratio <= 0 or self.adaln_lora_dim < 1:
            raise ValueError("Invalid World block dimensions")


@dataclass(frozen=True)
class ConditionerConfig:
    input: str = "rgb"
    input_size: int = 256
    temporal_patch: int = 2
    spatial_patch: int = 16
    width: int = 768
    depth: int = 12
    heads: int = 12
    feature_layers: tuple[int, ...] = (11,)
    mlp_ratio: float = 4.0
    # Upstream V-JEPA2.1 uses 256/patch_size for RoPE interpolation even in
    # the backbone named *_384. Do not infer this from the model name.
    pretrained_grid: int = 16
    context_mask_ratio: float = 0.0
    architecture: str = "causal"
    encoder_depth: int | None = None
    mask_stage: str = "after_e"
    vit_c_mlp_ratio: float | None = None

    def __post_init__(self):
        if self.vit_c_mlp_ratio is not None and self.vit_c_mlp_ratio <= 0:
            raise ValueError("vit_c_mlp_ratio must be positive")
        if self.vit_c_mlp_ratio is not None and self.architecture != "split_ec":
            raise ValueError("vit_c_mlp_ratio requires split_ec")
        if self.mask_stage not in {"after_e", "before_e"}:
            raise ValueError("mask_stage must be after_e or before_e")
        if self.mask_stage == "before_e" and self.architecture != "split_ec":
            raise ValueError("mask_stage=before_e requires architecture=split_ec")
        if self.architecture not in {"causal", "split_ec"}:
            raise ValueError("conditioner.architecture must be causal or split_ec")
        if self.encoder_depth is not None and (
            type(self.encoder_depth) is not int or not 1 <= self.encoder_depth < self.depth
        ):
            raise ValueError("encoder_depth must leave at least one layer in both ViT-E and ViT-C")
        if self.architecture == "causal" and self.encoder_depth is not None:
            raise ValueError("encoder_depth requires architecture=split_ec")
        if self.architecture == "split_ec":
            if self.depth < 2:
                raise ValueError("Split ViT requires total depth >= 2")
        if not 0 <= self.context_mask_ratio < 1:
            raise ValueError("context_mask_ratio must be in [0, 1)")
        if self.context_mask_ratio and self.input == "rgb" and self.architecture != "split_ec":
            raise ValueError("RGB context masking requires split_ec")
        if self.input not in {"rgb", "latent"}:
            raise ValueError("Conditioner input must be rgb or latent")
        if (
            min(
                self.input_size,
                self.spatial_patch,
                self.width,
                self.depth,
                self.heads,
                self.pretrained_grid,
            )
            < 1
        ):
            raise ValueError("Conditioner dimensions must be positive")
        if self.temporal_patch not in {2, 4}:
            raise ValueError("RGB temporal patch must be 2 or 4")
        if self.input == "latent" and self.temporal_patch != 2:
            raise ValueError("temporal_patch applies only to RGB; latent tokens are not repatched")
        if self.input_size % self.spatial_patch or self.width % self.heads:
            raise ValueError("Conditioner patch/head dimensions must divide input/width")
        if not self.feature_layers or len(set(self.feature_layers)) != len(self.feature_layers):
            raise ValueError("Feature layers must be nonempty and unique")
        if any(
            type(layer) is not int or not 0 <= layer < self.depth for layer in self.feature_layers
        ):
            raise ValueError("Feature layer index outside conditioner depth")
        if self.architecture == "split_ec" and any(
            i < self.vit_e_depth for i in self.feature_layers
        ):
            raise ValueError("Split feature_layers must select ViT-C layers in combined E+C depth")

    @property
    def vit_e_depth(self) -> int:
        if self.architecture == "causal":
            return 0
        return (
            self.encoder_depth if self.encoder_depth is not None else max(1, (self.depth + 2) // 4)
        )

    @property
    def vit_c_depth(self) -> int:
        return self.depth - self.vit_e_depth

    def tokens_per_state(self, geometry: CodecGeometry) -> int:
        if self.input == "latent":
            return geometry.tokens_per_state
        if geometry.rgb_frames % self.temporal_patch:
            raise ValueError("RGB state must contain complete temporal patches")
        return (
            geometry.rgb_frames
            // self.temporal_patch
            * (self.input_size // self.spatial_patch) ** 2
        )


@dataclass(frozen=True)
class RepaConfig:
    enabled: bool = False
    weight: float = 0.5
    align_layer: int = 1
    teacher_checkpoint: str = ""
    teacher_width: int = 768

    def __post_init__(self):
        if type(self.enabled) is not bool or self.weight < 0:
            raise ValueError("REPA requires a boolean enabled flag and nonnegative weight")
        if self.enabled and (not self.teacher_checkpoint or self.align_layer < 1):
            raise ValueError("REPA requires a teacher checkpoint and positive align_layer")
