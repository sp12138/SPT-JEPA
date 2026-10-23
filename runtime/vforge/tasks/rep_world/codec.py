"""Load released frozen codec assets without importing upstream applications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields

import torch

from vforge.models.rep_world.codecs.codec import FrozenCodec
from vforge.models.rep_world.codecs.encoders import FrozenFeatureEncoder
from vforge.models.rep_world.codecs.videorae import VideoRAE
from vforge.models.rep_world.codecs.vrae import (
    VRAE,
    TemporalAttentionPool,
    VRAEDecoder,
    VRAEDecoderConfig,
)
from vforge.models.rep_world.config import codec_geometry


def build_videorae(model_config, variant):
    """Accept the active released geometry, not unused generic tokenizer args."""
    codec_geometry("videorae", variant)
    expected_name = (
        "autoencoder_vjepa21_3d_c64_repa" if variant == "3d" else "autoencoder_vfm_ae_c32_repa"
    )
    if model_config.get("name") != expected_name:
        raise ValueError("VideoRAE checkpoint model name does not match variant")
    args = model_config.get("args", {})
    common = {"input_size": 256}
    expected = (
        dict(
            latent_dim=64,
            latent_temporal=4,
            latent_spatial=16,
            vjepa21_num_frames=16,
            vjepa21_tubelet_size=2,
            vjepa21_patch_size=16,
            vjepa21_teacher_out_layers=[2, 5, 8, 11],
            compressor_model_size="small",
            decoder_model_size="base",
        )
        if variant == "3d"
        else dict(
            num_latent_tokens=512,
            latent_dim=32,
            vjepa2_img_size=256,
            vjepa2_num_frames=16,
            vjepa2_tubelet_size=2,
            vjepa2_patch_size=16,
        )
    )
    for key, value in {**common, **expected}.items():
        if key in args and args[key] != value:
            raise ValueError(f"Unsupported released VideoRAE {key}: {args[key]!r}")
    # bottleneck_token_num=1024 in the 1D file is an unused generic setting;
    # the active released tokenizer has exactly 512 latent slots of width32.
    return VideoRAE(variant)


def load_videorae(
    path,
    variant,
    *,
    device,
    dtype=torch.float32,
    statistics_path=None,
    encode_batch_size=1,
    decode_batch_size=1,
):
    """Load a trusted release, including its embedded frozen teacher weights."""
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    config = payload.get("model") if isinstance(payload, Mapping) else None
    if not isinstance(config, Mapping) or not isinstance(config.get("sd"), Mapping):
        raise ValueError("VideoRAE checkpoint requires model config and model.sd")
    with torch.device("meta"):
        model = build_videorae(config, variant)
    ignored = {
        "feature_aligner.proj.0.weight",
        "feature_aligner.proj.0.bias",
        "feature_aligner.proj.2.weight",
        "feature_aligner.proj.2.bias",
    }
    if variant == "3d":
        ignored.add("teacher_model.img_mod_embed")
    state = {key: value for key, value in config["sd"].items() if key not in ignored}
    model.load_state_dict(state, strict=True, assign=True)
    model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    mean = std = None
    if statistics_path:
        stats = torch.load(statistics_path, map_location="cpu", weights_only=True)
        mean, std = stats["mean"], stats["std"]
    return FrozenCodec(
        model,
        codec_geometry("videorae", variant),
        mean=mean,
        std=std,
        encode_batch_size=encode_batch_size,
        decode_batch_size=decode_batch_size,
    )


def build_vrae(config, variant):
    """Translate only the released model config; reject altered encoder semantics."""
    geometry = codec_geometry("vrae", variant)
    encoder = config["encoder"]
    expected = {
        "eupe": ("eupe", "vit_b16", [5, 6, 7, 8, 9, 10, 11], "sum", 12, 1),
        "vjepa2_1": (
            "vjepa2_1",
            "vit_l16",
            [11, 13, 15, 17, 19, 21, 23],
            "mean_plus_final_spatial_mean",
            24,
            2,
        ),
    }[variant]
    names = ("name", "variant", "layers", "fusion", "num_blocks", "encoder_tubelet_size")
    for key, value in zip(names, expected, strict=True):
        if encoder.get(key) != value:
            raise ValueError(f"Unsupported V-RAE encoder {key}: {encoder.get(key)!r}")
    if encoder.get("hidden_size") != geometry.channels or encoder.get("patch_size") != 16:
        raise ValueError("V-RAE encoder geometry differs from selected variant")
    if encoder.get("pixel_normalization") != "imagenet":
        raise ValueError("Unsupported V-RAE pixel normalization")
    pool = config["pooling"]
    if (
        pool.get("name") != "temporal_attention"
        or pool.get("output_norm_affine", False)
        or pool.get("group_size") * expected[-1] != 4
    ):
        raise ValueError("Unsupported V-RAE pooling contract")
    decoder = dict(config["decoder"])
    decoder = dict(decoder.get("parameters", decoder))
    if decoder.get("rope_theta", 10000.0) != 10000.0:
        raise ValueError("Released V-RAE RoPE requires theta=10000")
    # Frozen inference always has zero attention dropout. These upstream
    # execution/initialization fields are not new Native configuration knobs.
    for name in (
        "name",
        "init",
        "checkpoint",
        "attention_backend",
        "gradient_checkpointing",
        "rope_theta",
        "attention_dropout",
    ):
        decoder.pop(name, None)
    unknown = set(decoder) - {field.name for field in fields(VRAEDecoderConfig)}
    if unknown:
        raise ValueError(f"Unknown V-RAE decoder settings: {sorted(unknown)}")
    decoder["input_dim"] = geometry.channels
    if "image_size" in decoder:
        decoder["image_size"] = tuple(decoder["image_size"])
    return VRAE(
        FrozenFeatureEncoder(variant),
        TemporalAttentionPool(
            geometry.channels,
            pool["group_size"],
            pool.get("num_heads", 16),
            pool.get("use_time_bias", True),
        ),
        VRAEDecoder(VRAEDecoderConfig(**decoder)),
    )


def load_vrae(
    path,
    variant,
    *,
    device,
    dtype=torch.float32,
    statistics_path=None,
    encode_batch_size=1,
    decode_batch_size=1,
    independent_state_decode=False,
):
    """Load an explicitly supplied, trusted official training checkpoint.

    Official files contain non-tensor RNG/config state and require PyTorch's
    pickle loader. Never pass an untrusted downloaded file here.
    """
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("resolved_config"), Mapping):
        raise ValueError("V-RAE checkpoint lacks its resolved model configuration")
    config = payload["resolved_config"].get("model", payload["resolved_config"])
    # Allocate only the final storage; frozen assets need no random initializer.
    with torch.device("meta"):
        model = build_vrae(config, variant)
    raw = payload.get("model")
    shadow = payload.get("ema", {}).get("shadow")
    if not isinstance(raw, Mapping) or not isinstance(shadow, Mapping):
        raise ValueError("V-RAE checkpoint must contain model and EMA shadow mappings")
    # Remove only the proven-unused image branch. Every active parameter and
    # persistent buffer still loads strictly, with no arbitrary ignored keys.
    removed = (
        {
            "encoder.backbone.img_mod_embed",
            "encoder.backbone.patch_embed_img.proj.weight",
            "encoder.backbone.patch_embed_img.proj.bias",
        }
        if variant == "vjepa2_1"
        else set()
    )
    state = {key: value for key, value in raw.items() if key not in removed}
    expected_ema = {
        key
        for key, value in model.state_dict().items()
        if key.startswith(("temporal_pool.", "decoder.")) and value.is_floating_point()
    }
    if set(shadow) != expected_ema:
        raise ValueError("V-RAE EMA does not exactly match temporal_pool and decoder")
    # Do not hide a corrupt online state behind otherwise complete EMA values.
    model.load_state_dict(state, strict=True, assign=True)
    model.load_state_dict({**state, **shadow}, strict=True, assign=True)
    model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    mean = std = None
    if statistics_path:
        stats = torch.load(statistics_path, map_location="cpu", weights_only=True)
        mean, std = stats["mean"], stats["std"]
    return FrozenCodec(
        model,
        codec_geometry("vrae", variant),
        mean=mean,
        std=std,
        encode_batch_size=encode_batch_size,
        decode_batch_size=decode_batch_size,
        independent_state_decode=independent_state_decode,
    )
