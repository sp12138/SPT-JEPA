"""Explicit weights-only import of the original two-tower training payload.

No optimizer, EMA, auxiliary predictor, step or data cursor is imported. Native
DCP remains the sole full-resume format. Only proven-unused ViT image branches
are discarded; the pure-TF derived RoPE cache is resized to the Native window.
Every active learned tensor is validated strictly.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from vforge.models.rep_world.model import RepWorld


def _strict_model(state, config):
    with torch.device("meta"):
        model = RepWorld(config.geometry, config.world, config.conditioner, config.flow)
    model.load_state_dict(state, strict=True, assign=True)
    return model


def load_native_model(path, config):
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, Mapping) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError(
            "Native initial weights must be a flat tensor state dict; "
            "use legacy for old training payloads"
        )
    return _strict_model(state, config)


def load_legacy_model(path, config):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise ValueError("Legacy RepWorld checkpoint requires model and meta mappings")
    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError(
            "Legacy checkpoint lacks model metadata; refusing implicit geometry guesses"
        )
    geo = config.geometry
    expected = dict(
        latent_codec=geo.family, rae_image_latent_channels=geo.channels, rae_group_size=geo.planes
    )
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Legacy {key}={meta.get(key)!r} does not match requested {value!r}")
    if meta.get("next_latent_use_full_history", False) and meta.get(
        "next_latent_use_previous_context", False
    ):
        raise ValueError(
            "Legacy checkpoint cannot enable both previous-state and full-history context"
        )
    mode = (
        "pure_tf"
        if meta.get("next_latent_use_full_history", False)
        else "zprev"
        if meta.get("next_latent_use_previous_context", False)
        else "no_zprev"
    )
    if mode != config.flow.mode:
        raise ValueError(f"Legacy flow mode {mode} does not match {config.flow.mode}")
    state = dict(payload["model"])
    if mode == "pure_tf":
        # The source precomputes a full training-window RoPE cache; Native
        # precomputes one state and builds longer/offset positions on demand.
        # These are derived buffers, not learned weights. Preserve the saved
        # base frequencies and the exact first-state cache values.
        tokens_per_plane = (geo.grid // config.world.patch_size) ** 2
        head_dim = config.world.width // config.world.heads
        native_tokens = geo.planes * tokens_per_plane
        for name in ("net.feat_rope.freqs_cos", "net.feat_rope.freqs_sin"):
            value = state.get(name)
            expected = (config.states * native_tokens, head_dim)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
                raise ValueError(f"Legacy pure-TF {name} must match training window {expected}")
            state[name] = value[:native_tokens].clone()
    condition = config.conditioner
    if condition is not None:
        source = payload.get("causal_semantic_encoder")
        if not isinstance(source, Mapping):
            raise ValueError("Legacy checkpoint lacks causal_semantic_encoder weights")
        if meta.get("causal_encoder_input", "rgb") != condition.input:
            raise ValueError(
                "Legacy conditioner input domain does not match requested configuration"
            )
        if meta.get("semantic_input_size", 224) != condition.input_size:
            raise ValueError("Legacy semantic_input_size does not match requested conditioner")
        if (
            condition.input == "rgb"
            and meta.get("causal_encoder_tubelet_size", 2) != condition.temporal_patch
        ):
            raise ValueError("Legacy temporal patch does not match requested conditioner")
        unused = {
            "encoder.img_mod_embed",
            "encoder.patch_embed_img.proj.weight",
            "encoder.patch_embed_img.proj.bias",
        }
        if condition.input == "latent":
            unused.add("encoder.video_mod_embed")
        state.update(
            {
                f"causal_semantic_encoder.{key}": value
                for key, value in source.items()
                if key not in unused
            }
        )
    model = _strict_model(state, config)
    if condition is not None:
        # The original sampler maps absent/None/empty metadata to the last
        # encoder layer, not to the full set of available hierarchical norms.
        layers = meta.get("causal_encoder_feature_layers") or [condition.depth - 1]
        if tuple(layers) != condition.feature_layers:
            raise ValueError("Legacy feature_layers do not match requested conditioner")
    return model
