"""Training-time condition dropout for Native Wan Taichu."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import torch

from vforge.tasks.wan_taichu.batch import fallback_stream_field, stream_field
from vforge.tasks.wan_taichu.variants import WanTaichuVariant


class ConditionDropoutPolicy:
    """Apply CFG dropout after batch canonicalization.

    Stream shapes stay fixed: a dropped stream is replaced by the VAE latent
    of its neutral pixel value, either read from cache or encoded once by a
    live frozen VAE. Text and auxiliary image features use zero tensors, which
    preserves the existing Taichu training contract.
    """

    def __init__(
        self,
        *,
        spec: WanTaichuVariant,
        streams: tuple[str, ...],
        extras: Mapping[str, Any],
    ) -> None:
        self.spec = spec
        self.streams = streams
        self.stream_probs = {name: _stream_drop_prob(name, extras) for name in streams}
        self.text_prob = float(extras.get("text_drop_prob", 0.0))
        self.image_prob = float(extras.get("rgb_cond_drop_prob", 0.0))
        self.audio_prob = float(extras.get("audio_drop_prob", 0.0))
        self.partial_normal = bool(extras.get("partial_normal_aug", False))
        self._live_fallbacks: dict[tuple[Any, ...], torch.Tensor] = {}

    @property
    def has_stream_dropout(self) -> bool:
        return any(value > 0.0 for value in self.stream_probs.values())

    def complete_missing_streams(
        self,
        bundle: Mapping[str, Any],
        *,
        encoder: Any | None,
    ) -> dict[str, Any]:
        """Fill an explicitly absent production-cache stream with its neutral latent.

        Some Taichu production sources intentionally cache ``None`` for a stream
        they do not own (Seedance has no gameplay grid, for example). Taichu
        completed that stream through its retained VAE before applying random
        dropout. Keep that source semantic in the Wan Taichu task rather than
        teaching the generic cache dataset about condition streams.
        """
        out = dict(bundle)
        reference = next(
            (
                out.get(stream_field(name))
                for name in self.streams
                if isinstance(out.get(stream_field(name)), torch.Tensor)
            ),
            None,
        )
        for name in self.streams:
            field = stream_field(name)
            if isinstance(out.get(field), torch.Tensor):
                continue
            fallback = out.get(fallback_stream_field(name))
            if fallback is None:
                if reference is None:
                    reference = self._reference_from_target(out)
                fallback = self._live_fallback(name, reference, encoder)
            out[field] = fallback
            if reference is None:
                reference = fallback
        return out

    def apply(self, bundle: Mapping[str, Any], *, encoder: Any | None) -> dict[str, Any]:
        out = dict(bundle)
        # Keep Taichu's draw order.  It is part of fixed-seed training parity:
        # reordering these independent Bernoulli draws changes which streams
        # are dropped even though their marginal probabilities stay equal.
        drop_rgb = self._drop_stream("rgb")
        drop_image = self.image_prob > 0.0 and random.random() < self.image_prob
        drop_normal = self._drop_stream("normal")
        drop_grid = self._drop_stream("grid")
        drop_text = self.text_prob > 0.0 and random.random() < self.text_prob
        audio_keep: dict[str, torch.Tensor] = {}
        for field in ("audio_features", "audio_features_l"):
            audio = out.get(field)
            if not isinstance(audio, torch.Tensor) or self.audio_prob <= 0.0:
                continue
            audio_keep[field] = torch.tensor(
                [random.random() >= self.audio_prob for _ in range(audio.shape[0])],
                device=audio.device,
                dtype=audio.dtype,
            ).view(audio.shape[0], 1, 1, 1)
        has_partial_normal = (
            self.partial_normal
            and "normal" in self.streams
            and isinstance(out.get("normal_fg_body"), torch.Tensor)
            and isinstance(out.get("normal_bg_body"), torch.Tensor)
        )
        partial_variant = random.randrange(3) if has_partial_normal else 0

        if drop_rgb:
            out["first_frame_latents"] = None
        if drop_text:
            out["prompt"] = torch.zeros_like(out["prompt"])
        if partial_variant:
            key = "normal_fg_body" if partial_variant == 1 else "normal_bg_body"
            body = out.get(key)
            assert isinstance(body, torch.Tensor)
            normal = out[stream_field("normal")]
            if self.spec.mask_channels:
                mask = torch.ones_like(normal[:, : self.spec.mask_channels])
                out[stream_field("normal")] = torch.cat([mask, body], dim=1)
            else:
                out[stream_field("normal")] = body
        out.pop("normal_fg_body", None)
        out.pop("normal_bg_body", None)
        for name, dropped in (("rgb", drop_rgb), ("grid", drop_grid), ("normal", drop_normal)):
            if not dropped:
                continue
            field = stream_field(name)
            fallback = out.get(fallback_stream_field(name))
            if fallback is None:
                fallback = self._live_fallback(name, out[field], encoder)
            out[field] = fallback
        if drop_image:
            for field in (
                "image_condition_latent_vae",
                "image_condition_latent_clip",
                "image_condition_latent_dino",
            ):
                if out.get(field) is not None:
                    out[field] = torch.zeros_like(out[field])
        for field, keep in audio_keep.items():
            audio = out.get(field)
            assert isinstance(audio, torch.Tensor)
            out[field] = audio * keep
        return out

    def _drop_stream(self, name: str) -> bool:
        probability = self.stream_probs.get(name, 0.0)
        return probability > 0.0 and random.random() < probability

    def _live_fallback(
        self,
        name: str,
        reference: torch.Tensor,
        encoder: Any | None,
    ) -> torch.Tensor:
        if encoder is None or getattr(encoder, "vae", None) is None:
            raise RuntimeError(
                f"dropping Wan Taichu stream {name!r} requires "
                f"{fallback_stream_field(name)!r} in the cache or a live VAE; "
                "zero tensors are not a valid fallback"
            )
        key = (
            name,
            tuple(reference.shape[1:]),
            reference.device,
            reference.dtype,
        )
        cached = self._live_fallbacks.get(key)
        if cached is None:
            cached = _encode_uniform_fallback(
                encoder,
                reference,
                pixel_value=0.5 if name == "normal" else 0.0,
                mask_channels=self.spec.mask_channels,
                mask_value=1.0 if name in {"grid", "normal"} else 0.0,
            )
            self._live_fallbacks[key] = cached[:1]
        return self._live_fallbacks[key].expand(reference.shape[0], -1, -1, -1, -1)

    def _reference_from_target(self, bundle: Mapping[str, Any]) -> torch.Tensor:
        frames = bundle.get("frame_latents")
        if not isinstance(frames, torch.Tensor) or frames.ndim != 5:
            raise RuntimeError(
                "filling a missing Wan Taichu condition stream requires a rank-5 "
                "frame_latents reference"
            )
        channels = self.spec.mask_channels + self.spec.z_dim
        return frames[:, :1].expand(-1, channels, -1, -1, -1)


def validate_dropout_config(
    spec: WanTaichuVariant,
    streams: tuple[str, ...],
    extras: Mapping[str, Any],
) -> None:
    """Validate condition-dropout knobs without allocating tensors or assets."""
    for name in streams:
        probability = _stream_drop_prob(name, extras)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name}_drop_prob must be in [0, 1]; got {probability}")
    for key in ("text_drop_prob", "rgb_cond_drop_prob"):
        probability = float(extras.get(key, 0.0))
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{key} must be in [0, 1]; got {probability}")
    audio_probability = float(extras.get("audio_drop_prob", 0.0))
    if not 0.0 <= audio_probability <= 1.0:
        raise ValueError(f"audio_drop_prob must be in [0, 1]; got {audio_probability}")
    if audio_probability > 0.0 and not spec.audio:
        raise ValueError("audio_drop_prob is only valid for an audio variant")
    stray = [
        f"{name}_drop_prob"
        for name in spec.allowed_condition_streams
        if name not in streams and float(extras.get(f"{name}_drop_prob", 0.0)) > 0.0
    ]
    if stray:
        raise ValueError(f"condition_streams={list(streams)} does not carry {stray}")
    if bool(extras.get("partial_normal_aug", False)) and "normal" not in streams:
        raise ValueError("partial_normal_aug requires the normal condition stream")


def _encode_uniform_fallback(
    encoder: Any,
    reference: torch.Tensor,
    *,
    pixel_value: float,
    mask_channels: int,
    mask_value: float,
) -> torch.Tensor:
    vae = encoder.vae
    z_dim = int(vae.config.z_dim)
    body_reference = reference[:, mask_channels:]
    if int(body_reference.shape[1]) != z_dim:
        raise ValueError(
            f"fallback reference has {body_reference.shape[1]} latent channels, expected {z_dim}"
        )
    t_lat, h_lat, w_lat = (int(value) for value in body_reference.shape[2:])
    scale_t = int(getattr(vae.config, "scale_factor_temporal", 4))
    scale_s = int(
        getattr(
            vae.config,
            "scale_factor_spatial",
            2 ** len(vae.temperal_downsample),
        )
    )
    pixels = torch.full(
        (1, 3, (t_lat - 1) * scale_t + 1, h_lat * scale_s, w_lat * scale_s),
        pixel_value,
        device=reference.device,
        dtype=encoder.training_dtype,
    )
    body = encoder.encode_image(encoder.scale_pixels(pixels))
    if mask_channels:
        mask = torch.full_like(body[:, :mask_channels], mask_value)
        body = torch.cat([mask, body], dim=1)
    if tuple(body.shape[1:]) != tuple(reference.shape[1:]):
        raise ValueError(
            f"uniform fallback shape {tuple(body.shape)} does not match {tuple(reference.shape)}"
        )
    return body.to(device=reference.device, dtype=reference.dtype)


def _stream_drop_prob(name: str, extras: Mapping[str, Any]) -> float:
    key = f"{name}_drop_prob"
    legacy_key = "rgb_first_frame_drop_prob" if name == "rgb" else key
    if name == "rgb" and key in extras and legacy_key in extras:
        raise ValueError(f"configure only one of {key} and {legacy_key}")
    return float(extras.get(key, extras.get(legacy_key, 0.0)))


__all__ = ["ConditionDropoutPolicy", "validate_dropout_config"]
