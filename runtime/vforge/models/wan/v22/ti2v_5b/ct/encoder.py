"""Frozen condition encoder for Wan 2.2 TI2V.

Differences from the Wan 2.1 encoder, all of which follow from TI2V using
temporal in-painting instead of a mask channel:

* the first frame is returned as its own ``first_frame_latents`` product rather
  than being folded into a condition stream; the training step writes it over
  latent frame 0 and marks those tokens clean via ``timestep=0``;
* every condition stream is a plain ``z_dim``-channel latent with no leading
  mask block, because knownness is expressed by the timestep;
* there is no CLIP / DINO / VAE image side branch.

The task resolves a mode into an ordered ``stream_fields`` mapping and injects
the raw field mapping alongside it.  Each stream lands in its own encoded field
and the training step concatenates them in declared order.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from diffusers.video_processor import VideoProcessor

from vforge.models.wan.encoder import WanFrozenAssetEncoder


def apply_partial_normal_augmentation(
    bundle: dict[str, Any],
    variant: int,
    *,
    partial_normal_sources: tuple[str, ...],
    normal_field: str,
) -> dict[str, Any]:
    """Swap the normal stream for one of its alternate bodies.

    ``variant`` 0 keeps the full normal; 1 and 2 select the foreground- and
    background-only bodies. The bodies are plain ``z_dim`` latents like the
    stream they replace, because TI2V expresses knownness through the per-token
    timestep and so carries no mask prefix to rebuild.

    The sources are consumed here: leaving them in the bundle would let a later
    stage augment twice.
    """
    out = dict(bundle)
    if variant > 0:
        try:
            key = partial_normal_sources[variant - 1]
        except IndexError:
            raise ValueError(
                f"partial-normal variant {variant} is outside [0, {len(partial_normal_sources)}]"
            ) from None
        source = out.get(key)
        if source is None:
            raise ValueError(
                f"partial_normal_aug selected {key!r}, but the bundle carries no "
                "such body; the cache was written without the augmentation sources"
            )
        out[normal_field] = source
    for key in partial_normal_sources:
        out.pop(key, None)
    return out


def apply_encoded_training_dropouts(
    bundle: dict[str, Any],
    *,
    stream_fields: Mapping[str, str],
    stream_drop_probs: Mapping[str, float],
    text_drop_prob: float,
    partial_normal_aug: bool = False,
    partial_normal_sources: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Apply the TI2V CFG dropout policy to an encoded bundle.

    Each stream draws independently and is zeroed as a whole block, so the
    model sees every combination of present conditions. Taichu instead fills a
    dropped stream with the latent of a uniform-pixel clip, which needs a live
    VAE the offline cache path does not have; zero-fill keeps online and
    offline training byte-identical.

    The in-painted first frame is never dropped: it is the geometry anchor and
    is always visible at inference, so dropping it would train a condition the
    sampler never presents.

    Every decision is drawn before any is applied, and the partial-normal
    variant is drawn last, so the RNG stream matches Taichu's for a given set of
    probabilities. Augmenting before dropping is what makes a dropped normal win
    over the augmentation rather than the other way round.
    """
    drop_text = text_drop_prob > 0.0 and random.random() < text_drop_prob
    dropped = []
    for name in stream_fields:
        probability = float(stream_drop_probs.get(name, 0.0))
        if probability > 0.0 and random.random() < probability:
            dropped.append(name)
    variant = random.randrange(len(partial_normal_sources) + 1) if partial_normal_aug else 0

    out = dict(bundle)
    if drop_text:
        out["prompt"] = torch.zeros_like(out["prompt"])
    if partial_normal_aug:
        out = apply_partial_normal_augmentation(
            out,
            variant,
            partial_normal_sources=partial_normal_sources,
            normal_field=stream_fields["normal"],
        )
    for name in dropped:
        field = stream_fields[name]
        if out.get(field) is not None:
            out[field] = torch.zeros_like(out[field])
    return out


class Wan22Ti2v5bConditionEncoder(WanFrozenAssetEncoder):
    """VAE + UMT5 encoder shared by the train, prepare and eval paths."""

    def __init__(
        self,
        vae: nn.Module | None,
        text_encoder: nn.Module | None,
        tokenizer,
        video_processor: VideoProcessor | None,
        training_dtype: torch.dtype,
        stream_fields: Mapping[str, str],
        raw_stream_keys: Mapping[str, str],
        stream_drop_probs: Mapping[str, float] | None = None,
        text_drop_prob: float = 0.0,
    ) -> None:
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            video_processor=video_processor,
            training_dtype=training_dtype,
        )
        self.stream_fields = dict(stream_fields)
        self.streams = tuple(self.stream_fields)
        self.raw_stream_keys = dict(raw_stream_keys)
        missing_raw = sorted(
            name for name in self.streams if name != "rgb" and name not in raw_stream_keys
        )
        if missing_raw:
            raise ValueError(f"Wan condition streams have no raw pixel mapping: {missing_raw}")
        drop_probs = {name: float(value) for name, value in (stream_drop_probs or {}).items()}
        unknown = sorted(set(drop_probs) - set(self.streams))
        if unknown:
            raise ValueError(
                f"resolved Wan streams {list(self.streams)} have no {unknown} stream to drop"
            )
        for name, value in (*drop_probs.items(), ("text_drop_prob", text_drop_prob)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")

        self.stream_drop_probs = drop_probs
        self.text_drop_prob = float(text_drop_prob)

    def _stream_pixels(
        self,
        name: str,
        batch_data: Mapping[str, Any],
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """Pixel video ``[B, T, C, H, W]`` in ``[-1, 1]`` backing one stream.

        ``frames`` is the already-scaled target video, which the ``rgb`` stream
        derives from: the reference frame followed by a blank tail, matching
        what the sampler can supply.
        """
        if name == "rgb":
            reference = frames[:, 0:1]
            tail_shape = (frames.shape[0], frames.shape[1] - 1, *frames.shape[2:])
            return torch.cat([reference, reference.new_zeros(tail_shape)], dim=1)
        key = self.raw_stream_keys[name]
        pixels = batch_data.get(key)
        if pixels is None:
            raise ValueError(
                f"resolved Wan streams {list(self.streams)} need a {name!r} "
                f"stream, but the batch carries no {key!r} video"
            )
        return self.scale_pixels(pixels).to(frames.device, frames.dtype)

    @torch.no_grad()
    def encode_batch(self, batch_data: dict[str, Any]) -> dict[str, Any]:
        """Deterministically encode a raw batch into the canonical latent bundle.

        Pure in ``batch_data``: no RNG, no dropout. Training-time CFG dropout
        lives in :meth:`apply_training_dropouts` so prepare and online encoding
        produce byte-identical bundles.
        """
        if self.vae is None:
            raise RuntimeError(
                "Wan22Ti2v5bConditionEncoder.encode_batch requires a VAE; a raw batch "
                "reached a rank that only has the offline cache"
            )
        device = next(self.vae.parameters()).device
        dtype = self.training_dtype

        frames = self.scale_pixels(batch_data["video_frames"]).to(device, dtype)

        text_inputs = batch_data["text_inputs"].to(device).long()
        text_mask = batch_data["text_mask"].to(device).long()
        prompt = self.encode_prompt(text_inputs, text_mask, frames.shape[0])

        bundle: dict[str, Any] = {
            "prompt": prompt,
            # VAE wants [B, C, T, H, W]. A single pixel frame collapses to one
            # latent frame, which is exactly latent position 0.
            "frame_latents": self.encode_image(frames.permute(0, 2, 1, 3, 4)),
            "first_frame_latents": self.encode_image(frames[:, 0:1].permute(0, 2, 1, 3, 4)),
        }
        for name in self.streams:
            pixels = self._stream_pixels(name, batch_data, frames)
            bundle[self.stream_fields[name]] = self.encode_image(pixels.permute(0, 2, 1, 3, 4))
        return bundle

    @torch.no_grad()
    def encode_inference_batch(self, batch_data: dict[str, Any]) -> dict[str, Any]:
        """Encode TI2V reference/condition streams without target frame latents."""
        if self.vae is None:
            raise RuntimeError("Wan22 TI2V inference condition encoding requires a VAE")
        device = next(self.vae.parameters()).device
        dtype = self.training_dtype
        image = batch_data["image_condition"].to(device, dtype)
        num_frames = int(batch_data["num_frames"])
        text_inputs = batch_data["text_inputs"].to(device).long()
        text_mask = batch_data["text_mask"].to(device).long()
        prompt = self.encode_prompt(text_inputs, text_mask, int(image.shape[0]))
        first = self.scale_pixels(image).to(device, dtype)
        bundle: dict[str, Any] = {
            "prompt": prompt,
            "first_frame_latents": self.encode_image(first.permute(0, 2, 1, 3, 4)),
        }
        for name in self.streams:
            if name == "rgb":
                tail_shape = (first.shape[0], num_frames - 1, *first.shape[2:])
                pixels = torch.cat([first, first.new_zeros(tail_shape)], dim=1)
            else:
                key = self.raw_stream_keys[name]
                raw = batch_data.get(key)
                if raw is None:
                    raise ValueError(f"Wan22 TI2V inference requires a {name!r} condition video")
                pixels = self.scale_pixels(raw).to(device, dtype)
            bundle[self.stream_fields[name]] = self.encode_image(pixels.permute(0, 2, 1, 3, 4))
        return bundle

    def apply_training_dropouts(self, bundle: dict[str, Any]) -> dict[str, Any]:
        return apply_encoded_training_dropouts(
            bundle,
            stream_fields=self.stream_fields,
            stream_drop_probs=self.stream_drop_probs,
            text_drop_prob=self.text_drop_prob,
        )


__all__ = [
    "Wan22Ti2v5bConditionEncoder",
    "apply_encoded_training_dropouts",
    "apply_partial_normal_augmentation",
]
