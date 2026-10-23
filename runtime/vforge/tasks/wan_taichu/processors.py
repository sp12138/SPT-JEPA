"""Wan Taichu-owned noising and training-input assembly.

The processors in this module translate one canonical encoded batch into the
exact inputs and targets consumed by :mod:`vforge.models.wan_taichu`.  They are
kept with the Taichu task instead of importing a versioned official-Wan task:
the two model families can now evolve or be removed independently.

Condition dropout is intentionally not implemented here.  Wan Taichu applies
its uniform-latent dropout policy once, after canonicalization, before calling
these processors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from vforge.tasks.wan.shared import WanConditionSpec
from vforge.tasks.wan_taichu.batch import stream_field
from vforge.utils.training import resolve_dtype


def _build_taichu_flow_match_scheduler(*, noise_shift: float):
    """Build the exact scheduler table used by Taichu's Wan training path.

    Diffusers' constructor uses ``linspace(1, 1/N, N)`` internally.  Taichu
    replaces that table with the diffsynth/Wan pre-shift sequence
    ``linspace(1, 0, N + 1)[:-1]``.  The two are close but not identical, so
    leaving the constructor table in place changes sampled timesteps and
    sigmas for a subset of otherwise identical training steps.
    """
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(shift=float(noise_shift))
    count = int(scheduler.config.num_train_timesteps)
    raw_sigmas = torch.linspace(1.0, 0.0, count + 1, dtype=torch.float32)[:-1]
    scheduler.set_timesteps(sigmas=raw_sigmas.tolist())
    return scheduler


def _build_taichu_bsmntw_weights(
    scheduler: Any,
    *,
    noise_shift: float,
) -> torch.Tensor:
    """Build Taichu's diffsynth middle-noise loss-weight table."""
    count = int(scheduler.config.num_train_timesteps)
    sigmas = scheduler.sigmas[:count]
    if float(noise_shift) == 1.0:
        return torch.ones_like(sigmas)
    positions = sigmas * count
    bell = torch.exp(-2 * ((positions - count / 2) / count) ** 2)
    shifted = bell - bell.min()
    return shifted * (count / shifted.sum())


def _resolve_loss_weight(
    *,
    weighting_scheme: str,
    bsmntw_weights: torch.Tensor | None,
    compute_loss_weighting: Any,
    index: torch.Tensor,
    sigma: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if weighting_scheme == "bsmntw":
        assert bsmntw_weights is not None
        return bsmntw_weights[index].to(device=device, dtype=torch.float32)
    return compute_loss_weighting(weighting_scheme, sigma)


class WanTaichuConditionProcessor:
    """Scalar-timestep flow matching used by Wan21-shaped variants."""

    def __init__(
        self,
        *,
        extras: Mapping[str, Any],
        precision: str,
        condition: WanConditionSpec,
    ) -> None:
        self._noise_shift = float(extras.get("noise_shift", 5.0))
        self._weight_mode = str(extras.get("weighting_scheme", "none"))
        self._dtype = resolve_dtype(precision)
        self._condition_spec = condition
        self._scheduler = None
        self._compute_loss_weighting = None
        self._bsmntw_weights = None

    def _ensure_diffusion_helpers(self) -> None:
        if self._scheduler is not None:
            return
        from diffusers.training_utils import compute_loss_weighting_for_sd3

        self._scheduler = _build_taichu_flow_match_scheduler(noise_shift=self._noise_shift)
        self._compute_loss_weighting = compute_loss_weighting_for_sd3
        if self._weight_mode == "bsmntw":
            self._bsmntw_weights = _build_taichu_bsmntw_weights(
                self._scheduler,
                noise_shift=self._noise_shift,
            )

    def process(
        self,
        encoded: dict[str, Any],
        *,
        step: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        apply_dropouts: bool = False,
    ) -> dict[str, Any]:
        if apply_dropouts:
            raise ValueError(
                "Wan Taichu dropout must run through ConditionDropoutPolicy "
                "before condition processing"
            )
        self._ensure_diffusion_helpers()
        assert self._scheduler is not None
        assert self._compute_loss_weighting is not None

        frame_latents = encoded["frame_latents"].to(device=device)
        condition = encoded.get("condition_latents_normal")
        if condition is not None:
            condition = condition.to(device=device)
        index = self._sample_timestep_index(
            step=step,
            device=device,
            generator=generator,
        )
        timesteps = self._scheduler.timesteps[index].to(device).unsqueeze(0)
        noise = torch.randn(
            frame_latents.shape,
            device=device,
            dtype=frame_latents.dtype,
            generator=generator,
        )
        sigma = self._scheduler.sigmas[index].to(device)
        loss_weights = _resolve_loss_weight(
            weighting_scheme=self._weight_mode,
            bsmntw_weights=self._bsmntw_weights,
            compute_loss_weighting=self._compute_loss_weighting,
            index=index,
            sigma=sigma,
            device=device,
        )
        noisy_latent = (1.0 - sigma) * frame_latents + noise * sigma
        hidden_states = (
            torch.cat(
                [
                    noisy_latent.to(dtype=self._dtype),
                    condition.to(dtype=self._dtype),
                ],
                dim=1,
            )
            if condition is not None
            else noisy_latent.to(dtype=self._dtype)
        )
        return {
            "model_inputs": {
                "hidden_states": hidden_states,
                "timestep": timesteps,
                "encoder_hidden_states": encoded["prompt"].to(
                    device=device,
                    dtype=self._dtype,
                ),
                "encoder_hidden_states_image_vae": _maybe_to(
                    encoded.get("image_condition_latent_vae"),
                    device,
                    self._dtype,
                ),
                "encoder_hidden_states_image_clip": _maybe_to(
                    encoded.get("image_condition_latent_clip"),
                    device,
                    self._dtype,
                ),
                "encoder_hidden_states_image_dino": _maybe_to(
                    encoded.get("image_condition_latent_dino"),
                    device,
                    self._dtype,
                ),
                "audio_input_s": _maybe_to(
                    encoded.get("audio_features"),
                    device,
                    self._dtype,
                ),
                "audio_input_l": _maybe_to(
                    encoded.get("audio_features_l"),
                    device,
                    self._dtype,
                ),
            },
            "targets": (noise - frame_latents).to(dtype=self._dtype),
            "loss_weights": loss_weights,
        }

    def _sample_timestep_index(
        self,
        *,
        step: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        del step
        assert self._scheduler is not None
        draw = torch.rand(1, device=device, generator=generator)
        return (draw.cpu() * self._scheduler.config.num_train_timesteps).long()[0]


class WanTaichuA14bConditionProcessor(WanTaichuConditionProcessor):
    """Range-first high/low expert timestep sampling for A14B."""

    def __init__(
        self,
        *,
        extras: Mapping[str, Any],
        precision: str,
        condition: WanConditionSpec,
        boundary_ratio: float,
        train_expert: str,
    ) -> None:
        super().__init__(extras=extras, precision=precision, condition=condition)
        self._boundary_ratio = boundary_ratio
        self._train_expert = train_expert
        self._patch_size = tuple(int(value) for value in extras.get("patch_size", (1, 2, 2)))

    def process(
        self,
        encoded: dict[str, Any],
        *,
        step: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        apply_dropouts: bool = False,
        patch_size: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        processed = super().process(
            encoded,
            step=step,
            device=device,
            generator=generator,
            apply_dropouts=apply_dropouts,
        )
        resolved_patch_size = tuple(int(value) for value in (patch_size or self._patch_size))
        model_inputs = processed["model_inputs"]
        model_inputs["ref_hidden_states"] = None
        model_inputs.pop("audio_input_l", None)
        model_inputs["timestep"] = build_per_token_timestep(
            encoded["frame_latents"],
            model_inputs["timestep"],
            resolved_patch_size,
            first_frame_zero=False,
        )
        return processed

    def _sample_timestep_index(
        self,
        *,
        step: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        assert self._scheduler is not None
        high = step % 2 == 0 if self._train_expert == "both" else self._train_expert == "high"
        boundary = self._boundary_ratio * self._scheduler.config.num_train_timesteps
        candidates = torch.nonzero(
            self._scheduler.timesteps >= boundary if high else self._scheduler.timesteps < boundary,
            as_tuple=False,
        ).flatten()
        if candidates.numel() == 0:
            raise ValueError("A14B selected timestep range is empty")
        draw = torch.rand(1, device=device, generator=generator)
        index = (draw.cpu() * candidates.numel()).long().clamp(max=candidates.numel() - 1)[0]
        return candidates[index]


class WanTaichuA14bTeacherForcingConditionProcessor(WanTaichuA14bConditionProcessor):
    """Build A14B's doubled sequence while preserving its expert sampler.

    A14B has no TI2V first-frame exception: the clean half is the original
    latent, every token in the noisy half receives the sampled timestep, and
    expert routing is based on that noisy half.
    """

    def process(
        self,
        encoded: dict[str, Any],
        *,
        step: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        apply_dropouts: bool = False,
        patch_size: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        processed = super().process(
            encoded,
            step=step,
            device=device,
            generator=generator,
            apply_dropouts=apply_dropouts,
            patch_size=patch_size,
        )
        frame_latents = encoded["frame_latents"].to(
            device=device,
            dtype=self._dtype,
        )
        condition = encoded.get("condition_latents_normal")
        clean = (
            torch.cat(
                [
                    frame_latents,
                    condition.to(device=device, dtype=self._dtype),
                ],
                dim=1,
            )
            if condition is not None
            else frame_latents
        )
        model_inputs = processed["model_inputs"]
        noisy = model_inputs["hidden_states"]
        noisy_timestep = model_inputs["timestep"]
        model_inputs["hidden_states"] = torch.cat([clean, noisy], dim=2)
        model_inputs["timestep"] = torch.cat(
            [torch.zeros_like(noisy_timestep), noisy_timestep],
            dim=1,
        )
        return processed


class WanTaichuTi2vConditionProcessor:
    """Per-token TI2V noising with a clean, in-painted first frame."""

    def __init__(
        self,
        *,
        extras: Mapping[str, Any],
        precision: str,
        streams: Sequence[str],
        ref_token: bool = False,
    ) -> None:
        self._noise_shift = float(extras.get("noise_shift", 5.0))
        self._weight_mode = str(extras.get("weighting_scheme", "none"))
        self._dtype = resolve_dtype(precision)
        self._streams = tuple(streams)
        self._ref_token = bool(ref_token)
        self._ref_num_frames_online = int(extras.get("ref_num_frames_online", 0))
        self._exclude_first_frame_from_loss = bool(
            extras.get("exclude_first_frame_from_loss", True)
        )
        self._patch_size = tuple(int(value) for value in extras.get("patch_size", (1, 2, 2)))
        self._scheduler = None
        self._compute_loss_weighting = None
        self._bsmntw_weights = None

    def _ensure_diffusion_helpers(self) -> None:
        if self._scheduler is not None:
            return
        from diffusers.training_utils import compute_loss_weighting_for_sd3

        self._scheduler = _build_taichu_flow_match_scheduler(noise_shift=self._noise_shift)
        self._compute_loss_weighting = compute_loss_weighting_for_sd3
        if self._weight_mode == "bsmntw":
            self._bsmntw_weights = _build_taichu_bsmntw_weights(
                self._scheduler,
                noise_shift=self._noise_shift,
            )

    def process(
        self,
        encoded: dict[str, Any],
        *,
        step: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        apply_dropouts: bool = False,
        patch_size: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        del step
        if apply_dropouts:
            raise ValueError(
                "Wan Taichu dropout must run through ConditionDropoutPolicy "
                "before condition processing"
            )
        self._ensure_diffusion_helpers()
        assert self._scheduler is not None
        assert self._compute_loss_weighting is not None

        frame_latents = encoded["frame_latents"].to(device=device)
        conditions = [
            self._require_stream(encoded, name).to(device=device, dtype=self._dtype)
            for name in self._streams
        ]
        first_frame_latents = encoded.get("first_frame_latents")
        if first_frame_latents is None:
            raise ValueError("Wan Taichu TI2V training requires first_frame_latents")
        first_frame_latents = first_frame_latents.to(device=device)

        draw = torch.rand(1, device=device, generator=generator)
        index = (draw.cpu() * self._scheduler.config.num_train_timesteps).long()[0]
        timestep = self._scheduler.timesteps[index].to(device)
        noise = torch.randn(
            frame_latents.shape,
            device=device,
            dtype=frame_latents.dtype,
            generator=generator,
        )
        sigma = self._scheduler.sigmas[index].to(device)
        noisy_latent = (1.0 - sigma) * frame_latents + noise * sigma
        noisy_latent[:, :, 0:1] = first_frame_latents.to(noisy_latent.dtype)

        resolved_patch_size = tuple(int(value) for value in (patch_size or self._patch_size))
        per_token_timestep = self._assemble_timestep(
            noisy_latent=noisy_latent,
            timestep=timestep,
            patch_size=resolved_patch_size,
        )
        hidden_states = self._assemble_hidden_states(
            noisy_latent=noisy_latent,
            frame_latents=frame_latents,
            first_frame_latents=first_frame_latents,
            conditions=conditions,
        )
        loss_weights = self._build_loss_weights(
            frame_latents,
            _resolve_loss_weight(
                weighting_scheme=self._weight_mode,
                bsmntw_weights=self._bsmntw_weights,
                compute_loss_weighting=self._compute_loss_weighting,
                index=index,
                sigma=sigma,
                device=device,
            ),
        )
        ref_hidden_states = self._resolve_ref_hidden_states(
            encoded,
            frame_latents=frame_latents,
            generator=generator,
        )
        model_inputs = {
            "hidden_states": hidden_states,
            "timestep": per_token_timestep,
            "encoder_hidden_states": encoded["prompt"].to(
                device=device,
                dtype=self._dtype,
            ),
        }
        if self._ref_token:
            model_inputs["ref_hidden_states"] = ref_hidden_states
        return {
            "model_inputs": model_inputs,
            "targets": (noise - frame_latents).to(dtype=self._dtype),
            "loss_weights": loss_weights,
        }

    def _resolve_ref_hidden_states(
        self,
        encoded: Mapping[str, Any],
        *,
        frame_latents: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor | None:
        if not self._ref_token:
            return None
        external = encoded.get("ref_frame_latents")
        if external is not None:
            if not isinstance(external, torch.Tensor):
                raise ValueError("ref_frame_latents must be a tensor")
            return external.to(device=frame_latents.device, dtype=self._dtype)
        sampled = sample_online_ref_frame_latents(
            frame_latents,
            max_frames=self._ref_num_frames_online,
            generator=generator,
        )
        return None if sampled is None else sampled.to(dtype=self._dtype)

    def _assemble_hidden_states(
        self,
        *,
        noisy_latent: torch.Tensor,
        frame_latents: torch.Tensor,
        first_frame_latents: torch.Tensor,
        conditions: list[torch.Tensor],
    ) -> torch.Tensor:
        del frame_latents, first_frame_latents
        return torch.cat([noisy_latent.to(dtype=self._dtype), *conditions], dim=1)

    def _assemble_timestep(
        self,
        *,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        patch_size: tuple[int, ...],
    ) -> torch.Tensor:
        return build_per_token_timestep(noisy_latent, timestep, patch_size)

    def _require_stream(self, encoded: Mapping[str, Any], name: str) -> torch.Tensor:
        field = stream_field(name)
        value = encoded.get(field)
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Wan Taichu TI2V with condition_streams={list(self._streams)} "
                f"requires tensor field {field!r}"
            )
        return value

    def _build_loss_weights(
        self,
        frame_latents: torch.Tensor,
        base_weight: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, frames, height, width = frame_latents.shape
        weight = base_weight.to(frame_latents.device, dtype=torch.float32).reshape(-1)
        if weight.numel() == 1:
            weight = weight.expand(batch)
        weight = (
            weight.view(batch, 1, 1, 1, 1)
            .expand(
                batch,
                channels,
                frames,
                height,
                width,
            )
            .clone()
        )
        if self._exclude_first_frame_from_loss and frames >= 1:
            weight[:, :, 0:1] = 0.0
        return weight


class WanTaichuTeacherForcingConditionProcessor(WanTaichuTi2vConditionProcessor):
    """Build a doubled ``[clean | noisy]`` TI2V sequence."""

    def _assemble_hidden_states(
        self,
        *,
        noisy_latent: torch.Tensor,
        frame_latents: torch.Tensor,
        first_frame_latents: torch.Tensor,
        conditions: list[torch.Tensor],
    ) -> torch.Tensor:
        clean_latent = frame_latents.clone()
        clean_latent[:, :, 0:1] = first_frame_latents.to(clean_latent.dtype)
        clean = torch.cat([clean_latent.to(dtype=self._dtype), *conditions], dim=1)
        noisy = torch.cat([noisy_latent.to(dtype=self._dtype), *conditions], dim=1)
        return torch.cat([clean, noisy], dim=2)

    def _assemble_timestep(
        self,
        *,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        patch_size: tuple[int, ...],
    ) -> torch.Tensor:
        return build_teacher_forcing_timestep(noisy_latent, timestep, patch_size)


def build_per_token_timestep(
    latent: torch.Tensor,
    scalar_timestep: torch.Tensor,
    patch_size: tuple[int, ...],
    *,
    first_frame_zero: bool = True,
) -> torch.Tensor:
    """Expand one timestep across tokens, optionally keeping frame zero clean."""
    batch, _, frames, height, width = latent.shape
    patch_t, patch_h, patch_w = patch_size
    post_patch_frames = frames // patch_t
    tokens_per_frame = (height // patch_h) * (width // patch_w)
    value = scalar_timestep.to(latent.device).reshape(())
    per_frame = torch.full(
        (post_patch_frames,),
        float(value),
        device=latent.device,
        dtype=torch.float32,
    )
    if first_frame_zero and post_patch_frames >= 1:
        per_frame[0] = 0.0
    per_token = per_frame.repeat_interleave(tokens_per_frame)
    return per_token.unsqueeze(0).expand(batch, -1).contiguous()


def build_teacher_forcing_timestep(
    latent: torch.Tensor,
    scalar_timestep: torch.Tensor,
    patch_size: tuple[int, ...],
) -> torch.Tensor:
    """Build ``[B, 2S]`` timesteps for a doubled clean/noisy sequence."""
    batch, _, frames, height, width = latent.shape
    patch_t, patch_h, patch_w = patch_size
    post_patch_frames = frames // patch_t
    tokens_per_frame = (height // patch_h) * (width // patch_w)
    value = scalar_timestep.to(latent.device).reshape(())
    clean = torch.zeros(post_patch_frames, device=latent.device, dtype=torch.float32)
    noisy = torch.full(
        (post_patch_frames,),
        float(value),
        device=latent.device,
        dtype=torch.float32,
    )
    noisy[0] = 0.0
    per_token = torch.cat([clean, noisy]).repeat_interleave(tokens_per_frame)
    return per_token.unsqueeze(0).expand(batch, -1).contiguous()


def select_ref_frame_indices(
    frame_count: int,
    count: int,
    *,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Select ordered reference slots with stratified random sampling."""
    if frame_count < 1:
        raise ValueError(f"frame_count must be positive; got {frame_count}")
    count = max(1, min(int(count), int(frame_count)))
    generator_device = generator.device if generator is not None else None
    segment = frame_count / count
    indices = []
    for index in range(count):
        lower = int(index * segment)
        upper = max(int((index + 1) * segment), lower + 1)
        offset = int(
            torch.randint(
                0,
                upper - lower,
                (1,),
                generator=generator,
                device=generator_device,
            ).item()
        )
        indices.append(lower + offset)
    return indices


def sample_online_ref_frame_latents(
    frame_latents: torch.Tensor,
    *,
    max_frames: int,
    generator: torch.Generator | None,
) -> torch.Tensor | None:
    """Sample Taichu-compatible GT latent slots for online ref-token training."""
    if max_frames <= 0 or frame_latents.ndim != 5:
        return None
    latent_frames = int(frame_latents.shape[2])
    max_count = min(latent_frames // 20, int(max_frames))
    if max_count <= 0:
        return None
    device = frame_latents.device
    count = int(
        torch.randint(
            0,
            max_count + 1,
            (1,),
            device=device,
            generator=generator,
        ).item()
    )
    if count == 0:
        return None
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(
        int(
            torch.randint(
                0,
                2**31 - 1,
                (1,),
                device=device,
                generator=generator,
            ).item()
        )
    )
    indices = select_ref_frame_indices(
        latent_frames,
        count,
        generator=cpu_generator,
    )
    index_tensor = torch.tensor(indices, device=device, dtype=torch.long)
    return frame_latents.index_select(2, index_tensor).contiguous()


def _maybe_to(value: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if value is None:
        return None
    return value.to(device=device, dtype=dtype)


__all__ = [
    "WanTaichuA14bConditionProcessor",
    "WanTaichuA14bTeacherForcingConditionProcessor",
    "WanTaichuConditionProcessor",
    "WanTaichuTeacherForcingConditionProcessor",
    "WanTaichuTi2vConditionProcessor",
    "build_per_token_timestep",
    "_build_taichu_bsmntw_weights",
    "build_teacher_forcing_timestep",
    "sample_online_ref_frame_latents",
    "select_ref_frame_indices",
]
