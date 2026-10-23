"""Dual-scheduler MiniMax-H3 inference over a cached packed layout."""

from __future__ import annotations

from typing import Any

import torch

from vforge.models.minimax_h3.encoder import _patchify_video_latents


def _unpatchify_video_rows(
    rows: torch.Tensor,
    *,
    num_frames: int,
    latent_height: int,
    latent_width: int,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Invert H3 row patchification into channel-first video latents."""
    if rows.ndim == 2:
        rows = rows.unsqueeze(0)
    patch_t, patch_h, patch_w = patch_size
    patch_volume = patch_t * patch_h * patch_w
    channels = rows.shape[-1] // patch_volume
    expected_rows = (num_frames // patch_t) * (latent_height // patch_h) * (latent_width // patch_w)
    if rows.shape[1] != expected_rows or rows.shape[-1] % patch_volume:
        raise ValueError(
            "H3 video rows do not match latent geometry: "
            f"rows={tuple(rows.shape)} frames={num_frames} "
            f"height={latent_height} width={latent_width} patch={patch_size}"
        )
    latents = rows.reshape(
        rows.shape[0],
        num_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    return latents.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
        rows.shape[0],
        channels,
        num_frames,
        latent_height,
        latent_width,
    )


class MiniMaxH3Inference:
    def __init__(
        self,
        *,
        model,
        engine_forward,
        vae,
        audio_vae,
        num_inference_steps: int,
        video_shift: float,
        audio_shift: float,
        keyframe_noise_aug: float,
        pixel_mean: tuple[float, float, float],
        pixel_std: tuple[float, float, float],
        audio_sample_rate: int,
        patch_size: tuple[int, int, int],
        trace_path: str | None = None,
        fixed_seed: int | None = None,
    ) -> None:
        from vforge.models.minimax_h3.trace import H3TraceRecorder

        self.model = model
        self.engine_forward = engine_forward
        self.vae = vae
        self.audio_vae = audio_vae
        self.num_inference_steps = num_inference_steps
        self.video_shift = video_shift
        self.audio_shift = audio_shift
        self.keyframe_noise_aug = keyframe_noise_aug
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.audio_sample_rate = audio_sample_rate
        self.patch_size = patch_size
        self.trace = H3TraceRecorder(trace_path)
        self.fixed_seed = fixed_seed

    def resolve_seed(self, sample_id: str, default: int) -> int:
        del sample_id
        return int(self.fixed_seed) if self.fixed_seed is not None else int(default)

    @torch.no_grad()
    def infer_sample(self, sample: dict[str, Any], *, seed: int) -> dict[str, Any]:
        from diffusers import MiniMaxH3Scheduler
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3SetTimestepsStep,
        )
        from diffusers.video_processor import VideoProcessor

        device = next(self.model.parameters()).device
        generator = torch.Generator().manual_seed(seed)
        prompt = sample["prompt_embeds"].to(device)
        if prompt.ndim == 2:
            prompt = prompt.unsqueeze(0)
        self.trace.tensor("encode.prompt_embeds", prompt)
        num_frames = int(sample["num_latent_frames"])
        latent_height = int(sample["latent_height"])
        latent_width = int(sample["latent_width"])
        patch_t, patch_h, patch_w = self.patch_size
        patch_volume = patch_t * patch_h * patch_w
        channels = int(self.model.config.in_channels)
        video_row_width = channels * patch_volume
        audio_row_width = int(self.model.config.audio_in_channels)
        condition = sample["condition_video_rows"]
        num_condition_video = int(sample["num_condition_video_rows"])
        num_condition_audio = int(sample["num_condition_audio_rows"])
        rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
        condition_frames = num_condition_video // rows_per_frame * patch_t
        if condition_frames:
            condition_clean = _unpatchify_video_rows(
                condition,
                num_frames=condition_frames,
                latent_height=latent_height,
                latent_width=latent_width,
                patch_size=self.patch_size,
            )
            self.trace.tensor("init.condition_clean", condition_clean)
            condition_noise = torch.randn(
                condition_clean.shape,
                generator=generator,
                dtype=torch.float32,
            ).to(condition_clean.device)
            self.trace.tensor("init.condition_noise", condition_noise)
            condition_t = self.keyframe_noise_aug
            condition = _patchify_video_latents(
                condition_t * condition_clean.float() + (1.0 - condition_t) * condition_noise,
                self.patch_size,
            ).unsqueeze(0)
            self.trace.tensor("init.condition_rows", condition)
        else:
            condition = condition.reshape(1, 0, video_row_width)
        video_latents = torch.randn(
            (1, channels, num_frames, latent_height, latent_width),
            generator=generator,
            dtype=torch.float32,
        )
        self.trace.tensor("init.video_latents", video_latents)
        video = _patchify_video_latents(video_latents, self.patch_size).unsqueeze(0)
        self.trace.tensor("init.video_rows", video)
        audio = torch.randn(
            (1, int(sample["num_audio_latents"]) * 2, audio_row_width),
            generator=generator,
            dtype=torch.float32,
        )
        self.trace.tensor("init.audio_rows", audio)
        condition = condition.to(device)
        video = video.to(device)
        audio = audio.to(device)
        structural = {
            key: sample[key].to(device)
            for key in (
                "token_tags",
                "position_ids",
                "video_indices",
                "audio_indices",
                "text_indices",
            )
        }
        scheduler = MiniMaxH3Scheduler(shift=self.video_shift)
        audio_scheduler = MiniMaxH3Scheduler(shift=self.audio_shift)
        scheduler.set_timesteps(self.num_inference_steps, device=device)
        audio_scheduler.set_timesteps(self.num_inference_steps, device=device)
        self.trace.tensor("scheduler.video_timesteps", scheduler.timesteps)
        self.trace.tensor("scheduler.audio_timesteps", audio_scheduler.timesteps)

        for step, (timestep, audio_timestep) in enumerate(
            zip(
                scheduler.timesteps,
                audio_scheduler.timesteps,
                strict=True,
            )
        ):
            timesteps, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
                structural["video_indices"],
                structural["audio_indices"],
                num_condition_video,
                num_condition_audio,
                int(structural["text_indices"].numel()),
                float(timestep),
                float(audio_timestep),
                max(float(timestep), self.keyframe_noise_aug),
                1.0,
            )
            self.trace.tensor("denoise.row_timesteps", timesteps, step=step)
            self.trace.tensor("denoise.timestep_indices", timestep_indices, step=step)
            model_video = torch.cat([condition, video], dim=1)
            self.trace.tensor("denoise.model_in_video", model_video, step=step)
            self.trace.tensor("denoise.model_in_audio", audio, step=step)
            output = self.engine_forward(
                hidden_states=model_video,
                audio_hidden_states=audio,
                encoder_hidden_states=prompt,
                timestep=timesteps.to(device),
                timestep_indices=timestep_indices.to(device),
                **structural,
            )
            self.trace.tensor("denoise.model_out_video", output.sample, step=step)
            self.trace.tensor(
                "denoise.model_out_audio",
                output.audio_sample,
                step=step,
            )
            video = scheduler.step(
                output.sample[:, num_condition_video:],
                timestep,
                video,
            ).prev_sample
            self.trace.tensor("denoise.video_rows", video, step=step)
            audio = audio_scheduler.step(
                output.audio_sample[:, num_condition_audio:],
                audio_timestep,
                audio,
            ).prev_sample
            self.trace.tensor("denoise.audio_rows", audio, step=step)

        video = video.reshape(
            -1,
            num_frames // patch_t,
            latent_height // patch_h,
            latent_width // patch_w,
            channels,
            patch_t,
            patch_h,
            patch_w,
        )
        video = video.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
            -1,
            channels,
            num_frames,
            latent_height,
            latent_width,
        )
        self.trace.tensor("decode.video_latents", video)
        audio = audio.reshape(2, int(sample["num_audio_latents"]), audio.shape[-1])
        audio = audio.permute(0, 2, 1).contiguous()
        self.trace.tensor("decode.audio_latents", audio)

        from vforge.distributed.parallel_state import get_parallel_state

        if not get_parallel_state().is_batch_leader:
            return {
                "sample_id": sample.get("sample_id", "sample"),
                "seed": seed,
            }
        if self.vae is None or self.audio_vae is None:
            raise RuntimeError("the H3 batch leader requires VAE assets for decoding")

        video_mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        video_std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        video_vae_input = video * video_std + video_mean
        self.trace.tensor("decode.video_vae_input", video_vae_input)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            decoded_video = self.vae.decode(video_vae_input, return_dict=False)[0]
        self.trace.tensor("decode.video_raw", decoded_video)
        pixel_mean = torch.tensor(self.pixel_mean, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(self.pixel_std, device=device).view(1, -1, 1, 1, 1)
        decoded_video = (decoded_video.float() * pixel_std + pixel_mean).clamp(0, 1)
        self.trace.tensor("decode.video_pixels", decoded_video)
        frames = VideoProcessor(
            vae_scale_factor=16,
            do_normalize=False,
        ).postprocess_video(decoded_video, output_type="pil")[0]

        audio_mean = torch.tensor(self.audio_vae.config.latents_mean, device=device).view(1, -1, 1)
        audio_std = torch.tensor(self.audio_vae.config.latents_std, device=device).view(1, -1, 1)
        decoded_audio = self.audio_vae.decode(
            audio * audio_std + audio_mean,
            return_dict=False,
        )[0]
        self.trace.tensor("decode.audio_raw", decoded_audio)
        decoded_audio = decoded_audio.float().permute(1, 0, 2)
        return {
            "sample_id": sample.get("sample_id", "sample"),
            "seed": seed,
            "frames": frames,
            "audio": decoded_audio[0].cpu(),
            "sampling_rate": self.audio_sample_rate,
        }


__all__ = ["MiniMaxH3Inference"]
