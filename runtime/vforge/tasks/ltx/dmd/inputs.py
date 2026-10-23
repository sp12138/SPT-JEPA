"""LTX DMD geometry, conditioning, modality sharding and sigma conversion."""

from dataclasses import dataclass, replace
from functools import lru_cache

import torch

from vforge.algorithms.dmd.model import DmdNoiseLevel, PreparedDmdBatch
from vforge.distributed.ulysses import gather_sequence, pad_to_multiple, slice_sequence
from vforge.models.ltx.embeddings_processor import convert_to_additive_mask
from vforge.models.ltx.modality import Modality
from vforge.models.ltx.scheduler import ltx_sigmas
from vforge.tasks.ltx.train import audio_positions, patchify_video, video_positions


def shard_modality(modality: Modality, state) -> Modality:
    """Split A/V token fields, preserving replicated text and logical lengths."""
    size = state.config.usp_size
    if size == 1:
        return modality
    fields = {}
    for name, dim in (("latent", 1), ("timesteps", 1), ("positions", 2)):
        padded, _ = pad_to_multiple(getattr(modality, name), size, dim=dim)
        fields[name] = slice_sequence(padded, dim=dim, group=state.usp_group, gather_grad=False)
    return replace(
        modality,
        **fields,
        attention_mask=None,
        global_sequence_length=modality.latent.shape[1],
    )


def gather_prediction(value: torch.Tensor, modality: Modality, state) -> torch.Tensor:
    """Restore the full prediction and discard tail padding."""
    if state.config.usp_size == 1:
        return value
    # SUM backward cancels the USP factor in FSDP's DP*USP mean.
    return gather_sequence(value, dim=1, group=state.usp_group).narrow(
        1, 0, modality.latent.shape[1]
    )


@dataclass(frozen=True)
class BlockLayout:
    frames: int
    height: int
    width: int
    video_ranges: tuple[tuple[int, int], ...]
    audio_ranges: tuple[tuple[int, int], ...]

    @property
    def frame_tokens(self):
        return self.height * self.width


def block_layout(frames, height, width, fps, first=4, following=3):
    if min(frames, height, width, first, following) < 1 or fps <= 0:
        raise ValueError("invalid LTX rollout geometry")
    count = 1 + max(0, frames - first) // following
    frames = first + (count - 1) * following
    audio_first = round((1 + (first - 1) * 8) / fps * 25)
    audio_next = round(following * 8 / fps * 25)
    if min(audio_first, audio_next) < 1:
        raise ValueError("audio blocks must contain tokens")
    video_ranges, audio_ranges = [], []
    v = a = 0
    for index in range(count):
        nv = (first if index == 0 else following) * height * width
        na = audio_first if index == 0 else audio_next
        video_ranges.append((v, v + nv))
        audio_ranges.append((a, a + na))
        v, a = v + nv, a + na
    return BlockLayout(frames, height, width, tuple(video_ranges), tuple(audio_ranges))


@dataclass(frozen=True)
class LtxDmdInputs:
    layout: BlockLayout
    fps: float
    reference_video: torch.Tensor
    reference_audio: torch.Tensor
    reference_video_positions: torch.Tensor
    reference_audio_positions: torch.Tensor
    video_positions: torch.Tensor
    audio_positions: torch.Tensor
    video_context: torch.Tensor
    audio_context: torch.Tensor
    negative: dict[str, torch.Tensor]
    ref_follow: bool
    prefix: dict[str, torch.Tensor] | None = None
    prefix_blocks: int = 0
    score_window: bool = False
    segment_chain: bool = False


@torch.no_grad()
def prepare_batch(
    batch,
    device,
    *,
    config,
    connectors,
    negative,
    latent_frames=None,
    contexts=None,
    segment_slots=None,
):
    video = batch["video_latents"]
    latent = video["latents"].to(device=device, dtype=torch.bfloat16)
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    if latent.shape[0] != 1 or latent.shape[1] != 128:
        raise ValueError("LTX DMD expects one video with 128 latent channels")
    fps = float(video["fps"]) if video.get("fps") is not None else float(config["frame_rate"])
    frames = (
        min(latent.shape[2], int(config["max_train_frames"]) // 8 + 1)
        if latent_frames is None
        else latent_frames
    )
    if segment_slots is not None:
        frames = int(config["first_block_latent_frames"]) + (segment_slots - 1) * int(
            config["block_latent_frames"]
        )
    layout = block_layout(
        frames,
        latent.shape[3],
        latent.shape[4],
        fps,
        int(config["first_block_latent_frames"]),
        int(config["block_latent_frames"]),
    )
    ref_video = patchify_video(latent[:, :, :1]).clone()
    ref_audio = torch.zeros(
        1,
        int(float(config["ref_audio_seconds"]) * 25 + 0.5),
        128,
        device=device,
        dtype=latent.dtype,
    )
    vp = video_positions(
        num_frames=layout.frames,
        height=layout.height,
        width=layout.width,
        fps=fps,
        batch_size=1,
        device=device,
    )
    ap = audio_positions(num_time_steps=layout.audio_ranges[-1][1], batch_size=1, device=device)
    rv = vp[:, :, : layout.frame_tokens].clone()
    ra = audio_positions(num_time_steps=ref_audio.shape[1], batch_size=1, device=device)
    gap = float(config["ref_position_gap"])
    rv[:, 0] -= gap + rv[:, 0].max()
    ra[:, 0] -= gap + ra[:, 0].max()
    if contexts is None:
        cond = batch["conditions"]
        vf = cond["video_prompt_embeds"].to(device=device, dtype=latent.dtype)
        af = cond.get("audio_prompt_embeds", cond["video_prompt_embeds"]).to(
            device=device, dtype=latent.dtype
        )
        mask = cond["prompt_attention_mask"].to(device)
        if vf.ndim == 2:
            vf, af = vf.unsqueeze(0), af.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        # Reference condition preparation runs outside the model autocast region.
        # In particular, connector RMSNorm must not promote cached features to FP32.
        with torch.autocast(device.type, enabled=False):
            vc, ac, _ = connectors.create_embeddings(
                vf, af, convert_to_additive_mask(mask, latent.dtype)
            )
        vc, ac = vc.to(latent.dtype), ac.to(latent.dtype)
    else:
        vc, ac = (
            contexts[key].to(device=device, dtype=latent.dtype)
            for key in ("video_context", "audio_context")
        )
    inputs = LtxDmdInputs(
        layout,
        fps,
        ref_video,
        ref_audio,
        rv,
        ra,
        vp,
        ap,
        vc,
        ac,
        {key: value.to(device=device, dtype=latent.dtype) for key, value in negative.items()},
        bool(config["ref_follow"]),
        segment_chain=segment_slots is not None,
    )
    # Only shape and the separate first-frame reference are consumed from data.
    return PreparedDmdBatch(
        clean={
            "video": latent.new_zeros(1, layout.video_ranges[-1][1], 128),
            "audio": latent.new_zeros(1, layout.audio_ranges[-1][1], 128),
        },
        model_inputs=inputs,
    )


def modality(latent, sigma, positions, context, *, first_frame_tokens=0):
    sigma = sigma.float().reshape(-1)
    if sigma.numel() != 1:
        raise ValueError("LTX DMD uses one sigma per sample, shared by both modalities")
    timesteps = torch.full(
        latent.shape[:2], float(sigma.item()), device=latent.device, dtype=latent.dtype
    )
    if first_frame_tokens:
        timesteps[:, :first_frame_tokens] = 0
    return Modality(
        enabled=True,
        latent=latent,
        sigma=sigma.to(latent.dtype),
        timesteps=timesteps,
        positions=positions,
        context=context,
    )


@lru_cache(maxsize=2)
def _grid(steps):
    return ltx_sigmas(steps)


def _level(sigma, normalized):
    return DmdNoiseLevel(
        normalized,
        {"video": sigma.reshape(1, 1, 1), "audio": sigma.reshape(1, 1, 1)},
        sigma.reshape(()),
    )


def resolve_score_noise(_scheduler, normalized, _batch):
    indices = (normalized.float() * 1000).round().long().clamp(0, 1000)
    sigma = _grid(1000).flip(0).to(normalized.device)[indices]
    return _level(sigma, normalized)


def resolve_rollout_noise(_scheduler, normalized, _batch):
    grid = _grid(40).to(normalized.device)
    sigma = grid[(grid - normalized.float()).abs().argmin()]
    return _level(sigma, normalized)


def valid_masks(batch):
    mask = torch.ones(
        batch.clean["video"].shape[:2] + (1,), dtype=torch.bool, device=batch.clean["video"].device
    )
    if not batch.model_inputs.prefix_blocks or batch.model_inputs.score_window:
        mask[:, : batch.model_inputs.layout.frame_tokens] = False
    return {"video": mask, "audio": None}


def loss_partitions(batch):
    layout = batch.model_inputs.layout
    prefix = batch.model_inputs.prefix_blocks
    vo = layout.video_ranges[prefix][0]
    ao = layout.audio_ranges[prefix][0]
    return [
        {"video": slice(v[0] - vo, v[1] - vo), "audio": slice(a[0] - ao, a[1] - ao)}
        for v, a in zip(layout.video_ranges[prefix:], layout.audio_ranges[prefix:], strict=True)
    ]
