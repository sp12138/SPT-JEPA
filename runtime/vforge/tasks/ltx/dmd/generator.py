"""Model-only operations for LTX's two-modality cached rollout."""

from contextlib import contextmanager
from dataclasses import replace
from functools import partial

import torch

from vforge.algorithms.dmd.model import DmdGeneratorAdapter, DmdRolloutBlock
from vforge.algorithms.dmd.runtime import DmdRolloutSchedule, rollout_generator
from vforge.distributed.parallel_state import get_parallel_state
from vforge.models.ltx.kv_cache import CacheMode, ModelKVCache
from vforge.models.ltx.scheduler import to_denoised
from vforge.tasks.ltx.dmd.inputs import (
    gather_prediction,
    modality,
    resolve_rollout_noise,
    shard_modality,
)


def _forward_with_cache(generator, *, video, audio, cache, cache_mode):
    if cache_mode == CacheMode.PREFILL_REF:
        # The short reference has no gradients. Keep its projections and
        # attention shapes independent of USP, then retain only local KV
        # heads. This avoids seeding every block with a different reference
        # from small-row BF16 kernels and input-layout changes.
        output = generator(video=video, audio=audio, cache=cache, cache_mode=cache_mode)
        state = get_parallel_state()
        size = state.config.usp_size
        if size > 1:
            for layer in cache.layers:
                for entry, heads, dim in (
                    (layer.video_self, generator.num_attention_heads, generator.inner_dim),
                    (
                        layer.audio_self,
                        generator.audio_num_attention_heads,
                        generator.audio_inner_dim,
                    ),
                ):
                    entry.shard_reference_heads(
                        heads=heads, head_dim=dim // heads, size=size, rank=state.usp_rank
                    )
        return output
    state = get_parallel_state()
    outputs = generator(
        video=shard_modality(video, state),
        audio=shard_modality(audio, state),
        cache=cache,
        cache_mode=cache_mode,
    )
    # Prefill only updates KV; its output shards are not consumed.
    if cache_mode != CacheMode.DENOISE or state.config.usp_size == 1:
        return outputs
    return tuple(
        gather_prediction(value, full, state)
        for value, full in zip(outputs, (video, audio), strict=True)
    )


class LtxDmdGeneratorAdapter(DmdGeneratorAdapter):
    def __init__(self, *, kv_cache_blocks=0, num_sink_blocks=0, chain=None, base_path=None):
        if min(kv_cache_blocks, num_sink_blocks) < 0 or (num_sink_blocks and not kv_cache_blocks):
            raise ValueError("sink blocks require a positive KV window")
        self.kv_cache_blocks = kv_cache_blocks
        self.num_sink_blocks = num_sink_blocks
        self.chain = chain
        self.base_path = base_path
        self._handover = None

    def chain_asset_contract(self):
        import torch.distributed as dist

        from vforge.models.ltx.video_vae import video_vae_identity

        distributed = dist.is_available() and dist.is_initialized()
        payload = [None, None]
        if not distributed or dist.get_rank() == 0:
            try:
                payload[0] = video_vae_identity(self.base_path)
            except Exception as exc:
                payload[1] = f"{type(exc).__name__}: {exc}"
        if distributed:
            dist.broadcast_object_list(payload, src=0)
        if payload[1] is not None:
            raise RuntimeError(f"video VAE identity failed: {payload[1]}")
        return {
            "vae_sha256": payload[0],
            "handover": "conv_bf16_decode_last_block_last_frame_encode_v1",
        }

    def prepare_segment(self, batch, carry):
        if carry is None:
            return batch
        device = batch.clean["video"].device
        carry = {key: value.to(device) for key, value in carry.items()}
        prefix = {
            key: torch.cat([carry[f"sink_{key}"], carry[f"local_{key}"]], dim=1)
            for key in batch.clean
        }
        prefix.update({key: carry[key] for key in ("handover_video", "handover_audio")})
        count = self.chain.sink_blocks + self.chain.local_blocks
        layout = batch.model_inputs.layout
        offsets = {"video": layout.video_ranges[count][0], "audio": layout.audio_ranges[count][0]}
        if any(prefix[key].shape[1] != offsets[key] for key in offsets):
            raise ValueError("chain prefix does not fit the configured slots")
        return replace(
            batch,
            clean={key: value[:, offsets[key] :] for key, value in batch.clean.items()},
            model_inputs=replace(batch.model_inputs, prefix=prefix, prefix_blocks=count),
        )

    @torch.no_grad()
    def advance_segment(self, generated, batch, carry):
        inputs = batch.model_inputs
        full = (
            generated
            if inputs.prefix is None
            else {
                key: torch.cat([inputs.prefix[key], value], dim=1)
                for key, value in generated.items()
            }
        )
        result = {}
        for key, ranges in (
            ("video", inputs.layout.video_ranges),
            ("audio", inputs.layout.audio_ranges),
        ):
            result[f"sink_{key}"] = (
                (
                    full[key][:, : ranges[0][1]]
                    if carry is None
                    else carry[f"sink_{key}"].to(full[key].device)
                )
                .detach()
                .clone()
            )
            result[f"local_{key}"] = (
                full[key][:, ranges[-self.chain.local_blocks][0] :].detach().clone()
            )
        import torch.distributed as dist

        from vforge.data.loader import broadcast_batch

        state = get_parallel_state()
        source = state.config.usp_size == 1 or state.usp_rank == 0
        handover = None
        error = [None]
        if source:
            try:
                if self._handover is None:
                    from vforge.models.ltx.video_vae import VideoHandover

                    self._handover = VideoHandover(self.base_path)
                handover = self._handover(
                    full["video"][:, slice(*inputs.layout.video_ranges[-1])],
                    height=inputs.layout.height,
                    width=inputs.layout.width,
                )
            except Exception as exc:
                error[0] = f"{type(exc).__name__}: {exc}"
        if state.config.usp_size > 1:
            dist.broadcast_object_list(
                error, group=state.usp_group, src=dist.get_global_rank(state.usp_group, 0)
            )
        if error[0] is not None:
            raise RuntimeError(f"video handover failed: {error[0]}")
        result["handover_video"] = (
            broadcast_batch(
                handover,
                group=state.usp_group,
                is_source=source,
                target_device=full["video"].device,
            )
            if state.config.usp_size > 1
            else handover
        )
        result["handover_audio"] = full["audio"][:, -1:].detach().clone()
        return result

    @contextmanager
    def open_rollout_context(self, generator, batch):
        inputs = batch.model_inputs
        layout = inputs.layout
        sink_count = min(self.num_sink_blocks, len(layout.video_ranges) - 1)
        cache = ModelKVCache(
            len(generator.transformer_blocks),
            dtype=inputs.reference_video.dtype,
            video_window_tokens=self.kv_cache_blocks * max(b - a for a, b in layout.video_ranges),
            audio_window_tokens=self.kv_cache_blocks * max(b - a for a, b in layout.audio_ranges),
            video_sink_tokens=sum(b - a for a, b in layout.video_ranges[:sink_count]),
            audio_sink_tokens=sum(b - a for a, b in layout.audio_ranges[:sink_count]),
        )
        cache.set_cond_pe(
            *generator.compute_block_cond_pe(
                inputs.reference_video_positions,
                inputs.reference_audio_positions,
                0.0,
            )
        )
        zero = torch.zeros((), device=inputs.reference_video.device)
        with torch.no_grad():
            _forward_with_cache(
                generator,
                video=modality(
                    inputs.reference_video,
                    zero,
                    inputs.reference_video_positions,
                    inputs.video_context,
                ),
                audio=modality(
                    inputs.reference_audio,
                    zero,
                    inputs.reference_audio_positions,
                    inputs.audio_context,
                ),
                cache=cache,
                cache_mode=CacheMode.PREFILL_REF,
            )
        if inputs.prefix is not None:
            with torch.no_grad():
                from vforge.tasks.ltx.dmd.inputs import resolve_rollout_noise

                level = resolve_rollout_noise(None, zero, batch)
                for v, a in zip(
                    inputs.layout.video_ranges[: inputs.prefix_blocks],
                    inputs.layout.audio_ranges[: inputs.prefix_blocks],
                    strict=True,
                ):
                    self.commit_block(
                        generator,
                        (v, a),
                        {
                            "video": inputs.prefix["video"][:, slice(*v)],
                            "audio": inputs.prefix["audio"][:, slice(*a)],
                        },
                        level,
                        batch,
                        cache,
                    )
        yield cache

    def block_layout(self, batch, initial_noise):
        layout = batch.model_inputs.layout
        count = batch.model_inputs.prefix_blocks
        vo, ao = layout.video_ranges[count][0], layout.audio_ranges[count][0]
        return [
            DmdRolloutBlock(
                selection=(v, a),
                initial_value={
                    "video": initial_noise["video"][:, v[0] - vo : v[1] - vo],
                    "audio": initial_noise["audio"][:, a[0] - ao : a[1] - ao],
                },
            )
            for v, a in zip(layout.video_ranges[count:], layout.audio_ranges[count:], strict=True)
        ]

    def _forward(self, generator, values, selection, level, batch, context, mode):
        inputs = batch.model_inputs
        v, a = selection
        start = v[0] / inputs.layout.frame_tokens * 8 / inputs.fps if inputs.ref_follow else 0.0
        context.set_cond_pe(
            *generator.compute_block_cond_pe(
                inputs.reference_video_positions,
                inputs.reference_audio_positions,
                start,
            )
        )
        return _forward_with_cache(
            generator,
            video=modality(
                values["video"],
                level.model_timestep,
                inputs.video_positions[:, :, slice(*v)],
                inputs.video_context,
            ),
            audio=modality(
                values["audio"],
                level.model_timestep,
                inputs.audio_positions[:, :, slice(*a)],
                inputs.audio_context,
            ),
            cache=context,
            cache_mode=mode,
        )

    def denoise_block(self, generator, noisy_block, selection, noise_level, batch, context):
        video, audio = self._forward(
            generator, noisy_block, selection, noise_level, batch, context, CacheMode.DENOISE
        )
        return {
            "video": to_denoised(noisy_block["video"], video, noise_level.model_timestep),
            "audio": to_denoised(noisy_block["audio"], audio, noise_level.model_timestep),
        }

    def commit_block(
        self, generator, selection, context_value, context_noise_level, batch, context
    ):
        with torch.no_grad():
            self._forward(
                generator,
                {k: v.detach() for k, v in context_value.items()},
                selection,
                context_noise_level,
                batch,
                context,
                CacheMode.PREFILL_CLEAN,
            )

    def concat_blocks(self, outputs, batch):
        return {key: torch.cat([value[key] for _, value in outputs], dim=1) for key in batch.clean}


class ForwardView:
    """Model metadata plus a framework-owned forward callback, without re-registering parameters."""

    def __init__(self, model, forward):
        self.model = model
        self.forward = forward

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


@torch.no_grad()
def generate(model, batch, *, seed, steps, kv_cache_blocks=0, num_sink_blocks=0):
    reference = batch.clean["video"]
    random = torch.Generator(device=reference.device).manual_seed(seed)

    def noise_like(value):
        return torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=random)

    return rollout_generator(
        model,
        LtxDmdGeneratorAdapter(kv_cache_blocks=kv_cache_blocks, num_sink_blocks=num_sink_blocks),
        batch,
        batch.clean,
        resolve_noise_level=partial(resolve_rollout_noise, None),
        schedule=DmdRolloutSchedule(tuple(step / 1000 for step in steps)),
        phase="critic",
        noise_fn=noise_like,
        initial_block_noise_fn=noise_like,
        noise_dtype=torch.float32,
        clean_context=True,
    )
