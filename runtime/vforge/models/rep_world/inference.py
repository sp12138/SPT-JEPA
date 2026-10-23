"""Strict-causal state rollout through the Native engine_forward boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from vforge.models.rep_world.model import pack_states, shift_timestep, unpack_states


def pyramid_window_starts(states: int, steps: int, offset: int, window: int) -> tuple[int, ...]:
    """Start each state as early as both its offset and a window slot allow."""
    if min(states, steps, window) < 1 or not 0 <= offset <= steps:
        raise ValueError("Pyramid states, steps and window must be positive; offset in [0, steps]")
    starts = []
    for index in range(states):
        desired = index * offset
        if index >= window:
            desired = max(desired, starts[index - window] + steps)
        starts.append(desired)
    return tuple(starts)


@dataclass(frozen=True)
class RolloutConfig:
    total_states: int
    history_states: int = 1
    history_window_states: int = 20
    trained_states: int = 20
    steps: int = 50
    cfg_scale: float = 2.0
    cfg_interval_min: float = 0.2
    cfg_interval_max: float = 1.0
    batch_size: int = 1
    use_vit_c_kv_cache: bool = False
    df_pyramid_offset: int | None = None
    df_pyramid_window_states: int | None = None

    def __post_init__(self):
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("rollout.batch_size must be a positive integer")
        if not 1 <= self.history_states < self.total_states:
            raise ValueError("Rollout requires 1 <= history_states < total_states")
        if min(self.steps, self.history_window_states, self.trained_states) < 1:
            raise ValueError("Rollout steps and windows must be positive")
        if not math.isfinite(self.cfg_scale) or self.cfg_scale < 1:
            raise ValueError("CFG scale must be finite and >= 1")
        if not 0 <= self.cfg_interval_min <= self.cfg_interval_max <= 1:
            raise ValueError("CFG interval must be within [0,1]")
        if type(self.use_vit_c_kv_cache) is not bool:
            raise ValueError("rollout.use_vit_c_kv_cache must be a boolean")
        if self.df_pyramid_offset is not None and (
            isinstance(self.df_pyramid_offset, bool)
            or not isinstance(self.df_pyramid_offset, int)
            or not 0 <= self.df_pyramid_offset <= self.steps
        ):
            raise ValueError("rollout.df_pyramid_offset must be an integer in [0, steps]")
        if self.df_pyramid_window_states is not None and (
            isinstance(self.df_pyramid_window_states, bool)
            or not isinstance(self.df_pyramid_window_states, int)
            or self.df_pyramid_window_states < 1
            or self.df_pyramid_offset is None
        ):
            raise ValueError("rollout.df_pyramid_window_states requires a positive window and offset")


class RepWorldInference:
    """Frozen codec calls are local; EVERY trainable forward uses the engine.

    ``rgb`` may contain GT future frames for output comparison, but only its
    observed prefix is passed to the codec or the trainable conditioner.
    No predictor, NextLat state, or GT future latent is used in rollout.
    """

    def __init__(
        self,
        engine_forward,
        codec,
        geometry,
        flow,
        conditioner,
        timestep_shift,
        config: RolloutConfig,
    ):
        self.engine_forward, self.codec, self.geometry = engine_forward, codec, geometry
        self.flow, self.conditioner, self.timestep_shift, self.config = (
            flow,
            conditioner,
            timestep_shift,
            config,
        )
        if flow.mode in {"pure_tf", "pure_df"}:
            if config.total_states > config.trained_states:
                raise ValueError("Pure TF/DF rollout cannot exceed its trained state window")
            if config.cfg_scale != 1:
                raise ValueError("Pure TF/DF has no semantic CFG; set cfg_scale=1")
        elif conditioner is None:
            raise ValueError("Semantic rollout needs a conditioner configuration")
        if config.df_pyramid_offset is not None and flow.mode != "pure_df":
            raise ValueError("Pyramid rollout requires flow.mode=pure_df")
        if (
            config.use_vit_c_kv_cache
            and config.history_window_states < config.total_states - 1
        ):
            raise ValueError("ViT-C KV cache currently requires full rollout history")

    def _schedule(self, reference):
        return shift_timestep(
            torch.linspace(
                1, 0, self.config.steps + 1, device=reference.device, dtype=reference.dtype
            ),
            self.timestep_shift,
        )

    def _denoise(self, noise, condition, previous):
        cfg = self.config
        z = noise.clone()
        schedule = self._schedule(condition)
        drop = torch.ones(z.shape[0], dtype=torch.bool, device=z.device)
        for t_now, t_next in zip(schedule[:-1], schedule[1:], strict=True):
            t = t_now.expand(z.shape[0], z.shape[2])
            conditional = self.engine_forward(
                operation="velocity", noisy=z, timestep=t, condition=condition, previous=previous
            )
            if cfg.cfg_scale > 1:
                unconditional = self.engine_forward(
                    operation="velocity",
                    noisy=z,
                    timestep=t,
                    condition=condition,
                    previous=previous,
                    condition_drop=drop,
                )
                # Preserve (low,high], including t=0 only when low=0.
                active = (t_now <= cfg.cfg_interval_max) & (
                    (cfg.cfg_interval_min == 0) | (t_now > cfg.cfg_interval_min)
                )
                scale = torch.where(
                    active, t_now.new_tensor(cfg.cfg_scale), t_now.new_tensor(1)
                ).reshape(1, 1, 1, 1, 1)
                velocity = unconditional + scale * (conditional - unconditional)
            else:
                velocity = conditional
            # A 0-D fp32 scalar times a bf16 velocity stays bf16 in PyTorch.
            # Preserve the source's broadcast-tensor CFG/Euler arithmetic.
            dt = (t_next - t_now).reshape(1, 1, 1, 1, 1)
            z = z - dt * velocity
        return z

    @torch.no_grad()
    def infer_sample(self, sample, *, seed):
        packed, predicted_rgb, observed_frames = self._rollout(sample["rgb"], seed)
        return self._result(sample["sample_id"], seed, packed, predicted_rgb, observed_frames)

    @torch.no_grad()
    def infer_batch(self, samples, *, seeds):
        """Roll out independent clips together without sharing random streams."""
        if not samples or len(samples) != len(seeds):
            raise ValueError("infer_batch requires one seed for each nonempty sample")
        observed_frames = self.config.history_states * self.geometry.rgb_frames
        prefixes = []
        for sample in samples:
            rgb = sample["rgb"]
            if (
                rgb.ndim != 5
                or rgb.shape[0] != 1
                or rgb.shape[1] != 3
                or rgb.shape[2] < observed_frames
            ):
                raise ValueError("Each batched sample must contain one RGB observed prefix")
            prefixes.append(rgb[:, :, :observed_frames])
        rgb = torch.cat(prefixes, dim=0)
        packed, predicted_rgb, observed_frames = self._rollout(rgb, list(seeds))
        return [
            self._result(
                sample["sample_id"],
                seed,
                packed[index : index + 1],
                predicted_rgb[index : index + 1],
                observed_frames,
            )
            for index, (sample, seed) in enumerate(zip(samples, seeds, strict=True))
        ]

    def _result(self, sample_id, seed, packed, predicted_rgb, observed_frames):
        cfg = self.config
        return {
            "sample_id": str(sample_id),
            "prediction": predicted_rgb,
            "latents": packed,
            "observed_frames": observed_frames,
            "metadata": {
                "seed": seed,
                "steps": cfg.steps,
                "cfg_scale": cfg.cfg_scale,
                "total_states": cfg.total_states,
                "history_states": cfg.history_states,
                "df_pyramid_offset": cfg.df_pyramid_offset,
                "df_pyramid_window_states": cfg.df_pyramid_window_states,
                "condition_input": None if self.conditioner is None else self.conditioner.input,
            },
        }

    def _rollout(self, rgb, seed):
        cfg, geo = self.config, self.geometry
        observed_frames = cfg.history_states * geo.rgb_frames
        if rgb.ndim != 5 or rgb.shape[1] != 3 or rgb.shape[2] < observed_frames:
            raise ValueError("RGB sample does not contain the requested observed prefix")
        observed_rgb = rgb[:, :, :observed_frames].clone()
        # Keep each case's frozen-codec invocation shape identical to singleton
        # evaluation; batching targets the trainable ViT and DiT.
        observed = (
            torch.cat([self.codec.encode(clip) for clip in observed_rgb.split(1)], dim=0)
            if isinstance(seed, list)
            else self.codec.encode(observed_rgb)
        )
        if isinstance(seed, list):
            generator = [
                torch.Generator(device=observed.device).manual_seed(value) for value in seed
            ]
        else:
            generator = torch.Generator(device=observed.device).manual_seed(seed)
        if self.flow.mode in {"pure_tf", "pure_df"}:
            packed = (
                self._pure_df_pyramid(observed, generator)
                if self.flow.mode == "pure_df" and cfg.df_pyramid_offset is not None
                else self._pure_tf(observed, generator)
            )
            stream = self.codec.stream()
            stream.prime(observed)
            # Full-attention decoders see only the current generated prefix.
            parts = [observed_rgb]
            for index in range(cfg.history_states, cfg.total_states):
                parts.append(stream.step(packed[:, :, index : index + 1]))
            predicted_rgb = torch.cat(parts, 2)
        else:
            packed, predicted_rgb = self._semantic(observed, observed_rgb, generator)
        return packed, predicted_rgb, observed_frames

    @staticmethod
    def _noise(shape, *, generator, device, dtype):
        if isinstance(generator, torch.Generator):
            return torch.randn(shape, generator=generator, device=device, dtype=dtype)
        if len(generator) != shape[0]:
            raise ValueError("Noise batch must match the number of independent generators")
        return torch.cat(
            [
                torch.randn((1, *shape[1:]), generator=stream, device=device, dtype=dtype)
                for stream in generator
            ],
            dim=0,
        )

    def _semantic(self, observed, observed_rgb, generator):
        geo, cfg = self.geometry, self.config
        packed, rgb = observed.clone(), observed_rgb.clone()
        stream = self.codec.stream()
        stream.prime(observed)
        batch = observed.shape[0]
        split_ec = getattr(self.conditioner, "architecture", "causal") == "split_ec"
        cached_vit_c = split_ec and cfg.use_vit_c_kv_cache
        encoded = None
        vit_c_cache = None
        if split_ec:
            source = (
                packed[:, :, -cfg.history_window_states :]
                if self.conditioner.input == "latent"
                else rgb[:, :, -cfg.history_window_states * geo.rgb_frames :]
            )
            encoded = self.engine_forward(operation="encode_states", source=source)
            if cached_vit_c:
                condition, vit_c_cache = self.engine_forward(
                    operation="prefill_condition_cached",
                    source=encoded,
                    cache_start_state=cfg.history_states - encoded.shape[1],
                )
        for index in range(cfg.history_states, cfg.total_states):
            if split_ec:
                if not cached_vit_c:
                    condition = self.engine_forward(operation="prefill_condition", source=encoded)
            elif self.conditioner.input == "latent":
                source = packed[:, :, -cfg.history_window_states :]
            elif not split_ec:
                source = rgb[:, :, -cfg.history_window_states * geo.rgb_frames :]
            if not split_ec:
                condition = self.engine_forward(operation="condition", source=source)
            noise = self._noise(
                (batch, geo.channels, geo.planes, geo.grid, geo.grid),
                generator=generator,
                device=condition.device,
                dtype=condition.dtype,
            )
            previous = None
            if self.flow.mode == "zprev":
                previous = (
                    unpack_states(packed[:, :, -1:], geo)[:, 0].contiguous().to(condition.dtype)
                )
            state = self._denoise(noise, condition, previous)
            next_packed = pack_states(state[:, None]).contiguous()
            packed = torch.cat((packed, next_packed), 2)
            # Decoded pixels enter the next conditioner only for RGB input.
            # Latent input retains generated latents directly; no re-encoding.
            rgb = torch.cat((rgb, stream.step(next_packed)), 2)
            if split_ec and index + 1 < cfg.total_states:
                source = (
                    next_packed
                    if self.conditioner.input == "latent"
                    else rgb[:, :, -geo.rgb_frames :]
                )
                latest = self.engine_forward(operation="encode_states", source=source)
                if cached_vit_c:
                    condition, vit_c_cache = self.engine_forward(
                        operation="prefill_condition_cached",
                        source=latest,
                        cache=vit_c_cache,
                        cache_start_state=index,
                    )
                else:
                    encoded = torch.cat((encoded, latest), dim=1)[:, -cfg.history_window_states :]
        return packed, rgb

    def _pure_df_pyramid(self, observed, generator):
        """Denoise all future states together with state-causal visibility.

        The future-state timestep tensor is [B, N]. The DiT receives its
        per-plane expansion [B, (history_states + N) * planes]. At every
        global iteration a single forward predicts every state's velocity.
        """
        if self.config.df_pyramid_window_states is not None:
            return self._pure_df_pyramid_window(observed, generator)
        geo, cfg = self.geometry, self.config
        batch = observed.shape[0]
        history = cfg.history_states
        future_states = cfg.total_states - history
        planes = geo.planes
        observed_states = unpack_states(observed, geo)
        explicit = observed_states.permute(0, 2, 1, 3, 4, 5).flatten(2, 3).contiguous()
        future = self._noise(
            (batch, geo.channels, future_states * planes, geo.grid, geo.grid),
            generator=generator,
            device=observed.device,
            dtype=observed.dtype,
        )
        offset = cfg.df_pyramid_offset
        global_iterations = cfg.steps + (future_states - 1) * offset
        global_step = torch.arange(
            global_iterations + 1, device=observed.device, dtype=torch.float32
        )[:, None]
        starts = (
            torch.arange(future_states, device=observed.device, dtype=torch.float32)[None]
            * offset
        )
        times = (1 - (global_step - starts) / cfg.steps).clamp_(0, 1)
        if not bool((times[-1] == 0).all()):
            raise AssertionError("Pyramid schedule must complete every future state")
        observed_time = torch.zeros(batch, history, device=observed.device, dtype=observed.dtype)

        for current, following in zip(times[:-1], times[1:], strict=True):
            active = following < current
            state_time = current[None].expand(batch, -1).to(observed.dtype)
            frame_time = torch.cat((observed_time, state_time), dim=1).repeat_interleave(
                planes, dim=1
            )
            full_sequence = torch.cat((explicit, future), dim=2)
            velocity = self.engine_forward(
                operation="world",
                noisy=full_sequence,
                timestep=frame_time,
                causal_state_planes=planes,
            )[:, :, history * planes :]
            dt = (following - current).to(future.dtype).repeat_interleave(planes)
            update = future - dt[None, None, :, None, None] * velocity
            # Unstarted noise and completed latents remain bitwise unchanged.
            future = torch.where(
                active.repeat_interleave(planes)[None, None, :, None, None],
                update,
                future,
            )

        states = torch.cat((explicit, future), dim=2).reshape(
            batch, geo.channels, cfg.total_states, planes, geo.grid, geo.grid
        )
        return pack_states(states.permute(0, 2, 1, 3, 4, 5)).contiguous()

    def _pure_df_pyramid_window(self, observed, generator):
        """Slide unfinished states across a fixed-KV completed prefix."""
        geo, cfg = self.geometry, self.config
        batch, planes = observed.shape[0], geo.planes
        history = cfg.history_states
        future_states = cfg.total_states - history
        window = min(cfg.df_pyramid_window_states, future_states)
        starts = pyramid_window_starts(
            future_states, cfg.steps, cfg.df_pyramid_offset, window
        )
        explicit = unpack_states(observed, geo).permute(0, 2, 1, 3, 4, 5)
        explicit = explicit.flatten(2, 3).contiguous()
        future = self._noise(
            (batch, geo.channels, future_states * planes, geo.grid, geo.grid),
            generator=generator,
            device=observed.device,
            dtype=observed.dtype,
        )
        zero_history = observed.new_zeros(batch, history * planes)
        _, cache = self.engine_forward(
            operation="world",
            noisy=explicit,
            timestep=zero_history,
            causal_state_planes=planes,
            return_key_values=True,
        )
        committed = 0
        for global_step in range(starts[-1] + cfg.steps):
            last = min(committed + window, future_states)
            start_planes, end_planes = committed * planes, last * planes
            window_latents = future[:, :, start_planes:end_planes]
            indices = range(committed, last)
            current = torch.tensor(
                [max(0, min(1, 1 - (global_step - starts[i]) / cfg.steps)) for i in indices],
                device=future.device,
                dtype=torch.float32,
            )
            following = torch.tensor(
                [max(0, min(1, 1 - (global_step + 1 - starts[i]) / cfg.steps)) for i in indices],
                device=future.device,
                dtype=torch.float32,
            )
            active = following < current
            frame_time = current[None].expand(batch, -1).to(future.dtype)
            frame_time = frame_time.repeat_interleave(planes, dim=1)
            velocity = self.engine_forward(
                operation="world",
                noisy=window_latents,
                timestep=frame_time,
                frame_offset=(history + committed) * planes,
                past_key_values=cache,
                causal_state_planes=planes,
            )
            dt = (following - current).to(future.dtype).repeat_interleave(planes)
            updated = window_latents - dt[None, None, :, None, None] * velocity
            future[:, :, start_planes:end_planes] = torch.where(
                active.repeat_interleave(planes)[None, None, :, None, None],
                updated,
                window_latents,
            )
            while committed < future_states and global_step + 1 >= starts[committed] + cfg.steps:
                if committed + 1 < future_states:
                    state = future[:, :, committed * planes : (committed + 1) * planes]
                    _, fresh = self.engine_forward(
                        operation="world",
                        noisy=state,
                        timestep=state.new_zeros(batch, planes),
                        frame_offset=(history + committed) * planes,
                        past_key_values=cache,
                        causal_state_planes=planes,
                        return_key_values=True,
                    )
                    # Replace one layer at a time. A list comprehension retains
                    # every old KV tensor until all new tensors are allocated.
                    for layer in range(len(cache)):
                        old_k, old_v = cache[layer]
                        new_k, new_v = fresh[layer]
                        cache[layer] = (
                            torch.cat((old_k, new_k), dim=-2),
                            torch.cat((old_v, new_v), dim=-2),
                        )
                        del old_k, old_v
                committed += 1
        if committed != future_states:
            raise AssertionError("Pyramid window left unfinished states")
        states = torch.cat((explicit, future), dim=2).reshape(
            batch, geo.channels, cfg.total_states, planes, geo.grid, geo.grid
        )
        return pack_states(states.permute(0, 2, 1, 3, 4, 5)).contiguous()

    def _pure_tf(self, observed, generator):
        geo, cfg = self.geometry, self.config
        batch = observed.shape[0]
        explicit = unpack_states(observed, geo).permute(0, 2, 1, 3, 4, 5).flatten(2, 3).contiguous()
        noise = self._noise(
            (batch, geo.channels, cfg.total_states * geo.planes, geo.grid, geo.grid),
            generator=generator,
            device=observed.device,
            dtype=observed.dtype,
        )
        zero = explicit.new_zeros(batch, explicit.shape[2])
        _, cache = self.engine_forward(
            operation="world",
            noisy=explicit,
            timestep=zero,
            causal_state_planes=geo.planes,
            return_key_values=True,
        )
        offset = explicit.shape[2]
        parts = [explicit.clone()]
        schedule = self._schedule(observed)
        for index in range(cfg.history_states, cfg.total_states):
            z = noise[:, :, index * geo.planes : (index + 1) * geo.planes].clone()
            for t_now, t_next in zip(schedule[:-1], schedule[1:], strict=True):
                velocity = self.engine_forward(
                    operation="world",
                    noisy=z,
                    timestep=t_now.expand(batch, geo.planes),
                    frame_offset=offset,
                    past_key_values=cache,
                    causal_state_planes=geo.planes,
                )
                dt = (t_next - t_now).reshape(1, 1, 1, 1, 1)
                z = z - dt * velocity
            parts.append(z)
            _, current = self.engine_forward(
                operation="world",
                noisy=z,
                timestep=z.new_zeros(batch, geo.planes),
                frame_offset=offset,
                past_key_values=cache,
                causal_state_planes=geo.planes,
                return_key_values=True,
            )
            cache = [
                (torch.cat((old_k, new_k), -2), torch.cat((old_v, new_v), -2))
                for (old_k, old_v), (new_k, new_v) in zip(cache, current, strict=True)
            ]
            offset += geo.planes
        explicit = torch.cat(parts, 2).reshape(
            batch, geo.channels, cfg.total_states, geo.planes, geo.grid, geo.grid
        )
        return pack_states(explicit.permute(0, 2, 1, 3, 4, 5)).contiguous()
