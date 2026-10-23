"""Wan 5B cache preparation for consistency distillation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from vforge.algorithms.cd.model import CdDataSupport, PreparedCdBatch
from vforge.data.loader import broadcast_batch
from vforge.distributed.parallel_state import get_parallel_state
from vforge.tasks.requests import TrainRequest
from vforge.tasks.wan.shared import build_train_dataloader
from vforge.tasks.wan_taichu.batch import (
    build_encoded_batch_codec,
    build_raw_collate_fn,
    canonicalize_encoded_batch,
    canonicalize_training_bundle,
    collate_encoded_bundles,
    stream_field,
)
from vforge.tasks.wan_taichu.cache import (
    open_taichu_encoded_cache_dataset,
    open_taichu_mixed_cache_dataset,
)
from vforge.tasks.wan_taichu.conditioning import ConditionDropoutPolicy
from vforge.tasks.wan_taichu.temporal import truncate_encoded_latents
from vforge.tasks.wan_taichu.train import _resolve_request
from vforge.utils.rng import RandomStreams
from vforge.utils.training import resolve_dtype


def build_wan_cd_data_support(request: TrainRequest) -> CdDataSupport:
    provider_request = _student_provider_request(request)
    training_dtype = resolve_dtype(provider_request.precision)
    spec, streams, extras = _resolve_request(provider_request)
    codec = build_encoded_batch_codec(spec, streams)
    dropout = ConditionDropoutPolicy(spec=spec, streams=streams, extras=extras)

    def build_train_stream(seed_offset: int):
        stream_request = replace(
            provider_request,
            sampler_seed=int(provider_request.sampler_seed) + int(seed_offset),
        )
        return build_train_dataloader(
            stream_request,
            codec=codec,
            collate_encoded=lambda samples: collate_encoded_bundles(codec, samples),
            make_raw_collate=lambda: build_raw_collate_fn(
                extras=extras,
                pretrained_path=provider_request.pretrained_path,
                streams=streams,
                use_audio=False,
            ),
            preprocess_fingerprint={
                "variant": spec.name,
                "condition_streams": list(streams),
                "max_frames": extras.get("max_frames"),
                "max_sequence_length": extras.get("max_sequence_length", 512),
                "model_revision": provider_request.pretrained_path,
                "grid_random_rot_deg": extras.get("grid_random_rot_deg", 0.0),
                "camera_to_grid": "taichu-v1",
                "cache_schema": "wan-taichu-per-stream-v3",
                "recipe": "cd",
            },
            force_mixed_encoded_cache=True,
            open_mixed_encoded_cache=lambda sources, **kwargs: open_taichu_mixed_cache_dataset(
                sources,
                load_ref_sidecar=False,
                **kwargs,
            ),
            open_indexed_encoded_cache=lambda cache_dir, train_manifest, **kwargs: (
                open_taichu_encoded_cache_dataset(
                    cache_dir,
                    train_manifest,
                    load_ref_sidecar=False,
                    **kwargs,
                )
            ),
        )

    def prepare_batch(
        batch: Any,
        device: torch.device,
        step: int,
        random_streams: RandomStreams,
    ) -> PreparedCdBatch:
        state = get_parallel_state()
        processed = None
        with torch.no_grad():
            if state.is_batch_leader:
                source = canonicalize_training_bundle(codec, batch)
                source = dropout.complete_missing_streams(source, encoder=None)
                encoded = canonicalize_encoded_batch(
                    source,
                    variant=spec,
                    streams=streams,
                )
                encoded = dropout.apply(encoded, encoder=None)
                encoded = truncate_encoded_latents(
                    encoded,
                    step=step,
                    generator=random_streams.generator(
                        "cd.length", seed=provider_request.seed, device=device
                    ),
                    device=device,
                )
                processed = _select_cd_tensors(
                    encoded,
                    variant=spec.name,
                    streams=streams,
                    latent_channels=spec.z_dim,
                    condition_channels=spec.mask_channels + spec.z_dim,
                    device=device,
                    dtype=training_dtype,
                )
            if state.batch_group is not None:
                processed = broadcast_batch(
                    processed,
                    group=state.batch_group,
                    is_source=state.is_batch_leader,
                    target_device=device,
                )
        if not isinstance(processed, dict):
            raise RuntimeError("Wan CD batch preparation produced no batch")
        return PreparedCdBatch(**processed)

    return CdDataSupport(
        build_train_stream=build_train_stream,
        prepare_batch=prepare_batch,
    )


def _select_cd_tensors(
    encoded: dict[str, Any],
    *,
    variant: str,
    streams: tuple[str, ...],
    latent_channels: int,
    condition_channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    names = {
        "clean_latent": "frame_latents",
        "prompt_embeds": "prompt",
    }
    values: dict[str, Any] = {"condition_latents": {}}
    for target, source in names.items():
        value = encoded.get(source)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Wan CD requires tensor field {source!r}")
        values[target] = value.to(device=device, dtype=dtype, non_blocking=True)
    for stream in streams:
        source = stream_field(stream)
        value = encoded.get(source)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Wan CD requires tensor field {source!r}")
        values["condition_latents"][stream] = value.to(
            device=device,
            dtype=dtype,
            non_blocking=True,
        )
    first_frame = encoded.get("first_frame_latents")
    values["first_frame_latent"] = (
        first_frame.to(device=device, dtype=dtype, non_blocking=True)
        if isinstance(first_frame, torch.Tensor)
        else None
    )
    clean = values["clean_latent"]
    if clean.ndim != 5 or int(clean.shape[1]) != latent_channels:
        raise ValueError(
            "Wan CD clean latents must be [B,C,T,H,W] with "
            f"C={latent_channels}; got {tuple(clean.shape)}"
        )
    conditions = values["condition_latents"]
    for stream, condition in conditions.items():
        if (
            condition.ndim != 5
            or int(condition.shape[1]) != condition_channels
            or condition.shape[0] != clean.shape[0]
            or condition.shape[2:] != clean.shape[2:]
        ):
            raise ValueError(
                f"Wan CD {stream!r} condition must be [B,{condition_channels},T,H,W] "
                "and match clean latent geometry"
            )
    frames = int(clean.shape[2])
    if variant == "wan22_ti2v_5b_causal_teacher":
        if frames < 2 or (frames - 1) % 3:
            raise ValueError(f"Wan 5B CD latent length must satisfy T=1+3n; got T={frames}")
        if not isinstance(values["first_frame_latent"], torch.Tensor):
            raise TypeError("Wan 5B CD requires tensor field 'first_frame_latents'")
    elif variant == "wan22_a14b_causal_teacher":
        if frames < 3 or frames % 3:
            raise ValueError(f"Wan A14B CD latent length must satisfy T=3n; got T={frames}")
        values["first_frame_latent"] = None
    else:
        raise ValueError(f"Wan CD does not support variant {variant!r}")
    return values


def _student_provider_request(request: TrainRequest) -> TrainRequest:
    student = request.training_models["student"]
    return replace(
        request,
        extras=dict(student.config),
        pretrained_path=student.pretrained_path,
        init_device=student.init_device,
        weight_loading=student.weight_loading,
    )


__all__ = ["build_wan_cd_data_support"]
