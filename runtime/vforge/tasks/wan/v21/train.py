"""Wan 2.1 training recipe: dataloader, optimizer, step semantics."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from vforge.data.cache import (
    CACHE_SCHEMA_VERSION,
    digest_sample_ids,
    read_cache_manifest,
)
from vforge.data.collate import build_raw_collate_fn
from vforge.data.dataset import (
    EncodedCacheDataset,
    RawSampleDataset,
    manifest_file_digest,
)
from vforge.data.loader import broadcast_batch, build_resumable_loader
from vforge.data.preprocess import preprocess_deferred_batch
from vforge.data.taichu_index_cache import open_encoded_cache_dataset
from vforge.distributed.parallel_state import get_parallel_state
from vforge.distributed.parallelize import unwrap_replicated_data_parallel
from vforge.models.factory import ModelSpec
from vforge.models.loading import PretrainedWeightSource
from vforge.tasks.requests import TrainRequest, single_offline_source
from vforge.tasks.wan.shared import (
    WanConditionSpec,
    build_state_dict_mapper,
    parameter_storage_dtype,
    validate_shared,
)
from vforge.tasks.wan.v21.shared import (
    WAN21_ENCODED_BATCH_CODEC,
    _build_condition_encoder,
    _build_wan21_model,
    _collate_encoded_bundles,
    _condition_spec,
    _condition_streams,
    project_to_encoded_batch,
)
from vforge.training.recipe import (
    OptimizerSpec,
    SchedulerSpec,
    StepContext,
    StepOutput,
    TrainComponents,
)
from vforge.utils.training import resolve_dtype

logger = logging.getLogger(__name__)


def build_train_components(request: TrainRequest) -> TrainComponents:
    validate_shared(
        latent_cache_mode=request.latent_cache_mode,
        usp_size=request.usp_size,
    )
    # Deferred rather than built here, because how a projection is split is the
    # engine's choice and no engine exists yet: mcore's parallel layers are
    # sharded by their constructor, so the decision has to reach this builder,
    # and the Trainer puts it in force just before calling this thunk.
    model_spec = ModelSpec(
        lambda: _build_wan21_model(
            extras=request.extras,
            init_device=request.init_device,
            pretrained_path=request.pretrained_path,
        )
    )
    dataloader, dataloader_resume_contract = _build_train_dataloader(request)
    processor = _ConditionProcessor(
        extras=request.extras,
        precision=request.precision,
    )
    encoder = None
    probe_encoder: list[Any | None] = [None]
    probe_collate = None
    probe_samples: dict[str, dict[str, Any]] = {}
    if request.latent_cache_mode == "online":
        state = get_parallel_state()
        # Only the leader encodes, so only it needs the frozen VAE/UMT5. Keyed
        # on the batch group rather than on ulysses, or a tp peer would load
        # several GiB of weights it never calls.
        if state.is_batch_leader:
            encoder = _build_condition_encoder(
                extras=request.extras,
                pretrained_path=request.pretrained_path,
                precision=request.precision,
                require_assets=True,
            )
    elif bool(request.extras.get("dual_encode_probe", False)):
        if not request.train_manifest:
            raise ValueError("model.extras.dual_encode_probe=true requires data.train_manifest")
        state = get_parallel_state()
        if state.is_batch_leader:
            probe_encoder[0] = _build_condition_encoder(
                extras=request.extras,
                pretrained_path=request.pretrained_path,
                precision=request.precision,
                require_assets=True,
            )
            probe_collate = build_raw_collate_fn(
                extras=request.extras,
                pretrained_path=request.pretrained_path,
            )
            raw_probe_dataset = RawSampleDataset.from_manifest(request.train_manifest)
            probe_samples = {
                sample.sample_id: raw_probe_dataset[index]
                for index, sample in enumerate(raw_probe_dataset.samples)
            }

    def step_fn(
        live_model: nn.Module,
        batch: Any,
        step: int,
        context: StepContext,
    ) -> StepOutput:
        return _wan_train_step(
            live_model,
            batch,
            step,
            context,
            latent_cache_mode=request.latent_cache_mode,
            seed=request.seed,
            processor=processor,
            encoder=encoder,
            probe_encoder=probe_encoder,
            probe_collate=probe_collate,
            probe_samples=probe_samples,
        )

    return TrainComponents(
        model=model_spec,
        dataloader=dataloader,
        optimizer_factory=_optimizer_factory(
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            betas=request.betas,
            eps=request.eps,
        ),
        step_fn=step_fn,
        scheduler_factory=_scheduler_factory(
            name=request.scheduler_name,
            factor=request.scheduler_factor,
            warmup_steps=request.warmup_steps,
            start_factor=request.scheduler_start_factor,
            end_factor=request.scheduler_end_factor,
            total_iters=request.scheduler_total_iters,
        ),
        optimizer_spec=OptimizerSpec(
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            betas=request.betas,
            eps=request.eps,
        ),
        scheduler_spec=SchedulerSpec(
            name=request.scheduler_name,
            factor=request.scheduler_factor,
            warmup_steps=request.warmup_steps,
            start_factor=request.scheduler_start_factor,
            end_factor=request.scheduler_end_factor,
            total_iters=request.scheduler_total_iters,
        ),
        pretrained_weights=_wan_pretrained_source(
            pretrained_path=request.pretrained_path,
            weight_loading=request.weight_loading,
        ),
        state_dict_mapper=build_state_dict_mapper(),
        dataloader_resume_contract=dataloader_resume_contract,
    )


def _wan_train_step(
    model: nn.Module,
    batch: Any,
    step: int,
    context: StepContext,
    *,
    latent_cache_mode: str,
    seed: int,
    processor: _ConditionProcessor,
    encoder: Any | None,
    probe_encoder: list[Any | None],
    probe_collate: Any | None,
    probe_samples: dict[str, dict[str, Any]],
) -> StepOutput:
    # Two questions that look alike and are not. Conditioning runs on one rank
    # and goes to everyone sharing its batch, which is batch_group. Gradient
    # scaling below asks instead how one full gradient reaches a single
    # reduction. Ulysses duplicates it and compensates here; CP contributes
    # disjoint token shards and is summed later by the Megatron engine. TP peers
    # hold different weight shards, so their gradients are not copies either.
    batch_group = context.parallel_state.batch_group
    is_leader = bool(context.parallel_state.is_batch_leader)
    usp_size = int(context.parallel_state.config.usp_size)

    with torch.no_grad():
        if is_leader:
            if latent_cache_mode == "offline":
                encoded = project_to_encoded_batch(batch)
                if step == 0 and probe_encoder[0] is not None:
                    sample_ids = batch.get("sample_id")
                    if isinstance(sample_ids, str):
                        sample_ids = [sample_ids]
                    if not isinstance(sample_ids, (list, tuple)):
                        raise ValueError("dual encode probe requires sample_id in cached batches")
                    assert probe_collate is not None
                    raw_batch = probe_collate(
                        [probe_samples[str(sample_id)] for sample_id in sample_ids]
                    )
                    online = project_to_encoded_batch(probe_encoder[0].encode_batch(raw_batch))
                    _assert_encoded_batch_equal(encoded, online)
                    probe_encoder[0] = None
                    del raw_batch, online
                    if torch.cuda.is_available():
                        import gc

                        gc.collect()
                        torch.cuda.empty_cache()
                if encoder is not None:
                    encoded = encoder.apply_training_dropouts(encoded)
            else:
                assert encoder is not None
                encoder.eval()
                # A no-op unless the source deferred its resize to this rank.
                preprocess_deferred_batch(batch, device=context.device)
                encoded = project_to_encoded_batch(encoder.encode_batch(batch))
                encoded = encoder.apply_training_dropouts(encoded)
            generator = context.random_streams.generator(
                "wan.input", seed=seed, device=context.device
            )
            processed = processor.process(
                encoded,
                step=step,
                device=context.device,
                generator=generator,
                apply_dropouts=encoder is None,
            )
        else:
            processed = None

        if batch_group is not None:
            processed = broadcast_batch(
                processed,
                group=batch_group,
                is_source=is_leader,
                target_device=context.device,
            )

    assert processed is not None
    output = model(**processed["model_inputs"])
    if isinstance(output, tuple):
        output = output[0]
    loss = _weighted_mse_loss(
        output,
        processed["targets"],
        processed["loss_weights"],
    )
    loss_for_backward = loss / usp_size if usp_size > 1 else loss
    targets = processed["targets"]
    logical_model = unwrap_replicated_data_parallel(model)
    patch_size = tuple(
        int(value)
        for value in getattr(
            getattr(logical_model, "config", None),
            "patch_size",
            (1, 2, 2),
        )
    )
    if targets.ndim != 5 or len(patch_size) != 3:
        raise ValueError(
            "Wan token accounting requires targets shaped [B,C,F,H,W] "
            f"and a 3D patch size; got {tuple(targets.shape)} and {patch_size}"
        )
    batch_size, _, frames, height, width = targets.shape
    patch_t, patch_h, patch_w = patch_size
    if frames % patch_t != 0 or height % patch_h != 0 or width % patch_w != 0:
        raise ValueError(
            "Wan latent dimensions must be divisible by patch size for "
            f"token accounting; got {(frames, height, width)} and {patch_size}"
        )
    tokens = int(batch_size * (frames // patch_t) * (height // patch_h) * (width // patch_w))
    metadata = {}
    if isinstance(batch, dict) and "sample_id" in batch:
        sample_ids = batch["sample_id"]
        metadata["sample_ids"] = (
            [sample_ids] if isinstance(sample_ids, str) else [str(value) for value in sample_ids]
        )
    return StepOutput(
        loss=loss_for_backward,
        metrics={"loss_scalar": loss.detach()},
        workload={
            "local_tokens": tokens,
            "local_batch": int(processed["targets"].shape[0]),
        },
        metadata=metadata,
    )


def _weighted_mse_loss(
    output: torch.Tensor,
    targets: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute Wan's weighted MSE with an explicit FP32 loss boundary."""
    residual = output - targets
    squared_residual = residual.float().square()
    weighted_squared_residual = loss_weights.float() * squared_residual
    return weighted_squared_residual.mean()


class _ConditionProcessor:
    """Per-step noise / CFG dropout / flow-matching target construction."""

    def __init__(
        self,
        *,
        extras: Mapping[str, Any],
        precision: str,
        condition: WanConditionSpec | None = None,
    ) -> None:
        self._noise_shift = float(extras.get("noise_shift", 5.0))
        self._weight_mode = str(extras.get("weighting_scheme", "none"))
        self._dtype = resolve_dtype(precision)
        self._text_drop_prob = float(extras.get("text_drop_prob", 0.0))
        self._rgb_first_frame_drop_prob = float(extras.get("rgb_first_frame_drop_prob", 0.0))
        self._rgb_cond_drop_prob = float(extras.get("rgb_cond_drop_prob", 0.0))
        self._normal_drop_prob = float(extras.get("normal_drop_prob", 0.0))
        self._condition_spec = condition or _condition_spec(extras)
        self._scheduler = None
        self._compute_loss_weighting = None

    def _ensure_diffusion_helpers(self) -> None:
        if self._scheduler is not None:
            return
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from diffusers.training_utils import compute_loss_weighting_for_sd3

        self._scheduler = FlowMatchEulerDiscreteScheduler(shift=self._noise_shift)
        self._compute_loss_weighting = compute_loss_weighting_for_sd3

    def process(
        self,
        encoded: dict[str, Any],
        *,
        step: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        apply_dropouts: bool = True,
    ) -> dict[str, Any]:
        self._ensure_diffusion_helpers()
        assert self._scheduler is not None
        assert self._compute_loss_weighting is not None
        enc = dict(encoded)
        if apply_dropouts:
            enc = self._apply_dropouts(enc)
        frame_latents = enc["frame_latents"].to(device=device)
        condition = enc["condition_latents_normal"]
        if condition is not None:
            condition = condition.to(device=device)
        indices = self._sample_timestep_index(step=step, device=device, generator=generator)
        timesteps = self._scheduler.timesteps[indices].to(device).unsqueeze(0)
        noise = torch.randn(
            frame_latents.shape,
            device=device,
            dtype=frame_latents.dtype,
            generator=generator,
        )
        sigmas = self._scheduler.sigmas[indices].to(device)
        loss_weights = self._compute_loss_weighting(self._weight_mode, sigmas)
        noisy_latent = (1.0 - sigmas) * frame_latents + noise * sigmas
        hidden = (
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
        model_inputs = {
            "hidden_states": hidden,
            "timestep": timesteps,
            "encoder_hidden_states": enc["prompt"].to(device=device, dtype=self._dtype),
            "encoder_hidden_states_image_vae": _maybe_to(
                enc.get("image_condition_latent_vae"), device, self._dtype
            ),
            "encoder_hidden_states_image_clip": _maybe_to(
                enc.get("image_condition_latent_clip"), device, self._dtype
            ),
            "encoder_hidden_states_image_dino": _maybe_to(
                enc.get("image_condition_latent_dino"), device, self._dtype
            ),
        }
        return {
            "model_inputs": model_inputs,
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
        """Draw a scheduler index uniformly across the whole timestep range.

        Split out so a multi-expert recipe can restrict the draw to the range
        owned by the expert it is training this step.
        """
        del step
        assert self._scheduler is not None
        time_step_u = torch.rand(1, device=device, generator=generator)
        return (time_step_u.cpu() * self._scheduler.config.num_train_timesteps).long()[0]

    def _apply_dropouts(self, bundle: dict[str, Any]) -> dict[str, Any]:
        from vforge.models.wan.v21.encoder import (
            apply_encoded_training_dropouts,
        )

        return apply_encoded_training_dropouts(
            bundle,
            rgb_first_frame_drop_prob=self._rgb_first_frame_drop_prob,
            rgb_cond_drop_prob=self._rgb_cond_drop_prob,
            normal_drop_prob=self._normal_drop_prob,
            text_drop_prob=self._text_drop_prob,
            streams=self._condition_spec.streams,
        )


def _build_train_dataloader(
    request: TrainRequest,
) -> tuple[Any, dict[str, Any]]:
    state = get_parallel_state()

    source = single_offline_source(request, recipe="Wan 2.1")
    if request.latent_cache_mode == "offline":
        cache_dir = source.cache_dir if source is not None else ""
        cache_manifest = source.manifest if source is not None else ""
        manifest = (
            read_cache_manifest(cache_dir, codec=WAN21_ENCODED_BATCH_CODEC) if cache_dir else None
        )
        if manifest is None:
            if request.max_steps > 0:
                if not cache_dir:
                    raise ValueError(
                        "offline training requires a cache path in data.cache_dir or data.sources"
                    )
                dataset = open_encoded_cache_dataset(
                    cache_dir,
                    cache_manifest,
                    codec=WAN21_ENCODED_BATCH_CODEC,
                )
                sample_ids = dataset.sample_ids
                manifest_fields = (
                    {
                        "manifest": cache_manifest,
                        "manifest_digest": manifest_file_digest(cache_manifest),
                    }
                    if cache_manifest
                    else {}
                )
                fingerprint = {
                    "source_kind": "encoded_cache",
                    "cache_dir": cache_dir,
                    "layout": "indexed",
                    "split": "train",
                    **manifest_fields,
                    "sample_ids_digest": digest_sample_ids(sample_ids),
                    "sample_count": len(sample_ids),
                    **_loader_fingerprint(request),
                }
            else:
                # Compose-only construction may not have a cache yet.
                dataset = EncodedCacheDataset(
                    cache_dir or ".",
                    [],
                    WAN21_ENCODED_BATCH_CODEC,
                )
                fingerprint = {
                    "source_kind": "encoded_cache",
                    "cache_dir": cache_dir,
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "empty": True,
                }
        else:
            if manifest.completed != len(manifest.sample_ids):
                raise ValueError(
                    "cache manifest is incomplete; run prepare again before "
                    "training or dataloader resume"
                )
            dataset = EncodedCacheDataset(
                cache_dir,
                manifest.sample_ids,
                WAN21_ENCODED_BATCH_CODEC,
            )
            fingerprint = {
                "source_kind": "encoded_cache",
                "cache_dir": cache_dir,
                "schema_version": manifest.schema_version,
                "bundle_version": manifest.bundle_version,
                "completed": manifest.completed,
                "encoding_fingerprint": manifest.encoding_fingerprint,
                "sample_ids_digest": (
                    manifest.sample_ids_digest or digest_sample_ids(manifest.sample_ids)
                ),
                "field_specs": manifest.field_specs or {},
                "sample_count": len(manifest.sample_ids),
                **_loader_fingerprint(request),
            }
    else:
        if not request.train_manifest:
            # Allow Hydra compose without a real manifest; fit will fail later.
            dataset = RawSampleDataset([])
            fingerprint = {
                "source_kind": "raw_video",
                "manifest": "",
                "empty": True,
            }
        else:
            raw = RawSampleDataset.from_manifest(request.train_manifest)
            dataset = raw
            fingerprint = {
                "source_kind": "raw_video",
                "manifest": request.train_manifest,
                "digest": raw.content_digest(),
                "sample_ids_digest": digest_sample_ids(
                    [sample.sample_id for sample in raw.samples]
                ),
                "sample_count": len(raw),
                **_loader_fingerprint(request),
                "preprocess": {
                    "max_frames": request.extras.get("max_frames"),
                    "max_sequence_length": request.extras.get(
                        "max_sequence_length",
                        512,
                    ),
                    "condition_streams": list(
                        _condition_streams(request.extras),
                    ),
                    "model_revision": request.pretrained_path,
                },
            }

    collate_fn = _collate_encoded_bundles if request.latent_cache_mode == "offline" else None
    if request.latent_cache_mode == "online" and request.pretrained_path:
        collate_fn = build_raw_collate_fn(
            extras=request.extras,
            pretrained_path=request.pretrained_path,
        )

    loader = build_resumable_loader(
        dataset,
        parallel_state=state,
        batch_size=request.batch_size,
        num_workers=request.num_workers,
        drop_last=request.drop_last,
        seed=request.sampler_seed,
        shuffle=request.shuffle,
        collate_fn=collate_fn,
        pin_memory=request.pin_memory,
        persistent_workers=request.persistent_workers,
        prefetch_factor=request.prefetch_factor,
        broadcast_batches=False,
    )
    # Preserve the existing Ulysses contract byte-for-byte so checkpoints made
    # before the loader became batch-axis-generic still resume. TP is a new
    # topology and therefore gets the generic spelling.
    if state.usp_enabled:
        fingerprint = {
            **fingerprint,
            "usp_size": int(state.config.usp_size),
            "source": "ulysses_broadcast_loader",
        }
    elif state.batch_group is not None:
        fingerprint = {
            **fingerprint,
            "batch_group_size": torch.distributed.get_world_size(state.batch_group),
            "source": "leader_broadcast_loader",
        }
    return loader, fingerprint


def _loader_fingerprint(request: TrainRequest) -> dict[str, Any]:
    """Loader knobs that a resume has to match, whatever the dataset is."""
    return {
        "batch_size": request.batch_size,
        "num_workers": request.num_workers,
        "drop_last": request.drop_last,
        "base_seed": request.base_seed,
        "sampler_seed": request.sampler_seed,
        "pin_memory": request.pin_memory,
        "persistent_workers": request.persistent_workers,
        "prefetch_factor": request.prefetch_factor,
    }


def _optimizer_factory(
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
):
    def factory(model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

    return factory


def _scheduler_factory(
    *,
    name: str,
    factor: float,
    warmup_steps: int,
    start_factor: float = 1.0,
    end_factor: float = 0.1,
    total_iters: int = 0,
):
    from vforge.utils.training import build_scheduler_factory

    return build_scheduler_factory(
        name=name,
        factor=factor,
        warmup_steps=warmup_steps,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=total_iters,
    )


def _wan_pretrained_source(
    *,
    pretrained_path: str,
    weight_loading: str,
) -> PretrainedWeightSource:
    from vforge.models.wan.v21.transformer import WanTransformer3DModel

    path = pretrained_path
    if not path:
        raise ValueError("model.pretrained_path is required")
    dtype = parameter_storage_dtype()

    def fallback_factory() -> nn.Module:
        transformer = WanTransformer3DModel.from_pretrained(
            path,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        _zero_wan_patch_embedding_image(transformer)
        return transformer

    return PretrainedWeightSource(
        path=path,
        subfolder="transformer",
        model_prefix="transformer.",
        loading_mode=weight_loading,
        dtype=dtype,
        allowed_missing_model_keys=frozenset(
            {
                "transformer.patch_embedding_image.weight",
                "transformer.patch_embedding_image.bias",
            }
        ),
        initialize_missing=_zero_wan_patch_embedding_image,
        fallback_model_factory=fallback_factory,
    )


def _zero_wan_patch_embedding_image(model: nn.Module) -> None:
    transformer = getattr(model, "transformer", model)
    embedding = getattr(transformer, "patch_embedding_image", None)
    if embedding is None:
        return
    with torch.no_grad():
        embedding.weight.zero_()
        if embedding.bias is not None:
            embedding.bias.zero_()


def _maybe_to(value: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if value is None:
        return None
    return value.to(device=device, dtype=dtype)


def _assert_encoded_batch_equal(
    cached: dict[str, Any],
    online: dict[str, Any],
) -> None:
    for key in cached:
        saved = cached.get(key)
        live = online.get(key)
        if saved is None or live is None:
            if saved is not None or live is not None:
                raise RuntimeError(f"dual encode probe mismatch for {key}: one value is None")
            continue
        if (
            saved.shape != live.shape
            or saved.dtype != live.dtype
            or not torch.equal(saved.detach().cpu(), live.detach().cpu())
        ):
            raise RuntimeError(
                f"dual encode probe mismatch for {key}: "
                f"cached=({tuple(saved.shape)}, {saved.dtype}), "
                f"online=({tuple(live.shape)}, {live.dtype})"
            )


__all__ = ["build_train_components"]
