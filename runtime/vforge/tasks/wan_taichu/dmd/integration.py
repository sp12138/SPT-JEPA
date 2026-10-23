"""Wan-specific hooks consumed by the common DMD family assembler."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from vforge.algorithms.dmd.model import DmdDataSupport, DmdFamilySpec
from vforge.models.factory import patch_diffusers_getattr
from vforge.models.loading import PretrainedWeightSource, load_pt_state_dict
from vforge.models.video_diffusion.wan import WAN_DEFAULT_NEGATIVE_PROMPT
from vforge.models.wan_taichu.encoder import WanTaichuConditionEncoder
from vforge.tasks.requests import TrainRequest
from vforge.tasks.wan.shared import build_train_dataloader, parameter_storage_dtype
from vforge.tasks.wan_taichu.batch import (
    build_encoded_batch_codec,
    build_raw_collate_fn,
    collate_encoded_bundles,
    stream_field,
)
from vforge.tasks.wan_taichu.conditioning import ConditionDropoutPolicy
from vforge.tasks.wan_taichu.dmd.algorithm import WanDistributionMatchingAlgorithm
from vforge.tasks.wan_taichu.dmd.critic import WanDmdCriticAdapter
from vforge.tasks.wan_taichu.dmd.generator import WanDmdGeneratorAdapter
from vforge.tasks.wan_taichu.dmd.inputs import (
    anchor_wan_score_noise,
    prepare_wan_dmd_batch,
    resolve_wan_noise_level,
    wan_loss_valid_mask,
    wan_score_noise_shape,
)
from vforge.tasks.wan_taichu.encoder import build_condition_encoder
from vforge.tasks.wan_taichu.train import (
    _build_model,
    _resolve_request,
    _StateDictCarrier,
    validate_train_request,
)
from vforge.tasks.wan_taichu.variants import require_variant
from vforge.utils.training import patched_token_count, resolve_dtype

_DMD_ROLES = ("generator", "real_score", "fake_score")

# A DMD recipe pairs one block-causal generator with two bidirectional score
# towers of the same architecture, keyed by the generator variant. The score
# towers must match the generator because the critic reads the generator's own
# latents.
_DMD_ROLE_VARIANTS = {
    "wan22_ti2v_5b_causal_teacher": {
        "generator": "wan22_ti2v_5b_causal_teacher",
        "real_score": "wan22_ti2v_5b",
        "fake_score": "wan22_ti2v_5b",
    },
    "wan22_a14b_causal_teacher": {
        "generator": "wan22_a14b_causal_teacher",
        "real_score": "wan22_a14b",
        "fake_score": "wan22_a14b",
    },
}

# Which condition streams each architecture's DMD checkpoints were trained
# with. 5B concatenates a single normal stream; A14B's y carries two streams,
# which is what makes its transformer 56-channel.
_DMD_CONDITION_STREAMS = {
    "wan22_ti2v_5b": (("normal",),),
    "wan22_a14b": (("rgb", "normal"),),
}

# Architectures whose incremental-KV rollout shards under Ulysses. Both DMD
# generators are Taichu towers and share one block forward, so the slice/gather
# that keeps the cache's ``num_heads // usp_size`` heads in step with the trunk
# covers them both. The official Wan 5B tower has a block forward of its own
# with no such path, but DMD never builds that one.
_DMD_USP_ARCHITECTURES = frozenset({"wan22_a14b", "wan22_ti2v_5b"})


def validate_wan_dmd(
    request: TrainRequest,
    engine: str,
) -> None:
    """Validate only Wan-specific DMD constraints."""

    _validate_models(request)
    if engine != "fsdp2":
        raise NotImplementedError("Wan DMD first-release support is FSDP2 only")
    provider_request = _generator_provider_request(request)
    validate_train_request(provider_request, engine)
    spec = require_variant(str(provider_request.extras.get("variant") or ""))
    _validate_dmd_usp(request, spec.architecture)
    streams = spec.resolve_condition_streams(provider_request.extras.get("condition_streams"))
    allowed = _DMD_CONDITION_STREAMS[spec.architecture]
    if streams not in allowed:
        raise ValueError(
            f"Wan DMD checkpoint contract for {spec.architecture} requires "
            f"condition_streams in {[list(option) for option in allowed]}, got {list(streams)}"
        )
    if bool(provider_request.extras.get("partial_normal_aug", False)):
        raise NotImplementedError("Wan DMD does not yet support partial_normal_aug")
    generator_config = request.training_models["generator"].config
    if int(generator_config.get("frame_per_block", 0)) < 1:
        raise ValueError("Wan DMD generator.config.frame_per_block must be positive")
    if not isinstance(generator_config.get("first_frame_own_block"), bool):
        raise TypeError("Wan DMD generator.config.first_frame_own_block must be a boolean")


def _validate_dmd_usp(request: TrainRequest, architecture: str) -> None:
    """Reject Ulysses on the architectures whose rollout cannot shard.

    The rollout denoises through the incremental KV cache, so Ulysses has to
    reach the cache as well as the trunk: the cache stores ``num_heads //
    usp_size`` heads per rank and the block forward has to shard its sequence
    to match. Only the Taichu tower carries that path, and only the sizes that
    divide both the head count and the tokens in one denoising block. The
    per-block token count is checked in the block forward, where the causal
    layout has already decided how many frames the block holds.
    """
    usp_size = int(request.usp_size)
    if usp_size == 1:
        return
    if architecture not in _DMD_USP_ARCHITECTURES:
        raise NotImplementedError(
            f"Wan DMD on {architecture} requires parallel.usp_size=1; "
            f"its incremental-KV rollout has no Ulysses path"
        )
    generator_config = request.training_models["generator"].config
    heads = int(generator_config.get("num_attention_heads", 0))
    if heads and heads % usp_size:
        raise ValueError(
            f"Wan DMD with parallel.usp_size={usp_size} needs it to divide "
            f"num_attention_heads={heads}"
        )


def _materialize_pixel_layout(batch: Any) -> Any:
    """Put stacked pixel streams back in the standard contiguous layout.

    Frames are decoded as a channels-last view, and stacking those into a
    batch produces 5D strides that no torch memory format recognizes. A
    host-to-device copy therefore materializes them contiguous, but a batch
    already sitting on the device keeps the decoder's layout -- and the VAE
    picks its conv kernels by memory format, so the two round differently.
    Normalizing here keeps that off the critical path.
    """
    if not isinstance(batch, dict):
        return batch
    repaired = {
        key: value.contiguous()
        for key, value in batch.items()
        if isinstance(value, torch.Tensor) and value.ndim >= 4 and not value.is_contiguous()
    }
    return {**batch, **repaired} if repaired else batch


def build_wan_data_support(
    request: TrainRequest,
) -> DmdDataSupport:
    """Reuse Wan data support and expose only model-specific hooks."""

    provider_request = _generator_provider_request(request)
    spec, streams, extras = _resolve_request(provider_request)
    codec = build_encoded_batch_codec(spec, streams)
    condition_fields = tuple(stream_field(name) for name in streams)
    dropout = ConditionDropoutPolicy(spec=spec, streams=streams, extras=extras)
    encoder = None
    if provider_request.latent_cache_mode == "online":
        encoder = build_condition_encoder(
            architecture=spec.architecture,
            streams=streams,
            mask_channels=spec.mask_channels,
            raw_stream_keys={"normal": "normal_condition", "grid": "grid_condition"},
            extras=extras,
            pretrained_path=provider_request.pretrained_path,
            precision=provider_request.precision,
        )
    negative_prompt: torch.Tensor | None = None
    compute_dtype = resolve_dtype(provider_request.precision)

    def prepare_batch(batch: Any, device: torch.device):
        nonlocal negative_prompt
        batch = _materialize_pixel_layout(batch)
        if provider_request.latent_cache_mode == "offline":
            encoded = codec.canonicalize(batch)
        else:
            if encoder is None:
                raise RuntimeError("online Wan DMD training requires a condition encoder")
            encoder.eval()
            encoded = codec.canonicalize(encoder.encode_batch(batch))
        encoded = codec.canonicalize(dropout.apply(encoded, encoder=encoder))
        encoded = {
            key: value.to(
                device,
                dtype=compute_dtype if value.is_floating_point() else value.dtype,
                non_blocking=True,
            )
            if isinstance(value, torch.Tensor)
            else value
            for key, value in encoded.items()
        }
        prompt = encoded.get("prompt")
        if not isinstance(prompt, torch.Tensor):
            raise TypeError("Wan DMD encoded batch prompt must be a tensor")
        if negative_prompt is None:
            # Online encoding shares the positive prompt's autocast context;
            # offline encoding uses the reference's CPU path. Cache either
            # result because the prompt and frozen text tower never change.
            with torch.random.fork_rng(devices=[]):
                negative_prompt = _build_negative_prompt_embedding(
                    provider_request,
                    encoder=encoder,
                )
        batch_negative = (
            negative_prompt.to(
                device=device,
                dtype=prompt.dtype,
                non_blocking=True,
            )
            .expand(int(prompt.shape[0]), -1, -1)
            .contiguous()
        )
        return prepare_wan_dmd_batch(
            encoded,
            negative_prompt_embeds=batch_negative,
            device=device,
            condition_fields=condition_fields,
            require_first_frame=spec.first_frame_inpaint,
        )

    return DmdDataSupport(
        build_train_stream=lambda seed_offset: _build_train_stream(
            provider_request,
            spec=spec,
            streams=streams,
            codec=codec,
            sampler_seed_offset=seed_offset,
        ),
        prepare_batch=prepare_batch,
        resolve_noise_level=lambda scheduler, normalized, batch: resolve_wan_noise_level(
            scheduler,
            normalized,
            batch,
        ),
        score_noise_shape=wan_score_noise_shape,
        # Both hooks exist only to protect the TI2V in-painted first frame: it
        # is held clean at score time and kept out of the DMD loss. A14B injects
        # its reference frame through y instead, so its first frame is an
        # ordinary generated frame on both counts.
        valid_masks=partial(
            wan_loss_valid_mask,
            exclude_first_frame=spec.first_frame_inpaint,
        ),
        postprocess_score_noise=anchor_wan_score_noise if spec.first_frame_inpaint else None,
        workload=_bundle_workload,
        metrics=_score_metrics,
    )


def _build_train_stream(
    request: TrainRequest,
    *,
    spec: Any,
    streams: tuple[str, ...],
    codec: Any,
    sampler_seed_offset: int,
) -> tuple[Any, dict[str, Any]]:
    stream_request = replace(
        request,
        sampler_seed=int(request.sampler_seed) + int(sampler_seed_offset),
    )
    return build_train_dataloader(
        stream_request,
        codec=codec,
        collate_encoded=lambda samples: collate_encoded_bundles(codec, samples),
        make_raw_collate=lambda: build_raw_collate_fn(
            extras=request.extras,
            pretrained_path=request.pretrained_path,
            streams=streams,
            use_audio=spec.audio,
        ),
        preprocess_fingerprint={
            "variant": spec.name,
            "max_frames": request.extras.get("max_frames"),
            "max_sequence_length": request.extras.get("max_sequence_length", 512),
            "condition_streams": list(streams),
            "cache_flavor": codec.name,
            "model_revision": request.pretrained_path,
            "sampler_seed_offset": int(sampler_seed_offset),
        },
    )


def _build_negative_prompt_embedding(
    request: TrainRequest,
    *,
    encoder: Any | None,
) -> torch.Tensor:
    """Encode the fixed CFG prompt once without changing the CT cache schema."""

    if encoder is not None:
        # The reference runs this prompt through the same call as the positive
        # one, deliberately, so that both embeddings come out of the same code.
        # Encoding it on the side instead drifted by a bf16 ULP on the largest
        # components -- same tokens, same weights, a different path through the
        # text tower -- and since only the unconditional CFG branch reads it,
        # that drift was the last thing separating the two real_score branches.
        max_len = int(request.extras.get("max_sequence_length", 512))
        tokenized = encoder.tokenizer(
            [_negative_prompt(request)],
            padding="max_length",
            max_length=max_len,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return (
            encoder.encode_prompt(
                tokenized.input_ids,
                tokenized.attention_mask,
                batch_size=1,
                max_sequence_length=max_len,
            )
            .detach()
            .cpu()
        )
    if not request.pretrained_path:
        raise ValueError("Wan DMD offline training requires generator pretrained_path")
    from transformers import AutoTokenizer, UMT5EncoderModel

    condition_root = str(request.extras.get("condition_pretrained_path") or request.pretrained_path)
    tokenizer = AutoTokenizer.from_pretrained(condition_root, subfolder="tokenizer")
    text_encoder = (
        UMT5EncoderModel.from_pretrained(
            condition_root,
            subfolder="text_encoder",
        )
        .eval()
        .requires_grad_(False)
    )
    # Taichu's offline path encodes this constant on CPU. BF16 UMT5 also
    # depends on the CPU reduction order, so keep that independent of host size.
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        max_len = int(request.extras.get("max_sequence_length", 512))
        # Use the configured text verbatim, as in the online Taichu encoder.
        # The generic inference helper normalizes punctuation before tokenizing.
        tokenized = tokenizer(
            [_negative_prompt(request)],
            padding="max_length",
            max_length=max_len,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return WanTaichuConditionEncoder.encode_prompt(
            SimpleNamespace(text_encoder=text_encoder, training_dtype=torch.float32),
            tokenized.input_ids,
            tokenized.attention_mask,
            batch_size=1,
            max_sequence_length=max_len,
        ).detach()
    finally:
        torch.set_num_threads(previous_threads)


def _negative_prompt(request: TrainRequest) -> str:
    """Return the CFG unconditional prompt this checkpoint was trained with.

    The published Wan string is English, but the A14B DMD lineage was distilled
    against its Chinese translation, and the real score's guidance direction is
    the difference between the conditional and unconditional branches -- so the
    wrong string here does not merely round differently, it points the
    guidance somewhere else.
    """
    configured = request.extras.get("negative_prompt")
    return WAN_DEFAULT_NEGATIVE_PROMPT if configured is None else str(configured)


def build_wan_role_model(
    request: TrainRequest,
    role: str,
) -> nn.Module:
    """Build one independent Wan role using the existing Wan model builder."""

    model_request = request.training_models[role]
    generator_request = request.training_models["generator"]
    shared_config = dict(generator_request.config)
    shared_config.update(model_request.config)
    role_request = TrainRequest(
        extras=shared_config,
        pretrained_path=model_request.pretrained_path or generator_request.pretrained_path,
        init_device=model_request.init_device,
        weight_loading=model_request.weight_loading,
    )
    spec, streams, extras = _resolve_request(role_request)
    model, _ = _build_model(
        spec=spec,
        streams=streams,
        extras=extras,
        pretrained_path=role_request.pretrained_path,
        init_device=role_request.init_device,
        # Match CT/TF's FP32 parameter storage; runtime precision controls
        # computation, including the KV rollout's FSDP forward entry.
        parameter_dtype=parameter_storage_dtype(),
    )
    towers = model.cache_transformers()
    for tower in towers:
        patch_diffusers_getattr(tower)
        tower.ulysses_gather_backward = "reduce_scatter"
    if len(towers) == 1:
        return towers[0]
    # A14B needs both experts in the role module: they are routed per timestep,
    # they are trained together, and each keeps its own KV cache. The model
    # object is what owns that routing, so it is what the role hands back.
    return model


def create_wan_generator_adapter(
    request: TrainRequest,
) -> WanDmdGeneratorAdapter:
    generator_config = request.training_models["generator"].config
    spec = require_variant(str(generator_config.get("variant") or ""))
    return WanDmdGeneratorAdapter(
        frame_per_block=int(generator_config["frame_per_block"]),
        first_frame_own_block=bool(generator_config["first_frame_own_block"]),
        first_frame_inpaint=spec.first_frame_inpaint,
        usp_size=request.usp_size,
    )


def create_wan_critic_adapter(
    request: TrainRequest,
    role: str,
) -> WanDmdCriticAdapter:
    del role
    return WanDmdCriticAdapter(compute_dtype=resolve_dtype(request.precision))


def build_wan_weight_sources(request: TrainRequest, roles: tuple[str, ...]):
    """Load independent roles, selecting A14B expert keys in the task layer."""
    sources = []
    for role in roles:
        model = request.training_models[role]
        if not model.init_weights:
            raise ValueError(f"DMD training.models.{role}.init_weights is required")
        source = PretrainedWeightSource(
            path=model.init_weights, model_path=role, loading_mode=model.weight_loading
        )
        spec = require_variant(_role_variant(request, role))
        experts = tuple(name for name in spec.experts if name)
        if not experts:
            sources.append(source)
            continue
        prefixes = _expert_checkpoint_prefixes(source.path, experts)
        for expert in experts:
            sources.append(
                replace(
                    source,
                    model_path=f"{role}.{expert}",
                    fallback_model_factory=partial(
                        _load_expert_weights, source.path, prefixes[expert]
                    ),
                )
            )
    return tuple(sources)


def _load_expert_weights(path: str, prefix: str) -> nn.Module:
    state = load_pt_state_dict(path)
    selected = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not selected:
        raise KeyError(f"checkpoint holds no keys under {prefix!r}")
    return _StateDictCarrier(selected)


# How the reference trainer names A14B's two experts. Its score towers are saved
# straight off the model, so they already carry this repo's attribute names,
# while its self-forcing products keep the Lightning bucket that owned each one.
_A14B_CHECKPOINT_PREFIXES = (
    {"wan": "wan.", "wan_2": "wan_2."},
    {"wan": "generator.model.transformer.", "wan_2": "generator_2.model.transformer."},
)


def _expert_checkpoint_prefixes(path: str, experts: tuple[str, ...]) -> dict[str, str]:
    """Pick the layout a multi-expert checkpoint was written in."""

    state = torch.load(path, map_location="meta", mmap=True, weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{path} is not a state dict")
    keys = tuple(str(key) for key in state)
    for candidate in _A14B_CHECKPOINT_PREFIXES:
        if set(candidate) != set(experts):
            continue
        if all(any(key.startswith(prefix) for key in keys) for prefix in candidate.values()):
            return candidate
    roots = sorted({key.split(".", 1)[0] for key in keys})
    raise ValueError(
        f"{path} does not hold one subtree per expert {list(experts)}; "
        f"its top-level names are {roots}"
    )


def _role_variant(request: TrainRequest, role: str) -> str:
    config = request.training_models[role].config
    variant = config.get("variant") or request.training_models["generator"].config.get("variant")
    return str(variant or "")


def _validate_models(request: TrainRequest) -> None:
    missing = sorted(set(_DMD_ROLES) - set(request.training_models))
    if missing:
        raise ValueError(f"Wan DMD training.models is missing required roles: {missing}")
    generator_variant = str(request.training_models["generator"].config.get("variant") or "")
    expected = _DMD_ROLE_VARIANTS.get(generator_variant)
    if expected is None:
        raise NotImplementedError(
            f"Wan DMD does not support generator=wan_taichu/{generator_variant}; "
            f"expected one of {sorted(_DMD_ROLE_VARIANTS)}"
        )
    for role, variant in expected.items():
        configured_variant = str(request.training_models[role].config.get("variant") or "")
        if configured_variant != variant:
            raise NotImplementedError(
                f"Wan DMD with generator=wan_taichu/{generator_variant} requires "
                f"{role}=wan_taichu/{variant}, got wan_taichu/{configured_variant}"
            )


def _generator_provider_request(request: TrainRequest) -> TrainRequest:
    """Adapt generator-owned settings to the existing Wan data-provider API."""

    generator = request.training_models["generator"]
    pretrained_paths = {
        model.pretrained_path for model in request.training_models.values() if model.pretrained_path
    }
    if len(pretrained_paths) > 1:
        raise NotImplementedError(
            "the first Wan DMD integration requires one shared structural pretrained_path"
        )
    return replace(
        request,
        extras=dict(generator.config),
        pretrained_path=generator.pretrained_path,
        init_device=generator.init_device,
        weight_loading=generator.weight_loading,
    )


def _bundle_workload(
    generated: dict[str, torch.Tensor],
    generator: nn.Module,
) -> dict[str, float | int]:
    latents = generated["latent"].permute(0, 2, 1, 3, 4)
    patch_size = tuple(int(value) for value in generator.config.patch_size)
    return {
        "local_tokens": patched_token_count(latents, patch_size),
        "local_batch": int(latents.shape[0]),
    }


def _score_metrics(level, _gradient) -> dict[str, torch.Tensor]:
    return {"score_timestep": level.model_timestep[:, 1:].float().mean()}


WAN_TAICHU_DMD = DmdFamilySpec(
    model_name="wan_taichu",
    validate=validate_wan_dmd,
    build_role_model=build_wan_role_model,
    build_data_support=build_wan_data_support,
    create_generator_adapter=create_wan_generator_adapter,
    create_critic_adapter=create_wan_critic_adapter,
    build_weight_sources=build_wan_weight_sources,
    algorithm_type=WanDistributionMatchingAlgorithm,
)


__all__ = ["WAN_TAICHU_DMD"]
