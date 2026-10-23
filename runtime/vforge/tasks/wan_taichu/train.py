"""Native Wan Taichu training recipe selected by an explicit variant."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from vforge.data.loader import broadcast_batch
from vforge.distributed.parallel_state import get_parallel_state
from vforge.models.loading import PretrainedWeightSource
from vforge.models.wan_taichu.model import (
    WanTaichuChannelLayout as WanChannelLayout,
)
from vforge.models.wan_taichu.model import (
    WanTaichuTransformer3DModel,
    published_z_dim,
    read_published_transformer_config,
)
from vforge.tasks.requests import TrainRequest
from vforge.tasks.wan.pretrained import condition_tail_initializer, patch_embedding_widener
from vforge.tasks.wan.shared import (
    WanConditionSpec,
    build_state_dict_mapper,
    build_train_dataloader,
    model_config_overrides,
    parameter_storage_dtype,
    validate_shared,
)
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
from vforge.tasks.wan_taichu.conditioning import (
    ConditionDropoutPolicy,
    validate_dropout_config,
)
from vforge.tasks.wan_taichu.encoder import (
    build_condition_encoder,
)
from vforge.tasks.wan_taichu.model import build_empty_model
from vforge.tasks.wan_taichu.optimizer import (
    build_fp32_master_optimizer_factory,
    build_wan_taichu_optimizer_factory,
    clip_wan_taichu_gradients,
)
from vforge.tasks.wan_taichu.processors import (
    WanTaichuA14bConditionProcessor,
    WanTaichuA14bTeacherForcingConditionProcessor,
    WanTaichuConditionProcessor,
    WanTaichuTeacherForcingConditionProcessor,
    WanTaichuTi2vConditionProcessor,
)
from vforge.tasks.wan_taichu.temporal import (
    MaxLatentsSchedule,
    truncate_encoded_latents,
)
from vforge.tasks.wan_taichu.variants import (
    WanTaichuVariant,
    require_train_variant,
)
from vforge.training.recipe import (
    OptimizerSpec,
    SchedulerSpec,
    StepContext,
    StepOutput,
    TrainComponents,
)
from vforge.utils.training import (
    build_optimizer_factory,
    build_scheduler_factory,
    model_patch_size,
    patched_token_count,
    resolve_dtype,
)

_RAW_STREAM_KEYS = MappingProxyType(
    {
        "normal": "normal_condition",
        "grid": "grid_condition",
    }
)


def build_train_components(request: TrainRequest) -> TrainComponents:
    """Build a real train recipe selected by ``model.extras.variant``."""
    spec, streams, extras = _resolve_request(request)
    codec = build_encoded_batch_codec(spec, streams)
    use_deepspeed_optimizer_parity = spec.architecture != "wan22_a14b"
    use_fp32_master = use_deepspeed_optimizer_parity and _use_single_rank_fp32_master(request)
    model, layouts = _build_model(
        spec=spec,
        streams=streams,
        extras=extras,
        pretrained_path=request.pretrained_path,
        init_device=request.init_device,
        parameter_dtype=(
            resolve_dtype(request.precision) if use_fp32_master else parameter_storage_dtype()
        ),
    )
    dataloader, dataloader_resume_contract = build_train_dataloader(
        request,
        codec=codec,
        collate_encoded=lambda samples: collate_encoded_bundles(codec, samples),
        make_raw_collate=lambda: build_raw_collate_fn(
            extras=extras,
            pretrained_path=request.pretrained_path,
            streams=streams,
            use_audio=spec.audio,
        ),
        preprocess_fingerprint={
            "variant": spec.name,
            "condition_streams": list(streams),
            "max_frames": extras.get("max_frames"),
            "max_sequence_length": extras.get("max_sequence_length", 512),
            "model_revision": request.pretrained_path,
            "audio_pretrained_path": extras.get("audio_pretrained_path") if spec.audio else None,
            "audio_type": extras.get("audio_type") if spec.audio else None,
            "audio_video_rate": extras.get("audio_video_rate") if spec.audio else None,
            "audio_fps": extras.get("audio_fps") if spec.audio else None,
            "audio_batch_frames": extras.get("audio_batch_frames") if spec.audio else None,
            "audio_bucket_m": extras.get("audio_bucket_m") if spec.audio else None,
            "audio_sample_rate": extras.get("audio_sample_rate") if spec.audio else None,
            "grid_random_rot_deg": (
                extras.get("grid_random_rot_deg", 0.0) if "grid" in streams else None
            ),
            "camera_to_grid": "taichu-v1" if "grid" in streams else None,
            "cache_schema": "wan-taichu-per-stream-v3",
        },
        open_mixed_encoded_cache=lambda sources, **kwargs: open_taichu_mixed_cache_dataset(
            sources,
            load_ref_sidecar=spec.ref_token,
            **kwargs,
        ),
        open_indexed_encoded_cache=lambda cache_dir, train_manifest, **kwargs: (
            open_taichu_encoded_cache_dataset(
                cache_dir,
                train_manifest,
                load_ref_sidecar=spec.ref_token,
                indexed_sample_count=(
                    request.sources[0].extras.get("indexed_sample_count")
                    if len(request.sources) == 1
                    else None
                ),
                **kwargs,
            )
        ),
    )
    processor = _build_condition_processor(
        spec=spec,
        streams=streams,
        extras=extras,
        precision=request.precision,
    )
    dropout = ConditionDropoutPolicy(spec=spec, streams=streams, extras=extras)
    max_latents_schedule = MaxLatentsSchedule.from_config(extras.get("max_latents_schedule"))
    encoder = _maybe_build_condition_encoder(
        request=request,
        spec=spec,
        streams=streams,
        extras=extras,
        dropout=dropout,
    )

    def step_fn(
        live_model: nn.Module,
        batch: Any,
        step: int,
        context: StepContext,
    ) -> StepOutput:
        return _train_step(
            live_model,
            batch,
            step,
            context,
            request=request,
            spec=spec,
            streams=streams,
            codec=codec,
            processor=processor,
            dropout=dropout,
            encoder=encoder,
            max_latents_schedule=max_latents_schedule,
        )

    return TrainComponents(
        model=model,
        dataloader=dataloader,
        # The 5B production baseline uses Taichu/DeepSpeed, so its Native
        # migration gate needs the task-local FP32-master and clipping adapters.
        # A14B production already runs through Taichu's VForge integration and
        # therefore keeps VForge's ordinary AdamW and logical grad-norm order.
        optimizer_factory=(
            build_fp32_master_optimizer_factory
            if use_fp32_master
            else (
                build_wan_taichu_optimizer_factory
                if use_deepspeed_optimizer_parity
                else build_optimizer_factory
            )
        )(
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            betas=request.betas,
            eps=request.eps,
        ),
        optimizer_spec=OptimizerSpec(
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            betas=request.betas,
            eps=request.eps,
        ),
        step_fn=step_fn,
        scheduler_factory=build_scheduler_factory(
            name=request.scheduler_name,
            factor=request.scheduler_factor,
            warmup_steps=request.warmup_steps,
        ),
        scheduler_spec=SchedulerSpec(
            name=request.scheduler_name,
            factor=request.scheduler_factor,
            warmup_steps=request.warmup_steps,
        ),
        gradient_clipper=(clip_wan_taichu_gradients if use_deepspeed_optimizer_parity else None),
        pretrained_weights=_pretrained_sources(
            spec=spec,
            model=model,
            layouts=layouts,
            pretrained_path=request.pretrained_path,
            weight_loading=request.weight_loading,
        ),
        # Single-expert Native wrappers own ``transformer.*`` internally but
        # retain the portable unprefixed PT key space. A14B already owns the
        # external ``wan`` / ``wan_2`` paths directly.
        state_dict_mapper=(
            None if spec.architecture == "wan22_a14b" else build_state_dict_mapper()
        ),
        dataloader_resume_contract=dataloader_resume_contract,
    )


def _use_single_rank_fp32_master(request: TrainRequest) -> bool:
    """Select the parity optimizer where FSDP cannot supply FP32 master weights."""
    return (
        get_parallel_state().config.dp_size == 1
        and request.usp_size == 1
        and resolve_dtype(request.precision) == torch.bfloat16
    )


def validate_train_request(request: TrainRequest, engine: str) -> None:
    spec, _, extras = _resolve_request(request)
    if request.tp_size != 1:
        raise NotImplementedError(
            "Wan Taichu does not support tensor parallelism yet; set parallel.tp_size=1"
        )
    if spec.architecture == "wan22_a14b" and engine == "megatron":
        if str(extras.get("train_expert", "both")) == "both":
            raise NotImplementedError(
                "the megatron engine does not support Wan Taichu A14B "
                "train_expert=both; use high or low"
            )


def _resolve_request(
    request: TrainRequest,
) -> tuple[WanTaichuVariant, tuple[str, ...], dict[str, Any]]:
    validate_shared(latent_cache_mode=request.latent_cache_mode, usp_size=request.usp_size)
    if "training_mode" in request.extras or "temporal_mode" in request.extras:
        raise ValueError(
            "Wan Taichu selects model semantics with model.extras.variant; "
            "training_mode and temporal_mode are not public configuration"
        )
    name = str(request.extras.get("variant") or "")
    if not name:
        raise ValueError("model.extras.variant is required for model.name=wan_taichu")
    spec = require_train_variant(name)
    streams = spec.resolve_condition_streams(request.extras.get("condition_streams"))
    extras = dict(request.extras)
    extras["condition_streams"] = list(streams)
    extras.setdefault("noise_shift", spec.noise_shift)
    _validate_source_latent_bounds(request)
    MaxLatentsSchedule.from_config(extras.get("max_latents_schedule"))
    if "grid_random_rot_deg" in extras:
        grid_random_rot_deg = float(extras["grid_random_rot_deg"])
        if grid_random_rot_deg < 0.0:
            raise ValueError(
                f"model.extras.grid_random_rot_deg must be non-negative; got {grid_random_rot_deg}"
            )
        extras["grid_random_rot_deg"] = grid_random_rot_deg
    if spec.boundary_ratio is not None:
        extras.setdefault("boundary_ratio", spec.boundary_ratio)
        boundary = float(extras["boundary_ratio"])
        if not 0.0 < boundary < 1.0:
            raise ValueError(f"boundary_ratio must be in (0, 1); got {boundary}")
        train_expert = str(extras.get("train_expert", "both"))
        if train_expert not in {"high", "low", "both"}:
            raise ValueError(
                f"model.extras.train_expert must be high, low, or both; got {train_expert!r}"
            )
    if spec.attention == "block_causal_teacher":
        extras.setdefault("frame_per_block", spec.frame_per_block)
        extras.setdefault("first_frame_own_block", spec.first_frame_own_block)
        frame_per_block = int(extras["frame_per_block"])
        if frame_per_block < 1:
            raise ValueError(f"frame_per_block must be >= 1; got {frame_per_block}")
        extras.setdefault("local_attn_size", -1)
        extras.setdefault("sink_size", 0)
    if spec.audio:
        extras.setdefault("audio_dim", spec.audio_dim)
        extras.setdefault("audio_num_layers", spec.audio_num_layers)
        extras.setdefault("num_audio_token", spec.num_audio_token)
        extras.setdefault("audio_inject_layers_s", list(spec.audio_inject_layers_s))
        extras.setdefault("audio_inject_layers_l", list(spec.audio_inject_layers_l))
        extras.setdefault("audio_window_s", spec.audio_window_s)
        extras.setdefault("audio_window_l", spec.audio_window_l)
        extras.setdefault("audio_type", "wav2vec2")
        extras.setdefault("audio_video_rate", 25)
        extras.setdefault("audio_fps", 25)
        extras.setdefault("audio_batch_frames", 121)
        extras.setdefault("audio_bucket_m", 0)
        extras.setdefault("audio_sample_rate", 16000)
        if int(extras["audio_dim"]) != spec.audio_dim:
            raise ValueError(
                f"{spec.name} fixes audio_dim={spec.audio_dim}; got {extras['audio_dim']}"
            )
        if int(extras["audio_num_layers"]) != spec.audio_num_layers:
            raise ValueError(
                f"{spec.name} fixes audio_num_layers={spec.audio_num_layers}; "
                f"got {extras['audio_num_layers']}"
            )
        if int(extras["num_audio_token"]) != spec.num_audio_token:
            raise ValueError(
                f"{spec.name} fixes num_audio_token={spec.num_audio_token}; "
                f"got {extras['num_audio_token']}"
            )
        for key in ("audio_inject_layers_s", "audio_inject_layers_l"):
            declared = extras[key]
            if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
                raise TypeError(f"{key} must be a sequence of block indices")
            layers = [int(index) for index in declared]
            if len(set(layers)) != len(layers):
                raise ValueError(f"{key} must not contain duplicate block indices")
            invalid = [index for index in layers if index < 0 or index >= 40]
            if invalid:
                raise ValueError(f"{key} must use block indices in [0, 40); got {invalid}")
            extras[key] = layers
        if not extras["audio_inject_layers_s"] and not extras["audio_inject_layers_l"]:
            raise ValueError("audio variant requires speak or listen injection layers")
        for key in ("audio_window_s", "audio_window_l"):
            extras[key] = int(extras[key])
            if extras[key] < -1:
                raise ValueError(f"{key} must be -1 (global) or a non-negative window")
        audio_type = str(extras.get("audio_type", "wav2vec2"))
        if audio_type not in {"wav2vec2", "wav2bert"}:
            raise ValueError(f"audio_type must be wav2vec2 or wav2bert; got {audio_type!r}")
        extras["audio_type"] = audio_type
        for key in ("audio_video_rate", "audio_fps", "audio_batch_frames", "audio_sample_rate"):
            value = int(extras[key])
            if value < 1:
                raise ValueError(f"{key} must be positive; got {value}")
            extras[key] = value
        if extras["audio_video_rate"] < extras["audio_fps"]:
            raise ValueError("audio_video_rate must be greater than or equal to audio_fps")
        bucket_m = int(extras.get("audio_bucket_m", 0))
        if bucket_m != 0:
            raise ValueError(f"{spec.name} fixes audio_bucket_m=0; got {bucket_m}")
        extras["audio_bucket_m"] = bucket_m
    else:
        audio_keys = [
            key
            for key in (
                "audio_pretrained_path",
                "audio_inject_layers_s",
                "audio_inject_layers_l",
                "audio_zero_init",
            )
            if extras.get(key) not in (None, False, (), [])
        ]
        if audio_keys:
            raise ValueError(f"{spec.name} does not support audio settings {audio_keys}")
    ref_num_frames_online = int(extras.get("ref_num_frames_online", 0))
    if ref_num_frames_online < 0:
        raise ValueError(
            f"model.extras.ref_num_frames_online must be >= 0; got {ref_num_frames_online}"
        )
    if not spec.ref_token and ref_num_frames_online:
        raise ValueError("model.extras.ref_num_frames_online is only valid for a ref-token variant")
    if spec.architecture != "wan21":
        unsupported = [
            key
            for key in ("use_vae_condition", "use_clip_condition", "use_dino_condition")
            if bool(extras.get(key, False))
        ]
        if unsupported:
            raise NotImplementedError(
                f"{spec.name} image branches {unsupported} are not part of the "
                "current migration phase"
            )
    validate_dropout_config(spec, streams, extras)
    return spec, streams, extras


def _validate_source_latent_bounds(request: TrainRequest) -> None:
    for index, source in enumerate(request.sources):
        minimum = source.min_latents
        maximum = source.max_latents
        if minimum is not None and minimum < 1:
            raise ValueError(f"data.sources[{index}].min_latents must be positive; got {minimum}")
        if maximum is not None and maximum < 1:
            raise ValueError(f"data.sources[{index}].max_latents must be positive; got {maximum}")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(
                f"data.sources[{index}] min_latents={minimum} must be <= max_latents={maximum}"
            )


def _build_model(
    *,
    spec: WanTaichuVariant,
    streams: tuple[str, ...],
    extras: Mapping[str, Any],
    pretrained_path: str,
    init_device: str,
    parameter_dtype: torch.dtype,
) -> tuple[nn.Module, dict[str, WanChannelLayout]]:
    if not pretrained_path:
        raise ValueError("model.pretrained_path is required")
    if init_device != "meta":
        raise ValueError("Wan Taichu requires model.init_device=meta")
    overrides = model_config_overrides(extras)
    config_path = _transformer_config_path(pretrained_path, extras)
    requested_in = spec.input_channels(streams)
    configs: dict[str, dict[str, Any]] = {}
    layouts: dict[str, WanChannelLayout] = {}

    if spec.architecture == "wan22_a14b":
        for name in spec.experts:
            config = read_published_transformer_config(
                config_path,
                subfolder=_A14B_EXPERT_SUBFOLDERS[name],
            )
            _validate_published_z_dim(spec, config)
            configs[name], layouts[name] = _widen_a14b_config(
                config,
                in_channels=requested_in,
                **overrides,
            )
    elif spec.architecture == "wan22_ti2v_5b":
        published = read_published_transformer_config(config_path)
        _validate_published_z_dim(spec, published)
        configs[""], layouts[""] = _widen_ti2v_config(
            published,
            in_channels=requested_in,
            **overrides,
        )
        configs[""].update(
            use_ref_token=spec.ref_token,
            in_channels_ref=spec.z_dim,
        )
    else:
        published = read_published_transformer_config(config_path)
        _validate_published_z_dim(spec, published)
        declared_in = int(published["in_channels"])
        if requested_in < declared_in:
            raise ValueError(
                f"{spec.name} requested in_channels={requested_in} is narrower than "
                f"the published width {declared_in}"
            )
        config = dict(published)
        config.update(overrides)
        config["in_channels"] = requested_in
        config["out_channels"] = spec.z_dim
        # These are model structure flags, while the same extras also decide
        # which frozen encoders are loaded.
        for key in ("use_vae_condition", "use_clip_condition", "use_dino_condition"):
            if key in extras:
                config[key] = bool(extras[key])
        if spec.audio:
            config.update(
                use_audio_condition=True,
                audio_dim=int(extras["audio_dim"]),
                audio_num_layers=int(extras["audio_num_layers"]),
                num_audio_token=int(extras["num_audio_token"]),
                audio_inject_layers_s=list(extras["audio_inject_layers_s"]),
                audio_inject_layers_l=list(extras["audio_inject_layers_l"]),
                audio_window_s=int(extras["audio_window_s"]),
                audio_window_l=int(extras["audio_window_l"]),
                audio_zero_init=bool(extras.get("audio_zero_init", False)),
            )
        configs[""] = config
        layouts[""] = WanChannelLayout(
            z_dim=spec.z_dim,
            pretrained_in_channels=declared_in,
            in_channels=requested_in,
        )

    model = build_empty_model(
        variant=spec.name,
        transformer_configs=configs,
        parameter_dtype=parameter_dtype,
        frame_per_block=int(extras.get("frame_per_block", 3)),
        first_frame_own_block=bool(extras.get("first_frame_own_block", True)),
        local_attn_size=int(extras.get("local_attn_size", -1)),
        sink_size=int(extras.get("sink_size", 0)),
        use_fsdp_precision_contract=bool(extras.get("wan_use_fsdp_precision_contract", True)),
        upcast_shared_text_context=bool(extras.get("wan_upcast_shared_text_context", True)),
    )
    if spec.architecture == "wan22_a14b":
        train_expert = str(extras.get("train_expert", "both"))
        assert model.wan is not None and model.wan_2 is not None
        model.wan.requires_grad_(train_expert in {"high", "both"})
        model.wan_2.requires_grad_(train_expert in {"low", "both"})
    if parameter_dtype == torch.bfloat16:
        # DeepSpeed casts the whole working model, including Diffusers'
        # ``_keep_in_fp32_modules``, before it creates the FP32 master weights.
        model.to(dtype=parameter_dtype)
    return model, layouts


def _transformer_config_path(pretrained_path: str, extras: Mapping[str, Any]) -> str:
    """Resolve structure separately for a direct Taichu state-dict warm-start.

    A production ``wan_taichu`` run does not convert ``.pt`` weights into a
    Diffusers directory.  The published Wan root remains necessary only to
    construct the Transformer and frozen condition encoders.
    """
    if Path(pretrained_path).suffix.lower() not in {".pt", ".pth"}:
        return pretrained_path
    config_path = str(extras.get("condition_pretrained_path") or "")
    if not config_path:
        raise ValueError(
            "Taichu .pt/.pth weights require model.extras.condition_pretrained_path "
            "to provide the published Wan Diffusers config and frozen assets"
        )
    return config_path


def _validate_published_z_dim(spec: WanTaichuVariant, config: Mapping[str, Any]) -> None:
    actual = published_z_dim(config)
    if actual != spec.z_dim:
        raise ValueError(f"{spec.name} expects z_dim={spec.z_dim}; checkpoint declares {actual}")


_A14B_EXPERT_SUBFOLDERS = {"wan": "transformer", "wan_2": "transformer_2"}


def _widen_a14b_config(
    published: Mapping[str, Any],
    *,
    in_channels: int,
    **overrides: Any,
) -> tuple[dict[str, Any], WanChannelLayout]:
    config = dict(published)
    z_dim = published_z_dim(config)
    declared_in = int(config["in_channels"])
    if int(in_channels) < declared_in:
        raise ValueError(
            f"the requested patch_embedding width {in_channels} is narrower than "
            f"the published width {declared_in}"
        )
    config.update(
        in_channels=int(in_channels),
        out_channels=z_dim,
        use_vae_condition=False,
    )
    config.update(overrides)
    return config, WanChannelLayout(
        z_dim=z_dim,
        pretrained_in_channels=declared_in,
        in_channels=int(in_channels),
    )


def _widen_ti2v_config(
    published: Mapping[str, Any],
    *,
    in_channels: int,
    **overrides: Any,
) -> tuple[dict[str, Any], WanChannelLayout]:
    config = dict(published)
    z_dim = published_z_dim(config)
    declared_in = int(config["in_channels"])
    if declared_in not in {z_dim, int(in_channels)}:
        raise ValueError(
            "Wan Taichu TI2V expects published in_channels to equal z_dim or "
            f"the requested width {in_channels}; got {declared_in}"
        )
    config.update(
        in_channels=int(in_channels),
        out_channels=z_dim,
        use_vae_condition=False,
    )
    config.update(overrides)
    return config, WanChannelLayout(
        z_dim=z_dim,
        pretrained_in_channels=declared_in,
        in_channels=int(in_channels),
    )


def _pretrained_sources(
    *,
    spec: WanTaichuVariant,
    model: nn.Module | None = None,
    layouts: Mapping[str, WanChannelLayout],
    pretrained_path: str,
    weight_loading: str,
) -> PretrainedWeightSource | tuple[PretrainedWeightSource, ...]:
    """Build the training-owned published-weight policy."""
    if spec.architecture == "wan22_a14b":
        return _a14b_pretrained_sources(
            layouts=layouts,
            pretrained_path=pretrained_path,
            weight_loading=weight_loading,
            expert_names=spec.experts,
        )
    layout = layouts[""]
    if spec.audio and model is None:
        raise ValueError("audio pretrained loading requires the assembled model")
    audio_missing = frozenset(
        key
        for key in (() if model is None else model.state_dict())
        if key.startswith(
            (
                "transformer.casual_audio_encoder.",
                "transformer.audio_injector.",
            )
        )
    )
    return _single_pretrained_source(
        architecture=spec.architecture,
        attention=spec.attention,
        layout=layout,
        path=pretrained_path,
        loading_mode=weight_loading,
        label=f"wan-taichu {spec.architecture}",
        initialize_image_embedding=spec.architecture == "wan21",
        initialize_ref_projection=spec.ref_token,
        allowed_missing_model_keys=audio_missing,
        initialize_audio=spec.audio,
        zero_audio_after_load=bool(
            spec.audio
            and model is not None
            and getattr(model.primary_transformer(), "_audio_zero_init", False)
        ),
    )


def _single_pretrained_source(
    *,
    architecture: str,
    attention: str,
    layout: WanChannelLayout,
    path: str,
    loading_mode: str,
    label: str,
    initialize_image_embedding: bool = False,
    initialize_ref_projection: bool = False,
    allowed_missing_model_keys: frozenset[str] = frozenset(),
    initialize_audio: bool = False,
    zero_audio_after_load: bool = False,
) -> PretrainedWeightSource:
    dtype = parameter_storage_dtype()
    initialize_tail = condition_tail_initializer(
        layout,
        transformer_of=lambda model: getattr(model, "transformer", model),
    )

    def initialize(model: nn.Module) -> None:
        initialize_tail(model)
        if initialize_image_embedding:
            _zero_image_embedding(model)
        if initialize_audio:
            _initialize_audio_parameters(model)

    if _is_taichu_state_dict(path):
        fallback_factory = _taichu_state_dict_factory(path, model_name="wan")
        loading_mode = _state_dict_loading_mode(loading_mode)
    else:

        def fallback_factory() -> nn.Module:
            model = WanTaichuTransformer3DModel.from_pretrained(
                path,
                subfolder="transformer",
                torch_dtype=dtype,
                architecture=architecture,
                attention=attention,
            )
            if initialize_image_embedding:
                _zero_image_embedding(model)
            return model

    missing = set(allowed_missing_model_keys)
    if initialize_image_embedding:
        missing.update(
            {
                "transformer.patch_embedding_image.weight",
                "transformer.patch_embedding_image.bias",
            }
        )
    if initialize_ref_projection:
        missing.update(
            {
                "transformer.patch_embedding_ref.weight",
                "transformer.patch_embedding_ref.bias",
            }
        )
    after_load = None
    if initialize_ref_projection or zero_audio_after_load:

        def after_load(model: nn.Module, missing_model_keys: frozenset[str]) -> None:
            if initialize_ref_projection:
                _warm_start_ref_projection(model, missing_model_keys)
            if zero_audio_after_load:
                _zero_audio_output_projections(model)

    return PretrainedWeightSource(
        path=path,
        subfolder="transformer",
        model_prefix="transformer.",
        loading_mode=loading_mode,
        dtype=dtype,
        allowed_missing_model_keys=frozenset(missing),
        initialize_missing=initialize,
        initialize_missing_after_load=after_load,
        resolve_shape_mismatch=patch_embedding_widener(layout, log_label=label),
        fallback_model_factory=fallback_factory,
    )


def _a14b_pretrained_sources(
    *,
    layouts: Mapping[str, WanChannelLayout],
    pretrained_path: str,
    weight_loading: str,
    expert_names: tuple[str, ...],
) -> tuple[PretrainedWeightSource, ...]:
    dtype = parameter_storage_dtype()
    sources = []
    taichu_factories = (
        _taichu_state_dict_factories(pretrained_path, expert_names=expert_names)
        if _is_taichu_state_dict(pretrained_path)
        else {}
    )
    for name in expert_names:
        subfolder = _A14B_EXPERT_SUBFOLDERS[name]
        layout = layouts[name]

        if taichu_factories:
            fallback_factory = taichu_factories[name]
            source_loading_mode = _state_dict_loading_mode(weight_loading)
        else:
            source_loading_mode = weight_loading

            def fallback_factory(subfolder: str = subfolder) -> nn.Module:
                return WanTaichuTransformer3DModel.from_pretrained(
                    pretrained_path,
                    subfolder=subfolder,
                    torch_dtype=dtype,
                    architecture="wan22_a14b",
                    attention="bidirectional",
                    use_vae_condition=False,
                )

        sources.append(
            PretrainedWeightSource(
                path=pretrained_path,
                subfolder=subfolder,
                model_path=name,
                loading_mode=source_loading_mode,
                dtype=dtype,
                initialize_missing=condition_tail_initializer(layout),
                resolve_shape_mismatch=patch_embedding_widener(
                    layout,
                    log_label="wan-taichu a14b widen",
                ),
                fallback_model_factory=fallback_factory,
            )
        )
    return tuple(sources)


class _StateDictCarrier(nn.Module):
    """Expose a task-normalized state dict to the framework broadcast loader."""

    def __init__(self, state: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self._state = dict(state)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        del args, kwargs
        return self._state


def _is_taichu_state_dict(path: str) -> bool:
    return Path(path).suffix.lower() in {".pt", ".pth"}


def _state_dict_loading_mode(mode: str) -> str:
    if mode == "distributed":
        raise ValueError(
            "Taichu .pt/.pth weights cannot use model.weight_loading=distributed; "
            "use auto or rank0_broadcast"
        )
    return "rank0_broadcast" if mode == "auto" else mode


def _taichu_state_dict_factory(path: str, *, model_name: str):
    """Read one Taichu weights-only PT and route an expert's unprefixed keys."""

    def load() -> nn.Module:
        return _StateDictCarrier(
            _route_taichu_expert_state(_read_taichu_state_dict(path), model_name=model_name)
        )

    return load


def _taichu_state_dict_factories(
    path: str,
    *,
    expert_names: Sequence[str],
) -> dict[str, Any]:
    """Load a dual-expert PT once, then hand each expert to the generic loader.

    ``torch.load`` cannot selectively read a pickle-based PT file.  Retaining
    one CPU mapping until both A14B experts have been routed avoids reading the
    production checkpoint twice without introducing a converted artifact.
    """
    cached_state: dict[str, torch.Tensor] | None = None
    remaining = set(expert_names)
    factories: dict[str, Any] = {}

    for expert_name in expert_names:

        def load(name: str = expert_name) -> nn.Module:
            nonlocal cached_state
            if cached_state is None:
                cached_state = _read_taichu_state_dict(path)
            routed = _route_taichu_expert_state(cached_state, model_name=name)
            remaining.discard(name)
            if not remaining:
                cached_state = None
            return _StateDictCarrier(routed)

        factories[expert_name] = load
    return factories


def _read_taichu_state_dict(path: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        isinstance(payload, Mapping)
        and "state_dict" in payload
        and not any(isinstance(value, torch.Tensor) for value in payload.values())
    ):
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise TypeError(
            "Taichu weights-only checkpoint must contain a state dict; got "
            f"{type(payload).__name__} from {path}"
        )
    return {str(key): value for key, value in payload.items() if isinstance(value, torch.Tensor)}


def _route_taichu_expert_state(
    state: Mapping[str, torch.Tensor],
    *,
    model_name: str,
) -> dict[str, torch.Tensor]:
    state = _strip_state_dict_prefix(state, "module.")
    prefix = f"{model_name}."
    if not any(key.startswith(prefix) for key in state):
        return dict(state)
    return {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}


def _strip_state_dict_prefix(
    state: Mapping[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state):
        return dict(state)
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value for key, value in state.items()
    }


def _zero_image_embedding(model: nn.Module) -> None:
    transformer = getattr(model, "transformer", model)
    embedding = getattr(transformer, "patch_embedding_image", None)
    if embedding is None:
        return
    with torch.no_grad():
        embedding.weight.zero_()
        if embedding.bias is not None:
            embedding.bias.zero_()


@torch.no_grad()
def _initialize_audio_parameters(model: nn.Module) -> None:
    """Recreate normal construction-time audio initialization after meta ``to_empty``."""
    transformer = getattr(model, "transformer", model)
    for root_name in ("casual_audio_encoder", "audio_injector"):
        root = getattr(transformer, root_name, None)
        if root is None:
            continue
        for module in root.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()

    encoder = getattr(transformer, "casual_audio_encoder", None)
    if encoder is not None:
        _local_tensor(encoder.weights).fill_(0.01)
        motion = getattr(encoder, "encoder", None)
        if motion is not None:
            _local_tensor(motion.padding_tokens).zero_()


@torch.no_grad()
def _zero_audio_output_projections(model: nn.Module) -> None:
    transformer = getattr(model, "transformer", model)
    zero = getattr(transformer, "zero_audio_output_projections", None)
    if callable(zero):
        zero()


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def _warm_start_ref_projection(
    model: nn.Module,
    missing_model_keys: frozenset[str],
) -> None:
    """Initialize a missing ref projection from the loaded video projection."""
    weight_key = "transformer.patch_embedding_ref.weight"
    bias_key = "transformer.patch_embedding_ref.bias"
    if weight_key not in missing_model_keys and bias_key not in missing_model_keys:
        return
    transformer = getattr(model, "transformer", model)
    main = getattr(transformer, "patch_embedding", None)
    ref = getattr(transformer, "patch_embedding_ref", None)
    if not isinstance(main, nn.Conv3d) or not isinstance(ref, nn.Conv3d):
        raise RuntimeError(
            "ref-token warm start requires patch_embedding and patch_embedding_ref Conv3d"
        )

    with torch.no_grad():
        if weight_key in missing_model_keys:
            target = _local_tensor(ref.weight)
            source = _local_tensor(main.weight)
            if target.shape[0] != source.shape[0] or target.shape[2:] != source.shape[2:]:
                raise RuntimeError(
                    "patch_embedding_ref is incompatible with patch_embedding: "
                    f"ref={tuple(target.shape)}, main={tuple(source.shape)}"
                )
            target.zero_()
            copied_channels = min(int(target.shape[1]), int(source.shape[1]))
            target[:, :copied_channels].copy_(source[:, :copied_channels])
        if bias_key in missing_model_keys and ref.bias is not None:
            target_bias = _local_tensor(ref.bias)
            if main.bias is None:
                target_bias.zero_()
            else:
                source_bias = _local_tensor(main.bias)
                if target_bias.shape != source_bias.shape:
                    raise RuntimeError(
                        "patch_embedding_ref bias is incompatible with patch_embedding bias"
                    )
                target_bias.copy_(source_bias)


def _maybe_build_condition_encoder(
    *,
    request: TrainRequest,
    spec: WanTaichuVariant,
    streams: tuple[str, ...],
    extras: Mapping[str, Any],
    dropout: ConditionDropoutPolicy,
) -> Any | None:
    state = get_parallel_state()
    if not state.is_batch_leader:
        return None
    online = request.latent_cache_mode == "online"
    live_fallback = bool(extras.get("load_vae_for_dropout_fallback", False))
    if not online and not live_fallback:
        return None
    encoder_extras = dict(extras)
    encoder_extras["use_audio_condition"] = spec.audio
    if spec.architecture == "wan22_a14b":
        encoder_extras.update(
            use_vae_condition=False,
            use_clip_condition=False,
            use_dino_condition=False,
        )
    return build_condition_encoder(
        architecture=spec.architecture,
        streams=streams,
        mask_channels=spec.mask_channels,
        raw_stream_keys=_RAW_STREAM_KEYS,
        extras=encoder_extras,
        pretrained_path=request.pretrained_path,
        precision=request.precision,
    )


def _build_condition_processor(
    *,
    spec: WanTaichuVariant,
    streams: tuple[str, ...],
    extras: Mapping[str, Any],
    precision: str,
) -> Any:
    if spec.architecture == "wan22_ti2v_5b":
        if spec.attention == "block_causal_teacher":
            return WanTaichuTeacherForcingConditionProcessor(
                extras=extras,
                precision=precision,
                streams=streams,
            )
        return WanTaichuTi2vConditionProcessor(
            extras=extras,
            precision=precision,
            streams=streams,
            ref_token=spec.ref_token,
        )
    condition = WanConditionSpec(streams=streams, mask_channels=spec.mask_channels)
    if spec.architecture == "wan22_a14b":
        processor_cls = (
            WanTaichuA14bTeacherForcingConditionProcessor
            if spec.attention == "block_causal_teacher"
            else WanTaichuA14bConditionProcessor
        )
        return processor_cls(
            extras=extras,
            precision=precision,
            condition=condition,
            boundary_ratio=float(extras["boundary_ratio"]),
            train_expert=str(extras.get("train_expert", "both")),
        )
    return WanTaichuConditionProcessor(
        extras=extras,
        precision=precision,
        condition=condition,
    )


def _train_step(
    model: nn.Module,
    batch: Any,
    step: int,
    context: StepContext,
    *,
    request: TrainRequest,
    spec: WanTaichuVariant,
    streams: tuple[str, ...],
    codec,
    processor: Any,
    dropout: ConditionDropoutPolicy,
    encoder: Any | None,
    max_latents_schedule: MaxLatentsSchedule | None = None,
) -> StepOutput:
    state = context.parallel_state
    is_leader = bool(state.is_batch_leader)
    patch_size = model_patch_size(model)
    sample_ids = None
    with torch.no_grad():
        if is_leader:
            batch = _convert_training_batch_precision(
                batch,
                resolve_dtype(request.precision),
            )
            if request.latent_cache_mode == "online":
                assert encoder is not None
                encoder.eval()
                source = canonicalize_training_bundle(
                    codec,
                    encoder.encode_batch(batch),
                    metadata=batch,
                )
            else:
                source = canonicalize_training_bundle(codec, batch)
            source = dropout.complete_missing_streams(source, encoder=encoder)
            encoded = canonicalize_encoded_batch(
                source,
                variant=spec,
                streams=streams,
            )
            sample_ids = encoded.get("sample_id")
            encoded = dropout.apply(encoded, encoder=encoder)
            encoded = truncate_encoded_latents(
                encoded,
                step=step,
                device=context.device,
                schedule=max_latents_schedule,
                generator=context.random_streams.generator(
                    "wan_taichu.length", seed=request.seed, device=context.device
                ),
            )
            processor_input = _processor_input(encoded, spec=spec, streams=streams)
            generator = context.random_streams.generator(
                "wan_taichu.input", seed=request.seed, device=context.device
            )
            kwargs: dict[str, Any] = {
                "step": step,
                "device": context.device,
                "generator": generator,
                "apply_dropouts": False,
            }
            if spec.architecture in {"wan22_ti2v_5b", "wan22_a14b"}:
                kwargs["patch_size"] = patch_size
            processed = processor.process(processor_input, **kwargs)
        else:
            processed = None
        if state.batch_group is not None:
            processed = broadcast_batch(
                processed,
                group=state.batch_group,
                is_source=is_leader,
                target_device=context.device,
            )
    assert processed is not None
    if spec.ref_token:
        processed["model_inputs"]["global_has_ref"] = _global_ref_present(
            processed["model_inputs"].get("ref_hidden_states"),
            hidden_states=processed["model_inputs"]["hidden_states"],
            parallel_state=state,
        )
    output = model(**processed["model_inputs"])
    if isinstance(output, tuple):
        output = output[0]
    loss = torch.mean(processed["loss_weights"] * (output - processed["targets"]) ** 2)
    metadata: dict[str, Any] = {}
    if sample_ids is not None:
        metadata["sample_ids"] = (
            [sample_ids] if isinstance(sample_ids, str) else [str(value) for value in sample_ids]
        )
    return StepOutput(
        loss=loss,
        metrics={"loss_scalar": loss.detach()},
        workload={
            "local_tokens": patched_token_count(processed["targets"], patch_size),
            "local_batch": int(processed["targets"].shape[0]),
        },
        metadata=metadata,
    )


def _convert_training_batch_precision(value: Any, dtype: torch.dtype) -> Any:
    """Match Taichu Application's floating-point batch conversion."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: _convert_training_batch_precision(item, dtype) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_convert_training_batch_precision(item, dtype) for item in value)
    if isinstance(value, list):
        return [_convert_training_batch_precision(item, dtype) for item in value]
    return value


def _processor_input(
    encoded: Mapping[str, Any],
    *,
    spec: WanTaichuVariant,
    streams: Sequence[str],
) -> dict[str, Any]:
    out = dict(encoded)
    if spec.architecture != "wan22_ti2v_5b":
        out["condition_latents_normal"] = torch.cat(
            [out[stream_field(name)] for name in streams],
            dim=1,
        )
    return out


def _global_ref_present(
    ref_hidden_states: torch.Tensor | None,
    *,
    hidden_states: torch.Tensor,
    parallel_state: Any,
) -> bool:
    """OR ref presence across ranks that shard the same FSDP model."""
    local_has_ref = ref_hidden_states is not None
    if not dist.is_available() or not dist.is_initialized():
        return local_has_ref
    flag = torch.tensor(
        int(local_has_ref),
        dtype=torch.int32,
        device=hidden_states.device,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=parallel_state.fsdp_group)
    return bool(flag.item())


__all__ = ["build_train_components", "validate_train_request"]
