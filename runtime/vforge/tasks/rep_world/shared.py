"""Configuration and lazy model/codec construction shared by Native verbs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch

from vforge.models.factory import ModelSpec
from vforge.models.loading import PretrainedWeightSource
from vforge.models.rep_world.config import (
    CodecGeometry,
    ConditionerConfig,
    RepaConfig,
    WorldConfig,
    codec_geometry,
)
from vforge.models.rep_world.horizon import (
    validate_quadrant_geometry,
    validate_token_mosaic_geometry,
)
from vforge.models.rep_world.model import FlowConfig, RepWorld
from vforge.tasks.rep_world.codec import load_videorae, load_vrae
from vforge.tasks.rep_world.ema import EmaConfig
from vforge.tasks.requests import single_offline_source
from vforge.utils.training import resolve_dtype


@dataclass(frozen=True)
class TaskConfig:
    geometry: CodecGeometry
    world: WorldConfig
    conditioner: ConditionerConfig | None
    flow: FlowConfig
    states: int
    frame_interval: int
    random_clip: bool
    use_muon: bool
    codec: Mapping
    rollout: Mapping
    initial_weights_format: str
    ema: EmaConfig = EmaConfig()
    next_state_tuning_start_step: int | None = None
    next_state_tuning_init: str = "online"
    cached_state_view: str = "full"
    repa: RepaConfig = RepaConfig()


def parse_config(extras):
    allowed = {
        "codec",
        "world",
        "conditioner",
        "flow",
        "states",
        "frame_interval",
        "random_clip",
        "use_muon",
        "rollout",
        "initial_weights_format",
        "next_state_tuning_start_step",
        "next_state_tuning_init",
        "cached_state_view",
        "ema",
        "repa",
    }
    if set(extras) - allowed:
        raise ValueError(f"Unknown RepWorld extras: {sorted(set(extras) - allowed)}")
    codec = dict(extras.get("codec", {}))
    allowed_codec = {
        "family",
        "variant",
        "checkpoint",
        "statistics",
        "encode_batch_size",
        "decode_batch_size",
        "independent_state_decode",
    }
    if set(codec) - allowed_codec:
        raise ValueError(f"Unknown codec settings: {sorted(set(codec) - allowed_codec)}")
    geometry = codec_geometry(codec.get("family", "videorae"), codec.get("variant", "3d"))
    repa = RepaConfig(**extras.get("repa", {}))
    if repa.enabled and (geometry.family, geometry.variant) != ("pixel", "rgb16"):
        raise ValueError("Video REPA requires the direct RGB16 PixelDiT experiment")
    flow = FlowConfig(**extras.get("flow", {}))
    condition_options = dict(extras.get("conditioner", {}))
    if "feature_layers" in condition_options:
        condition_options["feature_layers"] = tuple(condition_options["feature_layers"])
    conditioner = (
        None if flow.mode in {"pure_tf", "pure_df"} else ConditionerConfig(**condition_options)
    )
    world_options = dict(extras.get("world", {}))
    if set(world_options) & {"channels", "state_planes", "grid", "condition_width"}:
        raise ValueError("World codec/condition dimensions are derived, not independent overrides")
    if geometry.family == "pixel" and (
        conditioner is None or conditioner.input != "rgb" or world_options.get("pixel_depth", 0) < 1
    ):
        raise ValueError("Pixel RGB4 requires an RGB conditioner and two-level PixelDiT")
    if geometry.family == "wan22" and (
        conditioner is None or conditioner.input != "rgb" or conditioner.temporal_patch != 4
        or flow.mode != "no_zprev"
    ):
        raise ValueError("Wan2.2 RGB4 requires an RGB patch4 conditioner and no_zprev")
    if geometry.family != "pixel" and world_options.get("pixel_depth", 0):
        raise ValueError("PixelDiT is only for direct RGB4 prediction")
    world = WorldConfig(
        channels=geometry.channels,
        state_planes=geometry.planes,
        grid=geometry.grid,
        condition_width=conditioner.width if conditioner else 0,
        **world_options,
    )
    if conditioner is not None:
        conditioner.tokens_per_state(geometry)
        if conditioner.input == "latent" and flow.mode != "no_zprev":
            raise ValueError("Latent-input baseline supports flow.mode=no_zprev only")
    states = extras.get("states", 10 if geometry.family == "videorae" else 20)
    interval = extras.get("frame_interval", 1)
    if type(states) is not int or states < 2 or type(interval) is not int or interval < 1:
        raise ValueError("states >= 2 and frame_interval >= 1 must be integers")
    flow.target_count(states)
    if flow.target_layout == "random_token_mosaic":
        validate_token_mosaic_geometry(geometry, world.patch_size)
    elif flow.is_mosaic:
        validate_quadrant_geometry(geometry, world.patch_size)
    for field in ("random_clip", "use_muon"):
        if field in extras and type(extras[field]) is not bool:
            raise ValueError(f"{field} must be a boolean")
    weight_format = extras.get("initial_weights_format", "native")
    if (
        conditioner is not None
        and conditioner.architecture == "split_ec"
        and weight_format == "legacy"
    ):
        raise ValueError(
            "Split ViT requires native weights; legacy single-ViT mapping is unsupported"
        )
    if weight_format not in {"native", "legacy"}:
        raise ValueError("initial_weights_format must be native or legacy")
    tuning_start = extras.get("next_state_tuning_start_step")
    if tuning_start is not None:
        if type(tuning_start) is not int or tuning_start < 0:
            raise ValueError("next_state_tuning_start_step must be a nonnegative integer")
        if flow.mode != "no_zprev" or flow.target_origin_reserve is None:
            raise ValueError(
                "Next-state tuning requires no_zprev and explicit target_origin_reserve"
            )
    tuning_init = extras.get("next_state_tuning_init", "online")
    if tuning_init not in {"online", "ema_reset_optimizer"}:
        raise ValueError("next_state_tuning_init must be online or ema_reset_optimizer")
    if tuning_init == "ema_reset_optimizer" and (
        tuning_start is None or not EmaConfig(**extras.get("ema", {})).enabled
    ):
        raise ValueError("EMA tuning initialization requires EMA and a phase boundary")
    cached_state_view = extras.get("cached_state_view", "full")
    if cached_state_view not in {"full", "random_9_from_10"}:
        raise ValueError("cached_state_view must be full or random_9_from_10")
    if cached_state_view == "random_9_from_10" and (
        states != 9
        or (geometry.family, geometry.variant) != ("videorae", "3d")
        or (conditioner is not None and conditioner.input != "latent")
        or extras.get("random_clip", True)
    ):
        raise ValueError("random_9_from_10 requires fixed-window 9-state VideoRAE3D latent input")
    return TaskConfig(
        geometry,
        world,
        conditioner,
        flow,
        states,
        interval,
        extras.get("random_clip", True),
        extras.get("use_muon", True),
        codec,
        dict(extras.get("rollout", {})),
        weight_format,
        EmaConfig(**extras.get("ema", {})),
        tuning_start,
        tuning_init,
        cached_state_view,
        repa,
    )


def model_spec(config, init_device):
    if init_device not in {"meta", "cpu"}:
        raise ValueError("RepWorld model.init_device must be meta or cpu")

    def build():
        with torch.device(init_device):
            return RepWorld(config.geometry, config.world, config.conditioner, config.flow, config.repa)

    return ModelSpec(build)


def initialize_model(model):
    model.initialize_weights()


def pretrained_source(path, config, weight_loading="auto"):
    if not path:
        return None
    factory = None
    if config.initial_weights_format == "legacy":
        from vforge.tasks.rep_world.weights import load_legacy_model

        factory = partial(load_legacy_model, path, config)
    elif Path(path).suffix.lower() in {".pt", ".pth"}:
        from vforge.tasks.rep_world.weights import load_native_model

        # Refuse partial PT loads into uninitialized meta storage even when a
        # caller leaves Native's general compatible-load setting enabled.
        factory = partial(load_native_model, path, config)
    return PretrainedWeightSource(
        path=path, subfolder="", loading_mode=weight_loading, fallback_model_factory=factory
    )


class CodecProvider:
    """Load frozen assets lazily on the rank's actual device, never in workers."""

    def __init__(self, config, precision):
        self.config, self.dtype, self.codec = config, resolve_dtype(precision), None
        self.device = None

    def get(self, device):
        device = torch.device(device)
        if self.codec is not None:
            if device != self.device:
                raise ValueError("A frozen codec provider cannot move between rank devices")
            return self.codec
        options = self.config.codec
        if self.config.geometry.family == "pixel":
            if options.get("checkpoint") or options.get("statistics"):
                raise ValueError("Direct pixel training must not load a neural codec")
            from vforge.models.rep_world.pixel_codec import PixelRGB4Codec

            self.codec = PixelRGB4Codec(self.config.geometry)
            self.device = device
            return self.codec
        if self.config.geometry.family == "wan22":
            if options.get("statistics"):
                raise ValueError("Wan2.2 VAE uses its published built-in normalization")
            from vforge.models.rep_world.wan22_codec import Wan22RGB4Codec

            self.codec = Wan22RGB4Codec(
                self.config.geometry, options.get("checkpoint"), device, self.dtype
            )
            self.device = device
            return self.codec
        if not options.get("checkpoint"):
            raise ValueError("model.extras.codec.checkpoint is required for online RGB encoding")
        kwargs = dict(
            device=device,
            dtype=self.dtype,
            statistics_path=options.get("statistics") or None,
            encode_batch_size=options.get("encode_batch_size", 1),
            decode_batch_size=options.get("decode_batch_size", 1),
        )
        geometry = self.config.geometry
        if geometry.family == "vrae":
            kwargs["independent_state_decode"] = options.get("independent_state_decode", False)
            loader = load_vrae
        else:
            if options.get("independent_state_decode", False):
                raise ValueError("independent_state_decode is a V-RAE-only setting")
            loader = load_videorae
        self.codec = loader(options["checkpoint"], geometry.variant, **kwargs)
        self.device = device
        return self.codec


class RepaTeacherProvider:
    """Keep the frozen V-JEPA teacher outside the trainable model and DCP."""

    def __init__(self, config: RepaConfig, precision):
        self.config = config
        self.dtype = resolve_dtype(precision)
        self.teacher = None
        self.device = None

    def get(self, device):
        device = torch.device(device)
        if self.teacher is None:
            from vforge.models.rep_world.repa import VJEPA21VideoTeacher

            if self.config.teacher_width != 768:
                raise ValueError("Released V-JEPA 2.1 ViT-B teacher width is 768")
            self.teacher = VJEPA21VideoTeacher.from_checkpoint(
                self.config.teacher_checkpoint, device, self.dtype
            )
            self.device = device
        elif self.device != device:
            raise ValueError("Frozen REPA teacher cannot move between rank devices")
        return self.teacher


def validate_train_request(request, engine):
    config = parse_config(request.extras)
    if config.cached_state_view != "full" and request.latent_cache_mode != "offline":
        raise ValueError("cached_state_view requires offline latent cache training")
    if (
        config.conditioner
        and config.conditioner.architecture == "split_ec"
        and request.usp_size != 1
    ):
        raise ValueError("Split ViT-E/ViT-C currently requires USP=1; use DDP or FSDP")
    if config.conditioner and config.conditioner.context_mask_ratio and request.usp_size != 1:
        raise ValueError("RepWorld context token masking currently requires USP=1")
    if engine != "fsdp2" or max(request.tp_size, request.cp_size) != 1:
        raise ValueError(
            "RepWorld requires parallel.engine=fsdp2 and TP/CP sizes=1; "
            "Ulysses is supported with FSDP2 or HSDP; DDP requires USP=1"
        )
    if not request.train_manifest:
        raise ValueError("RepWorld requires the original ordered data.train_manifest")
    if request.latent_cache_mode == "online":
        if request.sources:
            raise ValueError(
                "Online RepWorld reads data.train_manifest; remove data.sources/cache_dir"
            )
        return
    if request.latent_cache_mode != "offline":
        raise ValueError("RepWorld latent_cache.mode must be online or offline")
    if (config.geometry.family, config.geometry.variant) != ("videorae", "3d"):
        raise ValueError("RepWorld offline shards currently support VideoRAE3D only")
    source = single_offline_source(request, recipe="RepWorld")
    if source is None or not source.cache_dir:
        raise ValueError("RepWorld offline training requires one data.sources entry with cache_dir")
    if (
        source.kind
        or source.index_files
        or source.manifest
        or source.weight != 1
        or source.min_latents is not None
        or source.max_latents is not None
    ):
        raise ValueError(
            "RepWorld cached source cannot filter, mix or replace the original manifest"
        )
    if source.extras:
        raise ValueError("RepWorld cached source does not accept raw dataset extras")
