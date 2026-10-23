"""Wan 2.2 TI2V-5B hooks for causal consistency distillation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch.nn as nn

from vforge.algorithms.cd.model import CdFamilySpec, CdTeacherForcingAdapter
from vforge.models.factory import patch_diffusers_getattr
from vforge.tasks.requests import TrainingModelRequest, TrainRequest
from vforge.tasks.wan.shared import parameter_storage_dtype
from vforge.tasks.wan_taichu.cd.adapter import (
    Wan5bCdTeacherForcingAdapter,
    WanA14bCdTeacherForcingAdapter,
)
from vforge.tasks.wan_taichu.cd.data import build_wan_cd_data_support
from vforge.tasks.wan_taichu.cd.schedule import create_wan_cd_schedule
from vforge.tasks.wan_taichu.train import (
    _build_model,
    _resolve_request,
    validate_train_request,
)

_ROLES = frozenset({"teacher", "student", "ema_model"})
_STRUCTURAL_FIELDS = (
    "variant",
    "condition_streams",
    "frame_per_block",
    "first_frame_own_block",
    "local_attn_size",
    "sink_size",
    "boundary_ratio",
)


def validate_wan_cd(request: TrainRequest, engine: str) -> None:
    if engine != "fsdp2":
        raise NotImplementedError("Wan CD first release supports FSDP2 only")
    if request.latent_cache_mode != "offline":
        raise NotImplementedError("Wan CD first release requires an offline latent cache")
    if set(request.training_models) != set(_ROLES):
        raise ValueError(f"Wan CD requires exactly roles {sorted(_ROLES)}")
    student = request.training_models["student"]
    student_variant = str(student.config.get("variant") or "")
    supported_variants = {
        "wan22_ti2v_5b_causal_teacher",
        "wan22_a14b_causal_teacher",
    }
    if student_variant not in supported_variants:
        raise NotImplementedError(
            f"Wan CD does not support variant {student_variant!r}; "
            f"expected one of {sorted(supported_variants)}"
        )
    for role, model in request.training_models.items():
        if model.model != "wan_taichu":
            raise ValueError(f"Wan CD role {role!r} must use model='wan_taichu'")
        variant = str(model.config.get("variant") or "")
        if variant != student_variant:
            raise ValueError(f"Wan CD role {role!r} disagrees with student.variant")
        for field in _STRUCTURAL_FIELDS:
            if model.config.get(field) != student.config.get(field):
                raise ValueError(f"Wan CD role {role!r} disagrees with student.{field}")
        if model.pretrained_path != student.pretrained_path:
            raise ValueError(f"Wan CD role {role!r} must share student.pretrained_path")
        if student_variant == "wan22_ti2v_5b_causal_teacher":
            if model.init_weights != student.init_weights:
                raise ValueError(f"Wan 5B CD role {role!r} must share student.init_weights")
            if model.weight_loading != student.weight_loading:
                raise ValueError(f"Wan 5B CD role {role!r} must share student.weight_loading")
            if model.config.get("init_source_prefix") != student.config.get("init_source_prefix"):
                raise ValueError(f"Wan 5B CD role {role!r} must share student.init_source_prefix")
    ema_model = request.training_models["ema_model"]
    if student_variant == "wan22_a14b_causal_teacher":
        if ema_model.init_weights != student.init_weights:
            raise ValueError("Wan A14B CD ema_model must initialize from student weights")
        if ema_model.weight_loading != student.weight_loading:
            raise ValueError("Wan A14B CD ema_model must share student.weight_loading")
        if ema_model.config.get("init_source_prefix") != student.config.get("init_source_prefix"):
            raise ValueError("Wan A14B CD ema_model must share student.init_source_prefix")
    streams = tuple(student.config.get("condition_streams") or ())
    if int(student.config.get("frame_per_block", 0)) != 3:
        raise ValueError("Wan CD requires frame_per_block=3")
    if student_variant == "wan22_ti2v_5b_causal_teacher":
        if streams != ("grid", "normal"):
            raise ValueError("Wan 5B CD requires condition_streams=[grid, normal]")
        if student.config.get("first_frame_own_block") is not True:
            raise ValueError("Wan 5B CD requires first_frame_own_block=true")
        if int(student.config.get("local_attn_size", 0)) != 31:
            raise ValueError("Wan 5B CD requires local_attn_size=31")
        if int(student.config.get("sink_size", -1)) != 10:
            raise ValueError("Wan 5B CD requires sink_size=10")
        if request.usp_size not in {1, 2, 4, 8}:
            raise ValueError("Wan 5B CD requires parallel.usp_size in {1,2,4,8}")
    else:
        if streams != ("rgb", "grid", "normal"):
            raise ValueError("Wan A14B CD requires condition_streams=[rgb, grid, normal]")
        if student.config.get("first_frame_own_block") is not False:
            raise ValueError("Wan A14B CD requires first_frame_own_block=false")
        if float(student.config.get("boundary_ratio", 0.0)) != 0.9:
            raise ValueError("Wan A14B CD requires boundary_ratio=0.9")
        if int(student.config.get("local_attn_size", 0)) != 30:
            raise ValueError("Wan A14B CD requires local_attn_size=30")
        if int(student.config.get("sink_size", -1)) != 9:
            raise ValueError("Wan A14B CD requires sink_size=9")
        if request.usp_size not in {1, 2, 4, 5, 8, 10, 20, 40}:
            raise ValueError("Wan A14B CD received an unsupported USP size")
    source_prefix = str(student.config.get("init_source_prefix") or "")
    if source_prefix and Path(student.init_weights).suffix not in {".pt", ".pth"}:
        raise ValueError("Wan CD init_source_prefix is only supported for .pt/.pth checkpoints")
    provider = _role_train_request(student, request=request)
    validate_train_request(provider, engine)


def build_wan_cd_role_model(request: TrainRequest, role: str) -> nn.Module:
    model_request = request.training_models[role]
    role_request = _role_train_request(model_request, request=request)
    spec, streams, extras = _resolve_request(role_request)
    model, _layouts = _build_model(
        spec=spec,
        streams=streams,
        extras=extras,
        pretrained_path=role_request.pretrained_path,
        init_device=role_request.init_device,
        parameter_dtype=parameter_storage_dtype(),
    )
    for transformer in model.fsdp_execution_roots():
        patch_diffusers_getattr(transformer)
        transformer.fsdp_forward_methods = lambda: ("forward_teacher_forcing",)
    if spec.name == "wan22_a14b_causal_teacher":
        return model
    return model.primary_transformer()


def create_wan_cd_adapter(request: TrainRequest) -> CdTeacherForcingAdapter:
    variant = str(request.training_models["student"].config.get("variant") or "")
    if variant == "wan22_a14b_causal_teacher":
        return WanA14bCdTeacherForcingAdapter(precision=request.precision)
    return Wan5bCdTeacherForcingAdapter(precision=request.precision)


def _role_train_request(
    model: TrainingModelRequest,
    *,
    request: TrainRequest,
) -> TrainRequest:
    return replace(
        request,
        extras=dict(model.config),
        pretrained_path=model.pretrained_path,
        init_device=model.init_device,
        weight_loading=model.weight_loading,
    )


WAN_TAICHU_CD = CdFamilySpec(
    model_name="wan_taichu",
    validate=validate_wan_cd,
    build_role_model=build_wan_cd_role_model,
    build_data_support=build_wan_cd_data_support,
    create_adapter=create_wan_cd_adapter,
    create_schedule=create_wan_cd_schedule,
)


__all__ = ["WAN_TAICHU_CD"]
