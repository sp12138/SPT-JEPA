"""Map a model name onto the builders that assemble its verbs.

The registry stores dotted paths rather than imported objects, so composing a
config never pays for a model's heavy dependencies: resolution happens when an
entry actually needs one builder. Keeping it a leaf module (no config imports)
is what lets the training layer stay unaware of which models exist.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

VERBS = ("train", "eval", "encode")


@dataclass(frozen=True)
class ModelTasks:
    """Optional ``package.module:attr`` paths to one model's builders.

    ``attr`` may traverse attributes, so both a free function and a method on a
    task class are valid targets. A model only registers the verbs it supports.
    """

    train: str | None = None
    eval: str | None = None
    encode: str | None = None
    train_preflight: str | None = None
    dmd: str | None = None
    cd: str | None = None

    def __post_init__(self) -> None:
        if not any(getattr(self, verb) for verb in VERBS) and self.dmd is None and self.cd is None:
            raise ValueError("model tasks must define at least one builder")


_REGISTRY: dict[str, ModelTasks] = {
    "rep_world": ModelTasks(
        train="vforge.tasks.rep_world.train:build_train_components",
        eval="vforge.tasks.rep_world.eval:build_eval_components",
        train_preflight="vforge.tasks.rep_world.shared:validate_train_request",
    ),
    "gen_engine": ModelTasks(
        train="vforge.tasks.gen_engine.train:build_train_components",
        eval="vforge.tasks.gen_engine.eval:build_eval_components",
        train_preflight="vforge.tasks.gen_engine.shared:validate_execution",
    ),
    "wan_taichu": ModelTasks(
        train="vforge.tasks.wan_taichu.train:build_train_components",
        eval="vforge.tasks.wan_taichu.eval:build_eval_components",
        train_preflight="vforge.tasks.wan_taichu.train:validate_train_request",
        dmd="vforge.tasks.wan_taichu.dmd.integration:WAN_TAICHU_DMD",
        cd="vforge.tasks.wan_taichu.cd.integration:WAN_TAICHU_CD",
    ),
    "wan21": ModelTasks(
        train="vforge.tasks.wan.v21.train:build_train_components",
        eval="vforge.tasks.wan.v21.eval:build_eval_components",
        encode="vforge.tasks.wan.v21.encode:build_encode_components",
    ),
    "wan22_a14b": ModelTasks(
        train="vforge.tasks.wan.v22.a14b.train:build_train_components",
        eval="vforge.tasks.wan.v22.a14b.eval:build_eval_components",
        encode="vforge.tasks.wan.v22.a14b.encode:build_encode_components",
        train_preflight="vforge.tasks.wan.v22.a14b.train:validate_train_request",
    ),
    "wan22_ti2v_5b": ModelTasks(
        train="vforge.tasks.wan.v22.ti2v_5b.ct.train:build_train_components",
        eval="vforge.tasks.wan.v22.ti2v_5b.eval:build_eval_components",
        encode="vforge.tasks.wan.v22.ti2v_5b.encode:build_encode_components",
    ),
    # Teacher forcing shares data, encode and weights with TI2V, while eval
    # builds its causal transformer and generated-history rollout.
    "wan22_ti2v_5b_tf": ModelTasks(
        train="vforge.tasks.wan.v22.ti2v_5b.teacher_forcing.train:build_train_components",
        eval="vforge.tasks.wan.v22.ti2v_5b.teacher_forcing.eval:build_eval_components",
        encode="vforge.tasks.wan.v22.ti2v_5b.encode:build_encode_components",
    ),
    "minimax_h3": ModelTasks(
        train="vforge.tasks.minimax_h3.train:build_train_components",
        eval="vforge.tasks.minimax_h3.eval:build_eval_components",
        encode="vforge.tasks.minimax_h3.encode:build_encode_components",
    ),
    "ltx23": ModelTasks(
        eval="vforge.tasks.ltx.dmd.eval:build_eval_components",
        dmd="vforge.tasks.ltx.dmd.integration:LTX23_DMD",
    ),
    "ltx23_ti2av_sft": ModelTasks(
        train="vforge.tasks.ltx.train:build_train_components",
    ),
    "ltx23_causal_ar_tf": ModelTasks(
        train="vforge.tasks.ltx.causal_train:build_train_components",
    ),
}


def register(name: str, tasks: ModelTasks) -> None:
    """Add a model, for tests and out-of-tree plugins."""
    if not name:
        raise ValueError("model name must be non-empty")
    _REGISTRY[name] = tasks


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def model_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def require_registered(name: str) -> ModelTasks:
    """Validate ``model.name`` without importing the model."""
    known = ", ".join(model_names()) or "<none>"
    if not name:
        raise ValueError(f"model.name is required; known models: {known}")
    tasks = _REGISTRY.get(name)
    if tasks is None:
        raise ValueError(f"unknown model.name={name!r}; known models: {known}")
    return tasks


def resolve_builder(name: str, verb: str) -> Callable[..., Any]:
    """Import and return the builder ``name`` registered for ``verb``."""
    if verb not in VERBS:
        raise ValueError(f"unknown verb={verb!r}; expected one of {', '.join(VERBS)}")
    tasks = require_registered(name)
    path = getattr(tasks, verb)
    if not path:
        supported = ", ".join(candidate for candidate in VERBS if getattr(tasks, candidate))
        raise ValueError(
            f"model {name!r} does not support verb={verb!r}; supported verbs: {supported}"
        )
    return _resolve_callable(name, verb, path)


def resolve_train_preflight(name: str) -> Callable[..., Any] | None:
    """Return a model's allocation-free train compatibility check, if any."""
    path = require_registered(name).train_preflight
    if path is None:
        return None
    return _resolve_callable(name, "train preflight", path)


def resolve_dmd_family(name: str):
    """Resolve the model family's single declarative DMD definition."""

    from vforge.algorithms.dmd.model import family_from_spec

    path = require_registered(name).dmd
    if path is None:
        raise ValueError(f"model {name!r} does not support DMD")
    return family_from_spec(_resolve_attribute(name, "dmd", path))


def resolve_cd_family(name: str):
    """Resolve the model family's single declarative CD definition."""

    from vforge.algorithms.cd.model import family_from_spec

    path = require_registered(name).cd
    if path is None:
        raise ValueError(f"model {name!r} does not support CD")
    return family_from_spec(_resolve_attribute(name, "cd", path))


def _resolve_callable(name: str, verb: str, path: str) -> Callable[..., Any]:
    target = _resolve_attribute(name, verb, path)
    if not callable(target):
        raise TypeError(f"model {name!r} {verb} builder {path!r} is not callable")
    return target


def _resolve_attribute(name: str, verb: str, path: str) -> Any:
    module_name, _, attr_path = path.partition(":")
    if not module_name or not attr_path:
        raise ValueError(
            f"model {name!r} {verb} builder must be 'package.module:attr', got {path!r}"
        )
    target: Any = importlib.import_module(module_name)
    for part in attr_path.split("."):
        try:
            target = getattr(target, part)
        except AttributeError as exc:
            raise ImportError(
                f"model {name!r} {verb} builder {path!r} does not define {part!r}"
            ) from exc
    return target


__all__ = [
    "VERBS",
    "ModelTasks",
    "model_names",
    "register",
    "require_registered",
    "resolve_builder",
    "resolve_cd_family",
    "resolve_dmd_family",
    "resolve_train_preflight",
    "unregister",
]
