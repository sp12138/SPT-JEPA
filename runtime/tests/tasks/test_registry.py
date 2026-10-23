"""Model registry resolution, validation and laziness."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vforge.algorithms.dmd.model import DmdModelFamily  # noqa: E402
from vforge.tasks.registry import (  # noqa: E402
    VERBS,
    ModelTasks,
    model_names,
    register,
    require_registered,
    resolve_builder,
    resolve_dmd_family,
    resolve_train_preflight,
    unregister,
)


@pytest.fixture
def dummy_model():
    name = "dummy_registry_model"
    register(
        name,
        ModelTasks(
            train=f"{__name__}:_build_train",
            eval=f"{__name__}:_Tasks.build_eval",
            encode=f"{__name__}:_missing_attr",
        ),
    )
    yield name
    unregister(name)


def _build_train(request):
    return ("train", request)


class _Tasks:
    @staticmethod
    def build_eval(request):
        return ("eval", request)


def test_wan21_is_registered() -> None:
    assert "wan21" in model_names()
    assert resolve_train_preflight("wan21") is None
    assert callable(resolve_train_preflight("wan22_a14b"))


def test_unknown_and_empty_names_report_the_known_set() -> None:
    with pytest.raises(ValueError, match="model.name is required"):
        require_registered("")
    with pytest.raises(ValueError, match="unknown model.name='nope'"):
        require_registered("nope")
    with pytest.raises(ValueError, match="wan21"):
        require_registered("nope")


def test_resolve_accepts_a_function_and_a_method(dummy_model) -> None:
    assert resolve_builder(dummy_model, "train")(1) == ("train", 1)
    assert resolve_builder(dummy_model, "eval")(2) == ("eval", 2)


def test_resolve_rejects_unknown_verb_and_missing_attribute(dummy_model) -> None:
    with pytest.raises(ValueError, match="unknown verb"):
        resolve_builder(dummy_model, "sample")
    with pytest.raises(ImportError, match="_missing_attr"):
        resolve_builder(dummy_model, "encode")


def test_model_can_register_only_the_verbs_it_supports() -> None:
    name = "train_only_registry_model"
    register(name, ModelTasks(train=f"{__name__}:_build_train"))
    try:
        assert resolve_builder(name, "train")(1) == ("train", 1)
        with pytest.raises(
            ValueError,
            match="does not support verb='eval'; supported verbs: train",
        ):
            resolve_builder(name, "eval")
    finally:
        unregister(name)


def test_model_tasks_require_at_least_one_builder() -> None:
    with pytest.raises(ValueError, match="at least one builder"):
        ModelTasks()


@pytest.mark.parametrize("name", model_names())
def test_every_registered_builder_resolves(name: str) -> None:
    """Catch a renamed builder here rather than at runtime on a GPU job."""
    tasks = require_registered(name)
    for verb in VERBS:
        if getattr(tasks, verb):
            assert callable(resolve_builder(name, verb))
    if tasks.dmd:
        assert isinstance(resolve_dmd_family(name), DmdModelFamily)


def test_validation_does_not_import_the_model(monkeypatch) -> None:
    """``require_registered`` must stay cheap enough for compose-time checks."""

    def forbidden(name: str, *args, **kwargs):
        raise AssertionError(f"compose-time validation imported {name!r}")

    monkeypatch.setattr(importlib, "import_module", forbidden)
    assert require_registered("wan21").train.startswith("vforge.tasks.wan.v21")
