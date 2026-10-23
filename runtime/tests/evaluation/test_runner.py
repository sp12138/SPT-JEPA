"""CPU tests for the generic evaluation lifecycle."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vforge.evaluation.runner import (
    EvalComponents,
    EvalOutputWriter,
    run_evaluation,
)
from vforge.training.engine import EngineRunContext


class _ToyDataset:
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"sample_id": f"s{index}", "x": torch.tensor([index], dtype=torch.float32)}


class _Engine:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def prepare_model(self, model, *, gradient_checkpointing_kwargs=None):
        self.calls.append(("prepare", model, gradient_checkpointing_kwargs))
        return model

    def materialize_model(self, model, device):
        self.calls.append(("materialize", model, torch.device(device)))
        return model.to(device)

    def prepare_data_parallel(self, _model):
        raise AssertionError("eval must not allocate data-parallel gradient buffers")

    def run_forward(self, forward, *, run):
        self.calls.append(("forward", run))
        with run.precision_context():
            return forward()


def _writer() -> EvalOutputWriter:
    def write_output(result: dict[str, Any], output_dir: str) -> str:
        path = Path(output_dir) / f"{result['sample_id']}.pt"
        torch.save(result, path)
        return str(path)

    return EvalOutputWriter(
        artifact_name=lambda sample_id: f"{sample_id}.pt",
        write_output=write_output,
    )


def _run(tmp_path: Path, components: EvalComponents, **kwargs: Any) -> tuple[dict, _Engine]:
    engine = _Engine()
    manifest = run_evaluation(
        components,
        engine=engine,  # type: ignore[arg-type]
        engine_run=EngineRunContext(precision_context=nullcontext),
        device=torch.device("cpu"),
        output_dir=str(tmp_path),
        **kwargs,
    )
    return manifest, engine


def test_run_evaluation_prepares_loads_builds_then_infers(tmp_path: Path) -> None:
    events: list[str] = []
    original = torch.nn.Linear(1, 1)

    class Inference:
        def __init__(self, model, engine_forward) -> None:
            assert model is original
            self.engine_forward = engine_forward

        def infer_sample(self, sample, *, seed: int):
            first = self.engine_forward(sample["x"])
            second = self.engine_forward(sample["x"])
            events.append(sample["sample_id"])
            return {
                "sample_id": sample["sample_id"],
                "seed": seed,
                "value": first + second,
            }

    components = EvalComponents(
        model=original,
        dataloader=_ToyDataset(),
        initial_weights=None,
        build_inference=Inference,
        output_writer=_writer(),
    )
    manifest, engine = _run(
        tmp_path,
        components,
        seed=7,
        num_samples=2,
        skip_existing=False,
        manifest_context={"resolved_config_hash": "abc"},
    )

    assert manifest["num_samples"] == 2
    assert events == ["s0", "s1"]
    assert len([call for call in engine.calls if call[0] == "forward"]) == 4
    assert [call[0] for call in engine.calls[:2]] == ["prepare", "materialize"]
    assert (tmp_path / "eval_manifest.json").is_file()


def test_inference_can_resolve_seed_from_the_global_sample_index(tmp_path: Path) -> None:
    observed: list[int] = []

    class Inference:
        def __init__(self, _model, _engine_forward) -> None:
            pass

        def resolve_sample_seed(self, sample, base_seed: int) -> int:
            return base_seed + int(sample["__vforge_eval_index__"])

        def infer_sample(self, sample, *, seed: int):
            observed.append(seed)
            return {"sample_id": sample["sample_id"], "seed": seed}

    components = EvalComponents(
        model=torch.nn.Linear(1, 1),
        dataloader=[
            {"sample_id": "arbitrary-id", "__vforge_eval_index__": 4},
        ],
        initial_weights=None,
        build_inference=Inference,
        output_writer=_writer(),
    )

    _run(tmp_path, components, seed=100, skip_existing=False)

    assert observed == [104]
    sidecar = json.loads((tmp_path / "arbitrary-id.pt.json").read_text())
    assert sidecar["seed"] == 104


def test_skip_requires_matching_provenance_sidecar(tmp_path: Path) -> None:
    calls = 0

    class Inference:
        def infer_sample(self, sample, *, seed: int):
            nonlocal calls
            calls += 1
            return {"sample_id": sample["sample_id"], "seed": seed, "value": sample["x"]}

    components = EvalComponents(
        model=torch.nn.Linear(1, 1),
        dataloader=_ToyDataset(),
        initial_weights=None,
        build_inference=lambda _model, _forward: Inference(),
        output_writer=_writer(),
    )
    (tmp_path / "s0.pt").write_bytes(b"legacy")
    _run(
        tmp_path,
        components,
        num_samples=1,
        skip_existing=True,
        manifest_context={"resolved_config_hash": "first"},
    )
    assert calls == 1

    _run(
        tmp_path,
        components,
        num_samples=1,
        skip_existing=True,
        manifest_context={"resolved_config_hash": "first"},
    )
    assert calls == 1

    _run(
        tmp_path,
        components,
        num_samples=1,
        skip_existing=True,
        manifest_context={"resolved_config_hash": "changed"},
    )
    assert calls == 2


def test_writer_rejects_path_that_disagrees_with_artifact_name(tmp_path: Path) -> None:
    def write_wrong_path(_result: dict[str, Any], output_dir: str) -> str:
        path = Path(output_dir) / "actual.pt"
        path.write_bytes(b"x")
        return str(path)

    writer = EvalOutputWriter(
        artifact_name=lambda _sample_id: "expected.pt",
        write_output=write_wrong_path,
    )
    with pytest.raises(RuntimeError, match="different"):
        writer.write(
            {"sample_id": "s0"},
            str(tmp_path),
            sample_id="s0",
            seed=1,
            provenance={},
        )


def test_training_checkpoint_source_uses_framework_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from vforge.evaluation import runner
    from vforge.training.checkpoint import TrainingCheckpointWeightSource

    observed: dict[str, Any] = {}

    @dataclass
    class Loaded:
        completed_steps: int

    def fake_load(model, path, **kwargs):
        assert all(parameter.requires_grad for parameter in model.parameters())
        observed.update(model=model, path=path, kwargs=kwargs)
        return Loaded(completed_steps=3)

    monkeypatch.setattr(runner, "load_model_weights", fake_load)
    model = torch.nn.Linear(1, 1)
    components = EvalComponents(
        model=model,
        dataloader=[],
        initial_weights=TrainingCheckpointWeightSource(
            path="/checkpoint",
            format="dcp",
        ),
        build_inference=lambda _model, _forward: SimpleNamespace(
            infer_sample=lambda _sample, seed: {}
        ),
        output_writer=_writer(),
    )
    manifest, _engine = _run(tmp_path, components, weights_strict=True)

    assert observed["model"] is model
    assert observed["path"] == "/checkpoint"
    assert observed["kwargs"]["format"] == "dcp"
    assert observed["kwargs"]["strict"] is True
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert manifest["checkpoint"]["sources"][0]["manifest"]["completed_steps"] == 3
