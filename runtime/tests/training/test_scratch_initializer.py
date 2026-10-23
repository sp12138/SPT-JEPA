import pytest
import torch

import vforge.training.trainer as trainer_module
from vforge import (
    FSDP2Config,
    ParallelConfig,
    ParallelizeConfig,
    StepOutput,
    TrainComponents,
    Trainer,
    TrainerConfig,
)
from vforge.models.loading import PretrainedWeightSource
from vforge.training.checkpoint import ResumeConfig


def _config(*, resume: ResumeConfig | None = None) -> TrainerConfig:
    return TrainerConfig(
        max_steps=0,
        parallelize=ParallelizeConfig(
            parallel=ParallelConfig(),
            fsdp2=FSDP2Config(enable=False),
        ),
        precision="fp32",
        resume=resume or ResumeConfig(),
    )


def _components(
    model: torch.nn.Module,
    *,
    model_initializer=None,
    pretrained_weights=None,
) -> TrainComponents:
    return TrainComponents(
        model=model,
        dataloader=[],
        optimizer_factory=lambda _model: None,
        step_fn=lambda *_args: StepOutput(loss=torch.tensor(0.0)),
        pretrained_weights=pretrained_weights,
        model_initializer=model_initializer,
    )


def test_pretrained_loading_suppresses_scratch_initializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(2, 1)
    source = PretrainedWeightSource(path="/unused/published")
    initialized: list[torch.nn.Module] = []
    loaded: list[tuple[torch.nn.Module, PretrainedWeightSource, bool]] = []

    monkeypatch.setattr(
        trainer_module,
        "stage_pretrained_weights",
        lambda candidate: candidate,
    )
    monkeypatch.setattr(
        trainer_module,
        "load_pretrained_weights",
        lambda candidate, staged, *, strict: loaded.append((candidate, staged, strict)),
    )

    Trainer(_config()).fit(
        _components(
            model,
            model_initializer=initialized.append,
            pretrained_weights=source,
        )
    )

    assert initialized == []
    assert loaded == [(model, source, False)]


class _StopAtResumeLoad(RuntimeError):
    pass


def test_non_strict_full_resume_uses_scratch_initializer_as_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    resume_path = tmp_path / "compatible.dcp"
    resume_path.mkdir()
    initialized: list[torch.nn.Module] = []
    monkeypatch.setattr(
        trainer_module,
        "read_manifest",
        lambda _path: (_ for _ in ()).throw(_StopAtResumeLoad),
    )

    with pytest.raises(_StopAtResumeLoad):
        Trainer(_config(resume=ResumeConfig(strict=False))).fit(
            _components(
                torch.nn.Linear(2, 1),
                model_initializer=initialized.append,
            ),
            resume_path=str(resume_path),
        )

    assert len(initialized) == 1


def test_compatible_meta_overlay_accepts_scratch_initializer_as_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    with torch.device("meta"):
        model = torch.nn.Linear(2, 1)
    seed = 23
    loaded_values: list[torch.Tensor] = []

    def initialize(candidate: torch.nn.Module) -> None:
        assert candidate is model
        assert all(not parameter.is_meta for parameter in candidate.parameters())
        with torch.no_grad():
            for parameter in candidate.parameters():
                parameter.fill_(seed)

    def load_overlay(candidate, *_args, **_kwargs) -> None:
        loaded_values.extend(parameter.detach().clone() for parameter in candidate.parameters())

    monkeypatch.setattr(trainer_module, "load_model_weights", load_overlay)

    Trainer(_config()).fit(
        _components(model, model_initializer=initialize),
        resume_path=str(tmp_path / "overlay.pt"),
    )

    assert loaded_values
    assert all(torch.all(value == seed) for value in loaded_values)


@pytest.mark.parametrize("resume_path", [None, "overlay.pt"])
def test_meta_model_without_weights_or_initializer_is_rejected(
    tmp_path,
    resume_path: str | None,
) -> None:
    with torch.device("meta"):
        model = torch.nn.Linear(2, 1)

    resolved_resume = None if resume_path is None else str(tmp_path / resume_path)
    with pytest.raises(RuntimeError, match="has no pretrained weights or scratch initializer"):
        Trainer(_config()).fit(
            _components(model),
            resume_path=resolved_resume,
        )
