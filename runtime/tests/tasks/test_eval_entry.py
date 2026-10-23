from __future__ import annotations

import pytest

from vforge.hydra_config import config_from_dict
from vforge.tasks.eval import _resolve_eval_checkpoints
from vforge.tasks.translate import eval_request


def test_eval_request_rejects_sampling_fields_under_model_extras() -> None:
    config = config_from_dict(
        {
            "model": {"extras": {"num_inference_steps": 40}},
            "evaluation": {
                "data_source": "validation",
                "manifest": "/data/val.jsonl",
            },
        }
    )

    with pytest.raises(ValueError, match="evaluation.num_inference_steps"):
        eval_request(config)


@pytest.mark.parametrize("legacy_field", ["eval_workflow", "gridnormal_variants", "rollout_mode"])
def test_eval_request_rejects_execution_fields_under_model_extras(
    legacy_field: str,
) -> None:
    config = config_from_dict(
        {
            "model": {"extras": {legacy_field: "legacy"}},
            "evaluation": {
                "data_source": "validation",
                "manifest": "/data/val.jsonl",
            },
        }
    )

    with pytest.raises(ValueError, match=legacy_field):
        eval_request(config)


def test_eval_latest_uses_the_shared_checkpoint_resolver(monkeypatch, tmp_path) -> None:
    observed = {}

    def resolve(value: str, directory: str) -> str:
        observed["args"] = (value, directory)
        return str(tmp_path / "00020.dcp")

    monkeypatch.setattr("vforge.tasks.eval.resolve_checkpoint_path", resolve)
    config = config_from_dict(
        {
            "checkpoint": {"dirpath": str(tmp_path)},
            "evaluation": {"checkpoint_path": "latest"},
        }
    )

    selections = _resolve_eval_checkpoints(config)

    assert observed["args"] == ("latest", str(tmp_path))
    assert selections[0].path == str(tmp_path / "00020.dcp")
    assert selections[0].output_name is None


def test_eval_all_checkpoints_uses_shared_discovery_and_stable_output_names(
    monkeypatch, tmp_path
) -> None:
    checkpoints = [
        (10, str(tmp_path / "00010.pt")),
        (20, str(tmp_path / "00020.dcp")),
    ]
    monkeypatch.setattr("vforge.tasks.eval.discover_checkpoints", lambda directory: checkpoints)
    monkeypatch.setattr("vforge.tasks.eval._checkpoint_matches_format", lambda *_args: True)
    config = config_from_dict(
        {
            "checkpoint": {"dirpath": str(tmp_path)},
            "evaluation": {"evaluate_all_checkpoints": True},
        }
    )

    selections = _resolve_eval_checkpoints(config)

    assert [(item.step, item.path, item.output_name) for item in selections] == [
        (10, checkpoints[0][1], "checkpoint-00010"),
        (20, checkpoints[1][1], "checkpoint-00020"),
    ]


def test_eval_all_checkpoints_rejects_an_explicit_single_checkpoint(tmp_path) -> None:
    config = config_from_dict(
        {
            "checkpoint": {"dirpath": str(tmp_path)},
            "evaluation": {
                "checkpoint_path": str(tmp_path / "00010.pt"),
                "evaluate_all_checkpoints": True,
            },
        }
    )

    try:
        _resolve_eval_checkpoints(config)
    except ValueError as exc:
        assert "must be empty, auto, or latest" in str(exc)
    else:
        raise AssertionError(
            "explicit checkpoint path must not be accepted when evaluating all checkpoints"
        )
