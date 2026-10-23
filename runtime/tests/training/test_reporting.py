import sys

import torch

from vforge import (
    WandbReporter,
    print_json_step,
    scalar_step_metrics,
    wandb_step_metrics,
)
from vforge.hydra_config import LoggingHydraConfig
from vforge.training.reporting import (
    ConsoleReporter,
    global_mean_step_payload,
    resolve_wandb_run_name,
)


def test_scalar_step_metrics_accepts_tensor_and_python_scalars():
    assert scalar_step_metrics(
        {
            "tensor": torch.tensor(1.5),
            "float": 2.5,
            "integer": 3,
            "vector": torch.ones(2),
            "ignored": "value",
        }
    ) == {
        "tensor": 1.5,
        "float": 2.5,
        "integer": 3.0,
    }


def test_print_json_step_remains_rank_zero_compatible(monkeypatch, capsys):
    monkeypatch.setenv("RANK", "0")
    print_json_step({"step": 1, "loss": 0.5})
    assert capsys.readouterr().out == ('VFORGE_TRAIN {"loss":0.5,"step":1}\n')


def test_console_reporter_emits_rank_zero_json(capsys):
    reporter = ConsoleReporter(rank=0)
    reporter.report_step({"step": 1, "loss": 0.5})
    reporter.close()

    assert capsys.readouterr().out == ('VFORGE_TRAIN {"loss":0.5,"step":1}\n')


def test_global_mean_step_payload_keeps_rank_zero_and_global_losses(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(values, *, op):
        assert op == torch.distributed.ReduceOp.SUM
        values.add_(2.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    payload = global_mean_step_payload(
        {
            "step": 2,
            "loss": 1.0,
            "loss_backward": 0.5,
            "metrics": {"loss_scalar": 1.0, "custom": 2.0},
            "timing": {"step_s": 4.0},
        },
        device=torch.device("cpu"),
    )

    assert payload["loss_rank0"] == 1.0
    assert payload["loss_global"] == 1.5
    assert payload["loss"] == 1.5
    assert payload["loss_backward_rank0"] == 0.5
    assert payload["loss_backward_global"] == 1.25
    assert payload["loss_backward"] == 1.25
    assert payload["metrics"] == {"loss_scalar": 1.5, "custom": 2.0}
    assert payload["timing"] == {"step_s": 3.0}


def test_wandb_metrics_separate_loss_and_effective_learning_rate():
    metrics = wandb_step_metrics(
        {
            "step": 2,
            "loss": 0.8,
            "loss_backward": 0.1,
            "learning_rate": 2e-7,
            "learning_rate_before_step": 2e-7,
            "learning_rate_after_step": 3e-7,
            "metrics": {"loss": 99.0, "loss_scalar": 0.75},
            "timing": {
                "step_s": 12.5,
                "step_compute_s": 11.0,
                "checkpoint_s": 1.5,
            },
            "perf": {
                "tokens_per_sec": 1000.0,
                "tokens_per_sec_per_gpu": 1000.0,
                "global_tokens": 11000.0,
            },
        }
    )

    assert metrics["trainer/global_step"] == 2
    assert metrics["train/loss"] == 0.1
    assert metrics["train/loss_backward"] == 0.1
    assert metrics["train/loss_scalar"] == 0.75
    assert metrics["train/learning_rate"] == 2e-7
    assert metrics["train/learning_rate_after_step"] == 3e-7
    assert metrics["time/step"] == 12.5
    assert metrics["time/step_compute"] == 11.0
    assert metrics["time/checkpoint"] == 1.5
    assert metrics["perf/tokens_per_sec"] == 1000.0
    assert metrics["perf/tokens_per_sec_per_gpu"] == 1000.0
    assert metrics["perf/global_tokens"] == 11000.0


def test_wandb_defaults_to_enabled_with_vforge_project():
    config = LoggingHydraConfig()

    assert config.wandb_enabled is True
    assert config.wandb_project == "vforge"
    assert config.wandb_name == ""


def test_wandb_run_name_uses_explicit_environment_and_output_fallbacks(monkeypatch):
    monkeypatch.delenv("WANDB_NAME", raising=False)
    monkeypatch.delenv("CI_TASK_NAME", raising=False)

    assert (
        resolve_wandb_run_name(
            "explicit-name",
            output_dir="/efs/user/runs/output-name",
            model_name="wan22_a14b",
        )
        == "explicit-name"
    )
    monkeypatch.setenv("CI_TASK_NAME", "kap-task")
    assert (
        resolve_wandb_run_name(
            "",
            output_dir="/efs/user/runs/output-name",
            model_name="wan22_a14b",
        )
        == "kap-task"
    )
    monkeypatch.setenv("WANDB_NAME", "environment-name")
    assert (
        resolve_wandb_run_name(
            "",
            output_dir="/efs/user/runs/output-name",
            model_name="wan22_a14b",
        )
        == "environment-name"
    )
    monkeypatch.delenv("WANDB_NAME")
    monkeypatch.delenv("CI_TASK_NAME")
    assert (
        resolve_wandb_run_name(
            "",
            output_dir="/efs/user/runs/output-name",
            model_name="wan22_a14b",
        )
        == "output-name"
    )


def test_wandb_initialization_failure_does_not_stop_training(monkeypatch, capsys, tmp_path):
    class FailingWandb:
        @staticmethod
        def init(**_kwargs):
            raise RuntimeError("service unavailable")

    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "wandb", FailingWandb())

    reporter = WandbReporter(
        enabled=True,
        project="vforge",
        name="test-run",
        save_dir=str(tmp_path),
    )

    assert reporter._run is None
    assert "continuing without W&B" in capsys.readouterr().out


def test_wandb_reporting_failure_disables_sink_without_stopping_training(capsys):
    class FailingRun:
        @staticmethod
        def log(_metrics, *, step, commit):
            raise RuntimeError(f"failed at {step=} {commit=}")

    reporter = WandbReporter.__new__(WandbReporter)
    reporter._run = FailingRun()

    reporter.report_step({"step": 500, "loss": 0.8})

    assert reporter._run is None
    assert "continuing without W&B" in capsys.readouterr().out


def test_reporter_commits_one_complete_record_per_optimizer_step():
    class FakeRun:
        def __init__(self):
            self.calls = []

        def log(self, metrics, *, step, commit):
            self.calls.append((metrics, step, commit))

    run = FakeRun()
    reporter = WandbReporter.__new__(WandbReporter)
    reporter._run = run
    reporter.report_step(
        {
            "step": 499,
            "loss": 0.8,
            "loss_backward": 0.1,
            "learning_rate": 0.0,
            "learning_rate_before_step": 0.0,
            "learning_rate_after_step": 1e-7,
            "timing": {"step_s": 12.5},
        }
    )
    reporter.report_step(
        {
            "step": 500,
            "loss": 0.9,
            "loss_backward": 0.2,
            "learning_rate": 1e-7,
            "learning_rate_before_step": 1e-7,
            "learning_rate_after_step": 2e-7,
            "timing": {"step_s": 11.5},
        }
    )

    assert len(run.calls) == 2
    assert [call[0]["trainer/global_step"] for call in run.calls] == [499, 500]
    assert [call[1] for call in run.calls] == [499, 500]
    assert all(commit for _, _, commit in run.calls)
    assert run.calls[0][0]["train/loss_scalar"] == 0.8
    assert run.calls[0][0]["train/learning_rate"] == 0.0
    assert run.calls[0][0]["time/step"] == 12.5
    assert run.calls[1][0]["train/loss_scalar"] == 0.9
    assert run.calls[1][0]["train/learning_rate"] == 1e-7
    assert run.calls[1][0]["time/step"] == 11.5


def test_reporter_can_match_taichu_timing_semantics():
    class FakeRun:
        def __init__(self):
            self.calls = []

        def log(self, metrics, *, step, commit):
            self.calls.append((metrics, step, commit))

    run = FakeRun()
    reporter = WandbReporter.__new__(WandbReporter)
    reporter._run = run
    reporter._include_optimizer_in_backward = True
    reporter._exclude_dataloader_from_step = True

    reporter.report_step(
        {
            "step": 0,
            "loss": 0.8,
            "timing": {
                "dataloader_s": 1.0,
                "backward_s": 3.0,
                "optimizer_s": 2.0,
                "step_s": 10.0,
            },
        }
    )

    metrics = run.calls[0][0]
    assert metrics["time/backward"] == 5.0
    assert metrics["time/optimizer"] == 2.0
    assert metrics["time/step"] == 9.0
