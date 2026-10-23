import logging
from contextlib import nullcontext

import pytest
import torch

import vforge.training.profiling as profiling
from vforge.training.profiling import (
    advance_profiler,
    build_torch_profiler,
    record_region,
    stop_profiler,
)


def test_profiler_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VFORGE_TORCH_PROFILE", raising=False)

    assert build_torch_profiler(rank=0) is None
    assert isinstance(record_region(None, "test"), nullcontext)


def test_profiler_only_builds_on_selected_rank(monkeypatch, tmp_path):
    monkeypatch.setenv("VFORGE_TORCH_PROFILE", "1")
    monkeypatch.setenv("VFORGE_TORCH_PROFILE_RANK", "1")
    monkeypatch.setenv(
        "VFORGE_TORCH_PROFILE_OUTPUT",
        str(tmp_path / "rank{rank}"),
    )

    assert build_torch_profiler(rank=0) is None


def test_profiler_requires_output_path(monkeypatch):
    monkeypatch.setenv("VFORGE_TORCH_PROFILE", "1")
    monkeypatch.setenv("VFORGE_TORCH_PROFILE_RANK", "0")
    monkeypatch.delenv("VFORGE_TORCH_PROFILE_OUTPUT", raising=False)

    with pytest.raises(ValueError, match="VFORGE_TORCH_PROFILE_OUTPUT"):
        build_torch_profiler(rank=0)


def test_profiler_builds_expected_schedule(monkeypatch, tmp_path):
    captured = {}
    schedule = object()
    profiler = object()
    monkeypatch.setenv("VFORGE_TORCH_PROFILE", "1")
    monkeypatch.setenv("VFORGE_TORCH_PROFILE_RANK", "0")
    monkeypatch.setenv(
        "VFORGE_TORCH_PROFILE_OUTPUT",
        str(tmp_path / "rank{rank}"),
    )
    monkeypatch.setenv("VFORGE_TORCH_PROFILE_WAIT", "2")
    monkeypatch.setenv("VFORGE_TORCH_PROFILE_WARMUP", "1")
    monkeypatch.setenv("VFORGE_TORCH_PROFILE_ACTIVE", "3")
    monkeypatch.setenv("VFORGE_TORCH_PROFILE_STACK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def fake_schedule(**kwargs):
        captured["schedule"] = kwargs
        return schedule

    def fake_profile(**kwargs):
        captured["profile"] = kwargs
        return profiler

    monkeypatch.setattr(torch.profiler, "schedule", fake_schedule)
    monkeypatch.setattr(torch.profiler, "profile", fake_profile)

    assert build_torch_profiler(rank=0) is profiler
    assert captured["schedule"] == {
        "wait": 2,
        "warmup": 1,
        "active": 3,
        "repeat": 1,
    }
    assert captured["profile"]["schedule"] is schedule
    assert captured["profile"]["activities"] == [torch.profiler.ProfilerActivity.CPU]
    assert captured["profile"]["record_shapes"] is False
    assert captured["profile"]["profile_memory"] is False
    assert captured["profile"]["with_stack"] is True


def test_profiler_emits_only_first_scheduled_trace(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setenv("VFORGE_TORCH_PROFILE", "1")
    monkeypatch.setenv(
        "VFORGE_TORCH_PROFILE_OUTPUT",
        str(tmp_path / "rank{rank}"),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.profiler,
        "schedule",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        torch.profiler,
        "profile",
        lambda **kwargs: captured.setdefault("kwargs", kwargs) or object(),
    )
    build_torch_profiler(rank=0)
    handler = captured["kwargs"]["on_trace_ready"]
    exports = []

    class Averages:
        @staticmethod
        def table(**_kwargs):
            return "table"

    class Profiler:
        step_num = 2

        @staticmethod
        def export_chrome_trace(path):
            exports.append(path)

        @staticmethod
        def key_averages():
            return Averages()

    profiler = Profiler()
    handler(profiler)
    profiler.step_num = 3
    handler(profiler)

    assert len(exports) == 1
    assert exports[0].endswith("trace-step2.json")
    assert (tmp_path / "rank0" / "key-averages-step2.txt").is_file()
    assert not (tmp_path / "rank0" / "key-averages-step3.txt").exists()


def test_cuda_profiler_prewarm_runs_one_disposable_session(monkeypatch):
    events = []
    monkeypatch.setattr(profiling, "_CUDA_PROFILER_WARMED", False)

    class Context:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_args):
            events.append("exit")

    class Tensor:
        def add_(self, value):
            events.append(("add", value))

    def fake_profile(**kwargs):
        events.append(("profile", kwargs["activities"]))
        return Context()

    monkeypatch.setattr(torch.profiler, "profile", fake_profile)
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: events.append("current_device") or 0,
    )
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: events.append(("empty", args, kwargs)) or Tensor(),
    )
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda: events.append("synchronize"),
    )

    profiling._warmup_cuda_profiler()
    profiling._warmup_cuda_profiler()

    assert [event for event in events if event == "enter"] == ["enter"]
    assert ("add", 1) in events
    assert events[-2:] == ["synchronize", "exit"]


def test_advance_profiler_synchronizes_recorded_cuda_step(monkeypatch):
    events = []
    profiler = type(
        "Profiler",
        (),
        {
            "current_action": torch.profiler.ProfilerAction.RECORD_AND_SAVE,
            "step": lambda _self: events.append("step"),
        },
    )()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: events.append(("synchronize", device)),
    )
    device = torch.device("cuda", 0)

    advance_profiler(profiler, device)

    assert events == [("synchronize", device), "step"]


def test_advance_profiler_does_not_sync_wait_step(monkeypatch):
    events = []
    profiler = type(
        "Profiler",
        (),
        {
            "current_action": torch.profiler.ProfilerAction.NONE,
            "step": lambda _self: events.append("step"),
        },
    )()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda _device: events.append("synchronize"),
    )

    advance_profiler(profiler, torch.device("cuda", 0))

    assert events == ["step"]


def test_stop_profiler_handles_optional_profiler():
    events = []

    class Profiler:
        @staticmethod
        def stop():
            events.append("stop")

    stop_profiler(None)
    stop_profiler(Profiler())

    assert events == ["stop"]


def test_stop_profiler_does_not_disrupt_cleanup(caplog):
    class Profiler:
        @staticmethod
        def stop():
            raise RuntimeError("boom")

    with caplog.at_level(logging.WARNING, logger=profiling.__name__):
        stop_profiler(Profiler())

    assert "Failed to stop profiler cleanly" in caplog.text
    assert caplog.records[-1].exc_info is not None
