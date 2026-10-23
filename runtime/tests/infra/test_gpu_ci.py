"""Independent CI jobs keep GPU groups isolated and clean every remote task."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml


@pytest.fixture
def runner(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "scripts/ci"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("vforge_ci_run", directory / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pipeline_jobs_select_distinct_gpu_tasks(runner, tmp_path, monkeypatch):
    pipeline = yaml.safe_load((runner.ROOT / ".ci/ci.yml").read_text())
    cpu, gpu = pipeline["stages"]
    assert cpu["name"] == "checks"
    assert gpu["fast-kill"] is False
    assert set(gpu["jobs"]) == {f"gpu_{group}" for group in runner.GPU_GROUPS}
    execute = Mock()
    monkeypatch.setattr(runner.kap_run, "execute", execute)
    names = set()
    for group in runner.GPU_GROUPS:
        job = gpu["jobs"][f"gpu_{group}"]
        step = job["steps"][-1]
        assert f"--gpu-group {group} " in step["run"]
        assert 'git checkout --detach "${SOURCE}"' in step["run"]
        assert "submodule update --init --checkout" in step["run"]
        assert job["timeout-minutes"] > step["timeout-minutes"] > 120
        runner.task("gpu", "b300", tmp_path, run_id="test", gpu_group=group)
        args, kwargs = execute.call_args
        spec = runner.kap_run.render(
            args[0],
            args[1],
            kwargs["name"],
            env=kwargs["env"],
            timeout_s=kwargs["timeout_s"],
        )
        assert spec["runPolicy"]["activeDeadlineSeconds"] == 7200
        assert spec["envVars"]["CI_GPU_GROUP"] == group
        names.add(spec["taskName"])
    assert len(names) == len(runner.GPU_GROUPS)
    execute.side_effect = RuntimeError("GPU regression failed")
    with pytest.raises(RuntimeError, match="GPU regression failed"):
        runner.task("gpu", "b300", tmp_path, run_id="test", gpu_group="ltx")


def test_final_cleanup_attempts_every_group_even_after_failure(runner, monkeypatch):
    visited = []

    def cleanup(spec):
        visited.append(spec["taskName"])
        if spec["taskName"] == runner.task_name("gpu-core", "test"):
            raise RuntimeError("cleanup failed")

    client = Mock()
    monkeypatch.setattr(runner.kap_run, "cleanup", cleanup)
    monkeypatch.setattr(runner, "artifact_store", lambda run_id: (client, "bucket", "key"))
    with pytest.raises(RuntimeError, match="CI cleanup failed"):
        runner.cleanup("mr", "test")
    assert set(visited) == {
        runner.task_name(phase, "test") for phase in ("cpu", "gpu-core", "gpu-ltx", "gpu-wan")
    }
    client.delete_object.assert_called_once_with(Bucket="bucket", Key="key")
