"""Select explicit GPU tests independently of CPU contracts."""

import importlib.util
import os
import signal
import subprocess
import sys
from contextlib import suppress

import pytest


@pytest.fixture
def gen_engine_dit():
    """Optional checkout for base CPU runs; mandatory in a Gen Engine CI bundle."""
    try:
        spec = importlib.util.find_spec("vforge.models.gen_engine.dit.model")
    except ModuleNotFoundError as exc:
        if exc.name not in {"vforge.models.gen_engine.dit", "vforge.models.gen_engine.dit.model"}:
            raise
        spec = None
    if spec is None:
        message = "private Gen Engine DiT is absent; initialize the pinned submodule"
        if os.environ.get("VFORGE_REQUIRE_GEN_ENGINE_DIT") == "1":
            pytest.fail(message)
        pytest.skip(message)
    # A broken implementation or missing transitive dependency must fail normally.
    return importlib.import_module("vforge.models.gen_engine.dit.model")


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", help="run explicitly marked GPU tests only")
    parser.addoption("--gpu-group", choices=("all", "core", "ltx", "wan"), default="all")


def gpu_group(nodeid):
    path = nodeid.split("::", 1)[0]
    if path.startswith(("tests/models/ltx/", "tests/tasks/ltx/")):
        return "ltx"
    if path.startswith(
        (
            "tests/models/wan_taichu/",
            "tests/tasks/wan_taichu/",
            "tests/models/wan/",
            "tests/tasks/wan/",
        )
    ):
        return "wan"
    return "core"


def pytest_collection_modifyitems(config, items):
    selected, deselected = [], []
    group = config.getoption("--gpu-group")
    if group != "all" and not config.getoption("--gpu"):
        raise pytest.UsageError("--gpu-group requires --gpu")
    for item in items:
        marker = item.get_closest_marker("gpu")
        if marker is not None:
            count = marker.kwargs.get("gpus")
            if marker.args or set(marker.kwargs) != {"gpus"} or type(count) is not int or count < 1:
                raise pytest.UsageError(f"{item.nodeid}: use @pytest.mark.gpu(gpus=<positive int>)")
        matches = bool(marker) == config.getoption("--gpu")
        if group != "all":
            matches = matches and gpu_group(item.nodeid) == group
        (selected if matches else deselected).append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def pytest_runtest_setup(item):
    marker = item.get_closest_marker("gpu")
    if marker:
        import torch

        required = marker.kwargs["gpus"]
        actual = torch.cuda.device_count()
        if actual < required:
            pytest.fail(f"requires {required} CUDA devices, found {actual}")


@pytest.fixture
def torchrun(request):
    """Run a distributed GPU test worker using the test's declared rank count."""
    marker = request.node.get_closest_marker("gpu")
    if marker is None:
        raise pytest.UsageError("the torchrun fixture requires @pytest.mark.gpu(gpus=N)")

    def launch(worker, *args, timeout=300, nproc=None):
        count = marker.kwargs["gpus"] if nproc is None else nproc
        if type(count) is not int or not 1 <= count <= marker.kwargs["gpus"]:
            raise ValueError("nproc must be within the test's declared GPU allocation")
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={count}",
            str(worker),
            *map(str, args),
        ]
        # All ranks belong to one group, including after the launcher exits.
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        ) as process:
            try:
                output, _ = process.communicate(timeout=timeout)
            except BaseException:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
                print(output, flush=True)
                raise
            print(output, flush=True)
            assert process.returncode == 0, f"torchrun {worker} failed ({process.returncode})"

    return launch
