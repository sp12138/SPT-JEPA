#!/usr/bin/env python3
"""Fixed worker phases. No scheduling and no test assertions live here."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def step(name, command):
    started = time.monotonic()
    print(f"[stage] START {name}", flush=True)
    try:
        subprocess.run(command, check=True)
    finally:
        print(f"[stage] END {name} elapsed={time.monotonic() - started:.1f}s", flush=True)


def run(mode):
    os.chdir(ROOT)
    # A crashing rank must return promptly, not wait for the host core-dump collector.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.environ.update(CI_STRICT="1", LINT_SCOPE="changed", LINT_FILES_FILE=".ci/lint-files")
    metadata = json.loads(Path(".ci/metadata.json").read_text())
    if "vforge/models/gen_engine/dit" in metadata.get("submodules", {}):
        os.environ["VFORGE_REQUIRE_GEN_ENGINE_DIT"] = "1"
    if mode in {"cpu", "docs"}:
        if os.environ["CI_HOST_GPU_NUM"] != "0":
            raise RuntimeError("CPU stages must run without GPU allocation")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    step("environment", ["bash", "scripts/ci/kap/install_vforge.sh"])
    import torch

    if mode in {"cpu", "docs"} and list(Path("/dev").glob("nvidia[0-9]*")):
        raise RuntimeError("CPU worker exposes GPU devices")
    print(f"[runtime] torch={torch.__version__} GPUs={torch.cuda.device_count()}", flush=True)
    if mode == "cpu":
        step("mr_check", [sys.executable, "scripts/ci/mr_check.py", ".ci/metadata.json"])
        step("lint", ["bash", "scripts/ci/lint.sh"])
        step(
            "cpu pytest",
            [sys.executable, "scripts/ci/pytest_run.py", "--junit=.ci-results/cpu.xml", "tests"],
        )
    if mode in {"cpu", "docs"}:
        step("docs build", ["bash", "scripts/ci/docs_build.sh"])
    if mode == "gpu":
        group = os.environ.get("CI_GPU_GROUP", "all")
        if group in {"ltx", "all"}:
            step(
                "LTX eval dependencies",
                ["bash", "scripts/ci/kap/install_ltx_eval.sh", sys.executable],
            )
        step(
            f"gpu pytest ({group})",
            [
                sys.executable,
                "scripts/ci/pytest_run.py",
                f"--junit=.ci-results/gpu-{group}.xml",
                "--gpu",
                f"--gpu-group={group}",
                "-v",
                "-s",
                "tests",
            ],
        )
    if mode in {"cpu", "docs"}:
        metadata = json.loads(Path(".ci/metadata.json").read_text())
        Path("site/.vforge-site.json").write_text(
            json.dumps(
                {"source_sha": metadata["source_sha"], "base_path": os.environ["DOCS_BASE_PATH"]}
            )
        )
        with tarfile.open("/tmp/vforge-site.tar.gz", "w:gz") as archive:
            archive.add("site", arcname="site")
        request = urllib.request.Request(
            os.environ["SITE_UPLOAD_URL"],
            method="PUT",
            data=Path("/tmp/vforge-site.tar.gz").read_bytes(),
        )
        # Never log the signed URL; a failed upload must fail the task.
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 200:
                    raise RuntimeError(f"upload returned {response.status}")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"site artifact upload failed: HTTP {exc.code}") from None
        except Exception as exc:
            raise RuntimeError(f"site artifact upload failed: {type(exc).__name__}") from None
        print("[stage] site artifact uploaded", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["cpu", "gpu", "docs"])
    run(parser.parse_args().mode)
