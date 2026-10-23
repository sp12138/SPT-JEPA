#!/usr/bin/env python3
"""KAP task lifecycle only: render, submit, follow, save logs and delete."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

MACHINE_DIR = Path(__file__).parent / "kap/machines"
LOG_TAIL = 20000


class TaskError(RuntimeError):
    pass


def koala(*args, timeout=60):
    return subprocess.run(
        ["koala-cli", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def checked(*args, timeout=60):
    result = koala(*args, timeout=timeout)
    if result.returncode:
        raise TaskError(f"koala-cli {args[0]} {args[1]} failed: {result.stderr.strip()}")
    return result.stdout


def load_machine(machine):
    path = Path(machine) if str(machine).endswith(".yaml") else MACHINE_DIR / f"{machine}.yaml"
    return yaml.safe_load(path.read_text())


def render(template, machine, name, *, bundle=None, overrides=None, env=None, timeout_s=1800):
    spec = yaml.safe_load(Path(template).read_text())
    profile = load_machine(machine)
    profile_env = profile.pop("env", {})
    expected_gpu = profile.pop("gpu", "")
    spec.update(profile)
    spec.update(overrides or {})
    hosts, gpus = int(spec["hostNum"]), int(spec["hostGpuNum"])
    if hosts < 1 or gpus < 0:
        raise TaskError("invalid host/GPU count")
    for field in ("hostCpuNum", "hostMemoryGb"):
        if field in spec and int(spec[field]) < 1:
            raise TaskError(f"{field} must be positive")
    spec.update(taskName=name, cleanPodPolicy="None", backoffLimit=0, ttlSecondsAfterFinished=300)
    spec["runPolicy"] = {"activeDeadlineSeconds": timeout_s}
    values = dict(spec.get("envVars", {}), **profile_env)
    values.update(
        CI_TASK_NAME=name,
        CI_MACHINE=Path(machine).stem,
        CI_HOST_NUM=str(hosts),
        CI_HOST_GPU_NUM=str(gpus),
        LOCAL_WORLD_SIZE=str(max(1, gpus)),
        CI_WORLD_SIZE=str(hosts * max(1, gpus)),
        CI_EXPECT_GPU=expected_gpu if gpus else "",
    )
    values.update(env or {})
    spec["envVars"] = {str(k): str(v) for k, v in values.items()}
    if bundle:
        spec["codePath"] = str(Path(bundle).resolve(strict=True))
    return spec


def routing(spec):
    return [
        "--cluster",
        spec["clusterId"],
        "--project-id",
        spec["projectId"],
        "--namespace",
        spec["namespace"],
    ]


def status_of(spec):
    result = koala("task", "get", spec["taskName"], *routing(spec), "--output", "json")
    if result.returncode:
        return "", result.stderr.strip()
    status = json.loads(result.stdout).get("tencStatus") or {}
    return status.get("status", ""), status.get("reason", "")


def worker_logs(spec, index):
    result = koala(
        "pod",
        "logs",
        f"{spec['taskName']}-worker-{index}",
        "-c",
        "worker",
        "--cluster",
        spec["clusterId"],
        "--namespace",
        spec["namespace"],
        "--tail",
        str(LOG_TAIL),
        timeout=30,
    )
    if result.returncode or "server got a panic" in result.stdout:
        return ""
    return result.stdout


def follow(spec, timeout_s, interval_s):
    deadline = time.monotonic() + timeout_s
    previous_status = None
    last_line = None
    while time.monotonic() < deadline:
        try:
            status, reason = status_of(spec)
        except subprocess.TimeoutExpired:
            status, reason = "", "status query timed out"
        if status != previous_status:
            print(f"[KAP] {spec['taskName']}: {status or 'pending'} {reason}", flush=True)
            previous_status = status
        try:
            lines = worker_logs(spec, 0).splitlines()
            start = 0
            if last_line in lines:
                start = len(lines) - list(reversed(lines)).index(last_line)
            for line in lines[start:]:
                print(line, flush=True)
            if lines:
                last_line = lines[-1]
        except subprocess.TimeoutExpired:
            print("[KAP] log query timed out", flush=True)
        if status in {"Succeeded", "Failed"}:
            return status
        time.sleep(interval_s)
    raise TaskError(f"{spec['taskName']} exceeded {timeout_s}s")


def cleanup(spec):
    checked("task", "delete", spec["taskName"], *routing(spec))
    print(f"[KAP] deleted {spec['taskName']}", flush=True)


def execute(
    template,
    machine,
    *,
    bundle=None,
    name=None,
    env=None,
    overrides=None,
    timeout_s=1800,
    interval_s=20,
    log_dir=None,
):
    name = name or f"vforge-ci-{time.strftime('%m%d%H%M')}-{os.urandom(2).hex()}"
    if len(name) > 40:
        raise TaskError("task name exceeds 40 characters")
    spec = render(
        template, machine, name, bundle=bundle, overrides=overrides, env=env, timeout_s=timeout_s
    )

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        with tempfile.TemporaryDirectory(prefix="vforge-kap-") as directory:
            config = Path(directory) / "task.yaml"
            config.write_text(yaml.safe_dump(spec, sort_keys=False))
            config.chmod(0o600)
            try:
                # Never echo the rendered manifest or submission output: publish env holds a key.
                checked("task", "start", f"--config={config}", timeout=180)
                print(f"[KAP] submitted {name}", flush=True)
                status = follow(spec, timeout_s, interval_s)
                if status != "Succeeded":
                    raise TaskError(f"{name}: {status}")
            finally:
                try:
                    if log_dir:
                        Path(log_dir).mkdir(parents=True, exist_ok=True)
                        for index in range(int(spec["hostNum"])):
                            try:
                                content = worker_logs(spec, index)
                                (Path(log_dir) / f"{name}-worker-{index}.log").write_text(content)
                            except Exception as exc:
                                print(f"[KAP] could not save log: {type(exc).__name__}", flush=True)
                finally:
                    cleanup(spec)
    finally:
        signal.signal(signal.SIGTERM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--machine", required=True, help="CI profile name or manual profile YAML")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--name", help="exact unique task name")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--timeout-min", type=int, default=30)
    parser.add_argument("--interval-s", type=int, default=20)
    parser.add_argument("--image")
    parser.add_argument("--host-num", type=int)
    parser.add_argument("--host-gpu-num", type=int)
    args = parser.parse_args()
    overrides = {
        k: v
        for k, v in {
            "image": args.image,
            "hostNum": args.host_num,
            "hostGpuNum": args.host_gpu_num,
        }.items()
        if v is not None
    }
    execute(
        args.config,
        args.machine,
        bundle=args.bundle,
        name=args.name,
        overrides=overrides,
        timeout_s=args.timeout_min * 60,
        interval_s=args.interval_s,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
