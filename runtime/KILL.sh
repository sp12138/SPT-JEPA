#!/usr/bin/env bash
set -euo pipefail

python3 - "$@" <<'PY'
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Stop all training processes owned by the current user."
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="print matching processes without sending signals",
)
parser.add_argument(
    "--grace-seconds",
    type=float,
    default=5.0,
    help="seconds to wait after SIGTERM before SIGKILL (default: 5)",
)
parser.add_argument(
    "--gpu-settle-seconds",
    type=float,
    default=30.0,
    help="seconds to wait for killed PIDs to leave GPU contexts (default: 30)",
)
args = parser.parse_args()
if args.grace_seconds < 0:
    parser.error("--grace-seconds must be non-negative")
if args.gpu_settle_seconds < 0:
    parser.error("--gpu-settle-seconds must be non-negative")

proc_root = Path("/proc")
if not proc_root.is_dir():
    raise SystemExit("KILL.sh requires Linux /proc")

patterns = (
    re.compile(r"(^|\s)(?:\S*/)?torchrun(?:\s|$)"),
    re.compile(r"(^|\s)(?:\S*/)?python(?:3(?:\.\d+)?)?\s+(?:\S*/)?train\.py(?:\s|$)"),
    re.compile(r"\s-m\s+vforge\.tasks\.train(?:\s|$)"),
    re.compile(r"(^|\s)(?:\S*/)?vforge-train(?:\s|$)"),
    re.compile(r"(^|\s)(?:\S*/)?scripts/train\.sh(?:\s|$)"),
)
uid = os.getuid()
current_pid = os.getpid()


def read_process(pid: int) -> tuple[int, int, str] | None:
    try:
        status = (proc_root / str(pid) / "status").read_text()
        fields = {}
        for line in status.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key] = value.strip()
        process_uid = int(fields["Uid"].split()[0])
        ppid = int(fields["PPid"])
        command = (
            (proc_root / str(pid) / "cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
            .strip()
        )
        return process_uid, ppid, command
    except (FileNotFoundError, KeyError, PermissionError, ProcessLookupError, ValueError):
        return None


def is_running(pid: int) -> bool:
    try:
        status = (proc_root / str(pid) / "status").read_text()
        state = next(
            line.split(":", 1)[1].strip()
            for line in status.splitlines()
            if line.startswith("State:")
        )
        return not state.startswith("Z")
    except (FileNotFoundError, PermissionError, ProcessLookupError, StopIteration):
        return False


def gpu_process_ids() -> set[int]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return set()
    return {
        int(line.strip())
        for line in output.splitlines()
        if line.strip().isdigit()
    }


processes: dict[int, tuple[int, str]] = {}
for entry in proc_root.iterdir():
    if not entry.name.isdigit():
        continue
    pid = int(entry.name)
    record = read_process(pid)
    if record is None:
        continue
    process_uid, ppid, command = record
    if process_uid == uid and command:
        processes[pid] = (ppid, command)

ancestors = {current_pid}
cursor = current_pid
while cursor in processes:
    cursor = processes[cursor][0]
    if cursor <= 1 or cursor in ancestors:
        break
    ancestors.add(cursor)

roots = {
    pid
    for pid, (_ppid, command) in processes.items()
    if pid not in ancestors and any(pattern.search(command) for pattern in patterns)
}
targets = set(roots)
changed = True
while changed:
    changed = False
    for pid, (ppid, _command) in processes.items():
        if pid not in ancestors and ppid in targets and pid not in targets:
            targets.add(pid)
            changed = True

if not targets:
    print("[kill-training] no matching training processes")
    raise SystemExit(0)

for pid in sorted(targets):
    print(f"[kill-training] pid={pid} cmd={processes[pid][1]}")

if args.dry_run:
    print(f"[kill-training] dry-run matched {len(targets)} process(es)")
    raise SystemExit(0)

target_gpu_pids = targets & gpu_process_ids()
for pid in sorted(targets, reverse=True):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

deadline = time.monotonic() + args.grace_seconds
remaining = set(targets)
while remaining and time.monotonic() < deadline:
    remaining = {pid for pid in remaining if is_running(pid)}
    if remaining:
        time.sleep(0.1)

for pid in sorted(remaining, reverse=True):
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

gpu_deadline = time.monotonic() + args.gpu_settle_seconds
remaining_gpu = target_gpu_pids & gpu_process_ids()
while remaining_gpu and time.monotonic() < gpu_deadline:
    time.sleep(0.5)
    remaining_gpu = target_gpu_pids & gpu_process_ids()
if remaining_gpu:
    print(
        f"[kill-training] GPU contexts did not clear for PIDs "
        f"{sorted(remaining_gpu)}",
        file=sys.stderr,
    )
    raise SystemExit(1)

print(
    f"[kill-training] stopped {len(targets)} process(es); "
    f"forced={len(remaining)}; gpu_contexts_cleared={len(target_gpu_pids)}"
)
PY
