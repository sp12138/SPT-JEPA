"""Run commands in a Koala pod via pseudo-tty when non-interactive exec hangs."""

from __future__ import annotations

import argparse
import os
import pty
import select
import subprocess
import sys
import time

DEFAULT_CLUSTER = "tenc-aws-yfxn-us-west"


def run_in_pod(
    pod_name: str,
    commands: str,
    *,
    container: str = "worker",
    cluster: str = DEFAULT_CLUSTER,
    timeout_s: float = 300.0,
) -> int:
    cmd = [
        "koala-cli",
        "pod",
        "exec",
        pod_name,
        "-c",
        container,
        "--cluster",
        cluster,
    ]
    master, slave = pty.openpty()
    proc = subprocess.Popen(cmd, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)

    sent = False
    buf = b""
    start = time.time()
    while time.time() - start < timeout_s:
        ready, _, _ = select.select([master], [], [], 0.5)
        if ready:
            data = os.read(master, 8192)
            if not data:
                break
            sys.stdout.write(data.replace(b"\x00", b"").decode("utf-8", "replace"))
            sys.stdout.flush()
            buf += data
            if not sent and (b"# " in buf or b"$ " in buf or b":/data/work#" in buf):
                payload = commands if commands.endswith("\n") else commands + "\n"
                os.write(master, payload.encode())
                sent = True
        if proc.poll() is not None:
            break

    if proc.poll() is None:
        os.write(master, b"exit\n")
        time.sleep(1)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    return proc.returncode if proc.returncode is not None else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pod_name")
    parser.add_argument("--container", default="worker")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Shell command to run")
    args = parser.parse_args()
    if not args.command:
        raise SystemExit("missing remote command")
    command = " ".join(args.command)
    raise SystemExit(
        run_in_pod(
            args.pod_name,
            command,
            container=args.container,
            cluster=args.cluster,
            timeout_s=args.timeout,
        )
    )


if __name__ == "__main__":
    main()
