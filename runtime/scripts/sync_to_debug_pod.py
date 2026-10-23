"""Incrementally sync local vforge sources into a running debug pod."""

from __future__ import annotations

import argparse
import base64
import io
import sys
import tarfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from koala_pod_run import DEFAULT_CLUSTER, run_in_pod

ROOT = Path(__file__).resolve().parents[1]
EXCLUDES = {
    ".git",
    ".venv",
    "__pycache__",
    "wandb",
    "outputs",
    "checkpoints",
    ".pycache",
}


def build_tarball() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(ROOT.rglob("*")):
            rel = path.relative_to(ROOT)
            if any(part in EXCLUDES for part in rel.parts):
                continue
            if path.is_dir():
                continue
            if path.suffix in {".pt", ".pth"}:
                continue
            archive.add(path, arcname=str(Path("vforge") / rel))
    return buffer.getvalue()


def render_remote_sync_script(payload_b64: str) -> str:
    wrapped_payload = "\n".join(textwrap.wrap(payload_b64, width=76))
    return f"""set -euo pipefail
cd /data/work
cat > /tmp/vforge_sync.b64 <<'B64'
{wrapped_payload}
B64
python - <<'PY'
import base64, datetime, pathlib, tarfile

payload = base64.b64decode(pathlib.Path("/tmp/vforge_sync.b64").read_text())
pathlib.Path("/tmp/vforge_sync.tgz").write_bytes(payload)
with tarfile.open("/tmp/vforge_sync.tgz", "r:gz") as archive:
    archive.extractall("/data/work")
manifest = pathlib.Path("/efs/alexlshen/vforge-debug/current_code_manifest.txt")
manifest.parent.mkdir(parents=True, exist_ok=True)
manifest.write_text(
    "\\n".join(
        [
            "task=vforge-b300-debug-8gpu",
            f"synced_at={{datetime.datetime.now().isoformat()}}",
            "workdir=/data/work",
            "note=incremental sync via scripts/sync_to_debug_pod.py",
        ]
    )
    + "\\n"
)
print("vforge sync ok", len(payload), "bytes")
PY
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", default="vforge-b300-debug-8gpu-worker-0")
    parser.add_argument("--container", default="worker")
    parser.add_argument("--cluster", default=DEFAULT_CLUSTER)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    payload = build_tarball()
    encoded = base64.b64encode(payload).decode("ascii")
    commands = render_remote_sync_script(encoded)
    exit_code = run_in_pod(
        args.pod,
        commands,
        container=args.container,
        cluster=args.cluster,
        timeout_s=args.timeout,
    )
    print(f"sync payload={len(payload)} bytes exit={exit_code}")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
