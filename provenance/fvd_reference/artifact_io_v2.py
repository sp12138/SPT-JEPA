"""Bounded exact-byte confirmation for ambiguous evaluator artifact renames."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import time

@contextmanager
def verified_artifact_renames(output_dir, *, verification_timeout_s=5.0, poll_interval_s=0.1):
    if verification_timeout_s < 0 or poll_interval_s <= 0:
        raise ValueError("Invalid verification wait")
    root = Path(output_dir).absolute()
    original = os.replace

    def replace(source, destination, *, src_dir_fd=None, dst_dir_fd=None):
        if src_dir_fd is not None or dst_dir_fd is not None:
            return original(source, destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        src, dst = Path(source).absolute(), Path(destination).absolute()
        feature = dst.name.startswith("fvd_") and (
            dst.name.endswith(".npy") or dst.name.endswith(".npy.json"))
        manifest = dst.name == "eval_manifest.json"
        allowed_source = src.name == dst.name + ".tmp" or (
            manifest and src.name == dst.name + ".protocol.tmp")
        if src.parent != root or dst.parent != root or not allowed_source or not (feature or manifest):
            return original(source, destination)
        intended = src.read_bytes()
        try:
            return original(source, destination)
        except FileNotFoundError:
            start = time.monotonic()
            deadline = start + verification_timeout_s
            attempts = 0
            while True:
                attempts += 1
                try:
                    completed = dst.read_bytes() == intended
                except OSError:
                    completed = False
                if completed:
                    print(json.dumps(dict(
                        event="artifact_rename_result_verified",
                        policy="verify_ambiguous_rename_bytes_v2_wait",
                        path=str(dst), bytes=len(intended),
                        sha256=hashlib.sha256(intended).hexdigest(),
                        attempts=attempts, elapsed_s=time.monotonic()-start,
                        rank=os.environ.get("RANK"))), flush=True)
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    print(json.dumps(dict(
                        event="artifact_rename_verification_failed",
                        policy="verify_ambiguous_rename_bytes_v2_wait",
                        path=str(dst), intended_sha256=hashlib.sha256(intended).hexdigest(),
                        attempts=attempts, elapsed_s=time.monotonic()-start,
                        rank=os.environ.get("RANK"))), flush=True)
                    raise
                time.sleep(min(poll_interval_s, remaining))
    os.replace = replace
    try:
        if os.environ.get("RANK", "0") == "0":
            print(json.dumps(dict(
                event="artifact_io_guard", policy="verify_ambiguous_rename_bytes_v2_wait",
                helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                output=str(root), verification_timeout_s=verification_timeout_s)), flush=True)
        yield
    finally:
        os.replace = original
