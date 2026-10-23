#!/usr/bin/env python3
"""CI orchestration: CPU -> docs preview -> GPU, or master docs publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import bundle
import kap_run

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = ROOT / "scripts/ci/kap"
GPU_GROUPS = ("core", "ltx", "wan")
GPU_TIMEOUT_SECONDS = 7200


def task_name(mode, run_id):
    if not run_id or "{{" in run_id:
        raise ValueError("CI run ID is missing or unresolved")
    return f"vforge-{mode}-{hashlib.sha256(run_id.encode()).hexdigest()[:20]}"


def artifact_store(run_id):
    import boto3
    from botocore.config import Config

    client = boto3.client("s3", region_name="us-west-2", config=Config(signature_version="s3v4"))
    return (
        client,
        kap_run.load_machine("cpu")["codeBucket"],
        f"ci-site/{task_name('docs', run_id)}.tar.gz",
    )


def cleanup(mode, run_id):
    # Independent CI finally job: no dependency on the interrupted job's filesystem.
    failures = []
    phases = (
        [("cpu", "cpu", "cpu"), *(("gpu", "b300", f"gpu-{group}") for group in GPU_GROUPS)]
        if mode == "mr"
        else [("docs", "cpu", "docs")]
    )
    for phase, machine, name in phases:
        try:
            kap_run.cleanup(
                kap_run.render(TEMPLATES / f"{phase}.task.yaml", machine, task_name(name, run_id))
            )
        except Exception as exc:
            failures.append(exc)
    try:
        client, bucket, key = artifact_store(run_id)
        client.delete_object(Bucket=bucket, Key=key)
    except Exception as exc:
        failures.append(exc)
    if failures:
        raise RuntimeError(f"CI cleanup failed ({len(failures)} operations)") from failures[0]


def task(mode, machine, code, *, run_id, env=None, gpu_group=None):
    name = mode
    if mode == "gpu":
        if gpu_group not in GPU_GROUPS:
            raise ValueError("GPU task requires a valid gpu_group")
        name = f"gpu-{gpu_group}"
        env = dict(env or {}, CI_GPU_GROUP=gpu_group)
    kap_run.execute(
        TEMPLATES / f"{mode}.task.yaml",
        machine,
        bundle=code,
        name=task_name(name, run_id),
        env=env,
        timeout_s=1800 if mode != "gpu" else GPU_TIMEOUT_SECONDS,
        log_dir=ROOT / ".ci-results",
    )


def latest_master():
    return bundle.git(ROOT, "ls-remote", "origin", "refs/heads/master").decode().split()[0]


def verify_site(site, revision, base_path):
    expected = {"source_sha": revision, "base_path": base_path}
    if json.loads((site / ".vforge-site.json").read_text()) != expected:
        raise RuntimeError("site artifact source or deployment path mismatch")
    if not all((site / page).is_file() for page in ("index.html", "next/index.html")):
        raise RuntimeError("site artifact is incomplete")


def build_site(code, metadata, directory, run_id, *, preview=False):
    base_path = "/debug/" if preview else "/"
    client, bucket, key = artifact_store(run_id)
    url = client.generate_presigned_url(
        "put_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=3600
    )
    try:
        task(
            "cpu" if preview else "docs",
            "cpu",
            code,
            run_id=run_id,
            env={"SITE_UPLOAD_URL": url, "DOCS_BASE_PATH": base_path},
        )
        archive = Path(directory) / "site.tar.gz"
        client.download_file(bucket, key, str(archive))
        with tarfile.open(archive) as contents:
            contents.extractall(directory, filter="data")
        site = Path(directory) / "site"
        verify_site(site, metadata["source_sha"], base_path)
    finally:
        client.delete_object(Bucket=bucket, Key=key)
    archive.unlink()
    return site


def publish(code, metadata, directory, run_id):
    revision = metadata["source_sha"]
    if revision != latest_master():
        print("[publish] obsolete revision; skipped", flush=True)
        return
    site = build_site(code, metadata, directory, run_id)
    if revision != latest_master():
        print("[publish] superseded while building; skipped", flush=True)
        return
    subprocess.run(["bash", str(ROOT / "scripts/ci/docs_deploy.sh"), str(site)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["mr", "publish"])
    parser.add_argument("--source", help="immutable source commit, not a merge preview")
    parser.add_argument("--target", help="target commit/ref for MR")
    parser.add_argument("--mr-title", default=os.environ.get("GIT_CI_MR_TITLE"))
    parser.add_argument(
        "--run-id", required=True, help="unique CI build ID, shared with the finally job"
    )
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument(
        "--include-submodule",
        action="append",
        default=[],
        help="initialized submodule to archive at the source revision's pinned SHA (repeatable)",
    )
    parser.add_argument(
        "--phase", choices=["cpu", "docs-debug", "gpu"], help="MR phase selected by the CI step"
    )
    parser.add_argument("--gpu-group", choices=GPU_GROUPS, help="GPU group owned by this CI job")
    args = parser.parse_args()
    task_name("cpu", args.run_id)
    if args.cleanup:
        cleanup(args.mode, args.run_id)
        return
    is_gpu = args.mode == "mr" and args.phase == "gpu"
    if is_gpu != (args.gpu_group is not None):
        parser.error("--gpu-group is required only with mr --phase gpu")
    if not args.source:
        parser.error("--source is required")
    preview_dir = ROOT / ".ci-results" / task_name("docs", args.run_id)
    if args.mode == "mr" and args.phase == "docs-debug":
        revision = bundle.git(ROOT, "rev-parse", "--verify", f"{args.source}^{{commit}}")
        try:
            site = preview_dir / "site"
            verify_site(site, revision.decode().strip(), "/debug/")
            subprocess.run(
                ["bash", str(ROOT / "scripts/ci/docs_deploy.sh"), str(site), "debug"], check=True
            )
        finally:
            if preview_dir.exists():
                shutil.rmtree(preview_dir)
        return
    if args.mode == "mr" and (not args.target or not args.phase):
        parser.error("--target and --phase are required for MR checks")
    with tempfile.TemporaryDirectory(prefix="vforge-ci-") as directory:
        code = Path(directory) / "bundle"
        metadata = bundle.prepare(
            ROOT,
            code,
            source=args.source,
            target=args.target if args.mode == "mr" else None,
            mr_title=args.mr_title,
            submodules=args.include_submodule,
        )
        if args.mode == "mr" and args.phase == "cpu":
            if preview_dir.exists():
                shutil.rmtree(preview_dir)
            preview_dir.mkdir(parents=True)
            try:
                build_site(code, metadata, preview_dir, args.run_id, preview=True)
            except BaseException:
                shutil.rmtree(preview_dir)
                raise
        elif args.mode == "mr":
            task("gpu", "b300", code, run_id=args.run_id, gpu_group=args.gpu_group)
        else:
            publish(code, metadata, directory, args.run_id)


if __name__ == "__main__":
    main()
