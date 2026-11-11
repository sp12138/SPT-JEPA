"""Finite four-host campaign; no LLM monitor and no automatic research round."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import itertools
import os
from pathlib import Path
import subprocess
import time

from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_compare.campaign import environment, TRAIN_PYTHON

MODULE = "experiments.robocasa_time_compare"
ARMS = {"dev13": ("robojepa", 1), "dev15": ("spt", 1),
        "dev6": ("spt64", 1), "dev6-2c": ("spt-grid16", 1)}


def initialize(root, code, budget=None):
    plan = json.loads((root/"plan.json").read_text())
    ready = [json.loads((root/f"STAGE_READY-{h}.json").read_text()) for h in ARMS]
    assert all(r["passed"] for r in ready)
    assert len({r["manifest_sha256"] for r in ready}) == 1
    (root/"code").symlink_to(code, target_is_directory=True)
    plan.update(training_budget_seconds=budget,
        budget_basis="User explicitly requested unlimited time until asked/stopped; no automatic algorithm changes",
        milestone_seconds=[3600], later_interval_seconds=3600,
        cases=[dict(task="reach", kitchen=k, seed=s) for k in (1, 4, 8, 10) for s in (19301, 19302)],
        external_encoder=plan["shared_external_encoder"],
        active_time_definition="Sum of slowest-rank data-load/transfer/forward/backward/optimizer iteration wall times; excludes setup, checkpoint and evaluation",
        allocated_time_also_reported=True, source_code=str(code),
        baseline_adaptation="Public300M architecture and TF/2+detachedK2/4; RGB-only zero-state, common224/FOV/caches; not23-dataset reproduction",
        primary_comparison="Four different algorithms, allseed1; no seed replication or significance claim",
        historical_SPT_cumulative_40000_updates_NOT_inherited=True,
        LR_schedule="Common wall-clock900second warmup then flat LR; method-native peakLR/betas/WD",
        no_automatic_extension=False, until_user_stop=True,
        common_global_batch=32, SPT_window=8, RoboJEPA_window=16,
        unequal_capacity_steps_and_image_FLOPs_disclosed=True, phase="entry")
    save(root/"plan.json", plan)
    for host, (method, seed) in ARMS.items():
        run = root/"training"/f"{method}-seed{seed}"; run.mkdir(parents=True)
        kind = "spt" if method.startswith("spt") else method
        cfg = dict(method=kind, host=host, seed=seed, encoder="vjepa", feature_kind="vjepa",
            objective="spt" if kind == "spt" else "native", global_batch=32,
            window=8 if kind == "spt" else 16, budget_seconds=budget,
            cache_manifest_sha256=ready[0]["manifest_sha256"],
            fresh_heads=True, inverse=False, future_action_target=False)
        if kind == "spt":
            cfg.update(target_refinement="plain", context_memory="spatial", code_chart="global",
                forecast_only=False, flow_logit_shift=0., state_slots=16, pool_layout="global")
        if method == "spt64": cfg.update(state_slots=4, context_memory=None)
        if method == "spt-grid16": cfg["pool_layout"] = "grid"
        save(run/"config.json", cfg)
    # Portable ffmpeg resolution: both existing node-local logging bundles are
    # legitimate dependencies; do not symlink another node's temporary path.
    (root/"bin").mkdir(exist_ok=True)
    executable = root/"bin/ffmpeg"
    executable.write_text('#!/bin/sh\nfor f in /tmp/spt-robojepa-literal-ui-20261111*/imageio_ffmpeg/binaries/ffmpeg-*; do\n  if [ -x "$f" ]; then exec "$f" "$@"; fi\ndone\nexit 127\n')
    executable.chmod(0o755)
    save(root/"INITIALIZED.json", dict(passed=True, cache_manifest_sha256=ready[0]["manifest_sha256"],
        fresh_heads=True, hosts=list(ARMS), budget_seconds=budget, source_code=str(code)))


def env(root, gpus, simulator=False):
    result = environment(root, gpus, simulator)
    local_ui = (Path("/tmp/spt-robojepa-literal-ui-20261111-sha7fbf1c56")
                if Path("/tmp/spt-robojepa-literal-ui-20261111-sha7fbf1c56").exists()
                else Path("/tmp/spt-robojepa-literal-ui-20261111-v1"))
    result["PYTHONPATH"] += ":" + str(local_ui)
    result["PATH"] = str(root/"bin") + ":" + result.get("PATH", "")
    return result


def call(root, argv, log, gpus, simulator=False, timeout=36*3600):
    assert not (root/"STOP").exists()
    with log.open("x") as stream:
        subprocess.run(argv, cwd=root/"code", env=env(root, gpus, simulator),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            check=True, timeout=timeout)


def worker(root, host, entry=False, attempt=1):
    method, seed = ARMS[host]; run = root/"training"/f"{method}-seed{seed}"
    logs = run/f"execution-{'entry' if entry else 'long'}-{attempt}"
    logs.mkdir(exist_ok=False)
    try:
        plan = json.loads((root/"plan.json").read_text())
        targets = [3600] if entry else itertools.count(3600, plan["later_interval_seconds"])
        if not entry:
            assert json.loads((root/"LONG_APPROVED.json").read_text())["passed"]
        for target in targets:
            label = "entry" if entry else f"hour-{round(target/3600):02d}"
            if not (run/label/"ready.json").exists():
                argv = [TRAIN_PYTHON, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                    "--nproc-per-node=8", "-m", "--", MODULE+".train", "--root", str(root),
                    "--run", str(run), "--local-data", "/tmp/spt-time-compare-20261112",
                    "--target-seconds", str(target)]
                if entry: argv.append("--entry")
                if (run/"last.ckpt").exists(): argv.append("--resume")
                call(root, argv, logs/f"train-{label}.log", "0,1,2,3,4,5,6,7")
            if entry: break
            def one(i):
                case = json.loads((root/"plan.json").read_text())["cases"][i]
                from experiments.robocasa_compare.evaluate import case_name
                result = root/"evaluation"/run.name/label/case_name(**case)/"summary.json"
                if result.exists():
                    assert json.loads(result.read_text())["passed"]; return
                call(root, [str(root/"environment/bin/python"), "-u", "-m", MODULE+".evaluate",
                    "--root", str(root), "--run", str(run), "--label", label,
                    "--case", str(i)], logs/f"evaluate-{label}-{i}.log", str(i), True, timeout=3600)
            with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(one, range(8)))
            from experiments.robocasa_time_compare.report import collect
            collect(root, run, label)
        save(root/f"{'ENTRY_TRAINED' if entry else 'DONE'}-{host}.json",
             dict(passed=True, host=host, method=method, seed=seed, last_label=label,
                  completed_unix=time.time()))
    except BaseException as error:
        save(logs/"ERROR.json", dict(error=repr(error), host=host, method_negative=False))
        raise


def launch(root, host, entry=False, attempt=1):
    assert json.loads((root/"INITIALIZED.json").read_text())["passed"]
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                      "--format=csv,noheader"], text=True).strip()
    assert not active, active
    mode = "entry" if entry else "long"
    owner = root/f"launch-{mode}-{host}-{attempt}.json"; assert not owner.exists()
    argv = [TRAIN_PYTHON, "-u", "-m", MODULE+".campaign", "--mode", "worker", "--root", str(root),
        "--host", host, "--attempt", str(attempt)]
    if entry: argv.append("--entry")
    with (root/f"worker-{mode}-{host}-{attempt}.log").open("x") as stream:
        process = subprocess.Popen(argv, cwd=root/"code", env=env(root, ""),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    row = dict(host=host, pid=process.pid, command=argv, created_unix=time.time(), entry=entry)
    save(owner, row); print(json.dumps(row), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", choices=("initialize", "worker", "launch"), required=True)
    p.add_argument("--host", choices=tuple(ARMS)); p.add_argument("--code", type=Path)
    p.add_argument("--budget", type=int); p.add_argument("--attempt", type=int, default=1)
    p.add_argument("--entry", action="store_true"); a = p.parse_args()
    if a.mode == "initialize": initialize(a.root, a.code, a.budget)
    elif a.mode == "launch": launch(a.root, a.host, a.entry, a.attempt)
    else: worker(a.root, a.host, a.entry, a.attempt)
