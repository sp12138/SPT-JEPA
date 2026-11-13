"""Open-ended authorized training with hourly frozen readouts and full resumes.

This controller executes a fixed recipe. Semantic algorithm improvements are
made by the agent at its hourly checks in the same conversation, never by this
script silently editing running code or selecting on confirmation outcomes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_time_compare.campaign import env
from experiments.robocasa_action_probe.scaling import summarize
from experiments.robocasa_compare.evaluate import case_name

MODULE = "experiments.robocasa_stable_scaling"
PYTHON = "/efs/prescottsun/project_jepa/envs/lewm/bin/python"
HOST_VARIANTS = {"dev13": "feature-delta", "dev15": "spt-rate"}


def corpus(root, version):
    path = root/"corpora"/f"version-{version:04d}.json"; path.parent.mkdir(exist_ok=True)
    with (root/"corpora/.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists(): return path
        plan = json.loads((root/"plan.json").read_text())
        cutoff = plan["training_clock_start"]+max(0, version-1)*3600
        assert time.time() >= cutoff
        card = json.loads((root/"data.json").read_text()); rows = []
        kitchens = {tuple(r["kitchen"]) for r in card["records"] if r["split"] == "train"}
        for record in card["records"]:
            if record["split"] != "train": continue
            ident = record["id"]; base = Path("/tmp/spt-time-compare-20261112")
            rows.append(dict(record, local_features=str(base/"features/vjepa"/(ident+".npy")),
                local_observations=str(base/"observations"/(ident+".npz")), original=True))
        for p in sorted((root/"stream/records").glob("*.json")):
            record = json.loads(p.read_text())
            if record["completed_unix"] > cutoff: continue
            assert record["passed"] and record["split"] == "train" and tuple(record["kitchen"]) in kitchens
            ident = record["id"]; base = Path("/tmp/spt-stable-scaling-20261114")
            rows.append(dict(record, local_features=str(base/"features/vjepa"/(ident+".npy")),
                local_observations=str(base/"observations"/(ident+".npz")), original=False))
        assert len(rows) >= 1120 and len({r["id"] for r in rows}) == len(rows)
        save(path, dict(passed=True, version=version, cutoff_unix=cutoff, records=rows,
            original_episodes=1120, new_episodes=len(rows)-1120, unique_transitions=len(rows)*16,
            fixed_probe_labels_do_not_grow=True, source_card_sha256=sha(root/"data.json")))
    return path


def stage(root, manifest, host):
    d = json.loads(manifest.read_text())
    marker = root/"corpora"/f"STAGED-{manifest.stem}-{host}.json"
    if marker.exists():
        proof = json.loads(marker.read_text())
        assert proof["passed"] and proof["manifest_sha256"] == sha(manifest)
        if all(Path(row[key]).exists() for row in d["records"] for key in ("local_features","local_observations")):
            return
    def one(row):
        for kind, local, digest in [("features", "local_features", "features_sha256"),
                ("observations", "local_observations", "observations_sha256")]:
            target = Path(row[local]); target.parent.mkdir(parents=True, exist_ok=True)
            if row["original"]:
                assert target.exists()
            else:
                tag = target.with_suffix(target.suffix+".sha256")
                if target.exists() and tag.exists():
                    assert tag.read_text().strip() == row[digest]
                else:
                    temp = target.with_suffix(target.suffix+".partial")
                    shutil.copyfile(row[kind], temp); assert sha(temp) == row[digest]
                    temp.replace(target); tag.write_text(row[digest]+"\n")
    with (root/"corpora"/f".stage-{host}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(one, d["records"]))
        save(marker, dict(passed=True,
            manifest_sha256=sha(manifest), episodes=len(d["records"]), time=time.time()))


def call(root, argv, log, gpus, simulator=False):
    plan = json.loads((root/"plan.json").read_text()); code = Path(plan["training_code"])
    child_env = env(root, gpus, simulator)
    child_env["PYTHONPATH"] = str(code)+":"+child_env["PYTHONPATH"]
    attempt = 1; selected = log
    while selected.exists():
        attempt += 1; selected = log.with_name(log.stem+f"-attempt{attempt}"+log.suffix)
    with selected.open("x") as stream:
        subprocess.run(argv, cwd=code, env=child_env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, timeout=12*3600, check=True)


def readout(root, run, label, host):
    plan = json.loads((root/"plan.json").read_text())
    result = root/"results"/run.name/(label+".json")
    if result.exists():
        previous = json.loads(result.read_text())
        assert previous["passed"] and previous["parent"]["nn_sha256"] == json.loads((run/label/"ready.json").read_text())["nn_sha256"]
        return
    cell = root/"hourly-readouts"/run.name/label; cell.mkdir(parents=True, exist_ok=True)
    if not (cell/"plan.json").exists():
        for name in ("code", "sources", "environment", "data.json"):
            (cell/name).symlink_to(root/name, target_is_directory=(root/name).is_dir())
        (cell/"cases").symlink_to(root/"case-generation/cases", target_is_directory=True)
        spec = dict(plan, cases=plan["monitor_cases"], comparison_checkpoint=label,
            source=str(root), source_models={"spt64-linear":str(run)})
        save(cell/"plan.json", spec)
    logs = cell/"logs"; logs.mkdir(exist_ok=True)
    arm = "spt64-linear"
    cfg=json.loads((run/"config.json").read_text());gpus=cfg["gpus"].split(",")
    def cache_lane(lane):
        for shard in range(lane,8,len(gpus)):
            proof = cell/"heads"/arm/"cache"/f"part-{shard}.json"
            if proof.exists(): assert json.loads(proof.read_text())["passed"]; continue
            call(root, [PYTHON, "-u", "-m", "experiments.robocasa_action_probe.probe", "--root", str(cell),
                "--mode", "cache", "--arm", arm, "--shard", str(shard)], logs/f"cache-{shard}.log", gpus[lane])
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool: list(pool.map(cache_lane, range(len(gpus))))
    directory = cell/"heads"/arm
    if not (directory/"ready.json").exists():
        call(root, [PYTHON, "-u", "-m", "experiments.robocasa_action_probe.probe", "--root", str(cell),
            "--mode", "fit", "--arm", arm], logs/"linear-fit.log", gpus[0])
    shards = len(gpus)*int(plan.get("readout_workers_per_gpu", 4))
    assert len(gpus) <= shards <= len(gpus)*4
    def control(shard):
        selected = [c for i,c in enumerate(plan["monitor_cases"]) if i % shards == shard]
        if all((cell/"evaluation"/arm/case_name(**c)/"summary.json").exists() for c in selected): return
        call(root, [str(root/"environment/bin/python"), "-u", "-m", MODULE+".campaign", "--root", str(root),
            "--mode", "control", "--cell", str(cell), "--shard", str(shard),"--shards",str(shards)],
            logs/f"control-{shard}-of-{shards}.log", gpus[shard % len(gpus)], True)
    with ThreadPoolExecutor(max_workers=shards) as pool: list(pool.map(control, range(shards)))
    rows = []
    for case in plan["monitor_cases"]:
        folder = cell/"evaluation"/arm/case_name(**case); d = json.loads((folder/"summary.json").read_text())
        assert d["passed"] and d["model_unchanged"] and d["physical_replay_pass"]
        assert sha(folder/"trace.npz") == d["trace_sha256"]
        rows.append(d)
    save(root/"results"/run.name/(label+".json"), dict(passed=True, variant=run.name,
        label=label, metrics=summarize(rows), head=json.loads((directory/"ready.json").read_text()),
        parent=json.loads((run/label/"ready.json").read_text()), cases_are_monitored_development=True,
        fixed_label_probe=True, readout="linear", source_cell=str(cell), time=time.time()))


def control(root, cell, shard, shards=4):
    from experiments.robocasa_action_probe import probe
    import torch
    torch.set_num_threads(2); torch.set_float32_matmul_precision("high")
    policy = probe.Policy(cell, "spt64-linear")
    plan = json.loads((cell/"plan.json").read_text())
    for index, case in enumerate(plan["cases"]):
        if index % shards != shard: continue
        path = cell/"evaluation/spt64-linear"/case_name(**case)/"summary.json"
        if path.exists(): assert json.loads(path.read_text())["passed"]; continue
        probe.evaluate(cell, "spt64-linear", index, policy=policy)


def worker(root, host, run_name):
    run = root/"training"/run_name;cfg=json.loads((run/"config.json").read_text())
    assert cfg["host"]==host;variant=cfg["variant"];gpus=cfg["gpus"]
    logs = run/"logs"; logs.mkdir(parents=True, exist_ok=True)
    try:
        while not (root/"PRETRAIN_READY.json").exists() or subprocess.check_output(["nvidia-smi","-i",gpus, "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"], text=True).strip():
            save(run/"controller.json", dict(phase="waiting_for_owned_diagnostics", time=time.time()))
            if (root/"STOP").exists() or (run/"STOP").exists(): return
            time.sleep(5)
        manifest = corpus(root, 1); stage(root, manifest, host)
        def learn(label, seconds, limit=None):
            if (run/label/"ready.json").exists(): return
            argv=[PYTHON,"-m","torch.distributed.run","--standalone","--nnodes=1","--nproc-per-node="+str(cfg["world_size"]),"-m","--",MODULE+".train",
                "--root",str(root),"--run",str(run),"--manifest",str(manifest),"--target-seconds",str(seconds),"--label",label]
            if limit is not None: argv += ["--max-updates",str(limit)]
            call(root,argv,logs/("train-"+label+".log"),gpus)
        # Retain startup updates and prove a real full-state continuation before long training.
        learn("entry",3600,8); learn("restore-probe",3600,8)
        hour=1
        while not (root/"STOP").exists() and not (run/"STOP").exists() and not (root/"FINISHED").exists():
            label=f"hour-{hour:03d}"
            manifest=corpus(root,hour if cfg["growing_data"] else 1);stage(root,manifest,host)
            save(run/"controller.json",dict(phase="training",hour=hour,variant=variant,manifest=str(manifest),time=time.time()))
            learn(label,hour*3600)
            if (root/"STOP").exists() or (run/"STOP").exists():break
            save(run/"controller.json",dict(phase="frozen_action_readout",hour=hour,variant=variant,time=time.time()))
            readout(root,run,label,host);hour+=1
    except BaseException as error:
        save(run/"ERROR.json",dict(error=repr(error),variant=variant,time=time.time(),method_negative=False));raise


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",choices=("worker","control","readout"),required=True);p.add_argument("--host",choices=HOST_VARIANTS)
    p.add_argument("--run-name");p.add_argument("--label");p.add_argument("--cell",type=Path);p.add_argument("--shard",type=int);p.add_argument("--shards",type=int,default=4);a=p.parse_args()
    if a.mode=="worker":worker(a.root,a.host,a.run_name)
    elif a.mode=="readout":
        run=a.root/"training"/a.run_name
        assert json.loads((run/"config.json").read_text())["host"]==a.host and a.label
        readout(a.root,run,a.label,a.host)
    else:control(a.root,a.cell,a.shard,a.shards)
