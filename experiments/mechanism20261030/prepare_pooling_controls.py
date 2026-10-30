"""Prepare a bounded, same-parameter spatial-addressing diagnostic.

This only writes frozen configurations and queues. Main training is not launched
until the separate execution/gradient preflight has been reviewed.
"""
import hashlib
import io
import json
import subprocess
import tarfile
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def main():
    base = Path("/efs/prescottsun/SPT-JEPA")
    code = base / "code"
    root = base / "runs/mechanism_pooling600_20261030"
    pre = base / "runs/mechanism_pooling_preflight_20261030"
    control = base / "runs/mechanism_rate600_20261030"
    py = "/efs/prescottsun/project_jepa/envs/lewm/bin/python"
    assert not root.exists() and not pre.exists()
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=code, text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
    root.mkdir(); pre.mkdir()
    source = root / "source-v1"
    source.mkdir()
    paths = ["as_jepa_lewm", "as_jepa_tools", "tests/as_jepa", "experiments/mechanism20261030"]
    archive = subprocess.check_output(["git", "archive", commit, *paths], cwd=code)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter="data")
    write(root / "source-v1.json", dict(source=str(source), source_commit=commit,
          files={str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()}))
    runtime_root = base / "runs/mechanism_cubeaudit_20261030/source-v1"
    src = dict(source=str(source), source_commit=commit, manifest=str(root / "source-v1.json"),
               runtime_source=str(runtime_root), runtime_manifest=str(runtime_root.parent / "source-v1.json"),
               pythonpath=":".join(map(str, [base / "environments/completion-overlay",
                   base / "environments/overlay", source, runtime_root / "runtime",
                   base / "sources/stable-worldmodel", base / "sources/clear-lewm-20261030"])))
    jobs, prejobs, records = [], [], []
    env = json.loads((control / "plan.json").read_text())["environment"]
    for task_index, task in enumerate(["tworoom", "cube"]):
        for seed in [1, 2]:
            reference = control / f"{task}-fresh-rate-on-seed{seed}"
            for method_index, method in enumerate(["quadrants", "shuffled"]):
                name = f"{task}-{method}-seed{seed}"
                out = root / name; out.mkdir()
                cfg = json.loads((reference / "config.json").read_text())
                cfg.update(output=str(out), checkpoint_every=200, retain_steps=[200, 400, 600], record_initial_weights=True)
                cfg["model"]["pool_partition"] = method
                config = out / "config.json"; write(config, cfg)
                gpu = task_index * 4 + (seed - 1) * 2 + method_index
                common = dict(host="dev13", gpu=gpu, source=src, task=task, method=method, training_seed=seed)
                jobs.append(dict(**common, id=name, output=str(out), config=str(config), config_sha256=digest(config),
                    phase="train600", command=[py, "-B", "-u", "-m", "as_jepa_tools.train_stage", "--config", str(config), "--stop-step", "600"]))
                health = root / ("health-" + name)
                jobs.append(dict(**common, id="health-" + name, output=str(health), depends_on=[name], phase="health",
                    command=[py, "-B", "-u", "-m", "as_jepa_tools.benchmark_health", "--run", str(out), "--output", str(health)]))
                info = root / ("info-" + name)
                jobs.append(dict(**common, id="info-" + name, output=str(info), depends_on=["health-" + name], phase="information",
                    command=[py, "-B", "-u", "-m", "as_jepa_tools.benchmark_information_audit", "--run", str(out), "--output", str(info)]))
                planning = root / ("planning-" + name)
                jobs.append(dict(**common, id="planning-" + name, output=str(planning), depends_on=["info-" + name], phase="planning",
                    environment={"MUJOCO_EGL_DEVICE_ID": str(gpu)},
                    command=[py, "-B", "-u", "-m", "as_jepa_tools.benchmark_evaluate", "--manifest", str(Path(cfg["data"]) / "manifests/validation-strict.json"),
                        "--checkpoint", str(out / "model-step0000600"), "--data", cfg["data"], "--output", str(planning)]))
                records.append(dict(id=name, original_control=str(reference), config_sha256=digest(config), task=task, seed=seed, method=method))
                if seed == 1:
                    po = pre / name; po.mkdir()
                    pcfg = json.loads(json.dumps(cfg)); pcfg.update(output=str(po), batch_size=2, workers=0, checkpoint_every=2, retain_steps=[2])
                    pc = po / "config.json"; write(pc, pcfg)
                    pgpu = task_index * 2 + method_index
                    c = dict(host="dev15", gpu=pgpu, source=src)
                    prejobs.append(dict(**c, id=name, output=str(po), config=str(pc), config_sha256=digest(pc),
                        command=[py, "-B", "-u", "-m", "as_jepa_tools.train_stage", "--config", str(pc), "--stop-step", "2"]))
                    go = pre / ("gradient-" + name)
                    prejobs.append(dict(**c, id="gradient-" + name, output=str(go), depends_on=[name],
                        command=[py, "-B", "-u", "-m", "as_jepa_tools.gradient_preflight", "--run", str(po), "--output", str(go)]))
    # The matching TwoRoom full-attention controls were already evaluated in the
    # channel campaign. Cube fresh600 needs these two explicit baseline scores.
    for seed in [1, 2]:
        reference = control / f"cube-fresh-rate-on-seed{seed}"
        out = root / f"planning-cube-full-seed{seed}"
        data = base / "data/hardtasks-v1/cube-metadata"
        jobs.append(dict(id=out.name, host="dev15", gpu=seed + 3, source=src, output=str(out), phase="planning-control",
            environment={"MUJOCO_EGL_DEVICE_ID": str(seed + 3)},
            command=[py, "-B", "-u", "-m", "as_jepa_tools.benchmark_evaluate", "--manifest", str(data / "manifests/validation-strict.json"),
                "--checkpoint", str(reference / "model-step0000600"), "--data", str(data), "--output", str(out)]))
    write(root / "plan.json", dict(root=str(root), deadline=None, environment=env, jobs=jobs))
    write(pre / "plan.json", dict(root=str(pre), deadline=None, environment=env, jobs=prejobs))
    write(root / "source-record.json", src)
    write(root / "protocol.json", dict(created_at=time.time(), source_commit=commit, runs=records,
        additional_updates=600, preflight=str(pre), comparison="Fresh same550 tensors, same data seed, optimizer, scheduler horizon, loss and64dim state. Full vs spatial vs shuffled addressing; CLS visible to every query.",
        motivation="Same-dimensional frozen probes localize weak TwoRoom position readout after global QueryPool. This does not prove useful dynamics or a sufficient solution.",
        limitations=["Fixed image quadrants introduce a spatial prior; this is a mechanism diagnostic, not a novelty claim.",
            "ViT tokens already have global receptive fields; only pooling address support is restricted.",
            "Shuffled partitions control support size but not all conditioning changes.",
            "600 steps remain in warmup; inspect full-state, gradients, functional use and task scores before continuation."],
        next_review="No automatic continuation beyond600. Preserve all old held campaigns and every seed."))
    print(json.dumps(dict(root=str(root), preflight=str(pre), source_commit=commit, jobs=len(jobs), preflight_jobs=len(prejobs))))


if __name__ == "__main__":
    main()
