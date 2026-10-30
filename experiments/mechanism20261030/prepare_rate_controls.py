"""Prepare paired short rate-removal diagnostics using the existing trainer."""
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    base = Path("/efs/prescottsun/SPT-JEPA")
    code = base / "code"
    root = base / "runs/mechanism_rate600_20261030"
    parent = base / "runs/hardtasks_learn2000_20261030"
    py = "/efs/prescottsun/project_jepa/envs/lewm/bin/python"
    assert not root.exists()
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=code, text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
    root.mkdir()
    source = root / "source-v1"
    source.mkdir()
    paths = ["as_jepa_lewm", "as_jepa_tools", "runtime", "experiments/mechanism20261030"]
    archive = subprocess.check_output(["git", "archive", commit, *paths], cwd=code)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter="data")
    manifest = dict(source=str(source), source_commit=commit, files={
        str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()})
    (root / "source-v1.json").write_text(json.dumps(manifest, indent=2))
    src = dict(source=str(source), source_commit=commit, manifest=str(root / "source-v1.json"),
               pythonpath=":".join(map(str, [base / "environments/completion-overlay",
                   base / "environments/overlay", source, source / "runtime",
                   base / "sources/stable-worldmodel", base / "sources/clear-lewm-20261030"])))
    jobs, copies, records = [], [], []
    for task, host in [("tworoom", "dev2"), ("cube", "dev13")]:
        gpu = 0
        for start in ["fresh", "resume2000"]:
            for seed in [1, 2]:
                old = parent / f"{task}-temporal-forecast-seed{seed}-learn2000"
                assert json.loads((old / "stage-complete.json").read_text())["stop_step"] == 2000
                for arm, beta in [("rate-on", .25), ("rate-off", 0.)]:
                    name = f"{task}-{start}-{arm}-seed{seed}"
                    out = root / name
                    out.mkdir()
                    stop = 600 if start == "fresh" else 2600
                    cfg = json.loads((old / "config.json").read_text())
                    cfg.update(output=str(out), checkpoint_every=200,
                               retain_steps=[stop - 400, stop - 200, stop], record_initial_weights=True)
                    cfg["model"]["beta"] = beta
                    config = out / "config.json"
                    config.write_text(json.dumps(cfg, indent=2) + "\n")
                    if start == "resume2000":
                        shutil.copy2(old / "last.ckpt", out / "last.ckpt")
                        digest = sha(old / "last.ckpt")
                        assert sha(out / "last.ckpt") == digest
                        copies.append(dict(id=name, parent=str(old), checkpoint_sha256=digest))
                    common = dict(host=host, gpu=gpu, source=src, task=task,
                                  method=arm, training_seed=seed, initialization=start)
                    command = [py, "-B", "-u", "-m", "as_jepa_tools.train_stage",
                               "--config", str(config), "--stop-step", str(stop)]
                    if start != "fresh":
                        command.append("--resume")
                    jobs.append(dict(**common, id=name, output=str(out), config=str(config),
                                     config_sha256=sha(config), phase="train600", command=command))
                    health = root / ("health-" + name)
                    jobs.append(dict(**common, id="health-" + name, output=str(health),
                                     depends_on=[name], phase="health", command=[py, "-B", "-u",
                                     "-m", "as_jepa_tools.benchmark_health", "--run", str(out),
                                     "--output", str(health)]))
                    records.append(dict(id=name, task=task, host=host, gpu=gpu, seed=seed,
                                        initialization=start, beta=beta, stop_step=stop))
                    gpu += 1
    prior_plan = json.loads((parent / "plan.json").read_text())
    plan = dict(root=str(root), deadline=None, environment=prior_plan["environment"], jobs=jobs)
    protocol = dict(root=str(root), created_at=time.time(), source_commit=commit,
                    runs=records, additional_updates=600, all_seeds_preserved=True,
                    hypotheses=["A positive rate penalty suppresses denoising-useful state information.",
                        "Fresh versus resumed rate removal tests dependence on learning history."],
                    limitations=["Beta zero removes both entry and surprise penalties; it does not isolate them.",
                        "The forecast-render branch still trains the predictor through pixel denoising.",
                        "Fresh/resume contrasts also differ in optimizer, data cursor and LR; they cannot prove an irreversible absorbing state.",
                        "Information recovery does not imply a usable latent goal metric or planning improvement.",
                        "These are diagnostic ablations, not a proposed new algorithm or an automatic long-run screen."],
                    comparison="New paired controls, same initialization seed or exact full checkpoint, data, RNG, model size, scheduler horizon and 600 updates. Only beta differs within a pair.",
                    next_review="Inspect complete state, gradients, state/region-use and trajectories. No automatic continuation; old held runs remain immutable.")
    for name, value in [("plan.json", plan), ("protocol.json", protocol), ("resume-audit.json", copies)]:
        (root / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(dict(root=str(root), source_commit=commit, training=16, health=16)))


if __name__ == "__main__":
    main()
