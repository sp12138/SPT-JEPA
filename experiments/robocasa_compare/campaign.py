"""Owned, finite three-host pilot. No old campaign adoption or LLM monitoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

from experiments.robocasa_compare.acquire import save, sha

TRAIN_PYTHON = "/efs/prescottsun/project_jepa/envs/lewm/bin/python"
HOST_ARMS = {"dev13": "temporal-forecast", "dev15": "spatial", "dev6": "long-rollout"}


def cache_data(root, host):
    source = root / "packed"
    local = Path("/tmp/prescottsun-robocasa-compare-20261110")
    destination = local / "packed"
    destination.mkdir(parents=True, exist_ok=True)
    card = json.loads((source / "dataset.json").read_text())
    tasks = sorted({r["shard"] for r in card["records"]})

    def one(task):
        manifest = json.loads((source / task / "manifest.json").read_text())
        target = destination / task
        target.mkdir(exist_ok=True)
        for name, key in [("pixels.npy", "pixels_sha256"), ("poses.npy", "poses_sha256")]:
            path = target / name
            if path.exists():
                assert sha(path) == manifest[key]
            else:
                temp = path.with_suffix(".partial")
                shutil.copyfile(source / task / name, temp)
                assert sha(temp) == manifest[key]
                temp.replace(path)
        shutil.copyfile(source / task / "manifest.json", target / "manifest.json")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(one, tasks))
    shutil.copyfile(source / "dataset.json", destination / "dataset.json")
    result = dict(passed=True, host=host, path=str(destination), dataset_sha256=sha(source / "dataset.json"),
                  all_array_checksums_pass=True, source_arrays_unchanged=True)
    save(root / f"local-data-{host}.json", result)
    return destination


def cache_simulator(root, host):
    assets = json.loads((root / "assets.json").read_text())
    base = Path("/tmp/prescottsun-robocasa-compare-20261110/simulator")
    base.mkdir(parents=True, exist_ok=True)
    if not (base / "READY.json").exists():
        for name in ["robocasa", "robosuite"]:
            target = base / name
            target.mkdir(exist_ok=False)
            archive = subprocess.Popen(["git", "archive", "HEAD"], cwd=root / "sources" / name, stdout=subprocess.PIPE)
            subprocess.run(["tar", "-xf", "-", "-C", str(target)], stdin=archive.stdout, check=True)
            archive.stdout.close()
            assert archive.wait() == 0
        paths = {"textures": "", "generative_textures": "", "fixtures_lightwheel": "",
                 "objaverse": "objects", "aigen_objs": "objects", "objects_lightwheel": "objects"}
        package = base / "robocasa/robocasa/models/assets"
        for group, relative in paths.items():
            archive = root / "downloads/assets" / (group + ".zip")
            assert sha(archive) == assets[group]["sha256"]
            target = package / relative
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as stream:
                assert all((target / n.filename).resolve().is_relative_to(target.resolve()) for n in stream.infolist())
                stream.extractall(target)
        save(base / "READY.json", dict(passed=True, asset_archives=assets,
                                      source_revisions=json.loads((root / "sources.json").read_text())))
    result = dict(passed=True, host=host, robocasa=str(base / "robocasa"),
                  robosuite=str(base / "robosuite"), marker_sha256=sha(base / "READY.json"))
    save(root / f"local-simulator-{host}.json", result)
    return result


def environment(root, gpu_list, simulator=False):
    paths = [str(root / "code"), str(root / "sources/robo_jepa"),
             str(root / "sources/robo_jepa/scripts/deployment")]
    if simulator:
        base = Path("/tmp/prescottsun-robocasa-compare-20261110/simulator")
        assert (base / "READY.json").exists()
        paths = [str(base / "robocasa"), str(base / "robosuite")] + paths
    project = root.parents[1]
    egl = project / "environments/egl-lib"
    return dict(os.environ, PYTHONPATH=":".join(paths), CUDA_VISIBLE_DEVICES=gpu_list,
                MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", PYNPUT_BACKEND="dummy",
                MUJOCO_EGL_DEVICE_ID=gpu_list.split(",")[0] if gpu_list else "0",
                LD_LIBRARY_PATH=str(egl / "usr/lib/x86_64-linux-gnu") + ":" + os.environ.get("LD_LIBRARY_PATH", ""),
                __EGL_VENDOR_LIBRARY_FILENAMES=str(egl / "10_nvidia.json"),
                OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                PYTHONHASHSEED="0",
                WANDB_MODE="disabled", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")


def command(argv, log, env, cwd, timeout):
    with Path(log).open("x") as stream:
        subprocess.run(argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL,
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def worker(root, host, seed, stop_step, attempt=1):
    arm = HOST_ARMS[host]
    run = root / "training" / f"{arm}-seed{seed}"
    assert (root / f"local-data-{host}.json").exists()
    data = json.loads((root / f"local-data-{host}.json").read_text())["path"]
    gpu = "0,1,2,3" if seed == 1 else "4,5,6,7"
    env = environment(root, gpu)
    try:
        cmd = [TRAIN_PYTHON, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=4",
               "-m", "experiments.robocasa_compare.train", "--root", str(root), "--run-directory", str(run),
               "--stop-step", str(stop_step), "--data-root", data]
        if stop_step > 2:
            assert json.loads((run / "audit-000002.json").read_text())["passed"]
            assert json.loads((root / "SIMULATOR_ENTRY_APPROVED.json").read_text())["passed"]
            cmd.append("--resume")
        suffix = "" if attempt == 1 else f"-attempt{attempt}"
        command(cmd, run / f"train-to-{stop_step}{suffix}.log", env, root / "code", 36 * 3600)
        single = "0" if seed == 1 else "4"
        command([TRAIN_PYTHON, "-m", "experiments.robocasa_compare.audit", "--root", str(root),
                 "--run", str(run), "--step", str(stop_step)], run / f"audit-{stop_step}{suffix}.log",
                environment(root, single), root / "code", 1800)
        save(run / f"completed-{stop_step}.json", dict(passed=True, step=stop_step, host=host))
    except BaseException as error:
        save(run / f"worker-error-{stop_step}-attempt{attempt}.json", dict(error=repr(error), host=host, seed=seed))
        raise


def launch(root, host, stop_step, attempt=1):
    assert host in HOST_ARMS
    assert (root / "training-initialized.json").exists()
    assert not (root / "STOP").exists()
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"], text=True).strip()
    assert not active, active
    suffix = "" if attempt == 1 else f"-attempt{attempt}"
    owner = root / f"launch-{host}-to-{stop_step}{suffix}.json"
    assert not owner.exists()
    records = []
    for seed in [1, 2]:
        cmd = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=60", str(38 * 3600), TRAIN_PYTHON,
               "-u", "-m", "experiments.robocasa_compare.campaign", "--mode", "worker", "--root", str(root),
               "--host", host, "--seed", str(seed), "--stop-step", str(stop_step), "--attempt", str(attempt)]
        env = environment(root, "")
        with (root / f"worker-{host}-seed{seed}-to-{stop_step}{suffix}.log").open("x") as stream:
            process = subprocess.Popen(cmd, cwd=root / "code", env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        records.append(dict(pid=process.pid, seed=seed, arm=HOST_ARMS[host], command=cmd))
    save(owner, dict(host=host, stop_step=stop_step, records=records, created_unix=time.time()))
    print(json.dumps(dict(host=host, stop_step=stop_step, pids=[r["pid"] for r in records])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", choices=list(HOST_ARMS), required=True)
    parser.add_argument("--mode", choices=["cache-data", "cache-simulator", "launch", "worker"], required=True)
    parser.add_argument("--seed", type=int, choices=[1, 2])
    parser.add_argument("--stop-step", type=int, choices=[2, 500, 5000], default=2)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "cache-data":
        cache_data(args.root, args.host)
    elif args.mode == "cache-simulator":
        cache_simulator(args.root, args.host)
    elif args.mode == "launch":
        launch(args.root, args.host, args.stop_step, args.attempt)
    else:
        worker(args.root, args.host, args.seed, args.stop_step, args.attempt)
