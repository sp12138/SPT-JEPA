"""Fixed-checkpoint detached-action readout curves, including fresh control cases.

No consequence-model updates, expert paths, or future-action labels. All raw
artifacts live outside Git. A cell is one representation/checkpoint/readout.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from experiments.robocasa_action_probe import probe
from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_compare.evaluate import case_name
from experiments.robocasa_time_compare.campaign import env

MODULE = "experiments.robocasa_action_probe.scaling"
HOSTS = ("dev13", "dev15", "dev6")


def initialize(root):
    plan = json.loads((root/"plan.json").read_text())
    for host in HOSTS:
        assert json.loads((root/f"PREPARED-{host}.json").read_text())["passed"]
    cases = plan["cases"]
    assert len(cases) == 40 and len({case_name(**c) for c in cases}) == 40
    assert len(plan["development_cases"]) == 8 and len(plan["confirmation_cases"]) == 32
    card = json.loads((root/"data.json").read_text())
    learning_kitchens = {tuple(r["kitchen"]) for r in card["records"]}
    assert not learning_kitchens & {(c["kitchen"], c["kitchen"]) for c in cases}
    for case in cases:
        path = root/"cases"/(case_name(**case)+".npz")
        proof = json.loads(path.with_suffix(".json").read_text())
        assert proof["passed"] and sha(path) == proof["sha256"]
    jobs = []
    for label in plan["comparison_labels"]:
        for arm in ("spt64-linear", "global16-linear"):
            jobs.append(dict(id=arm+"-"+label, arm=arm, label=label))
    jobs.append(dict(id="vjepa-linear-reference", arm="vjepa-linear", label="hour-17"))
    old = Path(plan["source_pilot"])
    for index, job in enumerate(jobs):
        job["host"] = HOSTS[index % len(HOSTS)]
        cell = root/"cells"/job["id"]; cell.mkdir(parents=True, exist_ok=False)
        for name in ("code", "sources", "environment", "data.json", "cases"):
            (cell/name).symlink_to(root/name, target_is_directory=(root/name).is_dir())
        spec = dict(plan, comparison_checkpoint=job["label"], scaling_cell=job)
        save(cell/"plan.json", spec)
        reuse = job["label"] == "hour-17" or job["arm"] == "vjepa-linear"
        job["reused_pilot_head_and_development_episodes"] = reuse
        if reuse:
            (cell/"heads").mkdir()
            (cell/"heads"/job["arm"]).symlink_to(old/"heads"/job["arm"], target_is_directory=True)
            directory = cell/"evaluation"/job["arm"]; directory.mkdir(parents=True)
            for case in plan["development_cases"]:
                name = case_name(**case)
                (directory/name).symlink_to(old/"evaluation"/job["arm"]/name, target_is_directory=True)
    plan.update(jobs=jobs, phase="readout_curves", all_cases_fixed_before_head_fitting=True)
    save(root/"plan.json", plan)
    save(root/"INITIALIZED.json", dict(passed=True, cells=len(jobs), physical_cases=40,
        total_model_case_records=len(jobs)*40, reused_development_episodes=24,
        fresh_control_episodes=len(jobs)*40-24, added_consequence_SGD=0,
        target_kitchens_disjoint_from_readout_train_validation=True))


def command(root, argv, log, gpu, simulator=False):
    assert not (root/"STOP").exists()
    child_env = env(root, str(gpu), simulator)
    with log.open("x") as stream:
        subprocess.run(argv, cwd=root/"code", env=child_env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=3600)


def control(root, cell_id, shard):
    torch.set_num_threads(2); torch.set_float32_matmul_precision("high")
    cell = root/"cells"/cell_id
    plan = json.loads((cell/"plan.json").read_text())
    arm = plan["scaling_cell"]["arm"]
    indices = [i for i in range(len(plan["cases"])) if i % 8 == shard]
    start = time.perf_counter(); policy = probe.Policy(cell, arm)
    startup = time.perf_counter()-start
    completed, reused = [], []
    for index in indices:
        case = plan["cases"][index]
        output = cell/"evaluation"/arm/case_name(**case)/"summary.json"
        if output.exists():
            row = json.loads(output.read_text())
            assert row["passed"] and row["model_unchanged"]
            reused.append(index)
        else:
            probe.evaluate(cell, arm, index, policy=policy)
            completed.append(index)
    assert policy.unchanged()
    save(cell/f"CONTROLLED-{shard}.json", dict(passed=True, completed=completed,
        reused=reused, encoder_loaded_once=True, startup_seconds=startup,
        model_unchanged=True, no_CEM=True))


def summarize(rows):
    return dict(cases=len(rows), mean_return=float(np.mean([r["score"] for r in rows])),
        final_distance_mm=float(np.mean([r["final_goal_distance_m"]*1000 for r in rows])),
        final_successes=sum(r["final_task_complete"] for r in rows),
        any_successes=sum(r["any_task_complete"] for r in rows),
        mean_episode_seconds=float(np.mean([r["elapsed_seconds"] for r in rows])),
        per_case=[dict(case=case_name(task=r["task"], kitchen=r["kitchen"], seed=r["seed"]),
            kitchen=r["kitchen"], score=r["score"], final_mm=r["final_goal_distance_m"]*1000,
            final_success=r["final_task_complete"], any_success=r["any_task_complete"])
            for r in rows])


def collect_cell(root, job):
    cell = root/"cells"/job["id"]; arm = job["arm"]
    plan = json.loads((cell/"plan.json").read_text()); rows = []
    for case in plan["cases"]:
        path = cell/"evaluation"/arm/case_name(**case)
        row = json.loads((path/"summary.json").read_text())
        proof = json.loads((root/"cases"/(case_name(**case)+".json")).read_text())
        assert row["passed"] and row["model_unchanged"] and row["physical_replay_pass"]
        assert row["case_sha256"] == proof["sha256"] and sha(path/"trace.npz") == row["trace_sha256"]
        steps = row["macro_steps"]
        assert row["CEM_calls"] == 0 and 1 <= steps <= 5 and row["encoder_calls"] == steps+1
        if steps < 5:
            assert row["environment_terminated"]
        with np.load(path/"trace.npz") as trace:
            assert trace["actions"].shape == (steps, 7) and np.all(trace["actions"][:, 3:] == 0)
            assert np.isfinite(trace["actions"]).all()
            assert np.all(np.linalg.norm(trace["actions"][:, :3], axis=-1) <= .300001)
            assert abs(float(trace["rewards"].mean())-row["score"]) < 1e-12
        rows.append(row)
    head = json.loads((cell/"heads"/arm/"ready.json").read_text())
    assert head["passed"] and head["actual_stopgradient_backward_pass"]
    assert head["no_action_loss_to_backbone"] and not head["target_case_feedback_used_for_selection"]
    result = dict(passed=True, **job, head=head, development=summarize(rows[:8]),
        confirmation=summarize(rows[8:]), total_model_case_records=40,
        no_new_consequence_updates=True, completed_unix=time.time())
    if arm != "vjepa-linear":
        run = Path(plan["source"])/"training"/plan["source_models"][arm]
        result["parent"] = json.loads((run/job["label"]/"ready.json").read_text())
    save(root/"results"/(job["id"]+".json"), result)


def worker(root, host):
    assert host in HOSTS and (root/"INITIALIZED.json").exists()
    plan = json.loads((root/"plan.json").read_text())
    jobs = [j for j in plan["jobs"] if j["host"] == host]
    try:
        compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader"], text=True).strip()
        assert not compute, compute
        for job in jobs:
            cell = root/"cells"/job["id"]; logs = root/"logs"/job["id"]
            logs.mkdir(parents=True, exist_ok=False)
            save(root/f"PROGRESS-{host}.json", dict(job=job, phase="cache_or_reuse", time=time.time()))
            if not job["reused_pilot_head_and_development_episodes"]:
                def cache(shard):
                    command(root, [probe.TRAIN_PYTHON, "-u", "-m", probe.MODULE,
                        "--root", str(cell), "--mode", "cache", "--arm", job["arm"],
                        "--shard", str(shard)], logs/f"cache-{shard}.log", shard)
                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(cache, range(8)))
                command(root, [probe.TRAIN_PYTHON, "-u", "-m", probe.MODULE,
                    "--root", str(cell), "--mode", "fit", "--arm", job["arm"]], logs/"fit.log", 0)
            save(root/f"PROGRESS-{host}.json", dict(job=job, phase="control", time=time.time()))
            def evaluate(shard):
                command(root, [str(root/"environment/bin/python"), "-u", "-m", MODULE,
                    "--root", str(root), "--mode", "control", "--cell", job["id"],
                    "--shard", str(shard)], logs/f"control-{shard}.log", shard, True)
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(evaluate, range(8)))
            collect_cell(root, job)
        save(root/f"DONE-{host}.json", dict(passed=True, completed_jobs=[j["id"] for j in jobs],
            completed_unix=time.time()))
    except BaseException as error:
        save(root/f"ERROR-{host}.json", dict(error=repr(error), method_negative=False))
        raise


def endpoint_statistics(early, late):
    # Paired kitchens are resampled as blocks; this is case uncertainty, never
    # uncertainty across independent training seeds (there is only one parent).
    a = {r["case"]: r for r in early["per_case"]}
    b = {r["case"]: r for r in late["per_case"]}; assert a.keys() == b.keys()
    kitchens = sorted({r["kitchen"] for r in a.values()})
    groups = [np.array([b[k]["score"]-a[k]["score"] for k in sorted(a)
        if a[k]["kitchen"] == kitchen]) for kitchen in kitchens]
    assert len(groups) == 4 and all(len(g) == 8 for g in groups)
    values = np.stack(groups); rng = np.random.default_rng(20261113)
    samples = rng.integers(0, len(groups), size=(10000, len(groups)))
    means = values.mean(1)[samples].mean(1)
    return dict(paired_mean_return_gain=float(values.mean()),
        kitchen_block_bootstrap_95pct=np.quantile(means, [.025, .975]).tolist(),
        physical_case_wins=sum(b[k]["score"] > a[k]["score"] for k in a),
        strict_final_success_gain=late["final_successes"]-early["final_successes"],
        strict_any_success_gain=late["any_successes"]-early["any_successes"],
        training_seeds=1, kitchen_clusters=4, physical_cases=32,
        not_a_power_law_fit_or_independent_seed_confirmation=True)


def collect(root):
    plan = json.loads((root/"plan.json").read_text())
    for host in HOSTS:
        assert json.loads((root/f"DONE-{host}.json").read_text())["passed"]
    rows = [json.loads((root/"results"/(j["id"]+".json")).read_text()) for j in plan["jobs"]]
    assert len(rows) == 17 and all(r["passed"] for r in rows)
    endpoint = {}
    for arm in ("spt64-linear", "global16-linear"):
        by_label = {r["label"]: r for r in rows if r["arm"] == arm}
        endpoint[arm] = endpoint_statistics(by_label["hour-01"]["confirmation"],
            by_label["hour-18"]["confirmation"])
    save(root/"results/summary.json", dict(passed=True, rows=rows, endpoint=endpoint,
        added_consequence_SGD=0, distinct_physical_cases=40, confirmation_physical_cases=32,
        total_model_case_records=680, reused_development_records=24,
        fresh_control_episodes=656, measured_grid_not_selected_best=True,
        source_plan_sha256=sha(root/"plan.json"), collected_unix=time.time()))
    save(root/"DONE.json", dict(passed=True, all17cells_complete=True,
        scaling_outcome_must_be_read_from_metrics_not_DONE_flag=True, completed_unix=time.time()))
    (root/"HOLD").write_text("Fixed-checkpoint scaling audit complete. Preserve all results, including flat or negative curves.\n")
    print(json.dumps(endpoint), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", choices=("initialize", "worker", "control", "collect"), required=True)
    p.add_argument("--host", choices=HOSTS); p.add_argument("--cell"); p.add_argument("--shard", type=int)
    args = p.parse_args()
    if args.mode == "initialize": initialize(args.root)
    elif args.mode == "worker": worker(args.root, args.host)
    elif args.mode == "control": control(args.root, args.cell, args.shard)
    else: collect(args.root)
