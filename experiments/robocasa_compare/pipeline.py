"""Finite training/evaluation plan; no recurring research or LLM monitor."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import subprocess
import time

from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_compare.campaign import HOST_ARMS, TRAIN_PYTHON, command, environment


HOSTS = tuple(HOST_ARMS)


def cases():
    return [dict(task=task, kitchen=kitchen, seed=seed)
            for task in ("reach", "push") for kitchen in range(1, 11) for seed in (7001, 7002)]


def case_id(case):
    return f"{case['task']}-k{case['kitchen']}-s{case['seed']}"


def jobs(host, phase):
    """Immutable assignment: every candidate sees every common case at final."""
    arm = HOST_ARMS[host]
    cohort = cases() if phase == "final" else [c for c in cases() if c["kitchen"] in (1, 6) and c["seed"] == 7001]
    result = []
    for seed in (1, 2):
        for case in cohort:
            result.append(dict(id=f"{arm}-seed{seed}-" + case_id(case), case=case,
                               run=f"{arm}-seed{seed}", step=5000 if phase == "final" else 500))
    if phase == "final":
        references = [dict(id=f"robojepa-{size}-mode{mode}-" + case_id(case),
                           case=case, reference=size, reference_mode=mode)
                      for size in ("50m", "2b") for mode in (2, 3) for case in cases()]
        references += [dict(id="zero-" + case_id(case), case=case, zero=True) for case in cases()]
        result += [job for i, job in enumerate(references) if i % len(HOSTS) == HOSTS.index(host)]
    return result


def evaluate_command(root, job, output, technical=False):
    args = [str(root / "environment/bin/python"), "-u", "-m", "experiments.robocasa_compare.evaluate",
            "--root", str(root), "--mode", "evaluate", "--output", str(output)]
    for key, value in job["case"].items():
        args += ["--" + key, str(value)]
    if job.get("run"):
        args += ["--run", str(root / "training" / job["run"]), "--step", str(job["step"])]
    elif job.get("reference"):
        args += ["--reference", job["reference"], "--reference-mode", str(job["reference_mode"])]
    else:
        assert job["zero"]
        args.append("--zero")
    if technical:
        args.append("--technical")
    return args


def initialize(root):
    plan = dict(version=1, hosts=HOST_ARMS, cases=cases(),
                jobs={host: {phase: jobs(host, phase) for phase in ("early", "final")} for host in HOSTS},
                stages=[2, 500, 5000], max_hours=72,
                target_scores_do_not_select_models=True, automatic_long_extension=False,
                external_references_are_not_equal_data_scratch_comparisons=True)
    path = root / "pipeline/plan.json"
    assert not path.exists()
    save(path, plan)
    return plan


def generate_cases(root, host):
    """Disjoint host assignments, preserving invalid goals in the ledger."""
    cohort = [c for i, c in enumerate(cases()) if i % len(HOSTS) == HOSTS.index(host)]
    todo = queue.Queue()
    for case in cohort:
        todo.put(case)
    logs = root / "logs/cases"
    logs.mkdir(parents=True, exist_ok=True)

    def one(gpu):
        rows = []
        while not todo.empty():
            try:
                case = todo.get_nowait()
            except queue.Empty:
                break
            marker = root / "cases" / (case_id(case) + ".json")
            try:
                if not marker.exists():
                    argv = [str(root / "environment/bin/python"), "-u", "-m", "experiments.robocasa_compare.evaluate",
                            "--root", str(root), "--mode", "case"]
                    for key, value in case.items():
                        argv += ["--" + key, str(value)]
                    command(argv, logs / (case_id(case) + ".log"), environment(root, str(gpu), True), root / "code", 1800)
                row = json.loads(marker.read_text())
                assert row["passed"] and sha(row["path"]) == row["sha256"]
                rows.append(dict(case=case, passed=True, marker=str(marker), sha256=row["sha256"]))
            except Exception as error:
                rows.append(dict(case=case, passed=False, error=repr(error), method_negative=False))
        return rows

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = sum(list(pool.map(one, range(8))), [])
    save(root / "pipeline" / f"cases-{host}.json", dict(passed=all(r["passed"] for r in rows), cases=rows))


def evaluate_jobs(root, host, phase, deadline):
    plan = json.loads((root / "pipeline/plan.json").read_text())
    todo = queue.Queue()
    for job in plan["jobs"][host][phase]:
        todo.put(job)

    def gpu_worker(gpu):
        records = []
        while not todo.empty() and time.time() < deadline and not (root / "STOP").exists():
            try:
                job = todo.get_nowait()
            except queue.Empty:
                break
            output = root / "evaluation" / phase / job["id"]
            status = root / "pipeline" / phase / (job["id"] + ".json")
            case = root / "cases" / (case_id(job["case"]) + ".json")
            if not case.exists() or not json.loads(case.read_text()).get("passed"):
                result = dict(passed=False, case_generation_failure=True, method_negative=False, job=job)
            else:
                try:
                    if output.exists():
                        raise RuntimeError(f"Output already exists; explicit review required: {output}")
                    argv = evaluate_command(root, job, output)
                    command(argv, root / "logs" / phase / (job["id"] + ".log"),
                            environment(root, str(gpu), simulator=True), root / "code",
                            min(8 * 3600, max(1, deadline - time.time())))
                    summary = json.loads((output / "summary.json").read_text())
                    assert summary["passed"] and not summary["technical_only"]
                    result = dict(passed=True, job=job, summary=str(output / "summary.json"),
                                  summary_sha256=sha(output / "summary.json"))
                except Exception as error:
                    result = dict(passed=False, execution_error=repr(error), method_negative=False, job=job)
            save(status, result)
            records.append(result)
        return records

    (root / "logs" / phase).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = sum(list(pool.map(gpu_worker, range(8))), [])
    save(root / "pipeline" / f"{host}-{phase}.json",
         dict(passed=len(results) == len(plan["jobs"][host][phase]) and all(r["passed"] for r in results),
              expected=len(plan["jobs"][host][phase]), disposed=len(results),
              execution_or_case_errors=sum(not r["passed"] for r in results), results=results))


def train_stage(root, host, step, deadline):
    def one(seed):
        argv = [TRAIN_PYTHON, "-u", "-m", "experiments.robocasa_compare.campaign", "--mode", "worker",
                "--root", str(root), "--host", host, "--seed", str(seed), "--stop-step", str(step)]
        command(argv, root / "pipeline" / f"{host}-seed{seed}-to-{step}.log",
                environment(root, ""), root / "code", min(38 * 3600, max(1, deadline - time.time())))
        run = root / "training" / f"{HOST_ARMS[host]}-seed{seed}"
        assert json.loads((run / f"completed-{step}.json").read_text())["passed"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(one, (1, 2)))


def run(root, host):
    owner = root / "pipeline" / f"launch-{host}.json"
    deadline = json.loads(owner.read_text())["deadline_unix"]
    try:
        for step in (500, 5000):
            assert not (root / "STOP").exists() and time.time() < deadline
            train_stage(root, host, step, deadline)
            evaluate_jobs(root, host, "early" if step == 500 else "final", deadline)
        phases = {phase: json.loads((root / "pipeline" / f"{host}-{phase}.json").read_text())
                  for phase in ("early", "final")}
        save(root / "pipeline" / f"completed-{host}.json",
             dict(passed=all(p["passed"] for p in phases.values()), queue_disposed=True,
                  finished_unix=time.time(), phases=phases))
        if host == "dev13":
            while time.time() < deadline and not (root / "STOP").exists():
                if all((root / "pipeline" / f"completed-{h}.json").exists() or
                       (root / "pipeline" / f"error-{h}.json").exists() for h in HOSTS):
                    break
                time.sleep(10)
            collect(root)
    except BaseException as error:
        save(root / "pipeline" / f"error-{host}.json", dict(error=repr(error), time_unix=time.time()))
        raise


def launch(root, host):
    plan = json.loads((root / "pipeline/plan.json").read_text())
    assert json.loads((root / "SIMULATOR_ENTRY_APPROVED.json").read_text())["passed"]
    for arm in HOST_ARMS.values():
        for seed in (1, 2):
            assert json.loads((root / "training" / f"{arm}-seed{seed}" / "completed-2.json").read_text())["passed"]
    assert not (root / "STOP").exists()
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    assert not active, active
    path = root / "pipeline" / f"launch-{host}.json"
    assert not path.exists()
    duration = plan["max_hours"] * 3600
    argv = ["/usr/bin/timeout", "--signal=TERM", "--kill-after=60", str(duration), TRAIN_PYTHON, "-u", "-m",
            "experiments.robocasa_compare.pipeline", "--root", str(root), "--host", host, "--mode", "run"]
    with (root / "pipeline" / f"{host}.log").open("x") as log:
        p = subprocess.Popen(argv, env=environment(root, ""), cwd=root / "code", stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(path, dict(pid=p.pid, command=argv, host=host, created_unix=time.time(),
                    deadline_unix=time.time() + duration, plan_sha256=sha(root / "pipeline/plan.json")))
    print(json.dumps(dict(host=host, pid=p.pid)), flush=True)


def collect(root):
    """One-shot collection; no polling process and no fabricated missing scores."""
    import numpy as np
    plan = json.loads((root / "pipeline/plan.json").read_text())
    groups, errors, pending = {}, [], []
    all_jobs = [j for host in HOSTS for j in plan["jobs"][host]["final"]]
    for job in all_jobs:
        status = root / "pipeline/final" / (job["id"] + ".json")
        if not status.exists():
            pending.append(job["id"])
            continue
        status = json.loads(status.read_text())
        if not status["passed"]:
            errors.append(status)
            continue
        assert sha(status["summary"]) == status["summary_sha256"]
        row = json.loads(Path(status["summary"]).read_text())
        if job.get("run"):
            label = job["run"].rsplit("-seed", 1)[0]
        elif job.get("reference"):
            label = f"robojepa-{job['reference']}-mode{job['reference_mode']}"
        else:
            label = "zero"
        groups.setdefault(label, {}).setdefault(row["task"], []).append(row)
    summaries = {}
    for label, tasks in groups.items():
        summaries[label] = {}
        for task, rows in tasks.items():
            by_case = {}
            for row in rows:
                by_case.setdefault((row["kitchen"], row["seed"]), []).append(row["score"])
            summaries[label][task] = dict(mean_score=float(np.mean([np.mean(v) for v in by_case.values()])),
                 episodes=len(rows), physical_cases=len(by_case), kitchens=len({k[0] for k in by_case}),
                 final_task_completion_rate=float(np.mean([r["final_task_complete"] for r in rows])),
                 any_task_completion_rate=float(np.mean([r["any_task_complete"] for r in rows])),
                 metric=rows[0]["metric"], complete=len(rows) == (40 if label in HOST_ARMS.values() else 20))
    # Conditional on these two trained models: bootstrap paired physical kitchens,
    # averaging model/physical seeds within each kitchen before resampling.
    comparisons = []
    for arm in HOST_ARMS.values():
        for reference in ("robojepa-2b-mode2", "robojepa-2b-mode3"):
            for task in ("reach", "push"):
                tables = []
                for label in (arm, reference):
                    table = {}
                    for row in groups.get(label, {}).get(task, []):
                        table.setdefault((row["kitchen"], row["seed"]), []).append(row["score"])
                    tables.append(table)
                common = sorted(tables[0].keys() & tables[1].keys())
                if not common:
                    continue
                diffs = {}
                for key in common:
                    diffs.setdefault(key[0], []).append(np.mean(tables[0][key]) - np.mean(tables[1][key]))
                values = np.array([np.mean(v) for v in diffs.values()])
                rng = np.random.default_rng(7311)
                boot = rng.choice(values, (10000, len(values)), replace=True).mean(1)
                comparisons.append(dict(arm=arm, reference=reference, task=task, mean_difference=float(values.mean()),
                     kitchen_cluster_95ci=np.quantile(boot, [.025, .975]).tolist(),
                     common_physical_cases=len(common), common_kitchens=len(values),
                     conditional_on_two_trained_models=True, equal_training_data_or_pretraining_budget=False))
    save(root / "results/summary.json", dict(queue_disposed=not pending, expected=len(all_jobs),
         scores=summaries, comparisons=comparisons, errors=errors, pending=pending,
         goal_generation_failures=[c for c in cases() if not (root/"cases"/(case_id(c)+".json")).exists()],
         public_reference_is_not_equal_budget_scratch_comparison=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", choices=HOSTS)
    parser.add_argument("--mode", choices=["initialize", "launch", "run", "cases", "collect"], required=True)
    args = parser.parse_args()
    if args.mode == "initialize":
        initialize(args.root)
    elif args.mode == "launch":
        launch(args.root, args.host)
    elif args.mode == "cases":
        generate_cases(args.root, args.host)
    elif args.mode == "collect":
        collect(args.root)
    else:
        run(args.root, args.host)
