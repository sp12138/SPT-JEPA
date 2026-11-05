"""Finish the existing 48 missing endpoints; no completed training/evaluation reruns."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def write(path, value):
    from repair_20261105 import durable_json
    durable_json(path, value)


def pending_jobs(root):
    plan = json.loads((root / 'plan.json').read_text())
    return [j for j in plan['evaluation_jobs']
            if not (root / 'evaluations' / j['name'] / 'result.json').exists()]


def evaluate(root, host, gpu):
    from concurrent.futures import ThreadPoolExecutor
    recovery = root / 'recovery-20261105'
    launch = json.loads((recovery / 'training-launch.json').read_text())
    deadline = launch['deadline_unix']
    jobs = json.loads((recovery / 'missing-jobs.json').read_text())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               HJEPA_HOME='/tmp/prescottsun-hjepa-droid-cc05cb8d', OMP_NUM_THREADS='3',
               OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    while time.time() < deadline:
        available = False
        for job in jobs:
            out = root / 'evaluations' / job['name']
            if (out / 'result.json').exists():
                continue
            available = True
            checkpoint = Path(job['checkpoint'])
            if not (checkpoint.parent / 'epoch100-ready.json').exists():
                continue
            claim = recovery / 'claims' / job['name']
            try:
                claim.mkdir(parents=True)
            except FileExistsError:
                continue
            out.mkdir(parents=True, exist_ok=True)
            command = ['/tmp/prescottsun-hjepa-env-97054ec/bin/python', '-u',
                       str(root / 'helper-v8/evaluate.py'), '--root', str(root), '--arm', job['arm'],
                       '--model-seed', str(job['model_seed']), '--planner-seed', str(job['planner_seed']),
                       '--clip', str(job['clip']), '--checkpoint', str(checkpoint), '--output', str(out)]
            with (out / 'recovery-output.log').open('ab') as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                write(out / 'recovery-launch.json', dict(pid=process.pid, command=command,
                                                        host=host, gpu=gpu, started=time.time()))
                while process.poll() is None and time.time() < deadline:
                    time.sleep(5)
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                if process.returncode or not (out / 'result.json').exists():
                    write(out / 'recovery-error.json', dict(exitcode=process.returncode, job=job))
                    return
        if not available:
            write(recovery / 'workers' / f'{host}-gpu{gpu}.json', dict(passed=True))
            return
        time.sleep(10)
    write(recovery / 'workers' / f'{host}-gpu{gpu}.json', dict(passed=False, reason='deadline'))


def collect(root):
    import numpy as np
    recovery = root / 'recovery-20261105'
    launch = json.loads((recovery / 'training-launch.json').read_text())
    plan = json.loads((root / 'plan.json').read_text())
    cached = {}
    pending = list(plan['evaluation_jobs'])
    while time.time() < launch['deadline_unix']:
        remaining = []
        for job in pending:
            path = root / 'evaluations' / job['name'] / 'result.json'
            if path.exists():
                v = json.loads(path.read_text())
                assert v['passed'] and v['state_unchanged']
                assert all(v[k] == job[k] for k in ['arm', 'model_seed', 'planner_seed', 'clip'])
                cached[job['name']] = dict(job=job, **v)
            else:
                remaining.append(job)
        pending = remaining
        records = list(cached.values())
        groups = {}
        for v in records:
            j = v['job']
            name = f"{j['category']}/{j['arm']}/epoch{j.get('epoch', 'public')}"
            groups.setdefault(name, []).append(v)
        metrics = {}
        for name, rows in groups.items():
            metrics[name] = dict(episodes=len(rows), training_seeds=sorted({v['model_seed'] for v in rows}),
                                 frechet_fidelity_mean=float(np.mean([v['metrics']['frechet/skill'] for v in rows])),
                                 ate_xyz_mean=float(np.mean([v['metrics']['ate/end_distance_xyz'] for v in rows])))
        complete = len(records) == len(plan['evaluation_jobs'])
        trainings = [root / 'training' / r['name'] / 'training-complete.json' for r in plan['recipes']]
        all_trained = all(p.exists() and json.loads(p.read_text())['step'] == 29200 for p in trainings)
        summary = dict(updated_unix=time.time(), completed_episodes=len(records), expected_episodes=1584,
                       training_complete=sum(p.exists() for p in trainings), training_total=12,
                       metrics=metrics, preliminary=not (complete and all_trained),
                       historical_failures_preserved=True, recovery='recovery-20261105',
                       note='Fréchet fidelity is offline trajectory fidelity, not physical robot success.')
        write(recovery / 'summary.json', summary)
        write(root / 'results/summary.json', summary)
        if complete and all_trained:
            write(recovery / 'HOLD', dict(passed=True, completed_episodes=1584, trainings=12,
                                         total_step_cap=29200, automatic_extension=False))
            return
        time.sleep(30)
    write(recovery / 'HOLD', dict(passed=False, reason='finite recovery deadline', automatic_extension=False))


def supervise(root):
    recovery = root / 'recovery-20261105'
    launch = json.loads((recovery / 'training-launch.json').read_text())
    pid = launch['pid']
    while time.time() < launch['deadline_unix']:
        path = Path(f'/proc/{pid}/cmdline')
        if not path.exists():
            break
        command = path.read_bytes().replace(b'\0', b' ').decode()
        if str(root / 'repair-20261105-v1.py') not in command:
            break
        time.sleep(15)
    else:
        path = Path(f'/proc/{pid}/cmdline')
        if path.exists() and str(root / 'repair-20261105-v1.py') in path.read_bytes().replace(b'\0', b' ').decode():
            os.killpg(pid, signal.SIGTERM)
        write(recovery / 'training-error.json', dict(reason='deadline'))
        return
    run = root / 'training/h-jepa-seed10000'
    path = run / 'training-complete.json'
    if not path.exists():
        write(recovery / 'training-error.json', dict(reason='resumed process exited without completion'))
        return
    assert json.loads(path.read_text())['step'] == 29200
    write(recovery / 'training-recovered.json', dict(passed=True, original_step=5800, final_step=29200,
                                                   new_updates=23400, no_completed_model_retrained=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['eval', 'collect', 'supervise'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--host')
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    root = Path(args.root)
    if args.mode == 'eval':
        evaluate(root, args.host, args.gpu)
    elif args.mode == 'collect':
        collect(root)
    else:
        supervise(root)
