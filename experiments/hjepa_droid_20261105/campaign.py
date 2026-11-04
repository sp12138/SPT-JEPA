"""Finite two-host queue; training, fixed evaluation, collect, then HOLD."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ARMS = ['temporal-forecast', 'innovation', 'lewm', 'h-jepa']
SEEDS = [1, 1000, 10000]
PYTHON = '/tmp/prescottsun-hjepa-env-97054ec/bin/python'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def environment(gpus):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
               HJEPA_HOME='/tmp/prescottsun-hjepa-droid-cc05cb8d', OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', HYDRA_FULL_ERROR='1')
    for key in list(env):
        if key.startswith('SLURM_'):
            del env[key]
    return env


def plan(root):
    assert not (root / 'plan.json').exists()
    recipes = []
    for i, arm in enumerate(ARMS):
        for seed in SEEDS:
            host = 'dev13' if seed == 1 or (seed == 10000 and i < 2) else 'dev15'
            recipes.append(dict(name=f'{arm}-seed{seed}', arm=arm, seed=seed, host=host, pair=i,
                                global_batch=256, epochs=100, updates=29200))
    jobs = []
    for arm in ['lewm', 'h-jepa', 'hwm']:
        for seed in SEEDS:
            for planner_seed in [1, 2, 3]:
                for clip in range(16):
                    name = f'reference-{arm}-s{seed}-p{planner_seed}-c{clip:02d}'
                    jobs.append(dict(name=name, arm=arm, model_seed=seed, planner_seed=planner_seed,
                                     clip=clip, category='reference'))
    for recipe in recipes:
        modelname = 'droid_hjepa_l2' if recipe['arm'] == 'h-jepa' else 'droid_lewm'
        for epoch in [10, 100]:
            checkpoint = root / 'training' / recipe['name'] / f'{modelname}_epoch_{epoch}_object.ckpt'
            for planner_seed in [1, 2, 3]:
                for clip in range(16):
                    jobs.append(dict(name=f"{recipe['name']}-e{epoch}-p{planner_seed}-c{clip:02d}",
                                     arm=recipe['arm'], model_seed=recipe['seed'],
                                     planner_seed=planner_seed, clip=clip, epoch=epoch,
                                     checkpoint=str(checkpoint), category='trained'))
    result = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  allowed_hosts=['dev13', 'dev15'], recipes=recipes, evaluation_jobs=jobs,
                  reference_episodes=432, trained_episodes=1152,
                  max_seconds_after_approval=72 * 3600,
                  automatic_search=False, automatic_extensions=False, llm_monitor=False,
                  candidate_loss='sample-level', pretrained_candidate_modules=False)
    write(root / 'plan.json', result)
    return result


def stopped(root, deadline):
    return (root / 'STOP').exists() or time.time() > deadline


def child(root, cmd, env, log, record, deadline):
    with Path(log).open('ab') as stream:
        process = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True,
                                   cwd=str(root / 'public/h_jepa'))
        write(record, dict(pid=process.pid, command=cmd, gpus=env['CUDA_VISIBLE_DEVICES'],
                           started=time.time(), log=str(log)))
        while process.poll() is None:
            if stopped(root, deadline):
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(5)
        return process.returncode


def eval_worker(root, host, gpu, deadline):
    jobs = json.loads((root / 'plan.json').read_text())['evaluation_jobs']
    helper = Path(__file__).resolve().parent
    while not stopped(root, deadline):
        selected = None
        unfinished = 0
        for job in jobs:
            out = root / 'evaluations' / job['name']
            if (out / 'result.json').exists() or (out / 'error.json').exists():
                continue
            unfinished += 1
            if 'checkpoint' in job and not Path(job['checkpoint']).is_file():
                continue
            if 'checkpoint' in job and not (Path(job['checkpoint']).parent / f"epoch{job['epoch']}-ready.json").exists():
                continue
            claim = root / 'claims' / job['name']
            try:
                claim.mkdir(parents=True)
            except FileExistsError:
                continue
            selected = job
            write(claim / 'owner.json', dict(host=host, gpu=gpu, pid=os.getpid(), time=time.time()))
            break
        if selected is None:
            if not unfinished or (root / 'training-disposed.json').exists():
                return
            time.sleep(10)
            continue
        job = selected
        out = root / 'evaluations' / job['name']
        out.mkdir(parents=True, exist_ok=True)
        command = [PYTHON, '-u', str(helper / 'evaluate.py'), '--root', str(root),
                   '--arm', job['arm'], '--model-seed', str(job['model_seed']),
                   '--planner-seed', str(job['planner_seed']), '--clip', str(job['clip']),
                   '--output', str(out)]
        if 'checkpoint' in job:
            command += ['--checkpoint', job['checkpoint']]
        rc = child(root, command, environment([gpu]), out / 'output.log',
                   out / 'launch.json', deadline)
        if rc or not (out / 'result.json').exists():
            write(out / 'error.json', dict(exitcode=rc, job=job, host=host, gpu=gpu,
                                         error_is_not_an_algorithm_score=True))


def pair_worker(root, host, pair, adopt=None):
    approval = json.loads((root / 'TRAINING_APPROVED.json').read_text())
    assert approval['passed'] and host in ['dev13', 'dev15']
    deadline = approval['deadline_unix']
    helper = Path(__file__).resolve().parent
    all_recipes = json.loads((root / 'plan.json').read_text())['recipes']
    original = next(r for r in all_recipes if r['host'] == host and r['pair'] == pair and
                    r['seed'] == (1 if host == 'dev13' else 1000))
    gpus = [2 * pair, 2 * pair + 1]
    if adopt is not None:
        # Only the finite operational parent changes. Its torchrun child is in
        # its own process group and keeps all already-executed SGD untouched.
        launch = json.loads((root / 'training' / original['name'] / 'launch.json').read_text())
        assert launch['pid'] == adopt
        while True:
            proc = Path(f'/proc/{adopt}/stat')
            alive = proc.exists() and proc.read_text().split()[2] != 'Z'
            if not alive:
                break
            if stopped(root, deadline):
                os.killpg(adopt, signal.SIGTERM)
                for _ in range(30):
                    if not Path(f'/proc/{adopt}').exists():
                        break
                    time.sleep(1)
                if Path(f'/proc/{adopt}').exists():
                    try:
                        os.killpg(adopt, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break
            time.sleep(5)
        run = root / 'training' / original['name']
        if not (run / 'training-complete.json').exists():
            write(run / 'error.json', dict(adopted_pid=adopt, exitcode_unavailable=True,
                                         error_is_not_an_algorithm_score=True))
    # The longest remaining arms start as soon as any pair becomes free.
    # Three seeds/method and every optimizer/data/evaluation budget stay fixed.
    while not stopped(root, deadline):
        recipe = None
        for candidate in all_recipes:
            run = root / 'training' / candidate['name']
            if any((run / marker).exists() for marker in ['training-complete.json', 'error.json', 'incomplete.json']):
                continue
            claim = root / 'training-claims' / candidate['name']
            try:
                claim.mkdir(parents=True)
            except FileExistsError:
                continue
            recipe = candidate
            write(claim / 'owner.json', dict(host=host, pair=pair, pid=os.getpid(), time=time.time()))
            break
        if recipe is None:
            break
        run = root / 'training' / recipe['name']
        run.mkdir(parents=True, exist_ok=True)
        if stopped(root, deadline):
            write(run / 'incomplete.json', dict(reason='deadline or STOP', recipe=recipe))
            continue
        command = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   '--nproc-per-node=2', str(root / 'helper-v8/train.py'), '--root', str(root),
                   '--host', host, '--arm', recipe['arm'], '--seed', str(recipe['seed'])]
        rc = child(root, command, environment(gpus), run / 'output.log', run / 'launch.json', deadline)
        if rc or not (run / 'training-complete.json').exists():
            write(run / 'error.json', dict(exitcode=rc, recipe=recipe,
                                         error_is_not_an_algorithm_score=True))
    write(root / 'workers' / f'{host}-pair{pair}-training-disposed.json', dict(passed=True))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda g: eval_worker(root, host, g, deadline), gpus))
    write(root / 'workers' / f'{host}-pair{pair}-complete.json', dict(passed=True))


def collect(root):
    import numpy as np
    frozen = json.loads((root / 'plan.json').read_text())
    deadline = json.loads((root / 'TRAINING_APPROVED.json').read_text())['deadline_unix']
    while True:
        disposed = []
        for recipe in frozen['recipes']:
            run = root / 'training' / recipe['name']
            disposed.append(any((run / name).exists() for name in
                                ['training-complete.json', 'error.json', 'incomplete.json']))
        if all(disposed):
            write(root / 'training-disposed.json', dict(passed=True, runs=len(disposed)))
        records, errors = [], []
        for job in frozen['evaluation_jobs']:
            out = root / 'evaluations' / job['name']
            if (out / 'result.json').exists():
                records.append(dict(job=job, **json.loads((out / 'result.json').read_text())))
            elif (out / 'error.json').exists():
                errors.append(job)
        groups = {}
        for record in records:
            job = record['job']
            key = f"{job['category']}/{job['arm']}/epoch{job.get('epoch', 'public')}"
            groups.setdefault(key, []).append(record)
        metrics = {}
        for key, rows in groups.items():
            metrics[key] = dict(episodes=len(rows), training_seeds=sorted({r['model_seed'] for r in rows}),
                               frechet_fidelity_mean=float(np.mean([r['metrics']['frechet/skill'] for r in rows])),
                               ate_xyz_mean=float(np.mean([r['metrics']['ate/end_distance_xyz'] for r in rows])))
        summary = dict(updated_unix=time.time(), training_disposed=sum(disposed),
                       training_total=len(disposed), completed_episodes=len(records),
                       expected_episodes=len(frozen['evaluation_jobs']), errors=errors,
                       metrics=metrics, preliminary=len(records) != len(frozen['evaluation_jobs']),
                       note='Repeated clips/planner seeds are not independent physical trials; Fréchet fidelity is not robot success.')
        write(root / 'results/summary.json', summary)
        workers = list((root / 'workers').glob('*-complete.json')) if (root / 'workers').exists() else []
        if len(workers) == 8 or stopped(root, deadline):
            write(root / 'HOLD', dict(finished=time.time(), automatic_extension=False,
                                     remaining_episodes=len(frozen['evaluation_jobs']) - len(records) - len(errors)))
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['plan', 'pair', 'collect', 'eval'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--host')
    parser.add_argument('--pair', type=int)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--adopt', type=int)
    args = parser.parse_args()
    root = Path(args.root)
    if args.mode == 'plan':
        print(plan(root))
    elif args.mode == 'pair':
        pair_worker(root, args.host, args.pair, args.adopt)
    elif args.mode == 'collect':
        collect(root)
    else:
        eval_worker(root, args.host, args.gpu, time.time() + 3600)
