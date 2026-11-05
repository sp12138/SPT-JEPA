"""Explicit finite launch/ownership records, restricted to dev13/dev15."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time


def wait_gpu_release(gpu, seconds=120):
    """Bounded driver/context teardown wait;never kill an unknown allocation."""
    from experiments.metaworld_eval_20261103.worker import free_gpu
    deadline = time.monotonic()+seconds
    while True:
        try:
            return free_gpu(gpu)
        except AssertionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)


BASE = Path('/efs/prescottsun/SPT-JEPA')
PYTHON = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def environment(source, gpu, evaluate=True):
    import sys
    old = BASE/'reports/metaworld_proposal_20261104/source-46e401b'
    helper = BASE/'reports/metaworld_improve_20261104/helper-698ae41'
    runtime = BASE/'reports/metaworld_improve_20261104/runtime-7b78f5d'
    sys.path[:0] = list(map(str, [old, helper, runtime]))
    from experiments.metaworld_proposal_20261104.campaign import environments
    env = environments(BASE/'runs/metaworld_proposal_20261104_v3', old, gpu)
    env['PYTHONPATH'] = str(source)+':'+env['PYTHONPATH']
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    return env


def prepare(root, source):
    root.mkdir(parents=True, exist_ok=False)
    original = json.loads((BASE/'runs/metaworld_proposal_20261104_v3/plan.json').read_text())
    tasks = ['bin-picking-v3', 'pick-place-v3', 'push-v3', 'drawer-open-v3',
             'door-open-v3', 'peg-insert-side-v3', 'reach-v3', 'reach-wall-v3']
    recipes = []
    for host, arm in [('dev13', 'tf-control'), ('dev15', 'native-control')]:
        old = next(r for r in original['recipes'] if r['arm'] == arm and r['seed'] == 1)
        for gpu, task in enumerate(tasks):
            name = f'{arm}-parent1-{task}'
            recipes.append(dict(name=name, arm=arm, host=host, gpu=gpu, task=task,
                                parent_seed=1, checkpoint=old['checkpoint'],
                                output=str(root/'diagnostics'/name)))
    save(root/'plan.json', dict(created_utc=now(), source=str(source), recipes=recipes,
         hosts=['dev13', 'dev15'], case_seeds=[42, 43, 1042, 1043],
         data=original['data'], manifest=original['case_manifest'],
         old_summary=str(BASE/'runs/metaworld_improve_20261104/results/summary.json'),
         policy_source=str(BASE/'sources/lerobot-mw-f5c9bc02'),
         deadline_unix=time.time()+12*3600, model_updates=0,
         scope='User-authorized new manipulation investigation. Known cases;experts/states diagnostic only;old campaigns remain closed.'))
    (root/'diagnostics').mkdir()


def launch(root, source, host, indices):
    assert host in ['dev13', 'dev15']
    plan = json.loads((root/'plan.json').read_text())
    records = []
    for r in plan['recipes']:
        if r['host'] != host or r['gpu'] not in indices:
            continue
        assert not Path(r['output']).exists(), r
        ownership = root/f"owner-{host}-gpu{r['gpu']}.json"
        assert not ownership.exists(), ownership
        env = environment(source, r['gpu'])
        command = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=30',
                   str(int(plan['deadline_unix']-time.time())), PYTHON, '-B', '-u',
                   str(source/'experiments/metaworld_contact_20261106/diagnose.py'),
                   '--plan', str(root/'plan.json'), '--name', r['name'], '--output', r['output']]
        with (root/f"{r['name']}.log").open('x') as stream:
            p = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(pid=p.pid, created_utc=now(), command=command, **r)
        save(ownership, record); records.append(record)
    print(json.dumps(dict(host=host, launched=[dict(pid=r['pid'], gpu=r['gpu']) for r in records])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['prepare', 'launch'], required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--host')
    p.add_argument('--gpus', type=int, nargs='+', default=list(range(8)))
    a = p.parse_args()
    if a.mode == 'prepare':
        prepare(a.root, a.source)
    else:
        launch(a.root, a.source, a.host, a.gpus)
