"""Finite frozen-model diagnostic launch; no SGD or subsequent auto-search."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess


def main(root, previous, host):
    from experiments.metaworld_improve_20261104.campaign import environments
    from experiments.metaworld_eval_20261103.worker import free_gpu, save, read
    base = Path('/efs/prescottsun/SPT-JEPA')
    runtime = base/'reports/metaworld_improve_20261104/runtime-7b78f5d'
    helper = base/'reports/metaworld_improve_20261104/helper-698ae41'
    python = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'
    assert read(previous/'results/summary.json')['passed']
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for recipe in read(previous/'plan.json')['recipes']:
        if recipe['host'] != host:
            continue
        gpu = recipe['gpu']
        free_gpu(gpu)
        work = root/'diagnostics'/recipe['name']
        work.mkdir(parents=True, exist_ok=False)
        cmd = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=60', '14400',
               python, '-B', '-u', str(helper/'experiments/metaworld_eval_20261103/diagnose.py'),
               '--checkpoint', str(Path(recipe['run'])/'model-step0016480'),
               '--data', read(previous/'dispatch.json')['data'], '--output', str(work/'output')]
        env = environments(previous, runtime, helper, gpu)
        with (work/'diagnose.log').open('x') as log:
            p = subprocess.Popen(cmd, env=env, cwd=runtime, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        jobs.append(dict(name=recipe['name'], gpu=gpu, pid=p.pid, command=cmd, work=str(work)))
    save(root/f'diagnostic-launch-{host}.json', dict(host=host, jobs=jobs,
         created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         scope='16 frozen20epoch models,50 tasks/model,atmost32 common heldoutwindows/task. '
               'RGB/action only,no modelupdates or simulator. Not online control performance.'))
    print(json.dumps(dict(host=host, jobs=len(jobs), pids=[r['pid'] for r in jobs])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--previous', required=True)
    p.add_argument('--host', choices=['dev13', 'dev15'], required=True)
    a = p.parse_args()
    main(Path(a.root), Path(a.previous), a.host)
