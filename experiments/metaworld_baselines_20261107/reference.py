"""Fresh full-50 evaluation of two existing scratch LeWM parents; no SGD."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

BASE = Path('/efs/prescottsun/SPT-JEPA')
ROOT = BASE / 'runs/metaworld_baselines_20261107'
SOURCE = BASE / 'reports/metaworld_input_inverse_20261107/source-faf40148'
PYTHON = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'
HOSTS = ['dev13', 'dev15', 'dev6', 'dev6-2c']


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def prepare(helper):
    from experiments.metaworld_input_inverse_20261107.campaign import digest
    ROOT.mkdir(parents=True, exist_ok=False)
    improve = BASE / 'runs/metaworld_improve_20261104'
    recipes = json.loads((improve / 'plan.json').read_text())['recipes']
    parents = [p for p in recipes if p['arm'] == 'native-control']
    assert len(parents) == 2
    original = BASE / 'reports/metaworld_improve_20261104/runtime-7b78f5d'
    assert digest(original/'as_jepa_lewm/native_lewm.py') == digest(SOURCE/'as_jepa_lewm/native_lewm.py')
    bank = BASE / 'runs/metaworld_input_inverse_20261107_v1/followup/cases/manifest.json'
    manifest = json.loads(bank.read_text())
    assert len(manifest['rows']) == 1000
    assert len({r['task'] for r in manifest['rows']}) == 50
    jobs = []
    for parent in parents:
        checkpoint = Path(parent['run']) / 'model-step0016480'
        assert (checkpoint/'weights.pt').is_file()
        for shard in range(16):
            host = HOSTS[shard//4]
            gpu = (parent['seed']-1)*4 + shard%4
            name = f"native20-seed{parent['seed']}-shard{shard:02}"
            selected = dict(manifest, rows=manifest['rows'][shard::16])
            casefile = ROOT/'reference/manifests'/(name+'.json')
            save(casefile, selected)
            technical = ROOT/'reference/manifests'/(name+'-technical.json')
            save(technical, dict(selected, rows=selected['rows'][:1]))
            jobs.append(dict(name=name, host=host, gpu=gpu, seed=parent['seed'],
                checkpoint=str(checkpoint), weights_sha256=digest(checkpoint/'weights.pt'),
                manifest=str(casefile), technical=str(technical), episodes=len(selected['rows'])))
    save(ROOT/'reference/plan.json', dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        helper=str(helper), source=str(SOURCE), parent_models=2, jobs=jobs, total_episodes=2000,
        source_manifest=str(bank), source_manifest_sha256=digest(bank), model_updates=0,
        deadline=time.time()+8*3600, scope='Two historical 20epoch scratch LeWM parents; not new baseline training or 32 independent learners.'))


def worker(name):
    from experiments.metaworld_input_inverse_20261107.campaign import environment, execute, digest
    plan = json.loads((ROOT/'reference/plan.json').read_text())
    job = next(j for j in plan['jobs'] if j['name'] == name)
    out = ROOT/'reference/workers'/name
    out.mkdir(parents=True, exist_ok=False)
    env = environment(SOURCE, job['gpu'])
    try:
        assert digest(Path(job['checkpoint'])/'weights.pt') == job['weights_sha256']
        for stage, budget in [('technical', 20), ('formal', 500)]:
            cmd = [PYTHON, '-m', 'experiments.metaworld_eval_20261103.evaluate',
                '--checkpoint', job['checkpoint'], '--data', str(BASE/'data/metaworld-lerobot-b46a1c18-packed-v1'),
                '--manifest', job['technical' if stage == 'technical' else 'manifest'],
                '--output', str(out/stage), '--budget', str(budget)]
            execute(cmd, out/(stage+'.log'), env, SOURCE, plan['deadline'], cap=8*3600)
            report = json.loads((out/stage/'metrics.json').read_text())
            assert report['passed'] and report['model_state_before'] == report['model_state_after']
            save(out/(stage+'-passed.json'), dict(passed=True, episodes=len(report['episodes']), model_updates=0))
        save(out/'complete.json', dict(passed=True, job=job, utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    except BaseException:
        save(out/'error.json', dict(passed=False, traceback=traceback.format_exc()))
        raise


def launch(host):
    plan = json.loads((ROOT/'reference/plan.json').read_text())
    helper = Path(plan['helper'])
    records = []
    for job in plan['jobs']:
        if job['host'] != host:
            continue
        env = os.environ.copy()
        env['PYTHONPATH'] = str(helper)+':'+str(SOURCE)
        log = ROOT/'reference'/(job['name']+'-worker.log')
        with log.open('x') as stream:
            proc = subprocess.Popen([PYTHON, str(helper/'experiments/metaworld_baselines_20261107/reference.py'),
                '--mode', 'work', '--name', job['name']], env=env, cwd=SOURCE,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        records.append(dict(name=job['name'], gpu=job['gpu'], pid=proc.pid, log=str(log)))
    save(ROOT/'reference'/('launch-'+host+'.json'), dict(host=host, workers=records))
    print(json.dumps(records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['prepare', 'launch', 'work'], required=True)
    parser.add_argument('--helper', type=Path)
    parser.add_argument('--host')
    parser.add_argument('--name')
    args = parser.parse_args()
    if args.mode == 'prepare': prepare(args.helper)
    elif args.mode == 'launch': launch(args.host)
    else: worker(args.name)
