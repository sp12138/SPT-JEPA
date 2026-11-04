"""One finite proposal-mechanism pilot on16 GPUs;no SGD/automatic extension."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from experiments.metaworld_eval_20261103.worker import read, save, digest, free_gpu, command


BASE = Path('/efs/prescottsun/SPT-JEPA')
RUNTIME = BASE/'reports/metaworld_improve_20261104/runtime-7b78f5d'
OLD_HELPER = BASE/'reports/metaworld_improve_20261104/helper-698ae41'
PYTHON = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def environments(root, source, gpu):
    from experiments.metaworld_improve_20261104.campaign import environments as old
    previous = Path(read(root/'plan.json')['previous'])
    env = old(previous, RUNTIME, OLD_HELPER, gpu, True)
    env['PYTHONPATH'] = str(source)+':'+env['PYTHONPATH']
    return env


def prepare(root, previous, source):
    assert read(previous/'results/summary.json')['passed']
    old = read(previous/'plan.json')['recipes']
    dispatch = read(previous/'dispatch.json')
    recipes = []
    arms = ['tf-control', 'native-control', 'high-noise', 'innovation']
    for host, seed in [('dev13',1), ('dev15',2)]:
        for gpu in range(8):
            arm, proposal = arms[gpu % 4], 'held' if gpu < 4 else 'library'
            parent = next(r for r in old if r['arm']==arm and r['seed']==seed)
            checkpoint = Path(parent['run'])/'model-step0016480'
            assert read(Path(parent['run'])/'audit-step16480.json')['passed']
            name = f'{arm}-{proposal}-seed{seed}'
            work = root/'pilot'/name
            work.mkdir(parents=True, exist_ok=False)
            recipes.append(dict(name=name, arm=arm, proposal=proposal, host=host, gpu=gpu,
                seed=seed, work=str(work), checkpoint=str(checkpoint),
                weights_sha256=digest(checkpoint/'weights.pt')))
    save(root/'plan.json', dict(created_utc=now(), previous=str(previous), source=str(source), recipes=recipes,
        tasks=['bin-picking-v3','door-open-v3','drawer-open-v3','peg-insert-side-v3',
               'pick-place-v3','push-v3','reach-v3','reach-wall-v3'], banks=['primary','fresh'],limit=2,
        data=dispatch['data'], case_manifest=dispatch['case_manifest'],
        deadline_unix=dispatch['deadline_unix'], deadline_utc=dispatch['deadline_utc'],
        endpoints=256, episodes=512, model_updates=0,
        scope='New user-authorized finite pilot.4500 forecast candidates/plan,H6,f5,execute3,500 any-success. '
              'Paired TF/native/highnoise/innovation20epoch checkpoints;no retraining. '
              'Archived standard-CEM cases reused as controls. Library adds factual train-memory access. '
              'Preserve independent primary/fresh and originalparent identities;no algorithm novelty claim.'))
    print(json.dumps(dict(prepared=True, recipes=16, endpoints=256, episodes=512)), flush=True)


def launch(root, source, host):
    rows = []
    plan = read(root/'plan.json')
    assert time.time() < plan['deadline_unix']
    for r in plan['recipes']:
        if r['host'] != host:
            continue
        work = Path(r['work'])
        cmd = ['/usr/bin/timeout','--signal=TERM','--kill-after=60',str(int(plan['deadline_unix']-time.time())),
               PYTHON,'-B','-u',str(source/'experiments/metaworld_proposal_20261104/campaign.py'),
               '--mode','work','--root',str(root),'--source',str(source),'--host',host,'--gpu',str(r['gpu'])]
        env = os.environ.copy()
        env.update(PYTHONPATH=str(source)+':'+str(OLD_HELPER)+':'+str(RUNTIME), PYTHONDONTWRITEBYTECODE='1')
        with (work/'worker.log').open('x') as stream:
            p = subprocess.Popen(cmd, env=env, cwd=source, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        rows.append(dict(name=r['name'],gpu=r['gpu'],pid=p.pid,command=cmd))
    save(root/f'launch-{host}.json',dict(host=host,jobs=rows,created_utc=now()))
    print(json.dumps(dict(host=host,pids=[r['pid'] for r in rows])),flush=True)


def work(root, source, host, gpu):
    import psutil
    plan = read(root/'plan.json')
    r = next(r for r in plan['recipes'] if r['host']==host and r['gpu']==gpu)
    work = Path(r['work'])
    try:
        diagnostic = next(j for j in read(root/f'diagnostic-launch-{host}.json')['jobs'] if j['gpu']==gpu)
        while time.time() < plan['deadline_unix']:
            try:
                p = psutil.Process(diagnostic['pid'])
                alive = p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
                owned = str(Path(diagnostic['work'])/'output') in p.cmdline()
            except psutil.NoSuchProcess:
                alive = owned = False
            if not alive or not owned:
                break
            time.sleep(10)
        diagnostic_metric = Path(diagnostic['work'])/'output/metrics.json'
        if diagnostic_metric.exists():
            d = read(diagnostic_metric)
            assert d['passed'] and d['model_state_before']==d['model_state_after'] and d['model_updates']==0
        else:
            save(work/'DIAGNOSTIC_MISSING.json',dict(passed=False,work=diagnostic['work'],
                 scope='Retain failed separate diagnostic;does not imply control-method failure.'))
        free_gpu(gpu)
        assert digest(Path(r['checkpoint'])/'weights.pt') == r['weights_sha256']
        env = environments(root,source,gpu)
        def evaluate(destination,task,bank,limit,budget):
            args = [PYTHON,'-B','-u',str(source/'experiments/metaworld_proposal_20261104/evaluate.py'),
                    '--checkpoint',r['checkpoint'],'--data',plan['data'],'--manifest',plan['case_manifest'],
                    '--output',str(destination/'output'),'--task',task,'--bank',bank,'--limit',str(limit),
                    '--budget',str(budget),'--proposal',r['proposal'],'--library',str(work/'library')]
            destination.mkdir()
            free_gpu(gpu)
            command(args,destination/'evaluate.log',env,RUNTIME,plan['deadline_unix'],cap=7200)
            report = read(destination/'output/metrics.json')
            assert report['passed'] and report['model_state_before']==report['model_state_after']
            assert report['model_updates']==0 and len(report['episodes'])==limit
            assert read(destination/'output/physical-audit.json')['physical_paths']==2
            save(destination/'complete.json',dict(passed=True,finished_utc=now(),
                 metrics_sha256=digest(destination/'output/metrics.json')))
        evaluate(work/'preflight','reach-wall-v3','primary',1,20)
        save(work/'PREFLIGHT_PASSED.json',dict(passed=True,finished_utc=now(),
             scope='Actual20step context/actionhistory interface plus2 real30step alternative paths;not scored task success.'))
        for task in plan['tasks']:
            for bank in plan['banks']:
                if (root/'STOP').exists():
                    raise RuntimeError('User stop marker')
                evaluate(work/f'{task}-{bank}',task,bank,plan['limit'],500)
        save(work/'complete.json',dict(passed=True,finished_utc=now(),endpoints=16,episodes=32))
    except Exception:
        save(work/'failed.json',dict(passed=False,finished_utc=now(),traceback=traceback.format_exc()))
        raise


def finish(root):
    plan = read(root/'plan.json')
    while time.time() < plan['deadline_unix']:
        if all(any((Path(r['work'])/x).exists() for x in ['complete.json','failed.json']) for r in plan['recipes']):
            break
        time.sleep(30)
    rows = []
    for r in plan['recipes']:
        work = Path(r['work'])
        for task in plan['tasks']:
            for bank in plan['banks']:
                p = work/f'{task}-{bank}'
                if not (p/'complete.json').exists():
                    continue
                d = read(p/'output/metrics.json')
                assert digest(p/'output/metrics.json') == read(p/'complete.json')['metrics_sha256']
                for e in d['episodes']:
                    rows.append(dict(arm=r['arm'],proposal=r['proposal'],parent_seed=r['seed'],
                                     bank=bank,task=task,case_seed=e['seed'],success=e['success']))
    previous = Path(plan['previous'])
    baselines = []
    summary = read(previous/'results/summary.json')
    for group in summary['groups']:
        if group['step'] != 16480:
            continue
        wanted = [42,43] if group['bank']=='primary' else [1042,1043]
        for e in group['observations']:
            if e['seed'] in wanted:
                baselines.append(dict(arm=group['arm'],proposal='archived-standard-CEM',
                    parent_seed=e['train_seed'],bank=group['bank'],task=group['task'],
                    case_seed=e['seed'],success=e['success']))
    passed = len(rows)==512 and all((Path(r['work'])/'complete.json').exists() for r in plan['recipes'])
    save(root/'results/summary.json',dict(passed=passed,finished_utc=now(),episodes=rows,
         baseline_cases=baselines,model_updates=0,
         scope='Proposal-mechanism pilot only;not equal train-memory/planner information. '
               'No automatic long training/new method round. Compare within frozen checkpoint,case,parent and bank.'))
    save(root/'HOLD',dict(finished_utc=now(),passed=passed))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode',choices=['prepare','launch','work','finish'],required=True)
    p.add_argument('--root',required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--previous')
    p.add_argument('--host',choices=['dev13','dev15'])
    p.add_argument('--gpu',type=int)
    a = p.parse_args()
    root, source = Path(a.root),Path(a.source)
    if a.mode=='prepare':prepare(root,Path(a.previous),source)
    elif a.mode=='launch':launch(root,source,a.host)
    elif a.mode=='work':work(root,source,a.host,a.gpu)
    else:finish(root)
