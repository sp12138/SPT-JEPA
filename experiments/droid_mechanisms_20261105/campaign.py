"""Finite paired screen and evidence-gated continuation using native workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from common import PYTHON, arm_root, bootstrap, environment, run_dir

ARMS = ['tf','tf-idm','f8','f8-idm','variance-fixed','variance-prior','variance-joint']
SEEDS = [1,1000]
HELPER = Path(__file__).resolve().parent


def claim(root, name):
    path = Path(root)/'claims'/name
    try:
        path.mkdir(parents=True)
        return True
    except FileExistsError:
        return False


def execute(root, command, gpus, output, deadline):
    dump = bootstrap(arm_root(root,'tf'))
    output = Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    with open(output.with_suffix('.log'),'ab') as log:
        process = subprocess.Popen(command,env=environment(gpus),stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT)
        dump(output.with_suffix('.launch.json'),dict(pid=process.pid,command=command,gpus=gpus,start=time.time()))
        try:
            code=process.wait(timeout=max(1,deadline-time.time()))
        except subprocess.TimeoutExpired:
            process.terminate()
            try: process.wait(timeout=60)
            except subprocess.TimeoutExpired: process.kill();process.wait()
            raise TimeoutError('Finite campaign deadline reached')
        if code:
            raise RuntimeError(f'Worker exit {code}: {output.with_suffix(".log")}')


def archive(root,arm,seed,stage):
    dump=bootstrap(arm_root(root,arm))
    source=run_dir(root,arm,seed)
    ready=json.loads((source/f'{stage}-ready.json').read_text())
    target=Path(root)/'checkpoints'/f'{arm}-seed{seed}'/stage
    target.mkdir(parents=True,exist_ok=False)
    for name,path in [('model.ckpt',Path(ready['checkpoint'])),('full.ckpt',source/'lightning_resume/last.ckpt')]:
        shutil.copyfile(path,target/name)
    ready.update(checkpoint=str(target/'model.ckpt'),full_checkpoint=str(target/'full.ckpt'))
    dump(target/'ready.json',ready)
    return target/'ready.json'


def train_job(a,arm,seed,stage):
    root=Path(a.root);dump=bootstrap(arm_root(root,arm))
    name=f'{arm}-seed{seed}'
    out=root/'workers'/name
    if stage=='screen':
        stages=['health','screen']
    else:
        stages=['long']
    for part in stages:
        command=[PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2',str(HELPER/'train.py'),
                 '--root',str(root),'--host',a.host,'--arm',arm,'--seed',str(seed),'--stage',part]
        execute(root,command,a.gpus,out/part,a.deadline)
        if part=='health':
            audit=[PYTHON,'-u',str(HELPER/'audit.py'),'restore','--root',str(root),'--arm',arm,'--seed',str(seed),'--stage','health']
            execute(root,audit,a.gpus[:1],out/'health-audit',a.deadline)
            proof=json.loads((run_dir(root,arm,seed)/'health-audit.json').read_text())
            assert proof['passed'] and proof['step']==600 and all(v>0 for v in proof['gradients'].values())
        ready=archive(root,arm,seed,part)
        dump(root/'stages'/f'{name}-{part}.json',dict(passed=True,arm=arm,seed=seed,stage=part,ready=str(ready),host=a.host,gpus=a.gpus))
    for pseed in [1,2,3]:
        for clip in range(16):
            name=f'{stage}-{arm}-s{seed}-p{pseed}-c{clip:02d}'
            dump(root/'eval-jobs'/f'{name}.json',dict(name=name,stage=stage,arm=arm,seed=seed,planner_seed=pseed,clip=clip,ready=str(ready)))


def eval_loop(a,gpu):
    root=Path(a.root);dump=bootstrap(arm_root(root,'tf'))
    for path in sorted((root/'eval-jobs').glob('*.json')):
        job=json.loads(path.read_text())
        if not claim(root,'eval-'+job['name']):continue
        output=root/'evaluations'/job['name']
        cmd=[PYTHON,'-u',str(HELPER/'evaluate.py'),'--root',str(root),'--arm',job['arm'],'--seed',str(job['seed']),
             '--planner-seed',str(job['planner_seed']),'--clip',str(job['clip']),'--ready',job['ready'],'--output',str(output)]
        try:
            execute(root,cmd,[gpu],root/'eval-logs'/job['name'],a.deadline)
        except Exception:
            dump(output/'error.json',dict(passed=False,job=job,traceback=traceback.format_exc()))


def actor(a):
    root=Path(a.root);dump=bootstrap(arm_root(root,'tf'))
    while time.time()<a.deadline:
        if (root/'HOLD').exists() or (root/'STOP').exists():break
        selection=json.loads((root/'selection.json').read_text()) if (root/'selection.json').exists() else None
        todo=[(arm,seed,'screen') for seed in SEEDS for arm in ARMS]
        if selection:
            todo += [(arm,seed,'long') for arm in selection['continue_arms'] for seed in SEEDS]
        work=False
        for arm,seed,stage in todo:
            if not claim(root,f'train-{stage}-{arm}-seed{seed}'):continue
            work=True
            try:train_job(a,arm,seed,stage)
            except Exception:
                dump(root/'errors'/f'{stage}-{arm}-seed{seed}.json',dict(passed=False,arm=arm,seed=seed,stage=stage,traceback=traceback.format_exc()))
            break
        if work:continue
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda gpu:eval_loop(a,gpu),a.gpus))
        time.sleep(10)
    dump(root/'actors'/f'{a.host}-g{a.gpus[0]}-exit.json',dict(exited=True,time=time.time()))


def select(rows):
    import numpy as np
    arrays={}
    for arm in ARMS:
        arrays[arm]={}
        for seed in SEEDS:
            subset=[r for r in rows if r['arm']==arm and r['model_seed']==seed]
            assert len(subset)==48
            arrays[arm][seed]=np.asarray([[np.mean([r['metrics'][key] for r in subset if r['clip']==c])
                                        for key in ['frechet/skill','ate/end_distance_xyz']] for c in range(16)])
    rng=np.random.default_rng(20261105)
    samples=rng.integers(0,16,(10000,16))
    candidates=[]
    summary={}
    for arm in ARMS:
        scores=np.stack([arrays[arm][s] for s in SEEDS])
        summary[arm]=dict(fidelity=float(scores[:,:,0].mean()),xyz_ate=float(scores[:,:,1].mean()))
        if arm=='tf':continue
        differences=np.stack([arrays[arm][s][:,0]-arrays['tf'][s][:,0] for s in SEEDS])
        by_case=differences.mean(0)
        ci=np.quantile(by_case[samples].mean(1),[.025,.975])
        qualifies=bool((differences.mean(1)>=.10).all() and scores[:,:,0].mean()>0 and ci[0]>0
                       and scores[:,:,1].mean()<=1.05*np.stack([arrays['tf'][s] for s in SEEDS])[:,:,1].mean())
        summary[arm].update(gain_by_seed=differences.mean(1).tolist(),gain_ci=ci.tolist(),qualifies=qualifies)
        if qualifies:candidates.append((float(by_case.mean()),arm))
    chosen=[arm for gain,arm in sorted(candidates,reverse=True)[:2]]
    return dict(continue_arms=['tf']+chosen if chosen else [],selected=chosen,summary=summary,
                qualification='Exploratory, conditional on two learners and 16 reused physical clips; not a SOTA or independent confirmation claim.')


def collect(a):
    root=Path(a.root);dump=bootstrap(arm_root(root,'tf'))
    while time.time()<a.deadline:
        errors=list((root/'errors').glob('*.json'))+list((root/'evaluations').glob('*/error.json'))
        stages=list((root/'stages').glob('*-screen.json'))
        records=[]
        for p in (root/'evaluations').glob('*/result.json'):
            row=json.loads(p.read_text())
            if row.get('passed'):records.append(row)
        screen=[r for r in records if r['stage']=='screen']
        dump(root/'results/summary.json',dict(status='ACTIVE',screen_trainings=len(stages),screen_endpoints=len(screen),errors=len(errors),updated=time.time()))
        if errors:
            dump(root/'HOLD',dict(status='EXECUTION_ERROR',files=[str(p) for p in errors],time=time.time()))
            return
        if len(screen)==672 and not (root/'selection.json').exists():
            selection=select(screen);dump(root/'selection.json',selection)
            if not selection['continue_arms']:
                dump(root/'results/summary.json',dict(status='NO_PROMOTION',screen_trainings=14,
                     screen_endpoints=672,errors=0,selection=selection,updated=time.time()))
                dump(root/'HOLD',dict(status='NO_PROMOTION',time=time.time()));return
        if (root/'selection.json').exists():
            selection=json.loads((root/'selection.json').read_text())
            expected=len(selection['continue_arms'])*2*48
            long=[r for r in records if r['stage']=='long']
            if expected and len(long)==expected:
                means={}
                for arm in selection['continue_arms']:
                    subset=[r for r in long if r['arm']==arm]
                    means[arm]={key:sum(r['metrics'][key] for r in subset)/len(subset) for key in ['frechet/skill','ate/end_distance_xyz']}
                dump(root/'results/long-summary.json',dict(passed=True,means=means,endpoints=len(long)))
                dump(root/'results/summary.json',dict(status='COMPLETE',screen_trainings=14,screen_endpoints=672,
                     long_endpoints=len(long),means=means,errors=0,updated=time.time()))
                dump(root/'HOLD',dict(status='COMPLETE',time=time.time()));return
        time.sleep(30)
    dump(root/'HOLD',dict(status='DEADLINE',time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['actor','collect'])
    p.add_argument('--root',required=True);p.add_argument('--host');p.add_argument('--gpus',type=int,nargs=2)
    p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args();actor(a) if a.mode=='actor' else collect(a)
