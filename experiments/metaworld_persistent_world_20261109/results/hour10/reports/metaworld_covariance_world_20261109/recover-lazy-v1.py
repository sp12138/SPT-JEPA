"""Warm the lazy linalg dispatcher serially; inherit four completed native2s.

No NN/loss/data/optimizer/source changes. Original failures remain. Two lowrank
workers failed on the FIRST DP forward, before SGD, and resume full step0.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,now,environment
from experiments.metaworld_covariance_world_20261109 import campaign as core

PARENT=BASE/'runs/metaworld_covariance_world_20261109'
ROOT=PARENT/'recovery-lazy-v1'
core.ROOT=ROOT


def prepare():
    p=read(PARENT/'plan.json');ROOT.mkdir(exist_ok=False)
    for e in p['evaluations']:e['output']=str(ROOT/'evaluation'/e['name'])
    p.update(recovery_utc=now(),helper_sha256=sha(__file__),original_errors_retained=True,
        correction='SeriallinalgdispatchwarmupbeforeDataParallel,noNN/loss/data/SGDchange. '
        'Inheritfouractual2updateCPUrestores/physical20stepchecks. TwolowrankworkersrestoreoriginalFULLstep0,0SGDrepeated.',
        original_error_records=len(list(PARENT.glob('errors/*.json'))))
    save(ROOT/'plan.json',p)


def native(config,stop,resume):
    import torch
    from experiments.metaworld_covariance_world_20261109 import train as original
    cfg=read(config);devices=list(range(torch.cuda.device_count()));assert len(devices)==4
    with torch.random.fork_rng(devices=devices):
        before=[torch.cuda.get_rng_state(d).clone() for d in devices];cpu=torch.get_rng_state().clone()
        for d in devices:
            with torch.cuda.device(d):
                value=torch.eye(4,device='cuda',requires_grad=True);chol=torch.linalg.cholesky(value);solved=torch.cholesky_solve(value,chol);(chol.square().sum()+solved.square().sum()).backward();torch.cuda.synchronize(d)
        assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(s,torch.cuda.get_rng_state(d)) for s,d in zip(before,devices))
    save(ROOT/'warmups'/f'{Path(cfg["output"]).name}-to{stop}.json',dict(passed=True,utc=now(),serial_devices=devices,
        exact_CPU_CUDA_RNG_unchanged=True,model_updates=0,helper_sha256=sha(__file__)))
    original.train(cfg,stop,resume)


def train_work(name):
    p=read(ROOT/'plan.json');j=next(j for j in p['jobs'] if j['name']==name);s=Path(p['source']);out=Path(j['output'])
    env=environment(s,j['gpus'][0])|{'CUDA_VISIBLE_DEVICES':','.join(map(str,j['gpus']))}
    try:
        previous=None
        for stop in core.STAGES:
            if previous is not None:core.wait(ROOT/f'COLLECTED-{previous:07d}.json',p)
            inherited=stop==2 and (out/'AUDIT-0000002.json').exists() and (out/'PREFLIGHT.json').exists()
            if not inherited:
                checkpoint=out/f'checkpoint-{(previous or 0):07d}.pt';assert checkpoint.exists()
                args=[PYTHON,str(Path(__file__).resolve()),'native','--config',j['config'],'--stop',str(stop),'--resume',str(checkpoint)]
                core.command(args,out/f'recovery-lazy-train-{stop}.log',env,s,p['deadline']);core.cpu_audit(j,stop,p)
            if stop==2:
                if not inherited:
                    core.command([PYTHON,'-m',core.MODULE+'.evaluate','--config',j['config'],'--step','2',
                        '--manifest',BASE/'runs/metaworld_transition_only_20261109/manifests/technical.json','--output',out/'technical','--budget','20'],
                        out/'technical.log',environment(s,j['gpus'][0]),s,p['deadline'])
                    assert read(out/'technical/metrics.json')['passed']
                    save(out/'PREFLIGHT.json',dict(passed=True,utc=now(),actual_new_updates=2,training_gpus=4,technical_steps=20,linalg_lazy_recovery=True))
                while not (ROOT/'TRAINING_APPROVED.json').exists():
                    core.gate();assert time.time()<p['deadline'] and not list(ROOT.glob('errors/*.json'));time.sleep(3)
            save(out/f'READY-{stop:07d}.json',dict(passed=True,utc=now(),step=stop,recovery=str(ROOT),inherited_native2=inherited))
            previous=stop
        save(out/'DONE.json',dict(passed=True,utc=now(),new_world_updates=2000,recovery=str(ROOT)))
    except BaseException:
        save(ROOT/'errors'/(name+'.json'),dict(utc=now(),traceback=traceback.format_exc()));raise


def launch(host):
    p=read(ROOT/'plan.json');s=Path(p['source']);workers=[]
    for j in p['jobs']:
        if j['host']!=host:continue
        with (ROOT/(j['name']+'-worker.log')).open('x') as f:
            child=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'train','--name',j['name']],cwd=s,
                env=environment(s,j['gpus'][0])|{'CUDA_VISIBLE_DEVICES':''},stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(kind='training',name=j['name'],gpus=j['gpus'],pid=child.pid))
    for gpu in range(8):
        with (ROOT/f'eval-{host}-{gpu}.log').open('x') as f:
            child=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'eval','--host',host,'--gpu',str(gpu)],cwd=s,
                env=environment(s,gpu)|{'CUDA_VISIBLE_DEVICES':''},stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(kind='frozen_eval_waiter',gpu=gpu,pid=child.pid))
    save(ROOT/f'launch-{host}.json',dict(utc=now(),host=host,workers=workers));print(host,workers,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','native','train','launch','eval','collect']);a.add_argument('--host');a.add_argument('--gpu',type=int);a.add_argument('--name');a.add_argument('--config',type=Path);a.add_argument('--stop',type=int);a.add_argument('--resume',type=Path);args=a.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='native':native(args.config,args.stop,args.resume)
    elif args.mode=='train':train_work(args.name)
    elif args.mode=='launch':launch(args.host)
    elif args.mode=='eval':core.evaluate_work(args.host,args.gpu)
    else:core.collect()
