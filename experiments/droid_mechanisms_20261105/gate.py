"""Disposable three-update native train, restore/backward and public planner."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

from common import PYTHON, arm_root, bootstrap, environment


def run(a):
    root = arm_root(a.root,a.arm); dump = bootstrap(root)
    helper = Path(__file__).resolve().parent
    target = root/a.tag/a.host/f'temporal-forecast-seed{a.seed}'
    target.mkdir(parents=True, exist_ok=True)
    evidence = target/a.attempt
    evidence.mkdir(exist_ok=True)
    commands = [
        [PYTHON,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2',str(helper/'train.py'),
         '--root',a.root,'--host',a.host,'--arm',a.arm,'--seed',str(a.seed),'--stage','preflight','--tag',a.tag],
        [PYTHON,'-u',str(helper/'audit.py'),'restore','--root',a.root,'--arm',a.arm,'--seed',str(a.seed),
         '--stage','preflight','--run',str(target)],
        [PYTHON,'-u',str(helper/'evaluate.py'),'--root',a.root,'--arm',a.arm,'--seed',str(a.seed),
         '--planner-seed','1','--clip','0','--ready',str(target/'preflight-ready.json'),'--output',str(target/'planning')]]
    try:
        for index,cmd in enumerate(commands):
            if index == 0 and a.reuse_preflight:
                prior = json.loads((target/'preflight-ready.json').read_text())
                assert prior['passed'] and prior['step']==3 and prior['arm']==a.arm
                continue
            with open(evidence/f'gate-{index}.log','ab') as log:
                subprocess.run(cmd,env=environment(a.gpus if index==0 else a.gpus[:1]),
                               stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True,timeout=2400)
        dump(evidence/'gate-complete.json',dict(passed=True,arm=a.arm,seed=a.seed,host=a.host,
             actual_updates=3,full_restore=True,real_gpu_backward=True,full_public_planner=True,finished=time.time()))
    except Exception:
        dump(evidence/'gate-error.json',dict(passed=False,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','host','arm']:p.add_argument('--'+key,required=True)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--gpus',type=int,nargs=2,required=True)
    p.add_argument('--tag',default='preflight-v1');p.add_argument('--attempt',default='gate-v2')
    p.add_argument('--reuse-preflight',action='store_true');run(p.parse_args())
