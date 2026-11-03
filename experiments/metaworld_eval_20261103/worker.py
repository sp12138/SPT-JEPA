"""Finite native training stages or shared MPC jobs; no research monitor/retries."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:json.dump(value, handle, indent=2)


def read(path):return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def timestamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def free_gpu(gpu):
    output = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'], text=True)
    memory = {int(s.split(',')[0]):int(s.split(',')[1]) for s in output.splitlines()}
    assert memory[gpu] <= 1, ('Assigned GPU is occupied; do not affect other jobs',gpu,output)
    return output


def command(cmd, log, env, cwd, deadline, cap=172800):
    remaining = min(cap, int(deadline-time.time()))
    if remaining <= 0:raise TimeoutError('Finite campaign deadline reached')
    with Path(log).open('x') as stream:
        subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=60',str(remaining),*cmd],
                       env=env,cwd=cwd,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                       check=True,timeout=remaining+120)


def archive(run, step):
    archive = run / f'archive-step{step:07d}'; archive.mkdir(exist_ok=False)
    # Installed checkpoint backend replaces last.ckpt atomically. Hard links
    # preserve each full native payload without duplicating 0.2--0.6GB files.
    for source,name in [(run/'last.ckpt','native.ckpt'),(run/'stage-complete.json','stage-complete.json'),
                        (run/'status.json','status.json'),(run/f'audit-step{step}.json','audit.json')]:
        if name=='native.ckpt':os.link(source,archive/name)
        else:shutil.copy2(source,archive/name)
    save(archive/'manifest.json',dict(step=step,created_utc=timestamp(),
         files={p.name:digest(p) for p in archive.iterdir() if p.is_file()}))


def main(args):
    root=Path(args.root); source=Path(args.source); helper=Path(args.helper)
    dispatch=read(root/'dispatch.json');deadline=dispatch['deadline_unix']
    env=os.environ.copy()
    paths=[helper,source,Path(args.base)/'environments/completion-overlay',Path(args.base)/'environments/overlay',
           Path(args.base)/'runs/mechanism_cubeaudit_20261030/source-v1/runtime',Path(args.base)/'sources/stable-worldmodel']
    if args.mode=='evaluate':paths.insert(0,Path(args.base)/'environments/metaworld-sanity-20261103/lib/python3.12/site-packages')
    env.update(PYTHONPATH=':'.join(map(str,paths)),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=str(args.gpu),
               HF_HUB_OFFLINE='1',WANDB_MODE='disabled',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    if args.mode=='evaluate':
        env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(args.gpu),
                   LD_LIBRARY_PATH=args.base+'/environments/egl-lib/usr/lib/x86_64-linux-gnu:'+env.get('LD_LIBRARY_PATH',''),
                   __EGL_VENDOR_LIBRARY_FILENAMES=args.base+'/environments/egl-lib/10_nvidia.json')
    gate=root/'TRAINING_APPROVED.json'
    while not gate.exists():
        if time.time()>=deadline or (root/'HOLD').exists() or (root/'STOP').exists():return
        time.sleep(15)
    assert read(gate)['passed']
    if args.mode=='train':
        run=Path(args.run);cfg=read(run/'config.json')
        assert read(gate)['config_sha256'][run.name]==digest(run/'config.json')
        assert read(run/'audit-step2.json')['passed']
        archive(run,2)
        for step in [600,2472,8240]:
            if (root/'STOP').exists() or (run/'STOP').exists():raise RuntimeError('User stop marker')
            free_gpu(args.gpu)
            command([args.python,'-B','-u',str(source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'),
                     '--config',str(run/'config.json'),'--stop-step',str(step),'--resume'],
                     run/f'train-stage{step}.log',env,source,deadline)
            command([args.python,'-B','-u',str(helper/'experiments/metaworld_eval_20261103/audit.py'),
                     '--run',str(run),'--step',str(step),'--output',str(run/f'audit-step{step}.json')],
                     run/f'audit-step{step}.log',env,source,deadline,cap=1800)
            assert read(run/f'audit-step{step}.json')['passed'];archive(run,step)
        save(run/'complete.json',dict(passed=True,finished_utc=timestamp(),step=8240))
        return
    jobs=read(root/'evaluation-plan.json')['jobs']
    while time.time()<deadline:
        if (root/'STOP').exists() or (root/'HOLD').exists():break
        disposed=0; ready=[]
        for job in jobs:
            work=Path(job['work'])
            if (work/'complete.json').exists() or (work/'failed.json').exists():disposed+=1;continue
            if (work/'claim').exists():continue
            audit=Path(job['run'])/f"audit-step{job['step']}.json"
            if audit.exists() and read(audit)['passed']:ready.append(job)
        if disposed==len(jobs):break
        if not ready:time.sleep(20);continue
        # Existing JSON order alternates models/tasks; all methods receive the
        # same frozen cases, budgets and seed banks. No outcome-based selection.
        for job in ready:
            work=Path(job['work']);work.mkdir(parents=True,exist_ok=True)
            try:(work/'claim').mkdir()
            except FileExistsError:continue
            save(work/'claim/owner.json',dict(host=os.uname().nodename,gpu=args.gpu,pid=os.getpid(),time=timestamp()))
            try:
                free_gpu(args.gpu)
                cmd=[args.python,'-B','-u',str(helper/'experiments/metaworld_eval_20261103/evaluate.py'),
                     '--checkpoint',job['checkpoint'],'--data',cfg_data(root),'--manifest',dispatch['case_manifest'],
                     '--output',str(work/'output'),'--task',job['task'],'--bank',job['bank'],'--limit',str(job['limit']),
                     '--budget','500','--success-mode','any']
                command(cmd,work/'evaluate.log',env,source,deadline,cap=21600)
                report=read(work/'output/metrics.json');assert report['passed'] and len(report['episodes'])==job['limit']
                save(work/'complete.json',dict(passed=True,finished_utc=timestamp(),metrics_sha256=digest(work/'output/metrics.json')))
            except Exception:
                save(work/'failed.json',dict(passed=False,finished_utc=timestamp(),traceback=traceback.format_exc()))
            break
    save(root/f'evaluation-worker-{args.host}-g{args.gpu}-complete.json',dict(finished_utc=timestamp(),deadline_reached=time.time()>=deadline))


def cfg_data(root):return read(root/'dispatch.json')['data']


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['train','evaluate'],required=True)
    parser.add_argument('--root',required=True);parser.add_argument('--source',required=True);parser.add_argument('--helper',required=True)
    parser.add_argument('--base',default='/efs/prescottsun/SPT-JEPA');parser.add_argument('--python',default='/efs/prescottsun/project_jepa/envs/lewm/bin/python')
    parser.add_argument('--gpu',type=int,required=True);parser.add_argument('--host',required=True);parser.add_argument('--run')
    args=parser.parse_args()
    try:main(args)
    except Exception:
        if args.run:
            save(Path(args.run)/'worker-failed.json',dict(passed=False,finished_utc=timestamp(),traceback=traceback.format_exc()))
        raise
