"""Finite native training/audit/evaluation chain; no periodic monitor."""
import argparse
import datetime
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from common import B, PY, read, sha, source_check, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--id',required=True)
    p.add_argument('--phase',choices=['preflight','gate'],required=True);a=p.parse_args()
    root=Path(a.root);protocol=read(root/'protocol.json');row=next(r for r in protocol['runs'] if r['id']==a.id)
    source=protocol['source'];source_check(source);run=Path(row['run']);cfg=read(run/'config.json')
    assert sha(run/'config.json')==row['config_sha256'] and not(run/'STOP').exists()
    if a.phase=='gate':
        approval=read(root/'gate-approval.json');assert approval['passed'] and a.id in approval['run_ids']
        assert read(root/'audits'/f'{a.id}-step2/audit.json')['passed']
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i',str(row['gpu'])],text=True)
    assert int(memory.strip())==0,memory
    env=dict(os.environ,**protocol['environment'],PYTHONPATH=source['pythonpath'],
             CUDA_VISIBLE_DEVICES=str(row['gpu']),MUJOCO_EGL_DEVICE_ID=str(row['gpu']),PYTHONDONTWRITEBYTECODE='1')
    target=2 if a.phase=='preflight' else 600
    record=root/'worker-records'/f'{a.id}-{a.phase}.json';assert not record.exists()
    write(record.with_name(record.stem+'-start.json'),dict(pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          gpu=row['gpu'],phase=a.phase,monitor=False))
    results=[];started=time.monotonic()
    def execute(name,cmd,timeout):
        log=root/'logs'/f'{a.id}-{a.phase}-{name}.log';log.parent.mkdir(exist_ok=True)
        with log.open('x') as f:
            q=subprocess.Popen(cmd,cwd=source['source'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            write(root/'processes'/f'{a.id}-{a.phase}-{name}.json',dict(pid=q.pid,command=cmd,host=row['host'],gpu=row['gpu']))
            try:code=q.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(q.pid,signal.SIGTERM)
                try:q.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(q.pid,signal.SIGKILL);q.wait()
                code=124
        results.append(dict(phase=name,returncode=code,log=str(log)))
        assert code==0,str(log)
    ok=False;error=None
    try:
        cmd=[PY,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(run/'config.json'),'--stop-step',str(target)]
        if a.phase=='gate':cmd.append('--resume')
        execute('train',cmd,1200 if target==2 else 3600)
        status=read(run/'status.json');assert status['state']=='finished' and status['step']==target
        dest=run/'full-archives'/f'step{target:07d}.ckpt';dest.parent.mkdir(exist_ok=True);assert not dest.exists()
        shutil.copy2(run/'last.ckpt',dest);assert sha(dest)==sha(run/'last.ckpt')
        helper=Path(source['source'])/'experiments/unified_three_20261101/audit.py'
        cmd=[PY,'-B','-u',str(helper),'--root',str(root),'--id',a.id,'--step',str(target)]
        if target==600:cmd.append('--diagnostics')
        execute('audit',cmd,1200)
        if target==600:
            checkpoint=run/'model-step0000600';out=root/'evaluations'/a.id
            if row['task']=='pusht':
                cmd=[PY,'-B','-u','-m','as_jepa_tools.planning','--checkpoint',str(checkpoint),'--output',str(out),
                     '--upstream',str(B/'sources/stable-worldmodel'),'--data-root',str(B/'data/hardtasks-v1/planning'),
                     '--seed','42','--episodes','50','--split','heldout','--test-episodes',str(Path(cfg['data'])/'test_episodes.npy')]
            else:
                cmd=[PY,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',str(Path(cfg['data'])/'manifests/validation-strict.json'),
                     '--checkpoint',str(checkpoint),'--data',cfg['data'],'--output',str(out)]
            execute('planning',cmd,1200)
        ok=True
    except Exception as exc:error=repr(exc)
    write(record,dict(passed=ok,run_id=a.id,phase=a.phase,step=target,results=results,error=error,
          seconds=time.monotonic()-started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),continuous_monitor=False))
    if not ok:raise RuntimeError(error)
    print('Finite worker complete',a.id,a.phase,flush=True)


if __name__=='__main__':main()
