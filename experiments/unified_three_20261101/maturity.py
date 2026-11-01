"""Evidence-reviewed1200/2000 native stages, fixed models and finite exit."""
import argparse
import datetime
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from common import PY, read, sha, source_check, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--id',required=True)
    p.add_argument('--approval',required=True);a=p.parse_args()
    root=Path(a.root);protocol=read(root/'protocol.json');row=next(x for x in protocol['runs'] if x['id']==a.id)
    approval=read(a.approval);assert approval['passed'] and a.id in approval['run_ids'] and approval['stages']==[1200,2000]
    if 'maturity_worker_sha256' in approval:assert sha(__file__)==approval['maturity_worker_sha256']
    source=protocol['source'];helper=approval['helper_source'];source_check(source);source_check(helper)
    assert read(root/'worker-records'/f'{a.id}-gate.json')['passed']
    run=Path(row['run']);assert sha(run/'config.json')==row['config_sha256'] and not(run/'STOP').exists()
    audit=read(root/'audits'/f'{a.id}-step600/audit.json');assert audit['passed'] and sha(run/'last.ckpt')==audit['full_checkpoint_sha256']
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i',str(row['gpu'])],text=True)
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader','-i',str(row['gpu'])],text=True).strip()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    assert int(free.strip())<=1 and uuid not in {x.strip() for x in processes}
    env=dict(os.environ,**protocol['environment'],PYTHONPATH=source['pythonpath']+':'+helper['source'],
             CUDA_VISIBLE_DEVICES=str(row['gpu']),MUJOCO_EGL_DEVICE_ID=str(row['gpu']),PYTHONDONTWRITEBYTECODE='1')
    record=root/'worker-records'/f'{a.id}-maturity2000.json';assert not record.exists()
    write(record.with_name(record.stem+'-start.json'),dict(pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          approval=str(a.approval),approval_sha256=sha(a.approval),gpu=row['gpu'],source=source,helper_source=helper,monitor=False))
    results=[];started=time.monotonic()
    def execute(name,command,timeout):
        log=root/'logs'/f'{a.id}-maturity-{name}.log'
        with log.open('x') as f:
            child=subprocess.Popen(command,env=env,cwd=source['source'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            write(root/'processes'/f'{a.id}-maturity-{name}.json',dict(pid=child.pid,command=command,source=source,helper_source=helper))
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=124
        results.append(dict(phase=name,returncode=code,log=str(log)));assert code==0,str(log)
    error=None
    try:
        for step in [1200,2000]:
            execute(f'train{step}',[PY,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(run/'config.json'),
                                    '--resume','--stop-step',str(step)],3600)
            status=read(run/'status.json');assert status['state']=='finished' and status['step']==step
            archive=run/'full-archives'/f'step{step:07d}.ckpt';assert not archive.exists()
            shutil.copy2(run/'last.ckpt',archive);assert sha(archive)==sha(run/'last.ckpt')
            portable=run/'portable-archives'/f'step{step:07d}';portable.parent.mkdir(exist_ok=True)
            shutil.copytree(run/f'model-step{step:07d}',portable)
            execute(f'audit{step}',[PY,'-B','-u',str(Path(source['source'])/'experiments/unified_three_20261101/audit.py'),
                 '--root',str(root),'--id',a.id,'--step',str(step),'--diagnostics'],1200)
            for case in (['primary'] if step==1200 else ['primary','fresh']):
                for mode in ['l2','ngc']:
                    execute(f'planning{step}-{case}-{mode}',[PY,'-B','-u',str(Path(helper['source'])/'experiments/unified_three_20261101/graph_evaluate.py'),
                         '--root',str(root),'--id',a.id,'--step',str(step),'--mode',mode,'--case-set',case],1800)
    except Exception as e:error=repr(e)
    write(record,dict(passed=error is None,run_id=a.id,results=results,error=error,stages=[1200,2000],
          seconds=time.monotonic()-started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),continuous_monitor=False))
    if error:raise RuntimeError(error)
    print('Bounded maturity complete',a.id,flush=True)


if __name__=='__main__':main()
