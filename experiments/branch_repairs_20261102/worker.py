"""One finite native train/audit/planning chain on one explicitly assignedGPU."""
import argparse
import datetime
import os
from pathlib import Path
import signal
import subprocess
import time

from common import B,PY,read,write,sha,source_check


def main(root,ident,gpu,phase):
    root=Path(root);protocol=read(root/'protocol.json');source=protocol['source'];source_check(source)
    row=next(r for r in protocol['runs'] if r['id']==ident);run=Path(row['run']);cfg=read(run/'config.json')
    assert sha(run/'config.json')==row['config_sha256'] and not(root/'HOLD').exists() and not(run/'STOP').exists()
    assert sha(Path(cfg['initial_encoder_checkpoint'])/'weights.pt')==cfg['frozen_encoder_sha256']
    for path,digest in protocol['manifest_sha256'].items():assert sha(path)==digest
    step=row['preflight_step'] if phase=='preflight' else row['stop_step']
    if phase=='gate':
        approval=read(root/'gate-approval.json');assert approval['passed'] and ident in approval['run_ids']
        assert read(root/'audits'/f'{ident}-step{row["preflight_step"]}/audit.json')['passed']
    memory=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i',str(gpu)],text=True).strip())
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader','-i',str(gpu)],text=True).strip()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    assert memory<=1 and uuid not in {x.strip() for x in active},(memory,uuid,active)
    env=dict(os.environ,**protocol['environment'],PYTHONPATH=source['pythonpath'],CUDA_VISIBLE_DEVICES=str(gpu),
             MUJOCO_EGL_DEVICE_ID=str(gpu),PYTHONDONTWRITEBYTECODE='1')
    Path(env['TMPDIR']).mkdir(exist_ok=True,parents=True)
    write(root/'worker-records'/f'{ident}-{phase}-start.json',dict(pid=os.getpid(),gpu=gpu,host=row['host'],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),initial_memory_mib=memory))
    results=[];started=time.monotonic();ok=False;error=None
    def execute(name,cmd,timeout):
        log=root/'logs'/f'{ident}-{phase}-{name}.log';log.parent.mkdir(exist_ok=True)
        with log.open('x') as f:
            child=subprocess.Popen(cmd,cwd=source['source'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            write(root/'processes'/f'{ident}-{phase}-{name}.json',dict(pid=child.pid,command=cmd,gpu=gpu,host=row['host'],cwd=source['source']))
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=124
        results.append(dict(phase=name,returncode=code,log=str(log)));assert code==0,str(log)
    def evaluate(case,limit=None):
        out=root/'evaluations'/f'{ident}-step{step}-{case}'
        out.mkdir(parents=True,exist_ok=False)
        if cfg['model_class']=='goal_query':env['ASJEPA_QUERY_AUDIT']=str(out/'query-objective-proof.json')
        if row['task']=='pusht':
            cmd=[PY,'-B','-u','-m','as_jepa_tools.planning','--checkpoint',str(run/f'model-step{step:07d}'),
                 '--output',str(out),'--upstream',str(B/'sources/stable-worldmodel'),
                 '--data-root',str(B/'data/hardtasks-v1/planning'),'--seed','42' if case=='primary' else '1042',
                 '--episodes',str(limit or 50),'--split','heldout','--test-episodes',str(Path(cfg['data'])/'test_episodes.npy')]
        else:
            manifest=protocol['manifests'][row['task']][case]
            cmd=[PY,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',str(manifest),
                 '--checkpoint',str(run/f'model-step{step:07d}'),'--data',cfg['data'],'--output',str(out)]
            if limit:cmd+=['--limit',str(limit)]
        execute('planning-'+case,cmd,1200 if limit else 3600)
        assert (out/'metrics.json').exists()
        if cfg['model_class']=='goal_query':
            proof=read(out/'query-objective-proof.json')
            assert proof['objective']=='GoalQueryObjective' and proof['finite'] and proof['cost_shape'][-1]==300
    try:
        cmd=[PY,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(run/'config.json'),'--stop-step',str(step)]
        if phase=='gate' or row['start_step']>0:cmd.append('--resume')
        execute('train',cmd,1800 if phase=='preflight' else 21600)
        status=read(run/'status.json');assert status['state']=='finished' and status['step']==step
        archive=run/'full-archives'/f'step{step:07d}.ckpt';archive.parent.mkdir(exist_ok=True)
        # Native saves use tmp+replace;this immutable inode survives laterlast updates.
        os.link(run/'last.ckpt',archive);assert sha(archive)==sha(run/'last.ckpt')
        audit=[PY,'-B','-u',str(Path(source['source'])/'experiments/branch_trials_20261102/audit.py'),
               '--root',str(root),'--id',ident,'--step',str(step)]
        if phase=='gate':audit.append('--diagnostics')
        execute('audit',audit,1800)
        evaluate('primary',2 if phase=='preflight' else None)
        if phase=='gate':evaluate('fresh')
        ok=True
    except Exception as exc:error=repr(exc)
    write(root/'worker-records'/f'{ident}-{phase}.json',dict(passed=ok,id=ident,host=row['host'],gpu=gpu,step=step,results=results,error=error,
          seconds=time.monotonic()-started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),continuous_monitor=False))
    if not ok:raise RuntimeError(error)
    print(dict(passed=True,id=ident,phase=phase))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--id',required=True);p.add_argument('--gpu',required=True,type=int);p.add_argument('--phase',choices=['preflight','gate'],required=True);a=p.parse_args();main(a.root,a.id,a.gpu,a.phase)
