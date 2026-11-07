"""Resume only the three H-JEPA queue-device failures from their saved step3."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import traceback

from experiments.metaworld_baselines_20261107.campaign import (
    ROOT,BASE,PYTHON,read,save,sha,env,execute,stage,audit,evaluate,now)


def prepare(helper):
    plan=read(ROOT/'plan.json');records=[]
    assert sha(helper/'experiments/metaworld_baselines_20261107/models.py')==sha(
        Path(plan['helper'])/'experiments/metaworld_baselines_20261107/models.py')
    for job in plan['jobs']:
        if job['method']!='h-jepa' or not (Path(job['output'])/'error.json').exists():continue
        old=Path(job['output'])
        assert (old/'error.json').exists() and not (old/'checkpoint-0000100.pt').exists()
        log=(old/'train-0000100.log').read_text()
        assert 'No backend type associated with device type cpu' in log and '_sync_module_states' in log
        parent=old/'checkpoint-0000003.pt';assert parent.exists()
        assert read(old/'ready-0000003.json')['step']==3
        new=ROOT/'training-device-v4'/job['name'];new.mkdir(parents=True,exist_ok=False)
        cfg=read(job['config']);cfg['output']=str(new);save(new/'config.json',cfg)
        save(new/'FORK.json',dict(parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),
            inherited_step=3,inherited_samples=768,public_model_source_unchanged=True,
            reason='CPU restore then CUDA moves custom queue buffer before NCCL DDP construction; no learning change',
            original_error=str(old/'error.json'),original_error_sha256=sha(old/'error.json'),helper=str(helper)))
        inherited=read(old/'PREFLIGHT_PASSED.json')
        save(new/'PREFLIGHT_PASSED.json',dict(inherited,scope='Inherited original3update/CPUrestore/FP32/MPC gate, not rerun',original_output=str(old)))
        claim=read(ROOT/'claims-v3'/(job['name']+'.json'))
        records.append(dict(name=job['name'],host=claim['host'],pair=claim['pair'],output=str(new),
            original_output=str(old),parent_checkpoint=str(parent)))
        job.update(output=str(new),config=str(new/'config.json'),original_device_failure=str(old),device_helper=str(helper))
    assert records
    owner=read(ROOT/'collector-launch-v3.json');proc=Path(f'/proc/{owner["pid"]}/cmdline')
    if proc.exists():
        command=proc.read_bytes().replace(b'\0',b' ').decode()
        assert 'helper-panel-v3' in command and '--mode collect' in command
        os.kill(owner['pid'],signal.SIGTERM)
    save(ROOT/'collector-retired-v3.json',dict(owner,retired_utc=now(),scope='Metadata collector replacement only;13 healthy training parents/children untouched'))
    save(ROOT/'plan-v3-before-device-recovery.json',read(ROOT/'plan.json'))
    save(ROOT/'plan.json',plan)
    save(ROOT/'DEVICE_RECOVERY_v4.json',dict(passed=True,helper=str(helper),records=records,
        original_failures=len(records),source_failure='NCCL initial module-state broadcast after custom queue CPU restore',
        completed_sgd_repeated=0,unsaved_sgd_repeated=0,three_original3update_checkpoints_retained=True,
        other13_training_processes_untouched=True,utc=now()))


def work(name):
    plan=read(ROOT/'plan.json');repair=read(ROOT/'DEVICE_RECOVERY_v4.json');helper=Path(repair['helper'])
    job=next(j for j in plan['jobs'] if j['name']==name);record=next(r for r in repair['records'] if r['name']==name)
    out=Path(job['output']);gpus=[record['pair']*2,record['pair']*2+1]
    try:
        cmd=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
            str(helper/'experiments/metaworld_baselines_20261107/train_native.py'),'--config',job['config'],
            '--stop','100','--resume',record['parent_checkpoint'],'--allow-output-transfer']
        execute(cmd,out/'train-0000100.log',env(helper,gpus),helper,plan['deadline'])
        audit(job,100,helper,gpus[0],plan['deadline'])
        # Full native hierarchy's actual gradient planner, not its fine-level-only CEM.
        execute([PYTHON,str(helper/'experiments/metaworld_baselines_20261107/native_hjepa_eval.py'),
            '--config',job['config'],'--checkpoint',str(out/'model-step0000100'),
            '--manifest',str(ROOT/'manifests/technical-0.json'),'--output',str(out/'native-controller-entry'),
            '--budget','20'],out/'native-controller-entry.log',env(helper,[gpus[1]]),helper,plan['deadline'])
        for before,stop in [(100,2500),(2500,8240),(8240,20600),(20600,41200)]:
            stage(job,stop,helper,gpus,before,plan['deadline']);audit(job,stop,helper,gpus[0],plan['deadline'])
            execute([PYTHON,str(helper/'experiments/metaworld_baselines_20261107/diagnose.py'),
                '--config',job['config'],'--checkpoint',str(out/f'checkpoint-{stop:07d}.pt'),
                '--output',str(out/f'diagnostic-{stop:07d}.json')],out/f'diagnostic-{stop:07d}.log',
                env(helper,[gpus[1]]),helper,plan['deadline'])
            evaluate(job,stop,helper,gpus,plan['deadline'])
        save(out/'complete.json',dict(passed=True,step=41200,host=record['host'],gpus=gpus,utc=now()))
        from experiments.metaworld_baselines_20261107.campaign import work as next_job
        next_job(record['host'],record['pair'])
    except BaseException:
        save(out/'error.json',dict(passed=False,traceback=traceback.format_exc(),utc=now()));raise


def launch(host):
    r=read(ROOT/'DEVICE_RECOVERY_v4.json');helper=Path(r['helper']);records=[]
    for item in r['records']:
        if item['host']!=host:continue
        out=Path(item['output']);gpus=[item['pair']*2,item['pair']*2+1]
        with (out/'worker.log').open('x') as stream:
            p=subprocess.Popen([PYTHON,str(helper/'experiments/metaworld_baselines_20261107/recover_device.py'),
                '--mode','work','--name',item['name']],cwd=helper,env=env(helper,gpus),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        records.append(dict(item,pid=p.pid))
    save(ROOT/f'launch-device-v4-{host}.json',dict(host=host,workers=records,utc=now()));print(records)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','launch','work'],required=True)
    p.add_argument('--helper',type=Path);p.add_argument('--host');p.add_argument('--name');a=p.parse_args()
    if a.mode=='prepare':prepare(a.helper)
    elif a.mode=='launch':launch(a.host)
    else:work(a.name)
