"""Finite paired density-ratio control tests,joining the first kernel workers."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import time

from experiments.metaworld_contact_20261106.launch import environment,PYTHON,wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command,TASKS
from experiments.metaworld_kernel_20261106.campaign import save,digest


ARMS=[('free-observation','free',False),('free-actuation','free',True),
      ('incremental-point','velocity',True),('transport-point','transport',True)]


def prepare(root,source,parent):
    root.mkdir(parents=True,exist_ok=False)
    old=json.loads((parent/'training-plan.json').read_text())
    launches={host:json.loads((parent/f'training-launch-{host}.json').read_text())for host in['dev13','dev15']}
    recipes=[]
    for r in old['recipes']:
        arm,passive,history=ARMS[r['gpu']%4]
        run=root/'training'/f'{arm}-seed{r["seed"]}';run.mkdir(parents=True)
        cache=Path(f'/tmp/prescottsun-kernel-features-seed{r["seed"]}')
        cfg=dict(output=str(run),data=old['data'],dataset_kind='density_transitions',model_class='predictive_density',
            feature_cache=str(cache),seed=r['seed'],batch_size=256,workers=2,cpu_threads=2,max_steps=10000,
            checkpoint_every=500,retain_steps=[2,500,2500,10000],precision='bf16-mixed',max_seconds=20*3600,
            record_initial_weights=True,record_runtime_source=True,
            model=dict(feature_seed=r['seed'],channels=32,width=128,blocks=6,variance='point',passive=passive,actuation_history=history,boundary=True,window=2))
        save(run/'config.json',cfg)
        join=next(x for x in launches[r['host']]['workers']if x['gpu']==r['gpu'])
        recipes.append(dict(arm=arm,seed=r['seed'],host=r['host'],gpu=r['gpu'],run=str(run),
            config_sha256=digest(run/'config.json'),join_pid=join['pid'],join_run=r['run'],feature_cache=str(cache)))
        save(run/'ORIGIN.json',dict(all_weights_random=True,old_weights_loaded=False,
            only_reused_object='Frozen RANDOM observation coordinate cache;0pretraining updates',
            no_reconstruction=True,no_action_targets_or_references=True,only_executed_current_control_input=True,
            targets='One NEXT observation queried with known-control,marginal,and future-goal information'))
    save(root/'training-plan.json',dict(source=str(source),parent=str(parent),data=old['data'],manifest=old['manifest'],
        recipes=recipes,stages=[2,500,2500,10000],objectives=['projection','marginal'],deadline_unix=time.time()+24*3600,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rationale='First-phase diagonal Gaussian suppresses useful motion and all manipulations fail at2500.'
                  'Test absolute versus incremental control effects and transported observed motion.'
                  'Past OWN executed control preserves force/holding context;never an action target.'
                  'Goal versus marginal prediction tests conditional-progress information.'
                  'Uncorrected density ratio omits behaviorprior,so is not the primary controller.'
                  'No RMS goal gate;3future-observation query scores per sample.No oldNNweights.'))


def work(root,source,host,gpu):
    plan=json.loads((root/'training-plan.json').read_text());r=next(x for x in plan['recipes']if x['host']==host and x['gpu']==gpu)
    run=Path(r['run']);deadline=plan['deadline_unix']
    try:
        process=Path(f'/proc/{r["join_pid"]}/cmdline')
        while process.exists():
            raw=process.read_bytes()
            if b'metaworld_kernel_20261106/campaign.py'not in raw:break
            assert time.time()<deadline;time.sleep(5)
        assert (Path(r['join_run'])/'complete.json').exists(),'First-phase worker did not complete normally;inspect retained error'
        assert (Path(r['feature_cache'])/'READY.json').exists()
        assert digest(run/'config.json')==r['config_sha256']
        env=environment(source,gpu)
        for step in plan['stages']:
            assert not(root/'STOP').exists()and not(run/'STOP').exists()
            wait_gpu_release(gpu)
            command([PYTHON,'-B','-u',str(source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'),
                '--config',str(run/'config.json'),'--stop-step',str(step)]+(['--resume']if step>2 else[]),
                run/f'train-{step}.log',env,source,deadline)
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/audit.py'),
                '--run',str(run),'--step',str(step),'--output',str(run/f'audit-{step}.json')],
                run/f'audit-{step}.log',env,source,deadline)
            assert json.loads((run/f'audit-{step}.json').read_text())['passed']
            archive=run/f'archive-step{step:07d}';archive.mkdir();shutil.copy2(run/'last.ckpt',archive/'native.ckpt')
            save(archive/'manifest.json',dict(step=step,native_sha256=digest(archive/'native.ckpt')))
            seeds=[42]if step==2 else[42,1042]if step==500 else[42,43,1042,1043]if step<10000 else list(range(42,52))+list(range(1042,1052))
            tasks=['reach-v3']if step==2 else TASKS
            for objective in plan['objectives']:
                output=run/f'eval-{objective}-step{step:07d}'
                command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/evaluate.py'),
                    '--run',str(run),'--step',str(step),'--manifest',plan['manifest'],'--output',str(output),
                    '--budget','20'if step==2 else'500','--objective',objective,'--tasks']+tasks+['--seeds']+list(map(str,seeds)),
                    run/f'eval-{objective}-{step}.log',env,source,deadline)
                assert json.loads((output/'summary.json').read_text())['passed']
            save(run/'progress.json',dict(step=step,phase='paired_stage_complete'))
        save(run/'complete.json',dict(passed=True,step=10000,objectives=plan['objectives']))
    except BaseException as error:
        save(run/'execution-error.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True)
    for key in['root','source']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parent',type=Path);p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.source,a.parent)
    else:work(a.root,a.source,a.host,a.gpu)
