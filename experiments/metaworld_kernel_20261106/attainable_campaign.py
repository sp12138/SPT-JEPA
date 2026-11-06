"""Own-world continuation:fit attainable goal forecasts,never action labels."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import time

from experiments.metaworld_contact_20261106.launch import environment,PYTHON,wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command,TASKS
from experiments.metaworld_kernel_20261106.campaign import save,digest


def prepare(root,source,parent):
    root.mkdir(parents=True,exist_ok=False);plan=json.loads((parent/'training-plan.json').read_text())
    launches={host:json.loads((parent/f'launch-{host}.json').read_text())for host in['dev13','dev15']};recipes=[]
    for r in plan['recipes']:
        slot=r['gpu']%4
        arm=['span-incremental','zero-field','stencil-field','span-free'][slot]
        head=slot in[0,3];steps=[2,500,2000,5000]if head else[2,500,2500,10000]
        world=next(x for x in plan['recipes']if x['seed']==r['seed']and x['arm']==('incremental-point'if slot==0 else'free-actuation'))if head else None
        run=root/'training'/f'{arm}-seed{r["seed"]}';run.mkdir(parents=True)
        model=dict(parent=world['run'],feature_seed=r['seed'],boundary=True,window=2)if head else dict(
            feature_seed=r['seed'],channels=32,width=128,blocks=6,variance='point',passive='zero'if slot==1 else'stencil',actuation_history=True,boundary=True,window=2)
        cfg=dict(output=str(run),data=plan['data'],dataset_kind='attainable_pairs'if head else'density_transitions',
            model_class='attainable_forecast'if head else'predictive_density',feature_cache=r['feature_cache'],
            seed=r['seed'],batch_size=512 if head else 256,workers=2,cpu_threads=2,max_steps=steps[-1],
            checkpoint_every=500,retain_steps=steps,precision='bf16-mixed',max_seconds=20*3600,
            record_initial_weights=True,record_runtime_source=True,model=model)
        save(run/'config.json',cfg);owner=next(x for x in launches[r['host']]['workers']if x['gpu']==r['gpu'])
        recipes.append(dict(arm=arm,kind='head'if head else'world',seed=r['seed'],host=r['host'],gpu=r['gpu'],run=str(run),
            parent=world['run']if head else None,resource_parent=r['run'],stages=steps,join_pid=owner['pid'],config_sha256=digest(run/'config.json')))
        save(run/'ORIGIN.json',dict(external_pretraining=False,reconstruction_or_action_supervised_ancestry=False,
            parent=world['run']if head else None,parent_updates=10000 if head else 0,
            new_updates=steps[-1],new_batch_size=cfg['batch_size'],total_training_windows=5120000 if head else 2560000,
            not_equal_image_FLOPs=True,no_reconstruction=True,no_action_targets_or_references=True,
            current_executed_input_present=not head,only_past_own_control_input=head,
            target='Observed NEXT random feature;current executed control is an INPUT only to world-model training',
            weights='Own strict same-data world and new4Dcoordinate head;entire parent cost counts'if head else'ALL random,0parent learned tensors'))
    save(root/'training-plan.json',dict(source=str(source),parent=str(parent),data=plan['data'],manifest=plan['manifest'],recipes=recipes,
        stages=[2,500,2000,2500,5000,10000],objectives=['projection'],deadline_unix=time.time()+24*3600,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rationale='Test mandatory control fields and control-span goal forecasts without action targets.'
                  'Zero-field delta=J(x)u;stencil delta=J(x)u+K(x)(u-u_previous),no state-only drift.'
                  'Free next-feature goal prediction has large components outside the learned control span.'
                  'Test a forecast parameterized through its OWN learned forward model,using only future-observation targets.'
                  'Frozen world prevents the goal learner from changing physics to fit its proposal.No action pseudolabels.'))


def work(root,source,host,gpu):
    plan=json.loads((root/'training-plan.json').read_text());r=next(x for x in plan['recipes']if x['host']==host and x['gpu']==gpu)
    run=Path(r['run']);resource=Path(r['resource_parent']);deadline=plan['deadline_unix']
    try:
        while not(resource/'audit-10000.json').exists():
            assert not(resource/'execution-error.json').exists(),'Resource parent failed before immutable10k world'
            assert time.time()<deadline;time.sleep(5)
        assert json.loads((resource/'audit-10000.json').read_text())['passed']
        if r['kind']=='head':
            parent=Path(r['parent'])
            while not(parent/'audit-10000.json').exists():
                assert time.time()<deadline;time.sleep(5)
            assert json.loads((parent/'audit-10000.json').read_text())['passed']
        assert digest(run/'config.json')==r['config_sha256'];env=environment(source,gpu)
        save(run/'OVERLAP.json',dict(parent_frozen_at10000=True,resource_parent=str(resource),
            own_learning_tensor_copies=True,parent_sgd_complete=True,overlap_may_include_frozen_evaluation=True))
        for step in r['stages']:
            assert not(root/'STOP').exists()and not(run/'STOP').exists()
            command([PYTHON,'-B','-u',str(source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'),
                '--config',str(run/'config.json'),'--stop-step',str(step)]+(['--resume']if step>2 else[]),run/f'train-{step}.log',env,source,deadline)
            auditor='audit_attainable.py'if r['kind']=='head'else'audit.py'
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106'/auditor),
                '--run',str(run),'--step',str(step),'--output',str(run/f'audit-{step}.json')],run/f'audit-{step}.log',env,source,deadline)
            assert json.loads((run/f'audit-{step}.json').read_text())['passed']
            archive=run/f'archive-step{step:07d}';archive.mkdir();shutil.copy2(run/'last.ckpt',archive/'native.ckpt')
            save(archive/'manifest.json',dict(step=step,native_sha256=digest(archive/'native.ckpt')))
            final=step==r['stages'][-1]
            seeds=[42]if step==2 else[42,1042]if step==500 else list(range(42,52))+list(range(1042,1052))if final else[42,43,1042,1043]
            output=run/f'eval-projection-step{step:07d}'
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/evaluate.py'),'--run',str(run),
                '--step',str(step),'--manifest',plan['manifest'],'--output',str(output),'--budget','20'if step==2 else'500',
                '--tasks']+(['reach-v3']if step==2 else TASKS)+['--seeds']+list(map(str,seeds)),run/f'eval-{step}.log',env,source,deadline)
            assert json.loads((output/'summary.json').read_text())['passed'];save(run/'progress.json',dict(step=step,phase='stage_complete'))
        save(run/'complete.json',dict(passed=True,step=r['stages'][-1],kind=r['kind'],parent_updates=10000 if r['kind']=='head'else 0,
            no_action_or_reconstruction_targets=True))
    except BaseException as error:
        save(run/'execution-error.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True)
    for key in['root','source']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parent',type=Path);p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.source,a.parent)
    else:work(a.root,a.source,a.host,a.gpu)
