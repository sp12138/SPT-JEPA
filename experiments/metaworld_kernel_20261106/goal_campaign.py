"""Finite observation-path generator experiments;no inverse-action training."""
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
    root.mkdir(parents=True,exist_ok=False);old=json.loads((parent/'training-plan.json').read_text());recipes=[];stages=[2,500,2500,10000]
    worlds={r['seed']:r for r in old['recipes']if r['arm']=='stencil'}
    for r in old['recipes']:
        mode,horizon=[('point1',1),('jump5',5),('compose5',5),('compose15',15)][r['gpu']%4]
        run=root/'training'/f'{mode}-seed{r["seed"]}';run.mkdir(parents=True);world=Path(worlds[r['seed']]['run'])
        worldcfg=json.loads((world/'config.json').read_text());assert worldcfg['model_class']=='causal_transport'
        assert json.loads((world/'ORIGIN.json').read_text())['all_weights_random']
        cfg=dict(output=str(run),data=old['data'],native_rgb_cache='/tmp/prescottsun-native-rgb128-20261106-v1',
            dataset_kind='goal_rgb_paths',model_class='goal_transport',seed=r['seed'],batch_size=128,workers=2,cpu_threads=2,
            max_steps=10000,max_seconds=24*3600,checkpoint_every=500,retain_steps=stages,precision='bf16-mixed',
            record_initial_weights=True,record_runtime_source=True,model=dict(mode=mode,horizon=horizon,width=128,blocks=4,window=2,
                world_config=worldcfg['model'],world_checkpoint=str(world/'model-step0020000/weights.pt'),world_audit=str(world/'audit-20000.json')))
        save(run/'config.json',cfg)
        save(run/'ORIGIN.json',dict(generator_all_random=True,frozen_world=str(world),frozen_world_origin='Own original scratch forward-only20k,all cost counts',
            total_pipeline_budget='20000world updates plus10000new observation-generator updates;sharedworld across4matched generator variants',
            external_pretraining=False,inverse_action_ancestry=False,no_action_labels_or_future_action_references=True,
            generator_dataset='Only raw RGB ordered observations;does not open command arrays',
            target='Strict interior future RGB;point1/jump5/compose5/compose15 test local fit versus path consistency',
            evaluator='Observation-likelihood projection and replan5.Same allowed5-native observation cadence,more decisions/compute than oldreplan15.' ))
        owner=next(x for x in json.loads((parent/f'launch-{r["host"]}.json').read_text())['workers']if x['gpu']==r['gpu'])
        recipes.append(dict(arm=mode,seed=r['seed'],host=r['host'],gpu=r['gpu'],run=str(run),world_run=str(world),
            resource_pid=owner['pid'],control_run=r['run'],control_arm=r['arm'],config_sha256=digest(run/'config.json')))
    save(root/'training-plan.json',dict(source=str(source),parent=str(parent),data=old['data'],manifest=old['manifest'],recipes=recipes,stages=stages,
        objectives=['projection'],deadline_unix=time.time()+36*3600,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Same original data/split/cases,RGB only,sample-level future observation learning.No inverse/action references.'
              'Matched frozen original models evaluated with same new planner first;costs and changed decision period explicit.No performance/novelty claim.'))


def evaluation(source,run,step,manifest,output,env,deadline,technical=False,seeds=None):
    command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/native_evaluate.py'),
        '--run',str(run),'--step',str(step),'--manifest',manifest,'--output',str(output),'--planner','path','--replan-native','5',
        '--budget','20'if technical else'500','--tasks']+(['reach-v3']if technical else TASKS)+['--seeds']+
        list(map(str,[42]if technical else seeds)),output.with_suffix('.log'),env,source,deadline)
    assert json.loads((output/'summary.json').read_text())['passed']


def work(root,source,host,gpu):
    plan=json.loads((root/'training-plan.json').read_text());r=next(x for x in plan['recipes']if x['host']==host and x['gpu']==gpu)
    run=Path(r['run']);deadline=plan['deadline_unix'];env=environment(source,gpu)
    try:
        process=Path(f'/proc/{r["resource_pid"]}/cmdline')
        while process.exists():
            command_line=process.read_bytes()
            if b'transport_campaign.py'not in command_line and b'goal_campaign.py'not in command_line:break
            assert time.time()<deadline;time.sleep(10)
        assert json.loads((Path(r['world_run'])/'audit-20000.json').read_text())['passed']
        wait_gpu_release(gpu)
        control=root/'controls'/f'{r["control_arm"]}-seed{r["seed"]}';control.mkdir(parents=True)
        evaluation(source,Path(r['control_run']),20000,plan['manifest'],control/'entry20',env,deadline,technical=True)
        evaluation(source,Path(r['control_run']),20000,plan['manifest'],control/'pilot',env,deadline,seeds=[42,1042])
        cfg=json.loads((run/'config.json').read_text());assert digest(run/'config.json')==r['config_sha256']
        save(run/'WORLD_BOUND.json',dict(checkpoint=cfg['model']['world_checkpoint'],sha256=digest(Path(cfg['model']['world_checkpoint'])),
            world_native_updates=20000,inverse_action_ancestry=False))
        for step in plan['stages']:
            assert not(root/'STOP').exists()and not(run/'STOP').exists();wait_gpu_release(gpu)
            command([PYTHON,'-B','-u',str(source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'),
                '--config',str(run/'config.json'),'--stop-step',str(step)]+(['--resume']if step>2 else[]),run/f'train-{step}.log',env,source,deadline)
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/native_audit.py'),
                '--run',str(run),'--step',str(step),'--output',str(run/f'audit-{step}.json')],run/f'audit-{step}.log',env,source,deadline)
            assert json.loads((run/f'audit-{step}.json').read_text())['passed']
            archive=run/f'archive-step{step:07d}';archive.mkdir();shutil.copy2(run/'last.ckpt',archive/'native.ckpt')
            save(archive/'manifest.json',dict(step=step,native_sha256=digest(archive/'native.ckpt')))
            seeds=[42,1042]if step==500 else[42,43,1042,1043]if step<10000 else list(range(42,52))+list(range(1042,1052))
            evaluation(source,run,step,plan['manifest'],run/f'eval-projection-step{step:07d}',env,deadline,technical=step==2,seeds=seeds)
            save(run/'progress.json',dict(step=step,phase='stage_complete'))
        save(run/'complete.json',dict(passed=True,step=10000,forward_observation_targets_only=True))
    except BaseException as error:save(run/'execution-error.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True)
    for key in ['root','source']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parent',type=Path);p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.source,a.parent)
    else:work(a.root,a.source,a.host,a.gpu)
