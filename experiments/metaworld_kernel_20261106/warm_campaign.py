"""Own-forward initialization control,original corpus only."""
import argparse
import json
from pathlib import Path
import shutil
import time
from experiments.metaworld_contact_20261106.launch import environment,PYTHON,wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command
from experiments.metaworld_kernel_20261106.campaign import save,digest
from experiments.metaworld_kernel_20261106.goal_campaign import evaluation


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=False);old=json.loads((a.parent/'training-plan.json').read_text());recipes=[]
    for host,seed in [('dev13',1),('dev15',2)]:
        owners=json.loads((a.parent/f'launch-generator-v3-{host}.json').read_text())['workers']
        for gpu,mode in[(2,'point1'),(6,'compose5')]:
            origin=next(x for x in old['recipes']if x['seed']==seed and x['arm']==mode);parent=Path(origin['run']);run=a.root/'training'/f'{mode}-seed{seed}';run.mkdir(parents=True)
            cfg=json.loads((parent/'config.json').read_text());cfg['output']=str(run);cfg['model_class']='goal_transport_warm';cfg['model']['init_from_world']=True
            save(run/'config.json',cfg);save(run/'ORIGIN.json',dict(parent=str(parent),frozen_world=cfg['model']['world_checkpoint'],
                initialization='Ownsame-data scratchforward20k;allcostcounts.No inverse/external ancestry.',
                supervision='FutureRGBonly,generatordatasetneveropensactions',comparison='Matching oldcoldgenerator seed/mode/data/stages/planner;initializer includes parentgoalqueryaffine.'))
            owner=next(x for x in owners if x['gpu']==gpu)
            recipes.append(dict(arm=mode,seed=seed,host=host,gpu=gpu,run=str(run),resource_pid=owner['pid'],config_sha256=digest(run/'config.json')))
    save(a.root/'training-plan.json',dict(source=str(a.source),data=old['data'],manifest=old['manifest'],recipes=recipes,stages=[2,500,2500,10000],
        objectives=['projection'],deadline_unix=time.time()+24*3600,scope='Original-data initialization diagnostic;no extra data,action labels or performance claim.'))


def work(a):
    plan=json.loads((a.root/'training-plan.json').read_text());r=next(x for x in plan['recipes']if x['host']==a.host and x['gpu']==a.gpu)
    run=Path(r['run']);source=Path(plan['source']);deadline=plan['deadline_unix']
    try:
        owner=Path(f'/proc/{r["resource_pid"]}/cmdline')
        while owner.exists()and b'goal_campaign.py'in owner.read_bytes():assert time.time()<deadline;time.sleep(10)
        wait_gpu_release(a.gpu);env=environment(source,a.gpu);assert digest(run/'config.json')==r['config_sha256']
        for step in plan['stages']:
            assert not(a.root/'STOP').exists()and not(run/'STOP').exists();wait_gpu_release(a.gpu)
            command([PYTHON,'-B','-u',str(source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'),'--config',str(run/'config.json'),
                '--stop-step',str(step)]+(['--resume']if step>2 else[]),run/f'train-{step}.log',env,source,deadline)
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/native_audit.py'),'--run',str(run),'--step',str(step),
                '--output',str(run/f'audit-{step}.json')],run/f'audit-{step}.log',env,source,deadline)
            assert json.loads((run/f'audit-{step}.json').read_text())['passed']
            archive=run/f'archive-step{step:07d}';archive.mkdir();shutil.copy2(run/'last.ckpt',archive/'native.ckpt');save(archive/'manifest.json',dict(step=step,native_sha256=digest(archive/'native.ckpt')))
            seeds=[42,1042]if step==500 else[42,43,1042,1043]if step<10000 else list(range(42,52))+list(range(1042,1052))
            evaluation(source,run,step,plan['manifest'],run/f'eval-projection-step{step:07d}',env,deadline,technical=step==2,seeds=seeds)
            save(run/'progress.json',dict(step=step,phase='stage_complete'))
        save(run/'complete.json',dict(passed=True,step=10000,forward_observation_targets_only=True))
    except BaseException as error:save(run/'execution-error.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True)
    for key in ['root','source','parent']:p.add_argument('--'+key,type=Path)
    p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    else:work(a)
