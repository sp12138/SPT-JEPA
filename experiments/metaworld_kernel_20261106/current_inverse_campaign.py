"""Scratch single-current-interval flow models;no future action trajectory targets."""
import argparse
import json
from pathlib import Path
import shutil
import time
from experiments.metaworld_contact_20261106.launch import environment,PYTHON,wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command
from experiments.metaworld_kernel_20261106.campaign import save,digest


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=False);recipes=[];owners=json.loads((a.reference/'launch-dev6.json').read_text())['workers']
    for seed in[1,2]:
        for index,(arm,anchors,balance,odds)in enumerate([('phase0','phase0','frames',1.),('native','all','frames',1.),('episodes','all','episodes',1.),('highnoise','all','episodes',4.)]):
            gpu=(seed-1)*4+index;run=a.root/'training'/f'{arm}-seed{seed}';run.mkdir(parents=True)
            cfg=dict(seed=seed,output=str(run),dataset_kind='native_inverse_pairs',data='/efs/prescottsun/SPT-JEPA/data/metaworld-lerobot-b46a1c18-packed-v1',
                native_inverse_cache='/tmp/prescottsun-native-rgb224-20261107-v1',inverse_anchors=anchors,inverse_balance=balance,model_class='temporal_bridge',
                model=dict(visual_source='local',objective='flow',chunk=1,window=2,width=192,action_dim=20,history_context=False,temporal_aux=False,condition_horizon='none',noise_odds=odds),
                batch_size=128,workers=2,max_steps=40000,checkpoint_every=1000,cpu_threads=2,max_seconds=24*3600,precision='bf16-mixed',
                record_runtime_source=True,record_initial_weights=True,retain_steps=[2,600,3000,10000,20000,40000],float32_matmul_precision='highest')
            save(run/'config.json',cfg);owner=next(x for x in owners if x['gpu']==gpu)
            save(run/'ORIGIN.json',dict(all_weights_random=True,no_initial_or_teacher_checkpoint=True,
                target='ONLYcurrentalready-executed5nativeinterval20D;NOactionatoraftert+5targeted.No optimality assertion.',
                learner_inputs='CurrentandlatergoalRGB224plusinternallysampledFMnoise/time;no environmentalclock/history/state/reward/success/taskID.',
                labels='Onefactualcurrent-inverseflowmatchingloss.Noauxiliaryforecast/reconstruction/time loss.',
                comparator='Old78.75%chunk6weightsNEVERloaded;oldmodelsarefrozenreferenceonly.',
                design='Phase0->allnativeanchors->uniformepisodeswithouttaskIDs->globalhighnoiseodds4;eachadjacentcontrastchangesonefactor.'))
            recipes.append(dict(arm=arm,seed=seed,host='dev6',gpu=gpu,run=str(run),resource_pid=owner['pid'],resource_marker='inverse_reference.py',config_sha256=digest(run/'config.json')))
    save(a.root/'training-plan.json',dict(source=str(a.source),recipes=recipes,stages=[2,600,3000,10000,20000,40000],
        manifest='/efs/prescottsun/SPT-JEPA/runs/metaworld_eval_20261103/cases-v2/manifest.json',deadline_unix=time.time()+36*3600,
        scope='Latestuserpermitsinverse,performancepriority.ScratchCURRENTinterval-only flow,8independenttrainingswithpairedinitialstateswithinseed.'
              'Originalcorpusonly,nocollectedextra data/pretrainedmodules/futurecommandlabels,one sample-level loss.'
              'NativeSGD2/600/3000entrygatesprecedematuretraining;allcontrolsuse40000LRhorizon.NopublicSOTA/equalFLOPclaim.'))


def work(a):
    plan=json.loads((a.root/'training-plan.json').read_text());r=next(x for x in plan['recipes']if x['gpu']==a.gpu);run=Path(r['run']);source=Path(plan['source']);deadline=plan['deadline_unix']
    try:
        owner=Path(f'/proc/{r["resource_pid"]}/cmdline')
        while owner.exists()and r['resource_marker'].encode()in owner.read_bytes():assert time.time()<deadline;time.sleep(10)
        wait_gpu_release(a.gpu)
        ready=Path('/tmp/prescottsun-native-rgb224-20261107-v1/READY.json')
        while not ready.exists():assert time.time()<deadline;time.sleep(10)
        assert json.loads(ready.read_text())['passed']and json.loads((a.root/'initial-contract.json').read_text())['passed']
        assert digest(run/'config.json')==r['config_sha256'];env=environment(source,a.gpu)
        for step in plan['stages']:
            assert not(a.root/'STOP').exists()and not(run/'STOP').exists();wait_gpu_release(a.gpu)
            command([PYTHON,'-B','-u',str(source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'),'--config',str(run/'config.json'),
                '--stop-step',str(step)]+(['--resume']if step>2 else[]),run/f'train-{step}.log',env,source,deadline)
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_contact_20261106/audit_bridge.py'),'--run',str(run),'--step',str(step),
                '--output',str(run/f'audit-{step}.json')],run/f'audit-{step}.log',env,source,deadline)
            assert json.loads((run/f'audit-{step}.json').read_text())['passed']
            archive=run/f'archive-step{step:07d}';archive.mkdir();shutil.copy2(run/'last.ckpt',archive/'native.ckpt');save(archive/'manifest.json',dict(step=step,native_sha256=digest(archive/'native.ckpt')))
            seeds=[42]if step==2 else[42,1042]if step<10000 else[42,43,1042,1043]if step<40000 else list(range(42,52))+list(range(1042,1052))
            endpoint=run/f'eval-step{step:07d}'
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/inverse_evaluate.py'),'--run',str(run),'--step',str(step),
                '--manifest',plan['manifest'],'--output',str(endpoint),'--execute','1','--budget','20'if step==2 else'500']+
                (['--tasks','bin-picking-v3']if step==2 else[])+['--seeds']+list(map(str,seeds)),endpoint.with_suffix('.log'),env,source,deadline)
            assert json.loads((endpoint/'summary.json').read_text())['passed'];save(run/'progress.json',dict(step=step,phase='stage_complete'))
        save(run/'complete.json',dict(passed=True,step=40000,current_interval_only=True))
    except BaseException as error:save(run/'execution-error.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True)
    for key in['root','source','reference']:p.add_argument('--'+key,type=Path)
    p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    else:work(a)
