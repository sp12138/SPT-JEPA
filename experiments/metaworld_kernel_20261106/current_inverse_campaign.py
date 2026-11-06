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
    plan=json.loads((a.root/'training-plan.json').read_text());r=next(x for x in plan['recipes']if x['gpu']==a.gpu and x['host']==a.host);run=Path(r['run']);source=Path(plan['source']);deadline=plan['deadline_unix']
    try:
        owner=Path(f'/proc/{r["resource_pid"]}/cmdline')
        while owner.exists()and r['resource_marker'].encode()in owner.read_bytes():assert time.time()<deadline;time.sleep(10)
        wait_gpu_release(a.gpu)
        ready=Path('/tmp/prescottsun-native-rgb224-20261107-v1/READY.json')
        while not ready.exists():assert time.time()<deadline;time.sleep(10)
        assert json.loads(ready.read_text())['passed']and json.loads((a.root/'initial-contract.json').read_text())['passed']
        assert digest(run/'config.json')==r['config_sha256'];env=environment(source,a.gpu)
        initial_step=r.get('initial_step',0)
        for step in plan['stages']:
            assert not(a.root/'STOP').exists()and not(run/'STOP').exists();wait_gpu_release(a.gpu)
            if step!=initial_step:
                command([PYTHON,'-B','-u',str(source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'),'--config',str(run/'config.json'),
                    '--stop-step',str(step)]+(['--resume']if step>2 else[]),run/f'train-{step}.log',env,source,deadline)
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_contact_20261106/audit_bridge.py'),'--run',str(run),'--step',str(step),
                '--output',str(run/f'audit-{step}.json')],run/f'audit-{step}.log',env,source,deadline)
            assert json.loads((run/f'audit-{step}.json').read_text())['passed']
            archive=run/f'archive-step{step:07d}';archive.mkdir();shutil.copy2(run/'last.ckpt',archive/'native.ckpt');save(archive/'manifest.json',dict(step=step,native_sha256=digest(archive/'native.ckpt')))
            entry=step in[2,initial_step,initial_step+2]
            seeds=[42]if entry else[42,1042]if step<10000 else[42,43,1042,1043]if step<40000 else list(range(42,52))+list(range(1042,1052))
            endpoint=run/f'eval-step{step:07d}'
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/inverse_evaluate.py'),'--run',str(run),'--step',str(step),
                '--manifest',plan['manifest'],'--output',str(endpoint),'--execute','1','--budget','20'if entry else'500']+
                (['--tasks','bin-picking-v3']if entry else[])+['--seeds']+list(map(str,seeds)),endpoint.with_suffix('.log'),env,source,deadline)
            assert json.loads((endpoint/'summary.json').read_text())['passed'];save(run/'progress.json',dict(step=step,phase='stage_complete'))
        if plan.get('confirmation'):
            confirmation=plan['confirmation'];manifest=Path(confirmation['manifest'])
            while not manifest.exists():assert time.time()<deadline;time.sleep(10)
            final=plan['stages'][-1];endpoint=run/'confirm-final'
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/inverse_evaluate.py'),'--run',str(run),'--step',str(final),
                '--manifest',str(manifest),'--output',str(endpoint),'--execute','1','--budget','500','--tasks']+confirmation['tasks']+['--seeds']+
                list(map(str,confirmation['seeds'])),run/'confirm-final.log',env,source,deadline)
            proof=json.loads((endpoint/'summary.json').read_text());assert proof['passed']and proof['model_unchanged']and len(proof['rows'])==1000
        save(run/'complete.json',dict(passed=True,step=plan['stages'][-1],current_interval_only=True,confirmation=bool(plan.get('confirmation'))))
    except BaseException as error:save(run/'execution-error.json',dict(error=repr(error)));raise


def prepare_mse(a):
    a.root.mkdir(parents=True,exist_ok=False);recipes=[]
    for host,seed in[('dev13',1),('dev15',2)]:
        owners=json.loads((a.wait_parent/f'launch-{host}.json').read_text())['workers']
        for gpu,arm,anchors,balance in[(1,'phase0','phase0','frames'),(5,'episodes','all','episodes')]:
            cfg=json.loads((a.flow_parent/'training'/f'{arm}-seed{seed}'/'config.json').read_text());run=a.root/'training'/f'{arm}-seed{seed}';run.mkdir(parents=True)
            cfg.update(output=str(run),inverse_anchors=anchors,inverse_balance=balance);cfg['model'].update(objective='mse',noise_odds=1.)
            save(run/'config.json',cfg);save(run/'ORIGIN.json',dict(all_weights_random=True,model_class='CURRENTintervalinversecontroller',
                loss='Onecurrent20Dregressionloss;no diffusion/noiseiterationsneeded.Noactiontargetat/aftert+5.',
                no_pretrained_or_teacher_or_future_chunk_weights=True,scope='Sameoriginalcorpus/architecture/currenttargets/pairedinitialseedasflowstudy;noextraselfcollectedcorpus.'))
            owner=next(x for x in owners if x['gpu']==gpu)
            recipes.append(dict(arm=arm,seed=seed,host=host,gpu=gpu,run=str(run),resource_pid=owner['pid'],resource_marker='recurrent_campaign.py',config_sha256=digest(run/'config.json')))
    old=json.loads((a.flow_parent/'training-plan.json').read_text())
    save(a.root/'training-plan.json',dict(source=str(a.source),recipes=recipes,stages=old['stages'],manifest=old['manifest'],deadline_unix=time.time()+36*3600,
        scope='ScratchCURRENT-onlyMSEversusflow;phase0andepisodebalancednativeanchors,twoindependentseeds.Same40000LRhorizon/batch/architecture/data.'
              'Singlecurrentintervaltarget,no futureactiontrajectory/clock/history/teacher.Thisiscontrollertraining,notpureJEPAforecastloss.'))


def prepare_long(a):
    a.root.mkdir(parents=True,exist_ok=False);(a.root/'results').mkdir();recipes=[]
    confirm=json.loads((a.confirm_root/'plan.json').read_text())
    for host,seeds in[('dev13',[3,4]),('dev15',[5,6])]:
        owners=json.loads((a.wait_parent/f'launch-{host}.json').read_text())['workers']
        for index,seed in enumerate(seeds):
            for j,representation in enumerate(['absolute','directed']):
                gpu=[0,2,3,4][index*2+j];run=a.root/'training'/f'{representation}-seed{seed}';run.mkdir(parents=True)
                cfg=json.loads((a.parent/'training/episodes-seed1/config.json').read_text())
                cfg.update(seed=seed,output=str(run),max_steps=120000,retain_steps=[2,600,10000,40000,80000,120000],inverse_goals='phase_mixture')
                cfg['model']['goal_representation']=representation;assert cfg['model']['chunk']==1 and cfg['model']['action_dim']==20
                save(run/'config.json',cfg);save(run/'ORIGIN.json',dict(all_weights_random=True,independent_new_seeds=[3,4,5,6],no_initial_checkpoint=True,
                    loss='OneCURRENT5native20Dflowloss,originalcorpusonly.Noaux/teacher/futurecommandblock.',
                    comparison='Pairedabsolute/directedsameCNN/decoderinitialseedand120kLRhorizon.Budget120kfromscratch,not40kresume/cosinerestart.',
                    promotion_basis='Pre-existingtwo-seed20case/task40k6manip90.4167%vs77.9167%currentbaseline;independent50-task20kscreen87.5%.Notall-taskperformancegoalachieved.',
                    confirmation='Pre-registered1000NEWphysicalcases/50tasks/seeds4042..4051and5042..5051;allcasesandexpertgoalfailuresretained.No taskID/state/reward/actionoracleinput.'))
                owner=next(o for o in owners if o['gpu']==gpu)
                recipes.append(dict(arm=representation,seed=seed,host=host,gpu=gpu,run=str(run),resource_pid=owner['pid'],resource_marker='frozen_goal_matrix-commit.py',config_sha256=digest(run/'config.json')))
    old=json.loads((a.parent/'training-plan.json').read_text())
    save(a.root/'training-plan.json',dict(source=str(a.source),recipes=recipes,stages=[2,600,10000,40000,80000,120000],manifest=old['manifest'],deadline_unix=time.time()+36*3600,
        confirmation=dict(manifest=confirm['manifest'],tasks=confirm['tasks'],seeds=confirm['seeds']),
        scope='Eightfreshscratchtrainingruns=4NEWindependentseedPAIRSx2representations;120ksharedLRhorizon/pairedweights.Within-seedabsolute/directedcontrast.'
              'OnecurrentintervalFMloss,originaldata,sensor5/500native,RGBonly.No publicSOTA/equalFLOPclaim.Longworkersfinishformal8taskstages+fresh50taskconfirmationandexit.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','prepare-mse','prepare-long','work'],required=True)
    for key in['root','source','reference','flow-parent','wait-parent','parent','confirm-root']:p.add_argument('--'+key,type=Path)
    p.add_argument('--gpu',type=int);p.add_argument('--host',default='dev6');a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    elif a.mode=='prepare-mse':prepare_mse(a)
    elif a.mode=='prepare-long':prepare_long(a)
    else:work(a)
