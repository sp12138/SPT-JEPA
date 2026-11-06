"""Finite original-data recurrent causal/goal-query comparison."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import time
from experiments.metaworld_contact_20261106.launch import environment,PYTHON,wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command
from experiments.metaworld_kernel_20261106.campaign import save,digest
from experiments.metaworld_kernel_20261106.goal_campaign import evaluation


def prepare(root,source,parent,horizon=5,gpus=(1,5)):
    root.mkdir(parents=True,exist_ok=False);old=json.loads((parent/'training-plan.json').read_text());recipes=[];stages=[2,500,2500,10000]
    for host,seed in [('dev13',1),('dev15',2)]:
        owners=json.loads((parent/f'launch-generator-v3-{host}.json').read_text())['workers']
        for gpu,mode in zip(gpus,['teacher','rollout']):
            run=root/'training'/f'{mode}-seed{seed}';run.mkdir(parents=True)
            donor=root.parent/'metaworld_native_transport_20261106/training'/f'free-seed{seed}'
            assert json.loads((donor/'ORIGIN.json').read_text())['all_weights_random']and json.loads((donor/'audit-20000.json').read_text())['passed']
            cfg=dict(output=str(run),data=old['data'],native_rgb_cache='/tmp/prescottsun-native-rgb128-20261106-v1',dataset_kind='query_rgb_paths',model_class='query_recurrent',
                seed=seed,batch_size=128,workers=2,cpu_threads=2,max_steps=10000,max_seconds=24*3600,checkpoint_every=500,retain_steps=stages,precision='bf16-mixed',
                record_initial_weights=True,record_runtime_source=True,model=dict(init_checkpoint=str(donor/'model-step0020000/weights.pt'),training_mode=mode,width=128,blocks=4,window=2,forecast_horizon=horizon))
            save(run/'config.json',cfg);owner=next(x for x in owners if x['gpu']==gpu)
            save(run/'ORIGIN.json',dict(parent=str(donor),parent_updates=20000,parent_scope='Ownscratchsame-dataforwardRGBonly;copyvision/initialCNN/decoder,allcostcounts',
                new_GRU_context_RGB_head_random=True,no_inverse_or_external_weights=True,targets=f'Same{horizon}actualfutureRGBframesforbotharms',
                teacher='Resetsbeliefoneachobservedtransition;extraactualintermediateobservations',rollout='Beliefcomposes5stepswithoutfutureimageinputs',
                known_action_word='Executedcontrolofeachcorrespondingforwardtrainingtransition INPUT only;neveroutputs,policyreferencesorplannerproposals.Goalqueryclearsallcommands.',no_state_reward_success_taskID_inputs=True))
            recipes.append(dict(arm=mode,seed=seed,host=host,gpu=gpu,run=str(run),resource_pid=owner['pid'],config_sha256=digest(run/'config.json')))
    save(root/'training-plan.json',dict(source=str(source),data=old['data'],manifest=old['manifest'],recipes=recipes,stages=stages,objectives=['projection'],
        deadline_unix=time.time()+36*3600,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        forecast_horizon=horizon,scope=f'Originalcorpusonly,shared2ownforwardparents x2matchedtrainingmodes,same{horizon}RGBtargetsandsamplestarts.Noinverse.Noextra trainingdata.'
              'OneRGB8likelihood,sharedrecurrentcausal/goalpredictor,latentmemoryandpassivechangeallowed.No performance/noveltyclaim.'))


def work(root,source,host,gpu):
    plan=json.loads((root/'training-plan.json').read_text());r=next(x for x in plan['recipes']if x['host']==host and x['gpu']==gpu);run=Path(r['run']);deadline=plan['deadline_unix']
    try:
        owner=Path(f'/proc/{r["resource_pid"]}/cmdline')
        while owner.exists()and b'goal_campaign.py'in owner.read_bytes():assert time.time()<deadline;time.sleep(10)
        wait_gpu_release(gpu);env=environment(source,gpu);assert digest(run/'config.json')==r['config_sha256']
        for step in plan['stages']:
            assert not(root/'STOP').exists()and not(run/'STOP').exists();wait_gpu_release(gpu)
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
    for key in ['root','source']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parent',type=Path);p.add_argument('--host');p.add_argument('--gpu',type=int);p.add_argument('--horizon',type=int,choices=[5,15],default=5)
    p.add_argument('--slots',type=int,nargs=2,default=[1,5]);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.source,a.parent,a.horizon,a.slots)
    else:work(a.root,a.source,a.host,a.gpu)
