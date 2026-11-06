"""Finite scratch native image-transport world learning;no inverse objective."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time

from experiments.metaworld_contact_20261106.launch import environment,PYTHON,wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command,TASKS
from experiments.metaworld_kernel_20261106.campaign import save,digest


def prepare(root,source,data,manifest):
    root.mkdir(parents=True,exist_ok=False);recipes=[];stages=[2,1000,5000,20000]
    for host,seeds in[('dev13',[1,3]),('dev15',[2,4])]:
        for offset,seed in enumerate(seeds):
            for slot,mode in enumerate(['free','linear','signed','stencil']):
                gpu=offset*4+slot;run=root/'training'/f'{mode}-seed{seed}';run.mkdir(parents=True)
                cfg=dict(output=str(run),data=str(data),native_rgb_cache='/tmp/prescottsun-native-rgb128-20261106-v1',
                    dataset_kind='native_rgb_pairs',model_class='causal_transport',seed=seed,batch_size=256,workers=2,cpu_threads=2,
                    max_steps=20000,max_seconds=24*3600,checkpoint_every=1000,retain_steps=stages,precision='bf16-mixed',
                    record_initial_weights=True,record_runtime_source=True,model=dict(mode=mode,width=128,blocks=4,window=2))
                save(run/'config.json',cfg)
                save(run/'ORIGIN.json',dict(all_weights_random=True,external_pretraining=False,no_inverse_action_ancestry_or_objective=True,
                    no_action_labels_pseudolabels_or_future_references=True,known_current_action_forward_input_only=True,
                    targets='Native NEXT RGB;goal-query target strictly between observed endpoints',
                    image_information='Observed image copied through a learned forward warp;no target-image corruption shortcut',
                    source_corpus='Same original LeRobot2500trajectories,all5phases interleaved;45/5split.No extra trainingdata.'))
                resource_pid=None
                if slot==3:
                    resource=Path(data).parents[1]/'runs/metaworld_signedfields_20261106'/f'launch-{host}.json'
                    owner=next(x for x in json.loads(resource.read_text())['workers']if x['gpu']==gpu)
                    resource_pid=owner['pid']
                recipes.append(dict(arm=mode,seed=seed,host=host,gpu=gpu,run=str(run),resource_pid=resource_pid,config_sha256=digest(run/'config.json')))
    save(root/'training-plan.json',dict(source=str(source),data=str(data),manifest=str(manifest),recipes=recipes,stages=stages,
        objectives=['projection'],deadline_unix=time.time()+36*3600,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Native forward image transforms and goal-conditioned future observations only.Scratch/sample-level/rewardfree.'
              'Same physicalcases/goals/500native cap/replan15;different native training and ordered native commands/compute.No SOTA claim.'))


def copy_data(data,local):
    local.mkdir()
    for shard in range(12):
        source=data/str(shard);target=local/str(shard);target.mkdir();m=json.loads((source/'manifest.json').read_text());assert m['passed']
        for name,key in[('pixels.npy','pixels_sha256'),('known_input.npy','known_input_sha256')]:
            shutil.copy2(source/name,target/name);assert digest(target/name)==m[key]
        shutil.copy2(source/'manifest.json',target/'manifest.json')
    save(local/'READY.json',dict(passed=True,frames=204806,source=str(data),no_model_updates=True))


def work(root,source,host,gpu):
    plan=json.loads((root/'training-plan.json').read_text());r=next(x for x in plan['recipes']if x['host']==host and x['gpu']==gpu)
    run=Path(r['run']);deadline=plan['deadline_unix'];data=Path(plan['data']);local=Path('/tmp/prescottsun-native-rgb128-20261106-v1')
    try:
        for shard in range(12):
            while not(data/str(shard)/'manifest.json').exists():assert time.time()<deadline;time.sleep(5)
        if gpu==0 and not(local/'READY.json').exists():
            try:copy_data(data,local)
            except BaseException as error:
                save(local/'COPY_ERROR.json',dict(error=repr(error)));raise
        while not(local/'READY.json').exists():
            assert not(local/'COPY_ERROR.json').exists(),'Native RGB node copy failed'
            assert time.time()<deadline;time.sleep(5)
        if r.get('resource_pid'):
            process=Path(f'/proc/{r["resource_pid"]}/cmdline')
            while process.exists():
                if b'forward_only_campaign.py'not in process.read_bytes():break
                assert time.time()<deadline;time.sleep(5)
        wait_gpu_release(gpu);env=environment(source,gpu);assert digest(run/'config.json')==r['config_sha256']
        for step in plan['stages']:
            assert not(root/'STOP').exists()and not(run/'STOP').exists();wait_gpu_release(gpu)
            command([PYTHON,'-B','-u',str(source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'),
                '--config',str(run/'config.json'),'--stop-step',str(step)]+(['--resume']if step>2 else[]),run/f'train-{step}.log',env,source,deadline)
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/native_audit.py'),
                '--run',str(run),'--step',str(step),'--output',str(run/f'audit-{step}.json')],run/f'audit-{step}.log',env,source,deadline)
            assert json.loads((run/f'audit-{step}.json').read_text())['passed']
            archive=run/f'archive-step{step:07d}';archive.mkdir();shutil.copy2(run/'last.ckpt',archive/'native.ckpt')
            save(archive/'manifest.json',dict(step=step,native_sha256=digest(archive/'native.ckpt')))
            seeds=[42]if step==2 else[42,1042]if step==1000 else[42,43,1042,1043]if step<20000 else list(range(42,52))+list(range(1042,1052))
            output=run/f'eval-projection-step{step:07d}'
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/native_evaluate.py'),'--run',str(run),
                '--step',str(step),'--manifest',plan['manifest'],'--output',str(output),'--budget','20'if step==2 else'500',
                '--tasks']+(['reach-v3']if step==2 else TASKS)+['--seeds']+list(map(str,seeds)),run/f'eval-{step}.log',env,source,deadline)
            assert json.loads((output/'summary.json').read_text())['passed'];save(run/'progress.json',dict(step=step,phase='stage_complete'))
        save(run/'complete.json',dict(passed=True,step=20000,forward_observation_targets_only=True))
    except BaseException as error:save(run/'execution-error.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True)
    for key in['root','source']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--data',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.source,a.data,a.manifest)
    else:work(a.root,a.source,a.host,a.gpu)
