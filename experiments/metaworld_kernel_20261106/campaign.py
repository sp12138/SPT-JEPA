"""Finite native stages for the strict predictive-kernel investigation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from experiments.metaworld_contact_20261106.launch import environment,PYTHON,wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command,TASKS


def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(root,source):
    plan=json.loads((root/'plan.json').read_text())
    stages=[2,250,1000,2500,10000]
    for r in plan['recipes']:
        run=Path(r['run']);run.mkdir(parents=True)
        cfg=dict(output=str(run),data=plan['data'],dataset_kind='kernel_transitions',model_class='predictive_kernel',
            feature_cache=str(root/f'cache-seed{r["seed"]}'),seed=r['seed'],batch_size=512,workers=2,cpu_threads=2,
            max_steps=10000,max_seconds=24*3600,checkpoint_every=500,retain_steps=stages,precision='bf16-mixed',
            record_initial_weights=True,record_runtime_source=True,
            model=dict(feature_seed=r['seed'],channels=32,width=128,blocks=6,boundary=r['boundary'],gaussian=r['gaussian'],affine=r['affine'],window=2))
        save(run/'config.json',cfg);r['config_sha256']=digest(run/'config.json')
        save(run/'ORIGIN.json',dict(all_weights_random=True,fixed_random_target=True,pretrained_weights=False,
            pixel_or_action_reconstruction=False,action_prediction_target=False,future_action_references=False,
            known_executed_control_is_dynamics_input_only=True,learned_outputs='NEXT observation features and uncertainty'))
    plan.update(training_source=str(source),stages=stages,phase='technical2_then_250_1000_2500_10000')
    save(root/'training-plan.json',plan)


def work(root,source,host,gpu):
    plan=json.loads((root/'training-plan.json').read_text());r=next(x for x in plan['recipes']if x['host']==host and x['gpu']==gpu)
    run=Path(r['run']);deadline=plan['deadline_unix']
    for shard in range(4):
        p=root/f'cache-seed{r["seed"]}'/str(shard)/'manifest.json'
        while not p.exists():
            assert time.time()<deadline;time.sleep(5)
        assert json.loads(p.read_text())['passed']
    wait_gpu_release(gpu)
    # Copy immutable feature-only data to node storage;never read action labels.
    local=Path(f'/tmp/prescottsun-kernel-features-seed{r["seed"]}')
    if r['shard']==0:
        local.mkdir()
        for shard in range(4):
            src=root/f'cache-seed{r["seed"]}'/str(shard);dst=local/str(shard);dst.mkdir()
            m=json.loads((src/'manifest.json').read_text())
            for name,expected in [('features.npy',m['features_sha256']),('executed_input.npy',m['executed_input_sha256'])]:
                shutil.copy2(src/name,dst/name);assert digest(dst/name)==expected
            shutil.copy2(src/'manifest.json',dst/'manifest.json')
        save(local/'READY.json',dict(passed=True))
    while not(local/'READY.json').exists():
        assert time.time()<deadline;time.sleep(5)
    cfg=json.loads((run/'config.json').read_text());cfg['feature_cache']=str(local)
    save(run/'CONFIG_RELOCATION.json',dict(original_sha256=r['config_sha256'],only_feature_cache_path_changed=True,local=str(local)))
    save(run/'config.json',cfg)
    env=environment(source,gpu)
    try:
        for step in plan['stages']:
            if (root/'STOP').exists()or(run/'STOP').exists():raise RuntimeError('Explicit stop')
            wait_gpu_release(gpu)
            command([PYTHON,'-B','-u',str(source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'),
                '--config',str(run/'config.json'),'--stop-step',str(step)]+(['--resume']if step>2 else[]),run/f'train-{step}.log',env,source,deadline)
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/audit.py'),'--run',str(run),
                '--step',str(step),'--output',str(run/f'audit-{step}.json')],run/f'audit-{step}.log',env,source,deadline)
            assert json.loads((run/f'audit-{step}.json').read_text())['passed']
            archive=run/f'archive-step{step:07d}';archive.mkdir();shutil.copy2(run/'last.ckpt',archive/'native.ckpt')
            save(archive/'manifest.json',dict(step=step,native_sha256=digest(archive/'native.ckpt')))
            seeds=[42]if step==2 else[42,1042]if step==250 else[42,43,1042,1043]if step<10000 else list(range(42,52))+list(range(1042,1052))
            tasks=['reach-v3']if step==2 else TASKS
            output=run/f'eval-step{step:07d}'
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/evaluate.py'),'--run',str(run),
                '--step',str(step),'--manifest',plan['manifest'],'--output',str(output),'--budget','20'if step==2 else'500',
                '--tasks']+tasks+['--seeds']+list(map(str,seeds)),run/f'eval-{step}.log',env,source,deadline)
            assert json.loads((output/'summary.json').read_text())['passed']
            save(run/'progress.json',dict(step=step,phase='stage_complete'))
        save(run/'complete.json',dict(passed=True,step=10000,no_performance_claim_without_summary=True))
    except BaseException as error:
        save(run/'execution-error.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.source)
    else:work(a.root,a.source,a.host,a.gpu)
