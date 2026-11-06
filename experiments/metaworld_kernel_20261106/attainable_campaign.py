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
        run=root/'training'/f'{r["arm"]}-seed{r["seed"]}';run.mkdir(parents=True)
        cfg=dict(output=str(run),data=plan['data'],dataset_kind='attainable_pairs',model_class='attainable_forecast',
            feature_cache=r['feature_cache'],seed=r['seed'],batch_size=512,workers=2,cpu_threads=2,max_steps=5000,
            checkpoint_every=500,retain_steps=[2,500,2000,5000],precision='bf16-mixed',max_seconds=20*3600,
            record_initial_weights=True,record_runtime_source=True,
            model=dict(parent=r['run'],feature_seed=r['seed'],boundary=True,window=2))
        save(run/'config.json',cfg);owner=next(x for x in launches[r['host']]['workers']if x['gpu']==r['gpu'])
        recipes.append(dict(arm=r['arm'],seed=r['seed'],host=r['host'],gpu=r['gpu'],run=str(run),parent=r['run'],
            join_pid=owner['pid'],config_sha256=digest(run/'config.json')))
        save(run/'ORIGIN.json',dict(external_pretraining=False,reconstruction_or_action_supervised_ancestry=False,
            parent=r['run'],parent_updates=10000,parent_batch_size=256,head_updates=5000,head_batch_size=512,
            total_training_windows=5120000,not_equal_image_FLOPs=True,
            policy_fitting_current_action_field=False,only_past_own_control_input=True,
            target='Observed NEXT random feature;forecast restricted to frozen own forward control span',
            weights='World and goal representation inherited from own strict same-data parent;new4Dcoordinate head.Entire parent cost counts.'))
    save(root/'training-plan.json',dict(source=str(source),parent=str(parent),data=plan['data'],manifest=plan['manifest'],recipes=recipes,
        stages=[2,500,2000,5000],objectives=['projection'],deadline_unix=time.time()+24*3600,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rationale='Free next-feature goal prediction has large components outside the learned control span.'
                  'Test a forecast parameterized through its OWN learned forward model,using only future-observation targets.'
                  'Frozen world prevents the goal learner from changing physics to fit its proposal.No action pseudolabels.'))


def work(root,source,host,gpu):
    plan=json.loads((root/'training-plan.json').read_text());r=next(x for x in plan['recipes']if x['host']==host and x['gpu']==gpu)
    run=Path(r['run']);parent=Path(r['parent']);deadline=plan['deadline_unix']
    try:
        process=Path(f'/proc/{r["join_pid"]}/cmdline')
        while process.exists():
            if b'metaworld_kernel_20261106/density_campaign.py'not in process.read_bytes():break
            assert time.time()<deadline;time.sleep(5)
        assert (parent/'complete.json').exists(),'Parent failed;inspect retained error rather than silently reinitializing'
        assert json.loads((parent/'audit-10000.json').read_text())['passed']
        assert digest(run/'config.json')==r['config_sha256'];env=environment(source,gpu)
        for step in plan['stages']:
            assert not(root/'STOP').exists()and not(run/'STOP').exists();wait_gpu_release(gpu)
            command([PYTHON,'-B','-u',str(source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'),
                '--config',str(run/'config.json'),'--stop-step',str(step)]+(['--resume']if step>2 else[]),run/f'train-{step}.log',env,source,deadline)
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/audit_attainable.py'),
                '--run',str(run),'--step',str(step),'--output',str(run/f'audit-{step}.json')],run/f'audit-{step}.log',env,source,deadline)
            assert json.loads((run/f'audit-{step}.json').read_text())['passed']
            archive=run/f'archive-step{step:07d}';archive.mkdir();shutil.copy2(run/'last.ckpt',archive/'native.ckpt')
            save(archive/'manifest.json',dict(step=step,native_sha256=digest(archive/'native.ckpt')))
            seeds=[42]if step==2 else[42,1042]if step==500 else[42,43,1042,1043]if step<5000 else list(range(42,52))+list(range(1042,1052))
            output=run/f'eval-projection-step{step:07d}'
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/evaluate.py'),'--run',str(run),
                '--step',str(step),'--manifest',plan['manifest'],'--output',str(output),'--budget','20'if step==2 else'500',
                '--tasks']+(['reach-v3']if step==2 else TASKS)+['--seeds']+list(map(str,seeds)),run/f'eval-{step}.log',env,source,deadline)
            assert json.loads((output/'summary.json').read_text())['passed'];save(run/'progress.json',dict(step=step,phase='stage_complete'))
        save(run/'complete.json',dict(passed=True,step=5000,parent_updates=10000,head_uses_only_future_observation_targets=True))
    except BaseException as error:
        save(run/'execution-error.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True)
    for key in['root','source']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parent',type=Path);p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.source,a.parent)
    else:work(a.root,a.source,a.host,a.gpu)
