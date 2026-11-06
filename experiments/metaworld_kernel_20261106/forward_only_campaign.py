"""Forward-only signed-field continuation after read-only physical probes."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import time

from experiments.metaworld_contact_20261106.launch import environment,PYTHON,wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command,TASKS
from experiments.metaworld_kernel_20261106.campaign import save,digest


def prepare(root,source,previous):
    root.mkdir(parents=True,exist_ok=False);old=json.loads((previous/'training-plan.json').read_text());recipes=[]
    for seed in[1,2,3,4]:
        original=next(r for r in old['recipes']if r['seed']==seed and r['arm']=='fullseq-stencil')
        host=original['host'];gpu=3 if seed<3 else 7
        probe=next(r for r in json.loads((previous/f'physical-response-launch-{host}.json').read_text())['workers']if r['gpu']==gpu)
        template=json.loads((Path(original['run'])/'config.json').read_text())
        run=root/'training'/f'signed-stencil-seed{seed}';run.mkdir(parents=True)
        cfg=dict(template,output=str(run),model_class='signed_density')
        save(run/'config.json',cfg)
        save(run/'ORIGIN.json',dict(all_weights_random=True,learned_weight_ancestry=False,external_pretraining=False,
            no_inverse_or_control_coordinate_objective=True,no_future_action_label_pseudolabel_or_reference=True,
            current_executed_commands_input_only=True,past_own_command_input=True,targets='NEXT observation features only',
            physical_probe_data_used_for_training=False,random_feature_cache_only=True))
        recipes.append(dict(arm='signed-stencil',seed=seed,host=host,gpu=gpu,run=str(run),probe_output=probe['output'],
            probe_pid=probe['pid'],config_sha256=digest(run/'config.json')))
    save(root/'training-plan.json',dict(source=str(source),data=old['data'],manifest=old['manifest'],recipes=recipes,
        stages=[2,500,2500,10000],objectives=['projection'],deadline_unix=time.time()+24*3600,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        constraint='Pure forward future-observation fitting,no inverse-action objective or future-action reference',
        hypothesis='Signed actuator response can express unequal positive/negative effects;linear fields cannot.'
                   'Past/current executed control stencil handles maintained actuation without state-only drift.'
                   'Exact16-chamber control optimization is planning only,never a training objective.'))


def work(root,source,host,gpu):
    plan=json.loads((root/'training-plan.json').read_text());r=next(x for x in plan['recipes']if x['host']==host and x['gpu']==gpu)
    run=Path(r['run']);deadline=plan['deadline_unix']
    try:
        probe=Path(r['probe_output'])/'summary.json'
        while not probe.exists():assert time.time()<deadline;time.sleep(5)
        assert json.loads(probe.read_text())['passed'];wait_gpu_release(gpu)
        env=environment(source,gpu);assert digest(run/'config.json')==r['config_sha256']
        for step in plan['stages']:
            assert not(root/'STOP').exists()and not(run/'STOP').exists();wait_gpu_release(gpu)
            command([PYTHON,'-B','-u',str(source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'),
                '--config',str(run/'config.json'),'--stop-step',str(step)]+(['--resume']if step>2 else[]),run/f'train-{step}.log',env,source,deadline)
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/audit.py'),
                '--run',str(run),'--step',str(step),'--output',str(run/f'audit-{step}.json')],run/f'audit-{step}.log',env,source,deadline)
            assert json.loads((run/f'audit-{step}.json').read_text())['passed']
            archive=run/f'archive-step{step:07d}';archive.mkdir();shutil.copy2(run/'last.ckpt',archive/'native.ckpt')
            save(archive/'manifest.json',dict(step=step,native_sha256=digest(archive/'native.ckpt')))
            seeds=[42]if step==2 else[42,1042]if step==500 else[42,43,1042,1043]if step<10000 else list(range(42,52))+list(range(1042,1052))
            output=run/f'eval-projection-step{step:07d}'
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/evaluate.py'),'--run',str(run),
                '--step',str(step),'--manifest',plan['manifest'],'--output',str(output),'--budget','20'if step==2 else'500',
                '--tasks']+(['reach-v3']if step==2 else TASKS)+['--seeds']+list(map(str,seeds)),run/f'eval-{step}.log',env,source,deadline)
            assert json.loads((output/'summary.json').read_text())['passed'];save(run/'progress.json',dict(step=step,phase='stage_complete'))
        save(run/'complete.json',dict(passed=True,step=10000,forward_observation_targets_only=True))
    except BaseException as error:save(run/'execution-error.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True)
    for key in['root','source']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--previous',type=Path);p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.source,a.previous)
    else:work(a.root,a.source,a.host,a.gpu)
