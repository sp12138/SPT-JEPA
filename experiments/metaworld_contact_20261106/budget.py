"""Longer original-world-model controls,without modifying old closed campaigns."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

from experiments.metaworld_contact_20261106.broad import wait_owner
from experiments.metaworld_contact_20261106.bridge_campaign import command, TASKS
from experiments.metaworld_contact_20261106.diagnose import save, digest
from experiments.metaworld_contact_20261106.launch import BASE, PYTHON, wait_gpu_release


def prepare(root,source):
    out=root/'budget-v1';out.mkdir()
    old=json.loads((BASE/'runs/metaworld_proposal_20261104_v3/plan.json').read_text())
    parent_plan=json.loads((root/'bridge-plan-v3.json').read_text())
    recipes=[]
    for host,seed in [('dev13',1),('dev15',2)]:
        owners=json.loads((root/'broad-repeat-v1'/f'launch-{host}.json').read_text())['jobs']
        for gpu,arm in [(1,'tf-control'),(2,'native-control')]:
            parent=Path(next(r for r in old['recipes'] if r['arm']==arm and r['seed']==seed)['checkpoint']).parent
            archive=parent/'archive-step0016480';proof=json.loads((archive/'manifest.json').read_text())
            expected=proof.get('native_sha256',proof.get('files',{}).get('native.ckpt'))
            assert expected and digest(archive/'native.ckpt')==expected
            name=f'{arm}-seed{seed}';run=out/'training'/name;run.mkdir(parents=True)
            config=copy.deepcopy(json.loads((parent/'config.json').read_text()))
            assert config['max_steps']==41200 and config['batch_size']==128
            config.update(output=str(run),retain_steps=[16480,16482,24982,40000])
            save(run/'config.json',config);os.link(archive/'native.ckpt',run/'last.ckpt')
            shutil.copytree(parent/'initial-model',run/'initial-model',copy_function=os.link)
            shutil.copytree(parent/'model-step0016480',run/'model-step0016480',copy_function=os.link)
            save(run/'FORK.json',dict(parent=str(parent),entry_step=16480,native_sha256=expected,
                parent_config=str(parent/'config.json'),full_optimizer_scheduler_rng_sample_cursor=True,
                objective_unchanged=True,NN_runtime='7b78f5d',completed_SGD_repeated=False,
                scope='New root only.Old campaigns/HOLD untouched;no new teacher/policylabels.'))
            recipes.append(dict(name=name,arm=arm,seed=seed,host=host,gpu=gpu,run=str(run),parent=str(parent),
                config_sha256=digest(run/'config.json'),wait_owner=next(r for r in owners if r['gpu']==gpu)))
    save(out/'plan.json',dict(source=str(source),recipes=recipes,data=old['data'],manifest=old['case_manifest'],
        deadline_unix=parent_plan['deadline_unix'],stages=[16482,24982,40000],
        scope='OriginalscratchTF/native continued20epoch->40k updates,batch128,original41200LRhorizon. '
              'No objective/data/sampler/optimizer changes;originalwindow8/all5phases,not CNNphase0/pairwindow. '
              'Matches total update/samplecount of40kCNNs,but not imageFLOPs,architecture or temporal sampling. '
              'Same20physicalcases/task,500native any-success;not publicSOTA or equalFLOPs.'))


def work(root,source,host,gpu):
    out=root/'budget-v1';plan=json.loads((out/'plan.json').read_text())
    r=next(x for x in plan['recipes'] if x['host']==host and x['gpu']==gpu)
    wait_owner(r['wait_owner'],plan['deadline_unix']);wait_gpu_release(gpu)
    run=Path(r['run']);proof=json.loads((run/'FORK.json').read_text())
    assert digest(run/'config.json')==r['config_sha256'] and digest(run/'last.ckpt')==proof['native_sha256']
    helper=BASE/'reports/metaworld_improve_20261104/helper-698ae41'
    runtime=BASE/'reports/metaworld_improve_20261104/runtime-7b78f5d'
    sys.path[:0]=[str(helper),str(runtime)]
    from experiments.metaworld_improve_20261104.campaign import environments
    env=environments(BASE/'runs/metaworld_improve_20261104',runtime,helper,gpu,True)
    paths=env['PYTHONPATH'].split(':');paths.insert(2,str(source));env['PYTHONPATH']=':'.join(paths)
    def audit(step,entry=False):
        cmd=[PYTHON,'-B','-u',str(helper/'experiments/metaworld_eval_20261103/audit.py'),
             '--run',str(run),'--step',str(step),'--output',str(run/f'audit-{step}.json')]
        if entry:cmd+=['--reference-config',proof['parent_config']]
        command(cmd,run/f'audit-{step}.log',env,runtime,plan['deadline_unix'])
        assert json.loads((run/f'audit-{step}.json').read_text())['passed']
    try:
        audit(16480,True)
        for step in plan['stages']:
            if (out/'STOP').exists() or (run/'STOP').exists():raise RuntimeError('Explicit stop marker')
            wait_gpu_release(gpu)
            command([PYTHON,'-B','-u',str(runtime/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'),
                     '--config',str(run/'config.json'),'--stop-step',str(step),'--resume'],
                    run/f'train-{step}.log',env,runtime,plan['deadline_unix'])
            audit(step)
            archive=run/f'archive-step{step:07d}';archive.mkdir();shutil.copy2(run/'last.ckpt',archive/'native.ckpt')
            save(archive/'manifest.json',dict(step=step,native_sha256=digest(archive/'native.ckpt')))
            if step==16482 or step==40000:
                seeds=[42] if step==16482 else list(range(42,52))+list(range(1042,1052))
                tasks=['reach-v3'] if step==16482 else TASKS
                command([PYTHON,'-B','-u',str(source/'experiments/metaworld_contact_20261106/evaluate.py'),
                    '--mode','cem','--parent',str(run/f'model-step{step:07d}'),'--data',plan['data'],
                    '--manifest',plan['manifest'],'--output',str(run/f'eval-step{step:07d}'),
                    '--budget','20' if step==16482 else '500','--seeds']+list(map(str,seeds))+['--tasks']+tasks,
                    run/f'eval-{step}.log',env,runtime,plan['deadline_unix'])
                assert json.loads((run/f'eval-step{step:07d}/summary.json').read_text())['passed']
        save(run/'complete.json',dict(passed=True,step=40000,scope=plan['scope']))
    except BaseException as error:
        save(run/'execution-error.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.source)
    else:work(a.root,a.source,a.host,a.gpu)
