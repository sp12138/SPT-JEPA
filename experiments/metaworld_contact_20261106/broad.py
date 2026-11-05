"""Frozen fifty-task coverage study,with parallel cached-goal construction."""
import argparse
import datetime
import json
from pathlib import Path
import time

import psutil

from experiments.metaworld_contact_20261106.bridge_campaign import command
from experiments.metaworld_contact_20261106.diagnose import save, digest
from experiments.metaworld_contact_20261106.launch import BASE, PYTHON, environment, wait_gpu_release
from experiments.metaworld_eval_20261103.environment import configuration


def prepare(root, source):
    out=root/'broad-v1';out.mkdir()
    old=json.loads((root/'bridge-plan-v3.json').read_text())
    long=json.loads((root/'long-v1/bridge-plan.json').read_text())
    controls=json.loads((BASE/'runs/metaworld_proposal_20261104_v3/plan.json').read_text())
    policy_source=BASE/'sources/lerobot-mw-f5c9bc02'
    tasks=sorted(configuration(policy_source)['TASK_POLICY_MAPPING'])
    assert len(tasks)==50 and len(set(tasks))==50
    recipes=[]
    for host,seed in [('dev13',1),('dev15',2)]:
        for gpu,arm in [(1,'local-flow'),(2,'local-mse'),(3,'tf-control'),(5,'native-control')]:
            if arm.startswith('local-'):
                learner=next(r for r in long['recipes'] if r['host']==host and r['arm']==arm)
                run=learner['run'];parent=learner['parent'];checkpoint=Path(run)/'model-step0040000'
                assert json.loads((Path(run)/'audit-40000.json').read_text())['passed']
            else:
                control=next(r for r in controls['recipes'] if r['arm']==arm and r['seed']==seed)
                run=None;parent=control['checkpoint'];checkpoint=Path(parent)
            if gpu in [2,3]:owner_root=root/'fresh-v1'
            elif gpu==1:owner_root=root/'vit-v1'
            else:owner_root=root/'vit-mse-v1'
            owners=json.loads((owner_root/f'launch-{host}.json').read_text())['jobs']
            recipes.append(dict(index=len(recipes),host=host,gpu=gpu,arm=arm,training_seed=seed,
                run=run,parent=parent,checkpoint=str(checkpoint),weights_sha256=digest(checkpoint/'weights.pt'),
                wait_owner=next(r for r in owners if r['gpu']==gpu)))
    for recipe in recipes:recipe['goal_tasks']=tasks[recipe['index']::len(recipes)]
    save(out/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=str(source),recipes=recipes,tasks=tasks,seeds=[42,43,44,45,1042,1043,1044,1045],
        data=old['data'],policy_source=str(policy_source),deadline_unix=old['deadline_unix'],
        scope='Exploratory coverage on all50 training tasks,not task-heldout generalization or canonicalMT50. '
              'Freeze successful40kCNNs and20epochTF/native;no SGD or model selection on this cohort. '
              'SameRGBgoals,physics,500native any-success,execute3;goals from diagnostic experts only. '
              'Eight physicalcases/task repeated across two trainedmodels/method. Retain every failed goal. '
              'Same independentLeRobotcorpus/split,but architecture,objective,phase sampling and compute differ. '
              'No externalpretraining/taskID/state/reward/optimal-action labels to learner or controller.'))


def wait_owner(owner, deadline):
    while True:
        try:
            p=psutil.Process(owner['pid'])
            owned=p.is_running() and p.status()!=psutil.STATUS_ZOMBIE and p.cmdline()==owner['command']
        except psutil.NoSuchProcess:owned=False
        if not owned:return
        assert time.time()<deadline
        time.sleep(5)


def prepare_repeats(root,source,study='broad-repeat-v1'):
    out=root/study;out.mkdir()
    original=json.loads((root/'broad-v1/plan.json').read_text())
    noh=study=='broad-noh-v1'
    trained_root=root/('marginal-local-v1' if noh else 'repeat-v1')
    trained=json.loads((trained_root/'bridge-plan.json').read_text())
    approval=json.loads((root/'broad-v1/GOALS_APPROVED.json').read_text())
    recipes=[]
    for host,seed in [('dev13',1 if noh else 3),('dev15',2 if noh else 4)]:
        owners=json.loads(((trained_root if noh else root/'broad-v1')/f'launch-{host}.json').read_text())['jobs']
        for gpu,arm in ([(0,'local-flow-no-h'),(4,'local-mse-no-h')] if noh else [(1,'local-flow'),(2,'local-mse')]):
            model=next(r for r in trained['recipes'] if r['host']==host and r['arm']==arm)
            run=Path(model['run']);assert json.loads((run/'audit-40000.json').read_text())['passed']
            checkpoint=run/'model-step0040000'
            recipes.append(dict(host=host,gpu=gpu,arm=arm,training_seed=seed,run=str(run),parent=model['parent'],
                checkpoint=str(checkpoint),weights_sha256=digest(checkpoint/'weights.pt'),
                wait_owner=next(r for r in owners if r['gpu']==gpu)))
    save(out/'plan.json',dict(original,source=str(source),recipes=recipes,inherited_goal_approval=approval,
         scope=original['scope']+(' Paired horizonmarginalization40k,trainingseeds1/2.' if noh else ' Independent randomtrainingseeds3/4.')+' Same400goals,not new physicalsamples.No SGD.'))
    save(out/'GOALS_APPROVED.json',approval)


def work(root, source, host, gpu, study='broad-v1'):
    out=root/study;plan=json.loads((out/'plan.json').read_text())
    r=next(v for v in plan['recipes'] if v['host']==host and v['gpu']==gpu)
    wait_owner(r['wait_owner'],plan['deadline_unix']);wait_gpu_release(gpu)
    env=environment(source,gpu);directory=out/f'{host}-gpu{gpu}';directory.mkdir()
    if study=='broad-v1':
        helper=BASE/'reports/metaworld_improve_20261104/helper-698ae41/experiments/metaworld_eval_20261103/prepare_cases.py'
        command([PYTHON,'-B','-u',str(helper),'--source',plan['policy_source'],'--output',str(directory/'cases'),
                 '--seeds']+list(map(str,plan['seeds']))+['--tasks']+r['goal_tasks'],directory/'goals.log',env,source,plan['deadline_unix'])
    if study=='broad-v1' and r['index']==0:
        files=[out/f'{x["host"]}-gpu{x["gpu"]}/cases/manifest.json' for x in plan['recipes']]
        while not all(p.exists() for p in files):
            assert time.time()<plan['deadline_unix'];time.sleep(5)
        rows=[]
        for file in files:
            value=json.loads(file.read_text());assert value['passed'];rows.extend(value['rows'])
        expected={(t,s) for t in plan['tasks'] for s in plan['seeds']}
        assert len(rows)==400 and {(r['task'],r['seed']) for r in rows}==expected
        assert all(digest(row['path'])==row['sha256'] for row in rows)
        manifest=out/'cases-manifest.json'
        save(manifest,dict(passed=True,rows=rows,scope=plan['scope']))
        save(out/'GOALS_APPROVED.json',dict(passed=True,manifest=str(manifest),sha256=digest(manifest),
             cases=len(rows),expert_goal_valid=sum(row['expert_any_success'] for row in rows),all_failures_retained=True))
    while not (out/'GOALS_APPROVED.json').exists():
        assert time.time()<plan['deadline_unix'];time.sleep(5)
    manifest=Path(json.loads((out/'GOALS_APPROVED.json').read_text())['manifest'])
    assert digest(manifest)==json.loads((out/'GOALS_APPROVED.json').read_text())['sha256']
    assert digest(Path(r['checkpoint'])/'weights.pt')==r['weights_sha256']
    extra=['--run',r['run'],'--step','40000'] if r['run'] else []
    command([PYTHON,'-B','-u',str(source/'experiments/metaworld_contact_20261106/evaluate.py'),
             '--mode','bridge' if r['run'] else 'cem','--parent',r['parent'],'--data',plan['data'],
             '--manifest',str(manifest),'--output',str(directory/'evaluation'),'--execute','3','--horizon','6']+
             extra+['--seeds']+list(map(str,plan['seeds']))+['--tasks']+plan['tasks'],
            directory/'evaluation.log',env,source,plan['deadline_unix'])
    assert json.loads((directory/'evaluation/summary.json').read_text())['passed']
    save(directory/'complete.json',dict(passed=True,no_sgd=True,scope=plan['scope']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--mode',choices=['prepare','prepare-repeats','prepare-noh','work'],required=True);p.add_argument('--host');p.add_argument('--gpu',type=int)
    p.add_argument('--study',choices=['broad-v1','broad-repeat-v1','broad-noh-v1'],default='broad-v1')
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.source)
    elif a.mode=='prepare-repeats':prepare_repeats(a.root,a.source)
    elif a.mode=='prepare-noh':prepare_repeats(a.root,a.source,'broad-noh-v1')
    else:work(a.root,a.source,a.host,a.gpu,a.study)
