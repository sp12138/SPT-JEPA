"""Preregistered new physical-case validation;no training or parameter search."""
import argparse
import datetime
import json
from pathlib import Path
import time

import psutil

from experiments.metaworld_contact_20261106.bridge_campaign import command, TASKS
from experiments.metaworld_contact_20261106.launch import BASE, PYTHON, environment, wait_gpu_release
from experiments.metaworld_contact_20261106.diagnose import save, digest


def prepare(root,source):
    out=root/'fresh-v1';out.mkdir()
    old=json.loads((root/'bridge-plan-v3.json').read_text());long=json.loads((root/'long-v1/bridge-plan.json').read_text())
    baseline=json.loads((BASE/'runs/metaworld_proposal_20261104_v3/plan.json').read_text())
    recipes=[]
    for host,seed in [('dev13',1),('dev15',2)]:
        owners=json.loads((root/'long-v1'/f'launch-{host}.json').read_text())['jobs']
        for gpu,arm,base in [(2,'local-mse','tf-control'),(3,'local-flow','native-control')]:
            learner=next(r for r in long['recipes']if r['host']==host and r['arm']==arm)
            checkpoint=Path(learner['run'])/'model-step0040000'
            assert json.loads((Path(learner['run'])/'audit-40000.json').read_text())['passed']
            control=next(r for r in baseline['recipes']if r['arm']==base and r['seed']==seed)
            recipes.append(dict(host=host,gpu=gpu,training_seed=seed,arm=arm,baseline_arm=base,
                run=learner['run'],parent=learner['parent'],baseline=control['checkpoint'],
                candidate_sha256=digest(checkpoint/'weights.pt'),baseline_sha256=digest(Path(control['checkpoint'])/'weights.pt'),
                wait_owner=next(r for r in owners if r['gpu']==gpu)))
    save(out/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(source),recipes=recipes,
         tasks=TASKS,seeds=list(range(2042,2052)),data=old['data'],deadline_unix=old['deadline_unix'],
         scope='New physicalseeds not used for design of these frozen40000-stepCNNs.PrimaryFlow/secondaryMSE declared before outcomes;matchedTF/native controls. '
               'SameRGB/actiononly,camera/goals/f5/context2forbaseline/execute3/500native any-success.No newSGD/optimal-actionlabels. '
               'Do not use this cohort for subsequent model selection;keep old20case study separately.No publicSOTAclaim.'))


def goals(root,source):
    fresh=root/'fresh-v1';plan=json.loads((fresh/'plan.json').read_text())
    script=BASE/'reports/metaworld_improve_20261104/helper-698ae41/experiments/metaworld_eval_20261103/prepare_cases.py'
    assert script.exists()
    env=environment(source,3)
    command([PYTHON,'-B','-u',str(script),'--source',str(BASE/'sources/lerobot-mw-f5c9bc02'),
             '--output',str(fresh/'cases'),'--seeds']+list(map(str,plan['seeds']))+['--tasks']+plan['tasks'],
            fresh/'goals.log',env,source,plan['deadline_unix'])
    manifest=fresh/'cases/manifest.json';value=json.loads(manifest.read_text());assert value['passed']and len(value['rows'])==80
    assert all(digest(row['path'])==row['sha256']for row in value['rows'])
    save(fresh/'GOALS_APPROVED.json',dict(passed=True,manifest=str(manifest),sha256=digest(manifest),cases=80,
         expert_goal_valid=sum(x['expert_any_success']for x in value['rows']),expert_failures_retained=True))


def work(root,source,host,gpu):
    fresh=root/'fresh-v1';plan=json.loads((fresh/'plan.json').read_text())
    r=next(x for x in plan['recipes']if x['host']==host and x['gpu']==gpu);owner=r['wait_owner']
    while True:
        try:
            p=psutil.Process(owner['pid']);owned=p.is_running()and p.status()!=psutil.STATUS_ZOMBIE and p.cmdline()==owner['command']
        except psutil.NoSuchProcess:owned=False
        if not owned:break
        assert time.time()<plan['deadline_unix'];time.sleep(5)
    while not(fresh/'GOALS_APPROVED.json').exists():
        assert time.time()<plan['deadline_unix'];time.sleep(5)
    wait_gpu_release(gpu)
    env=environment(source,gpu);directory=fresh/f'{host}-gpu{gpu}';directory.mkdir()
    assert digest(Path(r['run'])/'model-step0040000/weights.pt')==r['candidate_sha256']
    assert digest(Path(r['baseline'])/'weights.pt')==r['baseline_sha256']
    for name,mode,parent,extra in [(r['arm'],'bridge',r['parent'],['--run',r['run'],'--step','40000']),
                                   (r['baseline_arm'],'cem',r['baseline'],[])]:
        output=directory/name
        command([PYTHON,'-B','-u',str(source/'experiments/metaworld_contact_20261106/evaluate.py'),
                 '--mode',mode,'--parent',parent,'--data',plan['data'],'--manifest',str(fresh/'cases/manifest.json'),
                 '--output',str(output),'--execute','3','--horizon','6']+extra+['--seeds']+list(map(str,plan['seeds']))+['--tasks']+plan['tasks'],
                directory/f'{name}.log',env,source,plan['deadline_unix'])
        assert json.loads((output/'summary.json').read_text())['passed']
    save(directory/'complete.json',dict(passed=True,no_sgd=True,scope=plan['scope']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--mode',choices=['prepare','goals','work'],required=True);p.add_argument('--host');p.add_argument('--gpu',type=int)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.source)
    elif a.mode=='goals':goals(a.root,a.source)
    else:work(a.root,a.source,a.host,a.gpu)
