"""Unseen physical-case extension to all50tasks,with labelled image-goal protocol."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from experiments.metaworld_contact_20261106.launch import environment,PYTHON,wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command
from experiments.metaworld_kernel_20261106.campaign import save,digest


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=False);(a.root/'results').mkdir();recipes=[]
    policy_source='/efs/prescottsun/SPT-JEPA/sources/lerobot-mw-f5c9bc02'
    mapping=json.loads((Path(policy_source)/'src/lerobot/envs/metaworld_config.json').read_text())['TASK_POLICY_MAPPING'];tasks=sorted(mapping);assert len(tasks)==50
    slot=0
    for host,seed in[('dev13',1),('dev15',2)]:
        for gpu,objective,parent,wait_arm in[(1,'flow',a.flow_parent,'phase0'),(5,'mse',a.mse_parent,'episodes')]:
            run=parent/'training'/f'episodes-seed{seed}';cfg=json.loads((run/'config.json').read_text());assert cfg['model']['chunk']==1
            recipes.append(dict(host=host,gpu=gpu,seed=seed,arm=objective,run=str(run),output=str(a.root/f'{objective}-seed{seed}'),
                wait_run=str(a.mse_parent/'training'/f'{wait_arm}-seed{seed}'),case_tasks=tasks[slot::4],case_output=str(a.root/f'cases-slot{slot}')));slot+=1
    save(a.root/'plan.json',dict(source=str(a.source),recipes=recipes,tasks=tasks,case_seeds=[2042,2043,3042,3043],manifest=str(a.root/'cases/manifest.json'),
        policy_source=policy_source,deadline_unix=time.time()+24*3600,new_model_updates=0,
        scope='All50-taskimage-goalextension4NEWphysicalcases/task x2current-onlytrainedseeds;notstandardtask-IDMT50/publicJEPA-WMs.'
            'No newSGD.GoalexpertconstructsgoalRGBonly,neveractionsforlearner/controller.Allgoalconstructionfailuresretained/reportgoal-validsubset.'
            'Tasklist/seedsfrozenbeforeoutcomes,old8taskcasesnotreused.No success/novelty claimfromscreenalone.'))


def merge(root,plan):
    rows=[]
    for r in plan['recipes']:
        p=Path(r['case_output'])/'manifest.json'
        if not p.exists():return False
        m=json.loads(p.read_text());assert m['passed'];rows+=m['rows']
    assert len(rows)==200 and len({(r['task'],r['seed'])for r in rows})==200
    old=json.loads(Path('/efs/prescottsun/SPT-JEPA/runs/metaworld_eval_20261103/cases-v2/manifest.json').read_text());oldvectors={}
    for r in old['rows']:oldvectors.setdefault(r['task'],[]).append(np.load(r['path'])['rand_vec'])
    for r in rows:
        assert digest(Path(r['path']))==r['sha256'];v=np.load(r['path'])['rand_vec']
        assert not any(np.array_equal(v,x)for x in oldvectors.get(r['task'],[]))
    out=root/'cases';out.mkdir(exist_ok=True);save(out/'manifest.json',dict(passed=True,rows=rows,seeds=plan['case_seeds'],tasks=plan['tasks'],
        all_case_vectors_disjoint_old_160=True,expert_any_success_count=sum(r['expert_any_success']for r in rows),new_model_updates=0,
        scope=plan['scope']));return True


def work(a):
    plan=json.loads((a.root/'plan.json').read_text());r=next(r for r in plan['recipes']if r['host']==a.host and r['gpu']==a.gpu);source=Path(plan['source']);out=Path(r['output']);out.mkdir()
    try:
        while not(Path(r['wait_run'])/'complete.json').exists():
            assert not(Path(r['wait_run'])/'execution-error.json').exists();assert time.time()<plan['deadline_unix'];time.sleep(10)
        wait_gpu_release(a.gpu);env=environment(source,a.gpu)
        command([PYTHON,'-B','-u',str(source/'experiments/metaworld_eval_20261103/prepare_cases.py'),'--source',plan['policy_source'],'--output',r['case_output'],
            '--seeds']+list(map(str,plan['case_seeds']))+['--tasks']+r['case_tasks'],out/'prepare-cases.log',env,source,plan['deadline_unix'])
        if a.host=='dev13'and a.gpu==1:
            while not merge(a.root,plan):assert time.time()<plan['deadline_unix'];time.sleep(10)
        while not Path(plan['manifest']).exists():assert time.time()<plan['deadline_unix'];time.sleep(10)
        assert json.loads((Path(r['run'])/'audit-40000.json').read_text())['passed']
        base=[PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/inverse_evaluate.py'),'--run',r['run'],'--step','40000','--manifest',plan['manifest']]
        command(base+['--output',str(out/'entry20'),'--budget','20','--tasks','bin-picking-v3','--seeds',str(plan['case_seeds'][0])],out/'entry20.log',env,source,plan['deadline_unix'])
        command(base+['--output',str(out/'pilot'),'--seeds']+list(map(str,plan['case_seeds']))+['--tasks']+plan['tasks'],out/'pilot.log',env,source,plan['deadline_unix'])
        m=json.loads((out/'pilot/summary.json').read_text());assert m['passed']and m['model_unchanged']and len(m['rows'])==200
        save(out/'complete.json',dict(passed=True,episodes=200,new_model_updates=0))
    except BaseException as error:save(out/'execution-error.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True)
    for k in['root','source','flow-parent','mse-parent']:p.add_argument('--'+k,type=Path)
    p.add_argument('--gpu',type=int);p.add_argument('--host');a=p.parse_args();prepare(a)if a.mode=='prepare'else work(a)
