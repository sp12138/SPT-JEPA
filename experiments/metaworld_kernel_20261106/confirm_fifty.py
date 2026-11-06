"""Pre-register1000freshphysicalcases and confirm the two40krepresentations."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from experiments.metaworld_contact_20261106.launch import environment,PYTHON,wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command
from experiments.metaworld_kernel_20261106.campaign import save,digest


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=False);(a.root/'results').mkdir();old=json.loads((a.screen/'plan.json').read_text());recipes=[];slot=0
    for host,seed in[('dev13',1),('dev15',2)]:
        waiting=json.loads((a.wait_parent/'plan.json').read_text())['recipes']
        for gpu,arm,parent,parent_arm in[(6,'absolute',a.baseline,'episodes'),(7,'directed',a.candidate,'directed-flow')]:
            wait=next(r for r in waiting if r['host']==host and r['gpu']==gpu)
            recipes.append(dict(host=host,gpu=gpu,seed=seed,arm=arm,run=str(parent/'training'/f'{parent_arm}-seed{seed}'),output=str(a.root/f'{arm}-seed{seed}'),
                wait_output=wait['output'],case_tasks=old['tasks'][slot::4],case_output=str(a.root/f'cases-slot{slot}')));slot+=1
    save(a.root/'plan.json',dict(source=str(a.source),recipes=recipes,tasks=old['tasks'],seeds=list(range(4042,4052))+list(range(5042,5052)),
        manifest=str(a.root/'cases/manifest.json'),policy_source=old['policy_source'],deadline_unix=time.time()+30*3600,new_model_updates=0,
        scope='1000NEWphysicalcases=50tasksx20seeds,pre-registeredBEFOREanyoutcomes.Allcases/expertgoalfailuresretained.'
              'Original40kabsolute/directedcurrent-macrocontrollers,2trainedseedseach;0newSGD.No teacher/actionoracle/state/reward/taskIDinputs.'
              'SameRGBsensor5/500anysuccess/freshownlatent/12flowsteps.NotcanonicalMT50/publicJEPA-WMs.Upcoming4new-seed120klearners/armuseEXACTthiscohort.'))


def merge(a,plan):
    rows=[]
    for r in plan['recipes']:
        f=Path(r['case_output'])/'manifest.json'
        if not f.exists():return False
        m=json.loads(f.read_text());assert m['passed'];rows+=m['rows']
    assert len(rows)==1000 and len({(r['task'],r['seed'])for r in rows})==1000
    old=[]
    for f in[Path('/efs/prescottsun/SPT-JEPA/runs/metaworld_eval_20261103/cases-v2/manifest.json'),Path('/efs/prescottsun/SPT-JEPA/runs/metaworld_current50_20261107/cases/manifest.json')]:old+=json.loads(f.read_text())['rows']
    vectors={}
    for r in old:vectors.setdefault(r['task'],[]).append(np.load(r['path'])['rand_vec'])
    for r in rows:
        assert digest(Path(r['path']))==r['sha256'];v=np.load(r['path'])['rand_vec'];assert not any(np.array_equal(v,x)for x in vectors[r['task']])
    out=a.root/'cases';out.mkdir(exist_ok=True);save(out/'manifest.tmp',dict(passed=True,rows=rows,tasks=plan['tasks'],seeds=plan['seeds'],expert_goal_successes=sum(r['expert_any_success']for r in rows),
        vectors_disjoint_previous_160_and_200=True,scope=plan['scope']));(out/'manifest.tmp').replace(out/'manifest.json');return True


def work(a):
    plan=json.loads((a.root/'plan.json').read_text());r=next(r for r in plan['recipes']if r['host']==a.host and r['gpu']==a.gpu);source=Path(plan['source']);out=Path(r['output']);out.mkdir()
    try:
        while not(Path(r['wait_output'])/'complete.json').exists():
            assert not(Path(r['wait_output'])/'execution-error.json').exists();assert time.time()<plan['deadline_unix'];time.sleep(5)
        wait_gpu_release(a.gpu);env=environment(source,a.gpu)
        command([PYTHON,'-B','-u',str(source/'experiments/metaworld_eval_20261103/prepare_cases.py'),'--source',plan['policy_source'],'--output',r['case_output'],
            '--tasks']+r['case_tasks']+['--seeds']+list(map(str,plan['seeds'])),out/'case-preparation.log',env,source,plan['deadline_unix'])
        if a.host=='dev13'and a.gpu==6:
            while not merge(a,plan):assert time.time()<plan['deadline_unix'];time.sleep(5)
        while not Path(plan['manifest']).exists():assert time.time()<plan['deadline_unix'];time.sleep(5)
        assert json.loads((Path(r['run'])/'audit-40000.json').read_text())['passed']
        base=[PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/inverse_evaluate.py'),'--run',r['run'],'--step','40000','--manifest',plan['manifest'],'--execute','1']
        command(base+['--output',str(out/'entry20'),'--budget','20','--tasks','bin-picking-v3','--seeds','4042'],out/'entry20.log',env,source,plan['deadline_unix'])
        command(base+['--output',str(out/'pilot'),'--tasks']+plan['tasks']+['--seeds']+list(map(str,plan['seeds'])),out/'pilot.log',env,source,plan['deadline_unix'])
        m=json.loads((out/'pilot/summary.json').read_text());assert m['passed']and m['model_unchanged']and len(m['rows'])==1000
        save(out/'complete.json',dict(passed=True,episodes=1000,new_model_updates=0))
    except BaseException as e:save(out/'execution-error.json',dict(error=repr(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','work'],required=True)
    for k in['root','source','screen','wait-parent','baseline','candidate']:p.add_argument('--'+k,type=Path)
    p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args();prepare(a)if a.mode=='prepare'else work(a)
