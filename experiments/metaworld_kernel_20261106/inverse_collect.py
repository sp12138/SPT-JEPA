"""Finite actual-trace collection for current-only training or frozen references."""
import argparse
import json
from pathlib import Path
import time
from experiments.metaworld_kernel_20261106.collect import save
from experiments.metaworld_kernel_20261106.campaign import digest


def collect(root):
    training=(root/'training-plan.json').exists();plan=json.loads((root/('training-plan.json'if training else'plan.json')).read_text());runs=[];groups={};cases={};models={}
    for recipe in plan['recipes']:
        out=Path(recipe['run']if training else recipe['output']);entry=dict(recipe);entry['complete']=(out/'complete.json').exists();entry['error']=json.loads((out/'execution-error.json').read_text())if(out/'execution-error.json').exists()else None
        for step in(plan['stages']if training else[40000]):
            path=out/(f'eval-step{step:07d}'if training else'pilot')/'summary.json'
            if not path.exists():continue
            value=json.loads(path.read_text());assert value['passed']and value['model_unchanged']
            assert models.setdefault((recipe['run'],step),value['model_sha256'])==value['model_sha256']
            if training:assert json.loads((out/f'audit-{step}.json').read_text())['passed']
            for row in value['rows']:
                assert cases.setdefault((row['task'],row['seed']),row['case_sha256'])==row['case_sha256']
                assert digest(Path(row['trace']))==row['trace_sha256']
            if step in[2,recipe.get('initial_step',0),recipe.get('initial_step',0)+2]:continue
            groups.setdefault((recipe['arm'],step),[]).extend(dict(training_seed=recipe['seed'],**r)for r in value['rows'])
        runs.append(entry)
    scores=[]
    for(arm,step),rows in sorted(groups.items()):
        tasks=sorted(set(r['task']for r in rows));manip=[r for r in rows if r['task']not in['reach-v3','reach-wall-v3']]
        valid=[r for r in rows if r['expert_goal_success']]
        scores.append(dict(arm=arm,step=step,trained_seeds=sorted(set(r['training_seed']for r in rows)),episodes=len(rows),
            manipulation_successes=sum(r['success']for r in manip),manipulation_episodes=len(manip),manipulation_percent=100*sum(r['success']for r in manip)/len(manip),
            goal_valid_episodes=len(valid),goal_valid_percent=100*sum(r['success']for r in valid)/len(valid)if valid else None,
            task_scores={t:dict(successes=sum(r['success']for r in rows if r['task']==t),episodes=sum(r['task']==t for r in rows))for t in tasks}))
    return dict(runs=runs,scores=scores,complete=sum(r['complete']for r in runs),errors=sum(r['error']is not None for r in runs),training=training,
        performance_goal_achieved=False,scope='Current-onlycandidateorlabelledfuturechunkbaseline;originalcorpus/commonphysicalcases,notpublicSOTA/equalFLOPs.Noclock/oraclefuturecommandsat inference.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();training=(a.root/'training-plan.json').exists();plan=json.loads((a.root/('training-plan.json'if training else'plan.json')).read_text())
    while True:
        result=collect(a.root);save(a.root/'results/summary.json',result)
        if result['complete']+result['errors']==len(plan['recipes'])or time.time()>plan['deadline_unix']:break
        time.sleep(20)
    print(json.dumps(dict(complete=result['complete'],errors=result['errors'],scores=result['scores'])),flush=True)
