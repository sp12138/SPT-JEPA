"""Finite frozen-factorial collection with actual trace/case/NN identities."""
import argparse
import json
from pathlib import Path
import time
from experiments.metaworld_kernel_20261106.campaign import save,digest


def collect(root):
    plan=json.loads((root/'plan.json').read_text());runs=[];groups={};cases={};models={}
    for recipe in plan['recipes']:
        out=Path(recipe['output']);row=dict(recipe);error=out/'execution-error.json';complete=out/'complete.json'
        row['complete']=complete.exists();row['error']=json.loads(error.read_text())if error.exists()else None
        endpoint=out/'pilot/summary.json'
        if endpoint.exists():
            value=json.loads(endpoint.read_text());assert value['passed']and value['model_unchanged']
            assert models.setdefault(recipe['seed'],value['model_sha256'])==value['model_sha256']
            for r in value['rows']:
                assert cases.setdefault((r['task'],r['seed']),r['case_sha256'])==r['case_sha256']
                assert digest(Path(r['trace']))==r['trace_sha256']
            groups.setdefault((recipe['metric'],recipe['prior']),[]).extend(dict(training_seed=recipe['seed'],**r)for r in value['rows'])
        runs.append(row)
    scores=[]
    for (metric,prior),rows in sorted(groups.items()):
        manipulation=[r for r in rows if r['task']not in['reach-v3','reach-wall-v3']]
        scores.append(dict(metric=metric,control_prior=prior,trained_seeds=sorted(set(r['training_seed']for r in rows)),episodes=len(rows),
            manipulation_successes=sum(r['success']for r in manipulation),manipulation_episodes=len(manipulation),manipulation_percent=100*sum(r['success']for r in manipulation)/len(manipulation),
            per_task={task:dict(successes=sum(r['success']for r in rows if r['task']==task),episodes=sum(r['task']==task for r in rows))for task in sorted(set(r['task']for r in rows))}))
    return dict(runs=runs,scores=scores,complete=sum(r['complete']for r in runs),errors=sum(r['error']is not None for r in runs),new_updates=0,
        independent_trained_parents=2,unique_physical_cases=len(cases),performance_goal_achieved=False,
        scope='Frozenoriginal-data2500models;2trainedparents x4controller variants.Sharedphysicalcases,notindependentreplications.Nodataaugmentation/inverse/actionreferences.'
              'SameCNNrolloutbudget;reachablemetricaddsmeasuredFP64SVDOps.Noautomaticfullpromotionfromsmallpilot.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();plan=json.loads((a.root/'plan.json').read_text())
    while True:
        value=collect(a.root);save(a.root/'results/summary.json',value)
        if value['complete']+value['errors']==8 or time.time()>plan['deadline_unix']:break
        time.sleep(15)
    print(json.dumps(dict(complete=value['complete'],errors=value['errors'],scores=value['scores'])),flush=True)
