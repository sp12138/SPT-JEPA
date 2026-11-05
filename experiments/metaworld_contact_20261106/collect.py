"""Read-only finite result collection;never launches training or simulation."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np


MANIP = ['bin-picking-v3','pick-place-v3','push-v3','drawer-open-v3','door-open-v3','peg-insert-side-v3']
TASKS = MANIP+['reach-v3','reach-wall-v3']


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def summary(rows, tasks=TASKS):
    groups={}
    for task in tasks:
        selected=[r for r in rows if r['task']==task]
        valid=[r for r in selected if r['expert_goal_success']]
        if selected:
            groups[task]=dict(successes=sum(r['success'] for r in selected),episodes=len(selected),
                success_percent=100*sum(r['success'] for r in selected)/len(selected),
                goal_valid_successes=sum(r['success'] for r in valid),goal_valid_episodes=len(valid),
                goal_valid_percent=100*sum(r['success'] for r in valid)/len(valid) if valid else None)
    complete=all(t in groups for t in MANIP)
    macro=float(np.mean([groups[t]['success_percent'] for t in MANIP])) if complete else None
    non_reach=[t for t in tasks if t not in ['reach-v3','reach-wall-v3']]
    return dict(tasks=groups,manip_macro_percent=macro,episodes=len(rows),
                alltask_macro_percent=float(np.mean([groups[t]['success_percent'] for t in tasks])) if all(t in groups for t in tasks) else None,
                non_reach_macro_percent=float(np.mean([groups[t]['success_percent'] for t in non_reach])) if all(t in groups for t in non_reach) else None)


def ci(rows):
    """Resample physicalcases within task,keeping all trained-seed repeats."""
    values=[];generator=np.random.default_rng(20261106);common_cases=None
    for task in MANIP:
        selected=[r for r in rows if r['task']==task]
        cases=sorted({r['seed'] for r in selected});seeds=sorted({r['training_seed'] for r in selected})
        if len(cases)<20 or len(selected)!=len(cases)*len(seeds):return None
        if common_cases is None:common_cases=cases
        if common_cases!=cases:return None
        matrix=np.asarray([[next(r['success'] for r in selected if r['seed']==case and r['training_seed']==s) for s in seeds] for case in cases],np.float64)
        values.append(matrix)
    # Reuse the same resampled physical seed across tasks,retaining any
    # common-random-number correlation and all trained-model repetitions.
    banks=[[i for i,s in enumerate(common_cases) if (s>=1000)==fresh] for fresh in [False,True]]
    draws=[generator.choice(bank,size=(5000,len(bank)),replace=True) for bank in banks if bank]
    ids=np.concatenate(draws,axis=1)
    sampled=np.stack(values)[:,ids].mean((0,2,3))*100
    low,high=np.percentile(sampled,[2.5,97.5])
    return dict(lower=float(low),upper=float(high),conditional_on_trained_models=True,
                bootstrap='5000shared-physical-seed blocks across all tasks/models,stratifiedbyprimary/freshbank;not repeated model-cases independently')


def collect(root):
    groups=[];identities=[]
    def add(phase,arm,step,files,tasks=TASKS):
        rows=[];models=[]
        for seed,path in files:
            if not path.exists():continue
            value=json.loads(path.read_text())
            assert value['passed'] and value['model_unchanged']
            if phase=='confirmation10k':
                proof=path.parent/'parent-removal-proof.json'
                if not proof.exists():continue
                assert json.loads(proof.read_text())['passed'] and not value['frozen_parent_used']
            models.append(dict(training_seed=seed,learner_sha256=value['learner_sha256'],summary=str(path)))
            for r in value['rows']:
                rows.append(dict(r,training_seed=seed))
        if rows:
            groups.append(dict(phase=phase,arm=arm,step=step,models=models,**summary(rows,tasks),
                               case_cluster95=ci(rows) if len(models)>=2 else None,rows=rows))
    for arm in ['local-mse','local-flow']:
        add('confirmation10k',arm,10000,[(seed,root/'confirmation10k'/f'{arm}-seed{seed}'/'summary.json') for seed in [1,2]])
    for phase in ['screen','adaptation-v1','long-v1','repeat-v1','vit-v1','vit-mse-v1','marginal-local-v1']:
        folder=root if phase=='screen' else root/phase
        pointer=folder/'active-bridge-plan.json'
        plan_file=Path(json.loads(pointer.read_text())['path']) if pointer.exists() else folder/'bridge-plan.json'
        if not plan_file.exists():continue
        plan=json.loads(plan_file.read_text());recipes=plan['recipes']
        for arm in sorted({r['arm'] for r in recipes}):
            selected=[r for r in recipes if r['arm']==arm]
            for step in [600,3000,6000,10000,20000,40000]:
                add(phase,arm,step,[(r['seed'],Path(r['run'])/f'eval-step{step:07d}'/'summary.json') for r in selected])
            for r in selected:
                run=Path(r['run']);status=run/'status.json';error=run/'execution-error.json'
                identities.append(dict(phase=phase,arm=arm,seed=r['seed'],gpu=r['gpu'],host=r['host'],run=str(run),
                    status=json.loads(status.read_text()) if status.exists() else None,
                    complete=(run/'complete.json').exists(),
                    effective_error=json.loads(error.read_text()) if error.exists() else None))
    for phase in ['fresh-v1','broad-v1']:
        folder=root/phase;pointer=folder/'plan.json'
        if not pointer.exists():continue
        plan=json.loads(pointer.read_text());files={}
        for r in plan['recipes']:
            directory=folder/f'{r["host"]}-gpu{r["gpu"]}'
            if phase=='fresh-v1':
                names=[(r['arm'],40000),(r['baseline_arm'],16480)]
                for arm,step in names:files.setdefault((arm,step),[]).append((r['training_seed'],directory/arm/'summary.json'))
            else:
                step=40000 if r['run'] else 16480
                files.setdefault((r['arm'],step),[]).append((r['training_seed'],directory/'evaluation/summary.json'))
            identities.append(dict(phase=phase,arm=r['arm'],seed=r['training_seed'],host=r['host'],gpu=r['gpu'],
                run=str(directory),complete=(directory/'complete.json').exists(),status=None,effective_error=None))
        for (arm,step),entries in sorted(files.items()):add(phase,arm,step,entries,plan['tasks'])
    # Baseline rows are from the original matched20epoch cohort,not a new run.
    previous=root.parent/'metaworld_improve_20261104/results/summary.json'
    old=json.loads(previous.read_text());baseline=[]
    for arm in ['tf-control','native-control']:
        rows=[dict(r,training_seed=r['train_seed']) for g in old['groups'] if g['step']==16480 and g['arm']==arm for r in g['observations']]
        baseline.append(dict(arm=arm,step=16480,source=str(previous),**summary(rows)))
    for g in groups:
        if g['phase']=='confirmation10k':
            for row in g['rows']:
                matches=[x for v in old['groups'] if v['step']==16480 and v['arm']=='tf-control' and v['task']==row['task']
                         for x in v['observations'] if x['train_seed']==row['training_seed'] and x['seed']==row['seed']]
                assert len(matches)==1 and matches[0]['case_sha256']==row['case_sha256']
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline=baseline,groups=groups,runs=identities,
         scope='Same independentLeRobotcorpus/split,cachedRGBgoals/physicalcases,500native any-success.Strictscratch/noexternalteacher,no optimal-action labels. '
               'Candidate explicitly adds factual action-reconstruction controller;not a claim that pureJEPAforecastloss alone improved. '
               'Knowncases;conditional2-4trainedmodels;differentarchitectures/optimizers/compute.Not officialJEPA-WMs/MT50/SOTA comparison.')
    save(root/'review/summary.json',result)
    compact=[{k:g[k] for k in ['phase','arm','step','models','tasks','manip_macro_percent','alltask_macro_percent','non_reach_macro_percent','episodes','case_cluster95']} for g in groups]
    save(root/'review/compact.json',dict(created_utc=result['created_utc'],baseline=baseline,groups=compact,runs=identities,scope=result['scope']))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--watch',action='store_true');a=p.parse_args()
    deadline=json.loads((a.root/'bridge-plan-v3.json').read_text())['deadline_unix']
    while True:
        value=collect(a.root)
        print(json.dumps(dict(utc=value['created_utc'],groups=len(value['groups']),active_runs=sum(not r['complete'] for r in value['runs']))),flush=True)
        final=[r for r in value['runs'] if r['phase'] in ['long-v1','repeat-v1','vit-v1','vit-mse-v1','marginal-local-v1','fresh-v1','broad-v1']]
        if not a.watch or (len(final)==32 and all(r['complete'] for r in final)) or time.time()>deadline:
            if a.watch:save(a.root/'review/HOLD.json',dict(finalized=True,expected_runs=32,all_complete=len(final)==32 and all(r['complete'] for r in final),no_automatic_next_round=True))
            break
        time.sleep(30)
