"""CPU-only audit of all18 finished100epoch models and actual18,000 traces."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

BASE=Path('/efs/prescottsun/SPT-JEPA')
ROOT=BASE/'runs/metaworld_baselines_native_20261108'
OUT=BASE/'runs/metaworld_baseline_scaling_20261108/final100-audit'
HARD={'bin-picking-v3','door-open-v3','drawer-open-v3','peg-insert-side-v3','pick-place-v3','push-v3'}


def read(p):return json.loads(p.read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def check(j,bank,common):
    out=Path(j['output']);step=j['epoch_steps']['100']
    assert (out/'complete.json').exists() and not (out/'error.json').exists()
    rows=[];weights=set();states=set();reports={}
    for i in [0,1]:
        p=out/f'evaluation-{step:07d}-{i}/metrics.json';d=read(p)
        assert d['passed'] and d['model_state_before']==d['model_state_after']
        assert d['model_updates']==0 and d['budget']==500
        weights.add(d['weights_sha256']);states.add(d['model_state_before']);rows.extend(d['episodes'])
        reports[str(p)]=sha(p)
    assert len(rows)==1000 and len(weights)==len(states)==1
    assert {(r['task'],r['seed']):r['case_sha256'] for r in rows}==bank
    assert sha(out/f'model-step{step:07d}/weights.pt')==next(iter(weights))
    for r in rows:
        assert r['initial_rgb_replay']['passed'] and sha(r['trace'])==r['trace_sha256']
        with np.load(r['trace']) as trace:
            a,s=trace['actions'],trace['successes']
            assert a.shape==(r['steps'],4) and len(s)==len(a)<=500
            assert np.isfinite(a).all() and np.max(np.abs(a))<=1 and bool(s.any())==r['success']
    groups={}
    for task in sorted({r['task'] for r in rows}):
        selected=[r for r in rows if r['task']==task];assert len(selected)==20
        groups[task]=dict(successes=sum(r['success'] for r in selected),episodes=20,
            percent=100*sum(r['success'] for r in selected)/20)
    assert len(groups)==50
    curve={}
    for epoch in [20,50,100]:
        e=j['epoch_steps'][str(epoch)]
        stage=[r for i in [0,1] for r in read(out/f'evaluation-{e:07d}-{i}/metrics.json')['episodes']]
        selected=[r for r in stage if (r['task'],r['seed']) in common]
        assert len(selected)==100 and {(r['task'],r['seed']):r['case_sha256'] for r in selected}==common
        curve[str(epoch)]=sum(r['success'] for r in selected)
    return dict(name=j['name'],method=j['method'],seed=j['seed'],step=step,verified_traces=1000,
        weights_sha256=next(iter(weights)),metrics_sha256=reports,groups=groups,
        macro50=sum(v['percent'] for v in groups.values())/50,matched100_curve=curve,
        hard_successes=[dict(task=r['task'],seed=r['seed'],trace=r['trace'],trace_sha256=r['trace_sha256'])
            for r in rows if r['task'] in HARD and r['success']])


def main():
    OUT.mkdir(exist_ok=False)
    try:
        cases=read(BASE/'runs/metaworld_input_inverse_20261107_v1/followup/cases/manifest.json')['rows']
        bank={(r['task'],r['seed']):r['sha256'] for r in cases};assert len(bank)==1000
        for r in cases:assert sha(r['path'])==r['sha256']
        common={(r['task'],r['seed']):r['sha256'] for i in [0,1] for r in read(ROOT/f'manifests/intermediate-{i}.json')['rows']}
        assert len(common)==100
        jobs=read(ROOT/'plan.json')['jobs'];assert len(jobs)==18
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            rows=list(pool.map(lambda j:check(j,bank,common),jobs))
        methods={}
        for method in sorted({r['method'] for r in rows}):
            selected=[r for r in rows if r['method']==method];assert {r['seed'] for r in selected}=={1,2,3}
            scores=[r['macro50'] for r in selected]
            methods[method]=dict(mean_macro50=float(np.mean(scores)),sample_std_across3trained_seeds=float(np.std(scores,ddof=1)),
                seed_scores={str(r['seed']):r['macro50'] for r in selected},
                hard6_successes=sum(len(r['hard_successes']) for r in selected),hard6_model_case_outcomes=360,
                matched100_mean_curve={str(e):float(np.mean([r['matched100_curve'][str(e)] for r in selected])) for e in [20,50,100]})
        result=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),methods=methods,models=rows,
            actual_traces_sha_actions_success_cases_rgb_and_frozen_weights_verified=18000,unique_physical_cases=1000,
            expert_goal_failed_physical_cases=sum(not r['expert_any_success'] for r in cases),
            hard6_model_case_outcomes=2160,hard6_unique_physical_cases=120,hard6_successes=sum(len(r['hard_successes']) for r in rows),
            model_updates=0,neural_models_loaded=0,new_physical_episodes=0,render_calls=0,
            scope='Native100epoch fixed-LeRobot-data commonCEM scratch comparison. Nativewindows/capacity/images/FLOPs differ;not compute-matched/publicJEPA-WMs/MT50RLscores. '
                'Three independent trainingseeds crossed SAME1000physicalcases;not18000independentphysicalcases. '
                'All500stepany-success,allgoalconstructionfailuresretained. Matched100curves developmental,not full1000case estimates. '
                'No SOTA/novelty/strongmanipulation claim.')
        (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ['passed','methods','hard6_successes','expert_goal_failed_physical_cases']}))
    except BaseException:
        (OUT/'ERROR.json').write_text(json.dumps({'traceback':traceback.format_exc()},indent=2)+'\n')
        raise


if __name__=='__main__':main()
