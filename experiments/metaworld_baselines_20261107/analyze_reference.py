"""Verify and summarize the two-parent full-50 reference, never count shards as seeds."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path('/efs/prescottsun/SPT-JEPA/runs/metaworld_baselines_20261107')


def main():
    from experiments.metaworld_baselines_20261107.reference import save
    plan=json.loads((ROOT/'reference/plan.json').read_text())
    bank=json.loads(Path(plan['source_manifest']).read_text())
    expected={(r['task'],r['seed']):r for r in bank['rows']}
    records={1:[],2:[]};seen={1:set(),2:set()};banks={};checked=0
    for job in plan['jobs']:
        out=ROOT/'reference/workers'/job['name']
        assert (out/'complete.json').is_file(),job['name']
        report=json.loads((out/'formal/metrics.json').read_text())
        assert report['passed'] and report['model_updates']==0
        assert report['model_state_before']==report['model_state_after']
        assert report['weights_sha256']==job['weights_sha256']
        for row in report['episodes']:
            key=row['task'],row['seed'];assert key not in seen[job['seed']]
            seen[job['seed']].add(key)
            assert row['case_sha256']==expected[key]['sha256']
            assert row['initial_rgb_replay']['passed']
            trace=Path(row['trace']);assert hashlib.sha256(trace.read_bytes()).hexdigest()==row['trace_sha256']
            actual=np.load(trace);a=actual['actions'];s=actual['successes']
            assert a.shape==(row['steps'],4) and len(s)==row['steps']
            assert np.isfinite(a).all() and np.max(np.abs(a))<=1.
            assert bool(s.any())==row['success']==row['any_success']
            assert bool(s[-1])==row['final_success'] and row['steps']<=500
            if key in banks:assert banks[key]==row['first_bank_sha256']
            else:banks[key]=row['first_bank_sha256']
            records[job['seed']].append(dict(row,any_success_first100=bool(s[:100].any())))
            checked+=1
    assert all(s==set(expected) for s in seen.values())
    tasks=sorted({t for t,s in expected});groups={}
    for task in tasks:
        item={}
        for seed,rows in records.items():
            r=[row for row in rows if row['task']==task]
            item[str(seed)]=dict(episodes=len(r),successes=sum(row['success'] for row in r),
                success_percent=100*np.mean([row['success'] for row in r]),
                first100_any_success_percent=100*np.mean([row['any_success_first100'] for row in r]),
                goal_construction_failures=sum(not row['expert_goal_success'] for row in r))
        item['mean_percent']=float(np.mean([item[str(seed)]['success_percent'] for seed in [1,2]]))
        groups[task]=item
    manip=['bin-picking-v3','door-open-v3','drawer-open-v3','peg-insert-side-v3','pick-place-v3','push-v3']
    save(ROOT/'reference/results/summary.json',dict(passed=True,trained_models=2,shards=32,model_updates=0,
        actual_verified_physical_traces=checked,common_case_count=1000,tasks=50,
        macro50_percent=float(np.mean([r['mean_percent'] for r in groups.values()])),
        macro50_by_training_seed={str(seed):float(np.mean([r[str(seed)]['success_percent'] for r in groups.values()])) for seed in [1,2]},
        six_previous_manipulation_percent=float(np.mean([groups[t]['mean_percent'] for t in manip])),
        tasks_with_at_least_one_success=int(sum(r['mean_percent']>0 for r in groups.values())),groups=groups,
        scope='Two historical20epoch scratch NativeLeWM parents on1000fresh cases each; '
            'not new baseline panel training/convergence/published JEPA-WMs or canonical MT50. '
            'First100 any-success is trace-derived, not100step final-success. Physical contexts reused across two learners.'))


if __name__=='__main__':main()
