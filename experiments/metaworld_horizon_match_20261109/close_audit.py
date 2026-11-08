"""Read-only physical-trace closeout; no model loads or further optimization."""
from pathlib import Path
import numpy as np
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,sha,now
from experiments.metaworld_transition_only_20261109.campaign import HARD

ROOT=BASE/'runs/metaworld_horizon_match_20261109'


if __name__=='__main__':
    plan=read(ROOT/'plan.json');participants={j['name']:j for j in plan['jobs']}
    groups={};traces=[];identities={}
    for name,job in participants.items():
        assert read(Path(job['output'])/'audit-0012000.json')['passed']
        assert read(Path(job['output'])/'DONE.json')['passed']
    paths=list((ROOT/'evaluation').glob('*/metrics.json'))+list((ROOT/'extended-controller/evaluation').glob('*/metrics.json'))
    assert len(paths)==12
    for path in sorted(paths):
        d=read(path);assert d['passed'] and d['model_state_before']==d['model_state_after']
        rows=d['episodes'];assert len(rows)==(32 if 'extended-controller' in path.parts else 8)
        key=str(path.parent.relative_to(ROOT));counts={}
        for r in rows:
            assert sha(r['trace'])==r['trace_sha256'] and r['initial_rgb_replay']['passed']
            with np.load(r['trace']) as z:
                assert z['actions'].shape==(r['steps'],4) and z['successes'].shape==(r['steps'],)
                assert bool(z['successes'].any())==r['success'] and r['steps']<=500
                assert np.isfinite(z['actions']).all() and np.abs(z['actions']).max()<=1
            identity=(r['task'],r['seed'])
            assert identity not in identities or identities[identity]==r['case_sha256']
            identities[identity]=r['case_sha256']
            c=counts.setdefault(r['task'],dict(successes=0,episodes=0))
            c['episodes']+=1;c['successes']+=int(r['success'])
            traces.append(dict(path=r['trace'],sha256=r['trace_sha256']))
        groups[key]=dict(by_task=counts,hard_successes=sum(int(r['success']) for r in rows if r['task'] in HARD),
            hard_episodes=sum(r['task'] in HARD for r in rows),model_state_sha256=d['model_state_after'],
            horizon_native=d['horizon_native'],execute_native=d['execute_native'])
    assert len(traces)==192 and len(identities)==32
    hard_successes=sum(g['hard_successes'] for g in groups.values())
    assert hard_successes==0
    d=read(ROOT/'factual-12000/SUMMARY.json');assert d['passed']
    forecasts={m['name']:dict(open60=m['aggregate']['factual_open']['60'],
        observe2_60=m['aggregate']['observe2']['60'],observe15_60=m['aggregate']['observe15']['60']) for m in d['models']}
    save(ROOT/'CLOSEOUT_AUDIT.json',dict(passed=True,utc=now(),formal_endpoints=12,formal_episodes=192,
        unique_physical_cases=32,original_trained_parents=2,groups=groups,traces=traces,
        forecasts=forecasts,model_updates=0,physics_steps=0,new_model_loads=0,
        decision='CLOSED_NO_PROMOTION',hard_successes=hard_successes,
        scope='All192actualtracesverified;repeatedphysicalcases/parents/controllers,NOT192independentcases. '
            'MatchedH60lossimproveslongforecastbutdoesnotimprovehardcontrol;reachingregressionunderH60/e15. '
            'No longcontinuation,goalpathlearning,action-policytargetsornewbaselinebudget.'))
    print('PASS 192 actual traces; CLOSED_NO_PROMOTION',flush=True)
