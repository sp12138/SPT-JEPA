"""Audit physical floor controls and compare exactly shared cases to the reference."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/efs/prescottsun/SPT-JEPA/runs/metaworld_baselines_20261107')


def main():
    from experiments.metaworld_baselines_20261107.reference import save
    rows=[]
    for shard in range(16):
        report=json.loads((ROOT/'floors'/f'shard{shard:02}/summary.json').read_text())
        assert report['passed'] and report['policy_inputs']==[]
        rows.extend(report['rows'])
    assert len(rows)==3000
    expected=json.loads(Path('/efs/prescottsun/SPT-JEPA/runs/metaworld_input_inverse_20261107_v1/followup/cases/manifest.json').read_text())['rows']
    cases={(c['task'],c['seed']):c for c in expected};seen=set()
    for row in rows:
        key=row['task'],row['seed'];ident=(*key,row['policy']);assert ident not in seen;seen.add(ident)
        assert row['case_sha256']==cases[key]['sha256']
        trace=Path(row['trace']);assert hashlib.sha256(trace.read_bytes()).hexdigest()==row['trace_sha256']
        actual=np.load(trace);a=actual['actions'];s=actual['successes']
        assert len(a)==len(s)==row['steps'] and a.shape[1:]==(4,)
        assert np.isfinite(a).all() and np.max(np.abs(a))<=1.
        assert bool(s.any())==row['success'] and bool(s[:100].any())==row['first100_any_success']
    reference=json.loads((ROOT/'reference/results/summary.json').read_text())
    groups={};policies=['zero','uniform','train-mean']
    for task in sorted({r['task'] for r in rows}):
        groups[task]={}
        for policy in policies:
            chosen=[r for r in rows if r['task']==task and r['policy']==policy];assert len(chosen)==20
            groups[task][policy]=dict(episodes=20,successes=sum(r['success'] for r in chosen),
                success_percent=100*float(np.mean([r['success'] for r in chosen])))
        groups[task]['lewm20epoch_mean_percent']=reference['groups'][task]['mean_percent']
    save(ROOT/'floors/summary.json',dict(passed=True,episodes=3000,actual_verified_traces=3000,
        macro50_percent={policy:float(np.mean([g[policy]['success_percent'] for g in groups.values()])) for policy in policies},
        lewm20epoch_macro50_percent=reference['macro50_percent'],groups=groups,model_updates=0,
        scope='Same1000physicalcases,500nativeanysuccess. Floors are goal-blind, not trained policies. '
            'LeWM uses two historical20epoch trained parents; floors share cases across parents and must not be counted as2000independent contexts. '
            'Each floor is reported separately; no task-wise best-floor policy is constructed.'))


if __name__=='__main__':main()
