"""Audit one completed evaluation milestone,independent of outer worker state."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE=Path('/efs/prescottsun/SPT-JEPA')
ROOT=BASE/'runs/metaworld_baseline_scaling_20261108'
HARD={'bin-picking-v3','door-open-v3','drawer-open-v3','peg-insert-side-v3','pick-place-v3','push-v3'}


def read(p):return json.loads(p.read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()


def main(name,epoch):
    job=next(j for j in read(ROOT/'plan.json')['jobs'] if j['name']==name);step=job['epoch_steps'][str(epoch)]
    out=Path(job['output']);expected={(r['task'],r['seed']):r['sha256'] for r in read(BASE/'runs/metaworld_input_inverse_20261107_v1/followup/cases/manifest.json')['rows']}
    reports=[read(out/f'evaluation-{step:07d}-{i}/metrics.json') for i in [0,1]]
    assert all(r['passed'] and r['model_updates']==0 and r['model_state_before']==r['model_state_after'] for r in reports)
    weights={r['weights_sha256'] for r in reports};assert len(weights)==1 and sha(out/f'model-step{step:07d}/weights.pt')==next(iter(weights))
    rows=[r for d in reports for r in d['episodes']];assert len(rows)==1000
    assert {(r['task'],r['seed']):r['case_sha256'] for r in rows}==expected
    for r in rows:
        assert r['initial_rgb_replay']['passed'] and sha(r['trace'])==r['trace_sha256']
        with np.load(r['trace']) as t:
            a,s=t['actions'],t['successes'];assert a.shape==(r['steps'],4) and len(a)==len(s)<=500
            assert np.isfinite(a).all() and np.max(np.abs(a))<=1 and bool(s.any())==r['success']
    audit=read(out/f'audit-{step:07d}.json');assert audit['passed'] and audit['step']==step and audit['weights_unchanged']
    groups={}
    for task in sorted({r['task'] for r in rows}):
        selected=[r for r in rows if r['task']==task];assert len(selected)==20
        groups[task]=dict(episodes=20,successes=sum(r['success'] for r in selected),percent=5*sum(r['success'] for r in selected))
    assert len(groups)==50
    d=dict(passed=True,name=name,native_epoch=epoch,step=step,phase_only=True,actual_trace_files_verified=1000,
        unique_physical_cases=1000,weights_sha256=next(iter(weights)),macro50=sum(v['percent'] for v in groups.values())/50,
        hard6_successes=sum(r['success'] for r in rows if r['task'] in HARD),groups=groups,
        model_updates=0,neural_models_loaded=0,render_calls=0,new_physical_episodes=0,
        scope='Same1000physicalcasesas100epoch;phaseevaluationvalidity,notoutercampaigncompletion/independent1000newcases. '
            'CPUfileauditonly;RGBflagscheckedfromoriginalevaluation,notnewrender/physicalreplay. '
            '100+200nativeLRrestartcontinuation,not scratch300singlecosine orSOTAclaim.')
    path=ROOT/'audits'/f'{name}-epoch{epoch}.json';path.parent.mkdir(exist_ok=True)
    with path.open('x') as f:json.dump(d,f,indent=2)
    print(json.dumps({k:d[k] for k in ['passed','name','native_epoch','macro50','hard6_successes','actual_trace_files_verified']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--epoch',type=int,choices=[200,300],required=True)
    a=p.parse_args();main(a.name,a.epoch)
