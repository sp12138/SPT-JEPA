"""One bounded native100 controller pilot before one reserved continuation.

No timer,training change,or new policy learning. Existing active workers stay
on their immutable runtime. Only one unclaimed waiting pair is adopted.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import time
import traceback

from experiments.metaworld_baseline_scaling_20261108 import campaign as scale
from experiments.metaworld_baselines_20261107.campaign import BASE,PYTHON,env,execute,now,read,save,sha

ROOT = BASE / 'runs/metaworld_native100_pilot_20261108'
HARD = ['bin-picking-v3','door-open-v3','drawer-open-v3','peg-insert-side-v3','pick-place-v3','push-v3']


def prepare(helper):
    ROOT.mkdir(exist_ok=False)
    original = read(scale.PARENT / 'plan.json')
    native = read(BASE / 'runs/metaworld_baselines_20261107/native-controller/plan.json')
    for path, expected in native['source_hashes'].items():
        assert sha(path) == expected
    bank = read(scale.PARENT / 'manifests/intermediate-0.json')
    cases = []
    for shard in [0,1]:
        rows = [r for r in read(scale.PARENT / f'manifests/intermediate-{shard}.json')['rows'] if r['task'] in HARD]
        assert len(rows) == 6 and {r['task'] for r in rows} == set(HARD)
        cases.extend(rows)
        for label, selected in [('technical',rows[:1]),('formal',rows)]:
            save(ROOT / 'manifests' / f'{label}-{shard}.json',dict(bank,rows=selected))
    jobs = []
    for method, seed in [('lewm-idm',2),('h-jepa',2),('lewm-idm',3),('h-jepa',3)]:
        j = next(j for j in original['jobs'] if j['method']==method and j['seed']==seed)
        out = Path(j['output']); step=j['epoch_steps']['100']
        assert (out / 'complete.json').exists()
        expected = {(r['task'],r['seed']):r['sha256'] for r in cases}
        selected = []; hashes = set()
        for shard in [0,1]:
            report = read(out / f'evaluation-{step:07d}-{shard}/metrics.json')
            assert report['passed'] and report['model_state_before']==report['model_state_after']
            selected.extend(r for r in report['episodes'] if (r['task'],r['seed']) in expected)
            hashes.add(report['weights_sha256'])
        assert len(selected)==12 and len(hashes)==1
        assert {(r['task'],r['seed']):r['case_sha256'] for r in selected}==expected
        checkpoint=out/f'model-step{step:07d}'
        assert sha(checkpoint/'weights.pt')==next(iter(hashes))
        jobs.append(dict(name=j['name'],config=j['config'],checkpoint=str(checkpoint),
            weights_sha256=next(iter(hashes)),common_cem_rows=selected))
    save(ROOT/'plan.json',dict(jobs=jobs,helper=str(helper),evaluator=native['evaluator'],
        source_hashes=native['source_hashes'],cases=cases,formal_episodes=48,technical_episodes=8,
        max_seconds_after_claim=7200,deadline=original['deadline'],created_utc=now(),
        scope='Frozen100epoch4models x12same physicalcases=48model-case outcomes,12physicalcases. '
            'Developmental cohort previously inspected;not fresh confirmation/full50score. '
            'Native36macro/180nativeGD open-loop+learnedtrainactionqueue differsfromCEM '
            'H6/execute3 in horizon/solver/prior/feedback/compute jointly;not singlefactor causality. '
            'Scratch/RGB+executed-input/no rewards/state/taskID/optimalfutureaction labels.0SGD. '
            'Pilot runs on exactly1 released reserved pair,then resumes its normal continuation. '
            'Optional pilot failure doesnot discard a valid continuation.'))


def verify(report,manifest,weights):
    import numpy as np
    assert report['passed'] and report['model_updates']==0
    assert report['model_state_before']==report['model_state_after'] and report['weights_sha256']==weights
    expected={(r['task'],r['seed']):r['sha256'] for r in read(manifest)['rows']}
    assert {(r['task'],r['seed']):r['case_sha256'] for r in report['episodes']}==expected
    for r in report['episodes']:
        assert r['initial_rgb_replay']['passed'] and sha(r['trace'])==r['trace_sha256']
        with np.load(r['trace']) as trace:
            a,s=trace['actions'],trace['successes']
            assert a.shape==(r['steps'],4) and len(a)==len(s)<=report['budget']
            assert np.isfinite(a).all() and np.max(np.abs(a))<=1 and bool(s.any())==r['success']


def pilot(host,pair,reserved):
    plan=read(ROOT/'plan.json');helper=Path(plan['helper']);gpus=[2*pair,2*pair+1]
    deadline=min(plan['deadline'],time.time()+plan['max_seconds_after_claim'])
    save(ROOT/'OWNERSHIP.json',dict(host=host,pair=pair,gpus=gpus,pid=os.getpid(),
        reserved_scaling_job=reserved['name'],utc=now(),deadline=deadline))
    for path,expected in plan['source_hashes'].items():assert sha(path)==expected
    reports=[]
    for j in plan['jobs']:
        assert sha(Path(j['checkpoint'])/'weights.pt')==j['weights_sha256']
        for label,budget in [('technical',20),('formal',500)]:
            def one(shard,gpu):
                manifest=ROOT/'manifests'/f'{label}-{shard}.json'
                out=ROOT/j['name']/f'{label}-{shard}'
                out.parent.mkdir(parents=True,exist_ok=True)
                execute([PYTHON,plan['evaluator'],'--config',j['config'],'--checkpoint',j['checkpoint'],
                    '--manifest',str(manifest),'--output',str(out),'--budget',str(budget)],
                    out.parent/f'{label}-{shard}.log',env(helper,[gpu]),helper,deadline)
                r=read(out/'metrics.json');verify(r,manifest,j['weights_sha256']);return r
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(one,i,g) for i,g in enumerate(gpus)]
                current=[f.result() for f in futures]
            if label=='formal':
                rows=[r for d in current for r in d['episodes']]
                common={(r['task'],r['seed']):r for r in j['common_cem_rows']}
                pairs=[dict(task=r['task'],seed=r['seed'],native_success=r['success'],
                    cem_success=common[r['task'],r['seed']]['success']) for r in rows]
                reports.append(dict(name=j['name'],episodes=12,native_successes=sum(r['success'] for r in rows),
                    cem_successes=sum(r['success'] for r in common.values()),pairs=pairs))
                save(ROOT/'progress.json',dict(models=reports,utc=now()))
    save(ROOT/'summary.json',dict(passed=True,models=reports,formal_episodes=48,unique_physical_cases=12,
        technical_episodes=8,neural_updates=0,utc=now(),scope=plan['scope']))


def work(host,pair):
    plan=read(ROOT/'plan.json');deadline=plan['deadline']
    while time.time()<deadline:
        if scale.primary_busy(host,pair):time.sleep(20);continue
        j=scale.claim(host,pair)
        if j is None:time.sleep(20);continue
        # The ordinary scaling claim protects this pair from every standby
        # native worker even during CPU imports;no extra GPU ownership system.
        try:pilot(host,pair,j)
        except BaseException:
            save(ROOT/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc(),
                reserved_continuation_preserved=j['name']))
        save(ROOT/'CLOSED.json',dict(utc=now(),pilot_passed=(ROOT/'summary.json').exists(),
            handing_reserved_job_to_original_continuation=j['name']))
        scale.work(host,pair,preclaimed=j)
        return


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['prepare','work'],required=True)
    parser.add_argument('--helper',type=Path);parser.add_argument('--host');parser.add_argument('--pair',type=int)
    a=parser.parse_args()
    if a.mode=='prepare':prepare(a.helper)
    else:work(a.host,a.pair)
