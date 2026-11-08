"""Two frozen models,16formalepisodes;no optimization of learned parameters."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,now,environment
from experiments.metaworld_transition_only_20261109.campaign import command,HARD

ROOT=BASE/'runs/metaworld_sobolev_goal_20261109'
OLD=BASE/'runs/metaworld_transition_only_20261109'


def prepare(source):
    assert read(BASE/'reports/metaworld_sobolev_goal_20261109/CPU_AUDIT_v1.json')['passed'];ROOT.mkdir(exist_ok=False)
    jobs=[dict(seed=seed,host='dev13',gpu=gpu,config=str(OLD/f'training/h15-inverse1-seed{seed}/config.json'),
        baseline=str(OLD/f'evaluation/h15-inverse1-seed{seed}-s10000-H30-e15/metrics.json')) for seed,gpu in [(1,0),(2,2)]]
    save(ROOT/'plan.json',dict(utc=now(),source=str(source),jobs=jobs,deadline=time.time()+3600,
        manifest=str(OLD/'manifests/screen.json'),technical=str(OLD/'manifests/technical.json'),
        source_hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},
        hypothesis='RGBL2hasflatcostfornonoverlappingobjecttranslations. FixedDCTweight1/(1+k^2+l^2)couplesseparatedpixelswithoutfittingagoalhead. '
            'Toyplateau proof passes;factualrank214/240vsRGB216/240 isNEUTRAL,noperformanceevidence. Oneboundedphysicalprobeonly.',
        comparison='SamefrozenH15inverseparents10000/seed1,2;CEM64x4/H30/execute15/500native/same8physicalcases,seeds/bounds/firstbanks. '
            'Pixelstandard20stepactualaction/successparitybeforechangedcost;reuseactualsamecasebaselineformaltraces.16newformalepisodes,0world/headupdates.',
        allocation='2additionalfrozenprocessesco-locatedondev13gpu0/2,30percentmemorycap;oldworkersuntouched.24physicalGPUs,notextraGPUs.',
        automatic_expansion=False,novelty_or_SOTA_claim=False))
    print('Prepared2frozenmetricprobes',flush=True)


def work(seed):
    p=read(ROOT/'plan.json');j=next(j for j in p['jobs'] if j['seed']==seed);s=Path(p['source']);env=environment(s,j['gpu'])
    out=ROOT/f'seed{seed}';out.mkdir(exist_ok=False)
    try:
        while not Path(j['baseline']).exists():assert time.time()<p['deadline'];time.sleep(5)
        baseline=read(j['baseline']);assert baseline['passed'] and baseline['model_state_before']==baseline['model_state_after']
        def call(kind,manifest,dest,budget,log):
            command([PYTHON,'-m','experiments.metaworld_sobolev_goal_20261109.evaluate','--config',j['config'],
                '--manifest',manifest,'--output',dest,'--kind',kind,'--budget',str(budget)],out/log,env,s,p['deadline'],cap=3600)
        tech=out/'technical';call('pixel',p['technical'],tech,20,'technical.log');d=read(tech/'metrics.json');e=d['episodes'][0]
        reference=next(r for r in baseline['episodes'] if (r['task'],r['seed'])==(e['task'],e['seed']))
        assert sha(reference['trace'])==reference['trace_sha256']
        with np.load(reference['trace']) as a,np.load(e['trace']) as b:
            assert np.array_equal(a['actions'][:len(b['actions'])],b['actions']) and np.array_equal(a['successes'][:len(b['successes'])],b['successes'])
        assert e['planning'][0]['first_bank_prefix_sha256']==reference['planning'][0]['first_bank_prefix_sha256']
        save(out/'PREFLIGHT.json',dict(passed=True,utc=now(),actual_standard_actions_success_firstbank_exact=True,world_updates=0,head_updates=0))
        result=out/'spatial';call('spatial',p['manifest'],result,500,'spatial.log');spatial=read(result/'metrics.json')
        assert spatial['passed'] and spatial['model_state_before']==baseline['model_state_before']==spatial['model_state_after']
        bycase={(r['task'],r['seed']):r for r in baseline['episodes']};pairs=[]
        for row in spatial['episodes']:
            ref=bycase[(row['task'],row['seed'])];assert row['case_sha256']==ref['case_sha256']
            assert sha(row['trace'])==row['trace_sha256'] and sha(ref['trace'])==ref['trace_sha256']
            assert row['planning'][0]['first_bank_prefix_sha256']==ref['planning'][0]['first_bank_prefix_sha256']
            with np.load(row['trace']) as z:
                assert bool(z['successes'].any())==row['success'] and z['actions'].shape==(row['steps'],4)
                assert np.isfinite(z['actions']).all() and np.max(np.abs(z['actions']))<=1
            pairs.append(dict(task=row['task'],case_seed=row['seed'],case_sha256=row['case_sha256'],
                pixel_success=ref['success'],spatial_success=row['success'],pixel_trace=ref['trace'],spatial_trace=row['trace']))
        hard=[r for r in pairs if r['task'] in HARD]
        save(out/'DONE.json',dict(passed=True,utc=now(),training_seed=seed,pairs=pairs,hard=dict(episodes=len(hard),
            pixel_successes=sum(r['pixel_success'] for r in hard),spatial_successes=sum(r['spatial_success'] for r in hard)),
            world_model_updates=0,head_updates=0,sourceNN_unchanged=True))
    except BaseException:
        save(out/f'ERROR-{os.getpid()}.json',dict(utc=now(),traceback=traceback.format_exc()));raise


def launch():
    p=read(ROOT/'plan.json');s=Path(p['source']);workers=[]
    for j in p['jobs']:
        with (ROOT/f'worker-seed{j["seed"]}.log').open('x') as f:
            child=subprocess.Popen([PYTHON,'-m','experiments.metaworld_sobolev_goal_20261109.probe','work','--seed',str(j['seed'])],
                cwd=s,env=dict(environment(s,j['gpu']),CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(j,pid=child.pid))
    save(ROOT/'launch.json',dict(utc=now(),host='dev13',workers=workers));print(workers,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch','work']);p.add_argument('--source',type=Path);p.add_argument('--seed',type=int)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.source)
    elif a.mode=='launch':launch()
    else:work(a.seed)
