"""Four frozen matched models, paired with their completed white-CEM traces."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,now,environment
from experiments.metaworld_transition_only_20261109.campaign import command,HARD
from .evaluate import mixing

ROOT=BASE/'runs/metaworld_correlated_search_20261109'
MATCHED=BASE/'runs/metaworld_horizon_match_20261109'


def prepare(source):
    torch.set_num_threads(2)
    transform=mixing('cpu');covariance=transform.double()@transform.double().T
    assert torch.allclose(covariance.diag(),torch.ones(12,dtype=torch.float64),atol=2e-7,rtol=0)
    assert torch.linalg.eigvalsh(covariance).min()>0
    generator=torch.Generator().manual_seed(20261109)
    raw=torch.randn(100000,12,4,generator=generator)
    colored=torch.einsum('ij,nja->nia',transform,raw)
    assert colored.mean((0,2)).abs().max()<.01
    assert (colored.var((0,2),unbiased=False)-1).abs().max()<.015
    adjacent=float(torch.diag(covariance,1).mean())
    assert adjacent>.5
    ROOT.mkdir(exist_ok=False)
    old=read(MATCHED/'extended-controller/plan.json');jobs=[]
    for j in old['jobs']:
        reference=Path(j['output'])/'metrics.json';d=read(reference)
        assert d['passed'] and d['model_state_before']==d['model_state_after']
        assert len(d['episodes'])==32
        jobs.append(dict(name=j['name'],host=j['host'],gpu=j['gpu'],config=j['config'],reference=str(reference)))
    save(ROOT/'CPU_AUDIT.json',dict(passed=True,utc=now(),covariance=covariance.tolist(),
        gaussian_marginal_variance_min=float(colored.var((0,2),unbiased=False).min()),
        gaussian_marginal_variance_max=float(colored.var((0,2),unbiased=False).max()),adjacent_correlation=adjacent,
        action_examples_or_goal_data_fitted=False,model_updates=0))
    save(ROOT/'plan.json',dict(utc=now(),source=str(source),jobs=jobs,deadline=time.time()+7200,
        manifest=old['manifest'],technical=str(BASE/'runs/metaworld_transition_only_20261109/manifests/technical.json'),
        source_hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},formal_episodes=128,
        scope='FrozenH15/H60loss12000x2originalparents. Same32physicalcases/terminalRGBcost/64particlesx4iterations/held5/H60/execute15/native500. '
            'ChangeONLYfixedtimecovariance;N(0,1)per-timeper-dimensionmarginalretainedbeforeclipping. '
            'Matchedwhitebaselinesalreadycomplete;4actual20stepstandardparitygatesbeforecoloredformal. '
            'Previouslyinspecteddevelopmentalcases;notuntouchedconfirmation/publicSOTA/equaltrainingFLOPs.',
        rationale='Perfectworldversuslearnedworlddiagnosticandobservedcontactfailuressuggesttestingtrajectorysearchseparately. '
            'TemporallycorrelatedplanningisapublishediCEMprinciple;thisfixedDCTvariantisNOTcanonicaliCEMorclaimednovel. '
            'Spectrum1/(1+k^2),DCTreflectingboundaries,unitdiagonalnormalizationarefixedanalytically;noaction/goal/expertdatamarginalfitting.',
        reference='https://proceedings.mlr.press/v155/pinneri21a.html',
        allocation='Co-locatedfrozenjobsonthe4now-completedextended-controllercards;original24workersuntouched.24physicalGPUs.',
        model_updates=0,expert_action_proposals=False,automatic_expansion=False,automatic_long_promotion=False))


def work(name):
    plan=read(ROOT/'plan.json');j=next(j for j in plan['jobs'] if j['name']==name)
    source=Path(plan['source']);env=environment(source,j['gpu']);out=ROOT/name;out.mkdir(exist_ok=False)
    try:
        reference=read(j['reference']);bycase={(r['task'],r['seed']):r for r in reference['episodes']}
        def call(kind,manifest,dest,budget):
            command([PYTHON,'-m','experiments.metaworld_correlated_search_20261109.evaluate','--config',j['config'],
                '--manifest',manifest,'--output',str(dest),'--kind',kind,'--budget',str(budget)],
                out/(kind+'.log'),env,source,plan['deadline'],cap=7200)
        tech=out/'technical';call('white',plan['technical'],tech,20)
        for e in read(tech/'metrics.json')['episodes']:
            old=bycase[(e['task'],e['seed'])];assert sha(old['trace'])==old['trace_sha256']
            with np.load(old['trace']) as a,np.load(e['trace']) as b:
                assert np.array_equal(a['actions'][:len(b['actions'])],b['actions'])
                assert np.array_equal(a['successes'][:len(b['successes'])],b['successes'])
            assert e['planning'][0]['first_bank_prefix_sha256']==old['planning'][0]['first_bank_prefix_sha256']
        save(out/'PREFLIGHT.json',dict(passed=True,utc=now(),standard_actual_actions_success_bank_exact=True,model_updates=0))
        dest=out/'correlated';call('correlated',plan['manifest'],dest,500)
        result=read(dest/'metrics.json');assert result['passed']
        assert result['model_state_before']==result['model_state_after']==reference['model_state_before']
        pairs=[]
        for row in result['episodes']:
            old=bycase[(row['task'],row['seed'])];assert row['case_sha256']==old['case_sha256']
            assert sha(row['trace'])==row['trace_sha256'] and sha(old['trace'])==old['trace_sha256']
            with np.load(row['trace']) as z:
                assert z['actions'].shape==(row['steps'],4) and bool(z['successes'].any())==row['success']
                assert np.isfinite(z['actions']).all() and np.max(np.abs(z['actions']))<=1
            assert row['planning'][0]['first_bank_prefix_sha256']!=old['planning'][0]['first_bank_prefix_sha256']
            pairs.append(dict(task=row['task'],seed=row['seed'],case_sha256=row['case_sha256'],
                white_success=old['success'],correlated_success=row['success']))
        hard=[r for r in pairs if r['task'] in HARD]
        save(out/'DONE.json',dict(passed=True,utc=now(),pairs=pairs,hard=dict(episodes=len(hard),
            white_successes=sum(r['white_success'] for r in hard),correlated_successes=sum(r['correlated_success'] for r in hard)),
            model_updates=0,all_actual_traces_verified=True,learned_parameters_unchanged=True))
    except BaseException:
        save(out/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise


def launch(host):
    p=read(ROOT/'plan.json');source=Path(p['source']);workers=[]
    for j in p['jobs']:
        if j['host']!=host:continue
        with (ROOT/(j['name']+'-worker.log')).open('x') as f:
            child=subprocess.Popen([PYTHON,'-m','experiments.metaworld_correlated_search_20261109.probe','work','--name',j['name']],
                cwd=source,env=dict(environment(source,j['gpu']),CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
                stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(j,pid=child.pid))
    save(ROOT/('launch-'+host+'.json'),dict(utc=now(),host=host,workers=workers));print(workers,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch','work'])
    p.add_argument('--source',type=Path);p.add_argument('--host');p.add_argument('--name');a=p.parse_args()
    if a.mode=='prepare':prepare(a.source)
    elif a.mode=='launch':launch(a.host)
    else:work(a.name)
