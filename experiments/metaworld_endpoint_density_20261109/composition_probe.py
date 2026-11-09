"""Frozen controlled-imagination pilot; no new training or expert paths."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,now,environment
from experiments.metaworld_transition_only_20261109.campaign import command,HARD

ORIGINAL=BASE/'runs/metaworld_endpoint_density_20261109'
ROOT=ORIGINAL/'composition-v1'


def prepare(source):
    original=read(ORIGINAL/'plan.json');ROOT.mkdir(exist_ok=False)
    jobs=[j for j in original['jobs'] if j['mode'] in ['uniform','highnoise']];evaluations=[]
    cases=read(BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json')['rows']
    for case in cases:save(ROOT/'manifests'/(case['task']+'.json'),dict(rows=[case]))
    for kind in ['short','long','free_time']:
        for j in jobs:
            for case in cases:
                name=f'{kind}-{j["name"]}-{case["task"]}'
                evaluations.append(dict(name=name,kind=kind,job=j['name'],config=j['config'],
                    manifest=str(ROOT/'manifests'/(case['task']+'.json')),output=str(ROOT/'evaluation'/name)))
    save(ROOT/'plan.json',dict(utc=now(),source=str(source),jobs=jobs,evaluations=evaluations,deadline=time.time()+7200,
        source_hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},
        formal_episodes=96,model_updates=0,trained_parents=2,physical_cases=8,
        design='Same4frozenCFMheads2000,parent12000. Threecontrollers:singleH60;composedH120;composedH120withunknownterminaltime60or120marginalized. '
            'EverycontrollerCEM64x4/12actionknots/execute15/native500/49rendererqueriespercandidate. H60hold5,H120hold10;notahorizon-onlyorequalbackboneFLOPscomparison. '
            'SingleH60uses48transitionKLqueries+1endpoint. H120uses2samplesof16-stepreverseGaussianmidpoint(17queryeach),2conditionalgoalbounds5each,midpointgoalbound5:49total. '
            'SAMEdeclared16stepchainforgenerationandgoalbound. TwoimaginedfutureobservationsaregeneratedfromknowncandidatecontrolsWITHOUTgoalconditioning. '
            'ParentobserverassimilatesitsOWNgeneratedRGB,thenadvancesnext60nativecontrols. GoalONLYscorescandidateoutcomes;nofutureGT/expertpaths/actionlabels. '
            'FixedH120recordsmidpointgoaldiagnosticonly;free_timeintegratesgoal-time60/120withuniformprior. MonteCarloboundapproximations,notexactlikelihood.',
        rationale='Actualworldmodelsuseactions(95.7..97.3percentrealizedinputmatch),betterRGBpredictionstillweakcontrol. '
            'Testfree-terminal-timeboundaryinferenceandlongerforwardworldcompositioninsteadofaddinglabels/policytraining.',
        automatic_long_promotion=False,automatic_expansion=False))


def evaluate(config,manifest,output,kind):
    from . import evaluate as original
    from . import composition
    composition.KIND=kind;original.plan=composition.plan;original.RENDERER_QUERIES=49
    original.FORECAST_HORIZON=60 if kind=='short' else 120
    original.CONTROL_HOLD=original.FORECAST_HORIZON//12;original.PLANNER_OVERRIDE='controlled_observation_composition_'+kind
    original.PLANNER_SCOPE=('Frozenfactualworldcomposition:16stepreverseGaussianimagechain,same49rendererqueries/candidate. '
        'OwnsampledimaginedmidpointRGBonly,NOdesiredgoalinputtomidpointgeneration. Knowncandidatecontrolsandpastobservationsonly. '
        'H60hold5vsH120hold10,12actionknots;not equalbackboneFLOPsorhorizon-onlycausality. '
        'MonteCarloimagedensity/time-mixtureproxy,notexactNLL. Previouslyinspected8physicalcases/2originalparents,notpublicSOTA.')
    original.run(config,2000,manifest,output,500)


def work(host,gpu):
    p=read(ROOT/'plan.json');s=Path(p['source'])
    assert {str(x.relative_to(s)):sha(x) for x in s.rglob('*.py')}==p['source_hashes']
    while time.time()<p['deadline']:
        if (ROOT/'STOP').exists():return
        e=None
        for item in p['evaluations']:
            claim=ROOT/'claims'/(item['name']+'.json');claim.parent.mkdir(exist_ok=True)
            try:
                with claim.open('x') as f:__import__('json').dump(dict(host=host,gpu=gpu,pid=os.getpid(),utc=now()),f)
            except FileExistsError:continue
            e=item;break
        if e is None:return
        try:
            command([PYTHON,'-m','experiments.metaworld_endpoint_density_20261109.composition_probe','evaluate','--config',e['config'],
                '--manifest',e['manifest'],'--output',e['output'],'--kind',e['kind']],Path(e['output']).with_suffix('.log'),
                environment(s,gpu),s,p['deadline'])
            d=read(Path(e['output'])/'metrics.json');assert d['passed'] and d['parent_state_before']==d['parent_state_after'] and d['head_state_before']==d['head_state_after']
        except BaseException:
            save(ROOT/'errors'/(e['name']+'.json'),dict(utc=now(),traceback=traceback.format_exc()));raise


def launch(host):
    p=read(ROOT/'plan.json');s=Path(p['source']);workers=[]
    for gpu in range(8):
        with (ROOT/f'worker-{host}-{gpu}.log').open('x') as f:
            child=subprocess.Popen([PYTHON,'-m','experiments.metaworld_endpoint_density_20261109.composition_probe','work','--host',host,'--gpu',str(gpu)],
                cwd=s,env=dict(environment(s,gpu),CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(gpu=gpu,pid=child.pid))
    save(ROOT/f'launch-{host}.json',dict(host=host,utc=now(),workers=workers));print(host,workers,flush=True)


def collect():
    p=read(ROOT/'plan.json');cache={}
    while time.time()<p['deadline']:
        for e in p['evaluations']:
            path=Path(e['output'])/'metrics.json'
            if e['name'] in cache or not path.exists():continue
            d=read(path);assert d['passed'] and d['parent_state_before']==d['parent_state_after'] and d['head_state_before']==d['head_state_after']
            expected={(r['task'],r['seed']):r['sha256'] for r in read(e['manifest'])['rows']}
            assert {(r['task'],r['seed']):r['case_sha256'] for r in d['episodes']}==expected
            for r in d['episodes']:
                assert sha(r['trace'])==r['trace_sha256'] and r['initial_rgb_replay']['passed']
                with np.load(r['trace']) as z:
                    assert z['actions'].shape==(r['steps'],4) and bool(z['successes'].any())==r['success']
                    assert np.isfinite(z['actions']).all() and np.abs(z['actions']).max()<=1 and r['steps']<=500
            cache[e['name']]=dict(experiment=e,metrics=d)
        groups=__import__('collections').defaultdict(list)
        for x in cache.values():groups[x['experiment']['kind']+'-'+x['metrics']['mode']].extend(x['metrics']['episodes'])
        reports={k:dict(episodes=len(v),hard_successes=sum(r['success'] for r in v if r['task'] in HARD),hard_episodes=sum(r['task'] in HARD for r in v),
            by_task={t:dict(successes=sum(r['success'] for r in v if r['task']==t),episodes=sum(r['task']==t for r in v)) for t in sorted({r['task'] for r in v})}) for k,v in groups.items()}
        errors=[str(x) for x in ROOT.glob('errors/*.json')]
        save(ROOT/'results/summary.json',dict(utc=now(),endpoints=len(cache),planned_endpoints=96,groups=reports,errors=errors,model_updates=0))
        if len(cache)==96:
            save(ROOT/'CLOSED.json',dict(passed=not errors,utc=now(),automatic_long_promotion=False));return
        time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch','work','evaluate','collect']);p.add_argument('--source',type=Path)
    p.add_argument('--host');p.add_argument('--gpu',type=int);p.add_argument('--kind',choices=['short','long','free_time'])
    for key in ['config','manifest','output']:p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.source)
    elif a.mode=='launch':launch(a.host)
    elif a.mode=='work':work(a.host,a.gpu)
    elif a.mode=='evaluate':evaluate(a.config,a.manifest,a.output,a.kind)
    else:collect()
