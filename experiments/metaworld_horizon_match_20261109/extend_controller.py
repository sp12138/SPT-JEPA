"""Complete the missing H60/execute15 matched frozen-controller comparison."""
import argparse
from pathlib import Path
import subprocess
import time
import traceback
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,now,environment
from experiments.metaworld_transition_only_20261109.campaign import command

ROOT=BASE/'runs/metaworld_horizon_match_20261109'
EXT=ROOT/'extended-controller'


def prepare():
    plan=read(ROOT/'plan.json');EXT.mkdir(exist_ok=False);jobs=[]
    cards=[('dev13',0),('dev13',2),('dev15',0),('dev15',6)]
    for j,(host,gpu) in zip(plan['jobs'],cards):
        weights=Path(j['output'])/'model-step0012000/weights.pt'
        assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256']
        assert read(Path(j['output'])/'audit-0012000.json')['passed']
        jobs.append(dict(j,host=host,gpu=gpu,output=str(EXT/'evaluation'/j['name'])))
    save(EXT/'plan.json',dict(utc=now(),source=plan['source'],jobs=jobs,deadline=min(plan['deadline'],time.time()+2*3600),
        manifest=str(BASE/'runs/metaworld_transition_only_20261109/manifests/final.json'),step=12000,
        forecast_horizon_native=60,execute_native=15,formal_episodes=128,world_updates=0,
        trigger='OriginalH15inverse10000seed1/Push6042H60e15succeeds155native;SAMEweights/caseH30e15fails. Singletondevelopmentalnotrobustgain. '
            'AddmissingcombinedH60trained/H60plannedcomparison againstsameupdateH15control,2parentsx32samecases/model. '
            'ExistingH30e15/H60e2endpointsunchanged;notrepeatinganySGD orcompletedendpoint.',
        scope='Previouslyinspecteddevelopmentalcohort,notuntouchedconfirmation/publicSOTA;no goalpath/policytraining oraction-targetreference.',
        automatic_long_promotion=False))


def work(name):
    p=read(EXT/'plan.json');j=next(j for j in p['jobs'] if j['name']==name);s=Path(p['source'])
    try:
        command([PYTHON,'-m','experiments.metaworld_horizon_match_20261109.evaluate','--config',j['config'],'--step','12000',
            '--manifest',p['manifest'],'--output',j['output'],'--horizon','60','--execute','15'],
            EXT/f'{name}.log',environment(s,j['gpu']),s,p['deadline'],cap=7200)
        d=read(Path(j['output'])/'metrics.json');assert d['passed'] and d['model_state_before']==d['model_state_after']
        save(EXT/f'DONE-{name}.json',dict(passed=True,utc=now(),model_updates=0,episodes=len(d['episodes'])))
    except BaseException:
        save(EXT/f'ERROR-{name}.json',dict(utc=now(),traceback=traceback.format_exc()));raise


def launch(host):
    p=read(EXT/'plan.json');s=Path(p['source']);workers=[]
    for j in p['jobs']:
        if j['host']!=host:continue
        with (EXT/f'worker-{j["name"]}.log').open('x') as f:
            child=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'work','--name',j['name']],cwd=s,
                env=dict(environment(s,j['gpu']),CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(j,pid=child.pid))
    save(EXT/f'launch-{host}.json',dict(utc=now(),host=host,workers=workers));print(workers,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch','work']);p.add_argument('--host');p.add_argument('--name');a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='launch':launch(a.host)
    else:work(a.name)
