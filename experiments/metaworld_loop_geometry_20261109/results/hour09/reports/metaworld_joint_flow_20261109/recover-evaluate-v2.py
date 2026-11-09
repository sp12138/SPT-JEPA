"""Canonical frozen recovery: preserve96 episodes, finish48, add48 precision controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,now,environment
from experiments.metaworld_transition_only_20261109.campaign import command,HARD

PARENT=BASE/'runs/metaworld_joint_flow_20261109'
ROOT=PARENT/'recovery-v2'


def evaluate(config,step,manifest,output,precision,budget):
    import torch
    from experiments.metaworld_joint_flow_20261109 import evaluate as original
    if precision=='world32':
        from experiments.metaworld_transition_only_20261109.model import FactualRecurrent
        observe=FactualRecurrent.observe;advance=original.advance
        def fp32_observe(self,hidden,image):
            with torch.autocast('cuda',enabled=False):return observe(self,hidden.float(),image.float())
        def fp32_advance(parent,hidden,action):
            with torch.autocast('cuda',enabled=False):return advance(parent,hidden.float(),action.float())
        FactualRecurrent.observe=fp32_observe;original.advance=fp32_advance
    else:assert precision=='bf16'
    old_save=original.save
    def metadata(path,value):
        if Path(path).name=='metrics.json':
            value.update(world_compute_precision=precision,image_renderer_precision='bf16',
                precision_wrapper_sha256=sha(__file__),optimizer_updates=0,
                precision_scope='Samefrozenweights/cases/CEM64x4/H60/e15/10imagequeries;world32changesonlyobserver/knowncontroltransitionprecision. Not equalwallclock/FLOPs.')
        old_save(path,value)
    original.save=metadata;original.run(config,step,manifest,output,budget)


def verify(d):
    assert d['passed'] and d['state_before']==d['state_after']
    for r in d['episodes']:
        assert sha(r['trace'])==r['trace_sha256'] and r['initial_rgb_replay']['passed']
        with np.load(r['trace']) as z:
            assert z['actions'].shape==(r['steps'],4) and bool(z['successes'].any())==r['success']
            assert np.isfinite(z['actions']).all() and np.abs(z['actions']).max()<=1 and r['steps']<=500


def prepare():
    p=read(PARENT/'plan.json');ROOT.mkdir(exist_ok=False);inherited=[];evaluations=[]
    for e in p['evaluations']:
        if e['step']!=2000:inherited.append(e);continue
        for precision in ['bf16','world32']:
            name=e['name']+'-'+precision
            evaluations.append(dict(e,name=name,precision=precision,output=str(ROOT/'evaluation'/name)))
    assert len(inherited)==96 and len(evaluations)==96
    save(ROOT/'plan.json',dict(utc=now(),source=p['source'],source_hashes=p['source_hashes'],helper_sha256=sha(__file__),
        jobs=p['jobs'],inherited=inherited,evaluations=evaluations,deadline=p['deadline'],
        original_errors_retained=True,original_error_count=25,
        source_NN_loss_optimizer_data_LR_precision_unchanged_in_mean_training_recovery=True,
        recovery='Mean1resumeslastfull1500,atmost99unsavedupdatesreplayed. Initialpositivegradientproofretained;finitezero gradientsinmaturebatchesreportednotfatal. Fiveother2000learnersreusedwithoutSGD.',
        new_precision_test='Frozen2000weightsBF16vsFP32world/observer;imageheadBF16both. Gradientauditfixedweights1500/actualstep1600batch:BF16controlzero,FP32worldrestores1.9e-5. Not aperformanceclaim.',
        inherited_episodes=96,new_episodes=96,combined_episodes=192,physical_cases=8,original_trained_parents=2,
        automatic_long_promotion=False,automatic_expansion=False))


def preflight(name,gpu):
    p=read(ROOT/'plan.json');j=next(x for x in p['jobs'] if x['name']==name);s=Path(p['source']);out=ROOT/'technical'/name
    manifest=BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json'
    case=next(c for c in read(manifest)['rows'] if c['task']=='push-v3');manifest=ROOT/'manifests/push-v3.json'
    if not manifest.exists():save(manifest,dict(rows=[case]))
    command([PYTHON,str(Path(__file__).resolve()),'evaluate','--config',j['config'],'--step',500,'--manifest',manifest,
        '--output',out,'--precision','bf16','--budget',20],out.with_suffix('.log'),environment(s,gpu),s,p['deadline'])
    d=read(out/'metrics.json');verify(d);reference=read(PARENT/'evaluation'/f'{name}-s500-push-v3'/'metrics.json')
    a,b=d['episodes'][0],reference['episodes'][0]
    with np.load(a['trace']) as x,np.load(b['trace']) as y:
        assert np.array_equal(x['actions'],y['actions'][:20]) and np.array_equal(x['successes'],y['successes'][:20])
    assert a['planning'][0]['first_bank_prefix_sha256']==b['planning'][0]['first_bank_prefix_sha256']
    save(out/'PREFLIGHT.json',dict(passed=True,utc=now(),actual20_native_actions_success_firstbank_bitexact=True))


def work(host,gpu):
    p=read(ROOT/'plan.json');s=Path(p['source'])
    assert sha(__file__)==p['helper_sha256'] and {str(f.relative_to(s)):sha(f) for f in s.rglob('*.py')}==p['source_hashes']
    while time.time()<p['deadline']:
        if (ROOT/'STOP').exists():return
        selected=None
        for e in p['evaluations']:
            out=Path(read(e['config'])['output'])
            if not (out/'AUDIT-0002000.json').exists():continue
            claim=ROOT/'claims'/(e['name']+'.json');claim.parent.mkdir(exist_ok=True)
            try:
                with claim.open('x') as f:json.dump(dict(host=host,gpu=gpu,pid=os.getpid(),utc=now()),f)
            except FileExistsError:continue
            selected=e;break
        if selected is None:
            if all((ROOT/'claims'/(e['name']+'.json')).exists() for e in p['evaluations']):return
            time.sleep(5);continue
        e=selected
        try:
            command([PYTHON,str(Path(__file__).resolve()),'evaluate','--config',e['config'],'--step',2000,
                '--manifest',e['manifest'],'--output',e['output'],'--precision',e['precision']],Path(e['output']).with_suffix('.log'),environment(s,gpu),s,p['deadline'])
            verify(read(Path(e['output'])/'metrics.json'))
        except BaseException:
            save(ROOT/'errors'/(e['name']+'.json'),dict(utc=now(),traceback=traceback.format_exc()));raise


def launch(host):
    p=read(ROOT/'plan.json');s=Path(p['source']);workers=[]
    assert all(read(ROOT/'technical'/j['name']/'PREFLIGHT.json')['passed'] for j in p['jobs'])
    for gpu in range(8):
        with (ROOT/f'worker-{host}-{gpu}.log').open('x') as f:
            c=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'work','--host',host,'--gpu',str(gpu)],cwd=s,
                env=dict(environment(s,gpu),CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(gpu=gpu,pid=c.pid))
    save(ROOT/f'launch-{host}.json',dict(host=host,utc=now(),workers=workers));print(host,workers,flush=True)


def collect():
    p=read(ROOT/'plan.json');cache={}
    while time.time()<p['deadline']:
        for inherited,items in [(True,p['inherited']),(False,p['evaluations'])]:
            for e in items:
                key=('inherited-' if inherited else '')+e['name'];path=Path(e['output'])/'metrics.json'
                if key in cache or not path.exists():continue
                d=read(path);verify(d)
                expected={(c['task'],c['seed']):c['sha256'] for c in read(e['manifest'])['rows']}
                assert {(c['task'],c['seed']):c['case_sha256'] for c in d['episodes']}==expected
                cache[key]=dict(inherited=inherited,experiment=e,metrics=d,metadata_sha256=sha(path))
        groups=__import__('collections').defaultdict(list)
        for x in cache.values():
            d=x['metrics'];key=f'{d["mode"]}-step{d["step"]}-{d.get("world_compute_precision","bf16")}'
            groups[key].extend(d['episodes'])
        groups={k:dict(episodes=len(v),hard_successes=sum(r['success'] for r in v if r['task'] in HARD),hard_episodes=sum(r['task'] in HARD for r in v),
            by_task={t:dict(successes=sum(r['success'] for r in v if r['task']==t),episodes=sum(r['task']==t for r in v)) for t in sorted({r['task'] for r in v})}) for k,v in groups.items()}
        errors=[str(x) for x in ROOT.glob('errors/*.json')]
        save(ROOT/'results/summary.json',dict(utc=now(),combined_endpoints=len(cache),planned_combined_endpoints=192,
            inherited_endpoints=sum(x['inherited'] for x in cache.values()),new_endpoints=sum(not x['inherited'] for x in cache.values()),
            groups=groups,errors=errors,old_errors_retained_count=25,model_updates=0,
            scope='Twooriginaltrainedparents,eightdevelopmentalcases. Precisionchangescompute/numerics,notlabels. NoSOTA/untouchedconfirmation/longpromotion.'))
        if len(cache)==192:
            save(ROOT/'EPISODE_INDEX.json',{k:dict(experiment=x['experiment'],metadata_sha256=x['metadata_sha256'],inherited=x['inherited']) for k,x in cache.items()})
            save(ROOT/'CLOSED.json',dict(passed=not errors,utc=now(),all192trace_SHA_action_success_verified=True,automatic_long_promotion=False));return
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','preflight','launch','work','evaluate','collect'])
    p.add_argument('--host');p.add_argument('--gpu',type=int);p.add_argument('--name');p.add_argument('--step',type=int,default=2000)
    for key in ['config','manifest','output']:p.add_argument('--'+key,type=Path)
    p.add_argument('--precision',choices=['bf16','world32'],default='bf16');p.add_argument('--budget',type=int,default=500);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='preflight':preflight(a.name,a.gpu)
    elif a.mode=='launch':launch(a.host)
    elif a.mode=='work':work(a.host,a.gpu)
    elif a.mode=='evaluate':evaluate(a.config,a.step,a.manifest,a.output,a.precision,a.budget)
    else:collect()
