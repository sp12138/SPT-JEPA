"""Finite native-controller queue; primary training/evaluation always has priority."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback

from experiments.metaworld_baselines_20261107.campaign import ROOT,BASE,PYTHON,read,save,sha,env,execute,now
from experiments.metaworld_baselines_20261107.safe_metadata import read


def prepare(helper):
    main=read(ROOT/'plan.json');jobs=[]
    bank=read(BASE/'runs/metaworld_input_inverse_20261107_v1/followup/cases/manifest.json')
    for method in ['lewm-idm','h-jepa']:
        for seed in [1,2,3]:
            for shard in range(16):
                name=f'{method}-seed{seed}-shard{shard:02}'
                manifest=ROOT/'native-controller/manifests'/(name+'.json')
                save(manifest,dict(bank,rows=bank['rows'][shard::16]))
                jobs.append(dict(name=name,method=method,seed=seed,manifest=str(manifest)))
    native=BASE/'runs/hjepa_droid_20261105/public/stable_worldmodel/solver'
    evaluator=BASE/'reports/metaworld_baselines_20261107/native-hjepa-eval-v2.py'
    if (ROOT/'native-controller/plan.json').exists():
        assert not (ROOT/'native-controller/claims').exists()
        save(ROOT/'native-controller/plan-v1-before-flat-interface-fix.json',read(ROOT/'native-controller/plan.json'))
    save(ROOT/'native-controller/plan.json',dict(helper=str(helper),evaluator=str(evaluator),jobs=jobs,deadline=main['deadline'],
        models=6,episodes=6000,source_hashes={str(p):sha(p) for p in [native/'solver.py',native/'gd.py',
            BASE/'sources/metaworld-baselines-20261107/hjepa/h_jepa/hierarchical_solver.py',
            evaluator]},
        scope='Full native DROID GD/hierarchy transfer,36macro/180native open-loop blocks,then replan,500nativeanysuccess. '
            'Same1000physicalcases/model as shared-CEM final,reported separately because feedback/compute differ. '
            'No future expert controls; native learned action priors only. No extraSGD.'))


def owned_alive(pid):
    try:return 'metaworld_baselines_20261107' in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    except OSError:return False


def primary_busy(host,gpu):
    pair=gpu//2
    formal=BASE/'runs/metaworld_baselines_native_20261108/plan.json'
    if formal.exists():
        plan=read(formal);formal_root=formal.parent
        for job in plan['jobs']:
            if job['kind']=='external':continue
            if not (formal_root/'claims'/(job['name']+'.json')).exists():return True
        for owner in plan['legacy_owners']:
            if owner['host']==host and owner['pair']==pair and owned_alive(owner['pid']):return True
        for job in plan['jobs']:
            if job['kind']=='external':continue
            claim=read(formal_root/'claims'/(job['name']+'.json'))
            if claim['host']==host and claim['pair']==pair:
                out=Path(job['output'])
                if not (out/'complete.json').exists() and not (out/'error.json').exists():return True
        s=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True)
        return any(v.strip().isdigit() for v in s.splitlines())
    recovery=ROOT/'THREEHOST_RECOVERY.json'
    if recovery.exists():
        for transfer in read(recovery)['transfers']:
            f=ROOT/'threehost-recovery/claims'/(transfer['name']+'.json')
            if not f.exists():return True
            claim=read(f);out=Path(transfer['output'])
            if claim['host']==host and claim['pair']==pair and not (out/'complete.json').exists() and not (out/'error.json').exists():return True
    for file in [ROOT/f'launch-v3-{host}.json',ROOT/f'launch-device-v4-{host}.json']:
        if not file.exists():continue
        for record in read(file)['workers']:
            if record['pair']==pair and owned_alive(record['pid']):return True
    main=read(ROOT/'plan.json')
    # Never consume a pair needed by a still-unclaimed primary SGD recipe.
    if any(not (ROOT/main.get('claims_dir','claims')/(j['name']+'.json')).exists() for j in main['jobs']):return True
    for job in main['jobs']:
        c=ROOT/main.get('claims_dir','claims')/(job['name']+'.json')
        if not c.exists():continue
        claim=read(c)
        if claim['host']==host and claim['pair']==pair:
            out=Path(job['output'])
            if not (out/'complete.json').exists() and not (out/'error.json').exists():return True
    # Protect against orphaned CUDA children and other users' GPU processes.
    result=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True)
    return any(line.strip().isdigit() for line in result.splitlines())


def claim():
    main=read(ROOT/'plan.json');plan=read(ROOT/'native-controller/plan.json')
    for job in plan['jobs']:
        parent=next(j for j in main['jobs'] if j['method']==job['method'] and j['seed']==job['seed'])
        if not (Path(parent['output'])/'complete.json').exists():continue
        path=ROOT/'native-controller/claims'/(job['name']+'.json');path.parent.mkdir(exist_ok=True)
        try:
            with path.open('x') as stream:
                import json
                json.dump(dict(parent=parent,pid=os.getpid(),utc=now()),stream)
        except FileExistsError:continue
        return job,parent
    return None


def work(host,gpu):
    plan=read(ROOT/'native-controller/plan.json');helper=Path(plan['helper'])
    while time.time()<plan['deadline']:
        if primary_busy(host,gpu):time.sleep(30);continue
        selected=claim()
        if selected is None:
            if all((ROOT/'native-controller/results'/j['name']/'metrics.json').exists() or
                (ROOT/'native-controller/errors'/(j['name']+'.json')).exists() for j in plan['jobs']):return
            time.sleep(30);continue
        job,parent=selected;out=ROOT/'native-controller/results'/job['name'];out.parent.mkdir(exist_ok=True)
        try:
            execute([PYTHON,plan['evaluator'],
                '--config',parent['config'],'--checkpoint',str(Path(parent['output'])/'model-step0041200'),
                '--manifest',job['manifest'],'--output',str(out),'--budget','500'],out.parent/(job['name']+'.log'),
                env(helper,[gpu]),helper,plan['deadline'])
            report=read(out/'metrics.json')
            assert report['passed'] and report['model_state_before']==report['model_state_after']
        except BaseException:
            save(ROOT/'native-controller/errors'/(job['name']+'.json'),dict(passed=False,traceback=traceback.format_exc(),utc=now()))
            raise


def launch(host):
    p=read(ROOT/'native-controller/plan.json');helper=Path(p['helper']);records=[]
    for gpu in range(8):
        tag=p.get('launch_tag','')
        log=ROOT/'native-controller'/f'worker-{tag}{host}-{gpu}.log'
        with log.open('x') as stream:
            process=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'--mode','work','--host',host,'--gpu',str(gpu)],
                cwd=helper,env=env(helper,[gpu]),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        records.append(dict(gpu=gpu,pid=process.pid))
    save(ROOT/'native-controller'/f'launch-{p.get("launch_tag","")}{host}.json',dict(host=host,workers=records,utc=now()));print(records)


def collect():
    plan=read(ROOT/'native-controller/plan.json');cached={}
    while time.time()<plan['deadline']:
        errors=[]
        for job in plan['jobs']:
            f=ROOT/'native-controller/results'/job['name']/'metrics.json'
            if f.exists() and job['name'] not in cached:
                r=read(f);assert r['passed'] and r['model_updates']==0
                assert r['model_state_before']==r['model_state_after']
                expected={(v['task'],v['seed']):v['sha256'] for v in read(job['manifest'])['rows']}
                actual={(v['task'],v['seed']):v['case_sha256'] for v in r['episodes']};assert actual==expected
                import numpy as np
                for v in r['episodes']:
                    assert sha(v['trace'])==v['trace_sha256']
                    trace=np.load(v['trace']);actions=trace['actions'];success=trace['successes']
                    assert actions.shape==(v['steps'],4) and len(success)==v['steps']<=500
                    assert np.isfinite(actions).all() and np.max(np.abs(actions))<=1.
                    assert bool(success.any())==v['success'] and v['initial_rgb_replay']['passed']
                cached[job['name']]=dict(job=job,episodes=r['episodes'],weights_sha256=r['weights_sha256'])
            e=ROOT/'native-controller/errors'/(job['name']+'.json')
            if e.exists():errors.append(str(e))
        models={}
        for result in cached.values():
            j=result['job'];key=f"{j['method']}-seed{j['seed']}"
            models.setdefault(key,[]).extend(result['episodes'])
        reports={}
        for name,rows in models.items():
            assert len({(r['task'],r['seed']) for r in rows})==len(rows)
            groups={}
            for task in sorted({r['task'] for r in rows}):
                selected=[r for r in rows if r['task']==task]
                groups[task]=dict(episodes=len(selected),successes=sum(r['success'] for r in selected),
                    percent=100*sum(r['success'] for r in selected)/len(selected))
            reports[name]=dict(episodes=len(rows),complete=len(rows)==1000,groups=groups,
                macro50_percent=(sum(g['percent'] for g in groups.values())/50 if len(rows)==1000 else None))
        save(ROOT/'native-controller/summary.json',dict(utc=now(),completed_shards=len(cached),errors=errors,
            episodes=sum(len(r['episodes']) for r in cached.values()),models=reports,
            scope=plan['scope'],verification='ActualtraceSHA/actionbounds/success/cases/RGB/frozenNNidentity checked once per finished shard'))
        if len(cached)+len(errors)==len(plan['jobs']):
            save(ROOT/'native-controller/CLOSED.json',dict(utc=now(),completed_shards=len(cached),errors=errors));return
        time.sleep(30)
    save(ROOT/'native-controller/DEADLINE.json',dict(utc=now(),scope='Finite original five-day budget; pending work remains incomplete'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','launch','work','collect'],required=True)
    p.add_argument('--helper',type=Path);p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.helper)
    elif a.mode=='launch':launch(a.host)
    elif a.mode=='work':work(a.host,a.gpu)
    else:collect()
