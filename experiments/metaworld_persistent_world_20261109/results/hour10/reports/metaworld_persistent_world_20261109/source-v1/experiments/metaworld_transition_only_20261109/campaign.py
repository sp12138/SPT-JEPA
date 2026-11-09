"""One finite 24-model trial, followed by shared frozen evaluation work.

No old campaign is resumed. This is a new transition-only candidate/control
study, not another public-baseline extension. All weights start randomly.
"""
import argparse
import collections
import hashlib
import os
from pathlib import Path
import subprocess
import time
import traceback
from .common import BASE,ROOT,PYTHON,read,save,sha,now,environment

TASKS=['bin-picking-v3','door-open-v3','drawer-open-v3','peg-insert-side-v3',
       'pick-place-v3','push-v3','reach-v3','reach-wall-v3']
HARD=set(TASKS[:6]);HARD4={'bin-picking-v3','peg-insert-side-v3','pick-place-v3','push-v3'}


def prepare(source):
    ROOT.mkdir(parents=True,exist_ok=False)
    reports=BASE/'reports/metaworld_transition_only_20261109'
    contract=read(reports/'data-contract-entry-v1.json');assert contract['passed']
    for host in ['dev13','dev15','dev6']:assert read(reports/f'cache-{host}.json')['passed']
    bank=read(BASE/'runs/metaworld_input_inverse_20261107_v1/followup/cases/manifest.json')
    selected={}
    for task in TASKS:
        rows=sorted([r for r in bank['rows'] if r['task']==task],key=lambda r:r['seed'])
        assert len(rows)==20
        selected[task]=[rows[i] for i in [0,1,10,11]]
    for name,rows in [('screen',[selected[t][0] for t in TASKS]),
                      ('final',[r for t in TASKS for r in selected[t]]),
                      ('technical',[selected['reach-v3'][0]])]:
        save(ROOT/'manifests'/f'{name}.json',dict(bank,rows=rows))
    jobs=[]
    for seed in [1,2,3,4]:
        calibration=read(reports/f'calibration-seed{seed}.json');assert calibration['passed']
        assert calibration['model_source_sha256']==sha(source/'experiments/metaworld_transition_only_20261109/model.py')
        for horizon in [1,5,15]:
            for inverse in [False,True]:
                name=f'h{horizon}-inverse{int(inverse)}-seed{seed}';out=ROOT/'training'/name
                cfg=dict(name=name,output=str(out),seed=seed,horizon=horizon,width=128,
                    inverse_weight=calibration['inverse_weight'] if inverse else 0.,
                    batch_size=64,short_stop=10000,lr_horizon=30000,max_train_seconds=12*3600,
                    data='/tmp/prescottsun-factual-rgb128-20261109-v1',scratch=True,
                    supervision_contract='factual_forward_and_same_input_inverse',
                    source=str(source),calibration=str(reports/f'calibration-seed{seed}.json'),
                    expected_initial_parameters_sha256=calibration['parameters_sha256'])
                save(out/'config.json',cfg)
                index=len(jobs);host=['dev13','dev15','dev6'][index//8];gpu=index%8
                jobs.append(dict(name=name,host=host,gpu=gpu,config=str(out/'config.json'),output=str(out),seed=seed,
                    horizon=horizon,inverse=inverse))
    evaluations=[]
    for job in jobs:
        for step in [2000,10000]:
            for horizon in [30,60]:
                for execute in [2,15]:
                    name=f"{job['name']}-s{step}-H{horizon}-e{execute}"
                    evaluations.append(dict(name=name,job=job['name'],config=job['config'],step=step,
                        horizon=horizon,execute=execute,manifest=str(ROOT/'manifests'/('screen.json' if step==2000 else 'final.json')),
                        output=str(ROOT/'evaluation'/name)))
    sources={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')}
    save(ROOT/'plan.json',dict(created_utc=now(),source=str(source),source_hashes=sources,jobs=jobs,evaluations=evaluations,
        deadline=time.time()+36*3600,train_models=24,training_seeds=4,short_updates=10000,lr_horizon=30000,
        new_long_training_authorized_only_after_evidence=True,
        supervision='Expert-collectedLeRobotdatausedONLYasfactualtransitionlogs. FutureRGBprediction+sameexecutedforwardinputinverse. '
            'No desiredgoalinlearner,expertpath/policytargets,externalweights/reward/state/taskID/language. Goalimageonlyinplanner. '
            'One-phasefromrandomweights;noQueryRecurrentgoal-pathparents ordisqualifiedpolicylineage.',
        comparison='3traininghorizons1/5/15x2inverseconditionsx4pairedseeds. Sameall181608nativeanchors/history8/dataorder/initialparameters. '
            'InferenceH30/H60xexecute2/15,held5nativecontrols,CEM64x4/top10/terminalRGBMSE/500nativeanysuccess. '
            'Horizon/feedbackcomputeNOTequal;notcommon300x15/publicSOTA. Screen8/final32cases/model-controller;developmentalcasespreviouslyinspected.',
        monitor='Userrequestsactualchecksinhourlydirectsessionwait. No externalLaunchAgent/newgoal/secondLLMmonitor. '
            'Fixidentifiedproblemswithpreservedattempts;no automaticlongpromotionfromlossoronesuccess.'))
    print(dict(root=str(ROOT),models=24,evaluation_endpoints=len(evaluations)),flush=True)


def command(args,log,env,source,deadline,cap=12*3600):
    remaining=min(cap,int(deadline-time.time()));assert remaining>0
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    with Path(log).open('x') as stream:
        subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=60',str(remaining),*map(str,args)],
            cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            check=True,timeout=remaining+120)


def cpu_audit(job,step,source,env,deadline):
    root=Path(job['output']);dest=root/f'audit-{step:07d}.json'
    if dest.exists():assert read(dest)['passed'];return
    lock=root/f'audit-{step:07d}.lock'
    try:lock.mkdir()
    except FileExistsError:
        while not dest.exists():
            if list(root.glob(f'ERROR-audit-{step:07d}*.json')):raise RuntimeError('Another audit failed')
            assert time.time()<deadline;time.sleep(2)
        assert read(dest)['passed'];return
    try:
        cpu_env=dict(env,CUDA_VISIBLE_DEVICES='')
        command([PYTHON,'-m','experiments.metaworld_transition_only_20261109.audit','checkpoint',
            '--config',job['config'],'--checkpoint',root/f'checkpoint-{step:07d}.pt','--output',dest],
            root/f'audit-{step:07d}.log',cpu_env,source,deadline,cap=600)
        assert read(dest)['passed']
    except BaseException:
        save(root/f'ERROR-audit-{step:07d}-{os.getpid()}.json',dict(utc=now(),traceback=traceback.format_exc()));raise


def global_gate():
    plan=read(ROOT/'plan.json')
    if not all((Path(j['output'])/'PREFLIGHT.json').exists() for j in plan['jobs']):return
    try:(ROOT/'GLOBAL_GATE_LOCK').mkdir()
    except FileExistsError:return
    try:
        groups=collections.defaultdict(list);data_hashes=set();sample_groups=collections.defaultdict(list)
        for j in plan['jobs']:
            out=Path(j['output']);assert read(out/'PREFLIGHT.json')['passed']
            initial=read(out/'initial.json');cfg=read(j['config'])
            assert initial['parameters_sha256']==cfg['expected_initial_parameters_sha256']
            groups[j['seed']].append(initial['parameters_sha256']);data_hashes.add(sha(out/'data-contract.json'))
            sample_groups[j['seed']].append((out/'sample-trace.jsonl').read_text())
        assert all(len(set(v))==1 for v in groups.values()) and len(data_hashes)==1
        assert all(len(set(v))==1 for v in sample_groups.values())
        save(ROOT/'TRAINING_APPROVED.json',dict(passed=True,utc=now(),models=24,paired_initial_parameter_hashes=groups,
            exact_paired_first_two_batches=True,common_training_manifest=True,actual_two_update_restore_and_physical_gates=True))
    except BaseException:
        save(ROOT/'GLOBAL_GATE_ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise


def claim_evaluation(host,gpu):
    plan=read(ROOT/'plan.json');jobs={j['name']:j for j in plan['jobs']}
    for e in sorted(plan['evaluations'],key=lambda e:-e['step']):
        job=jobs[e['job']];out=Path(job['output'])
        if not (out/f'model-step{e["step"]:07d}/SEALED.json').exists():continue
        path=ROOT/'evaluation-claims'/(e['name']+'.json');path.parent.mkdir(exist_ok=True)
        try:
            with path.open('x') as f:__import__('json').dump(dict(host=host,gpu=gpu,pid=os.getpid(),utc=now()),f)
        except FileExistsError:continue
        return e,job
    return None


def work(host,gpu):
    plan=read(ROOT/'plan.json');source=Path(plan['source']);env=environment(source,gpu);deadline=plan['deadline']
    job=next(j for j in plan['jobs'] if j['host']==host and j['gpu']==gpu);out=Path(job['output'])
    try:
        # No process/GPU adoption: only the newly assigned idle card is used.
        active=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True)
        assert not any(s.strip().isdigit() for s in active.splitlines()),active
        command([PYTHON,'-m','experiments.metaworld_transition_only_20261109.train','--config',job['config'],'--stop','2'],
            out/'train-entry.log',env,source,deadline)
        cpu_audit(job,2,source,env,deadline)
        for horizon,execute in [(30,15),(60,2)]:
            dest=out/f'technical-H{horizon}-e{execute}'
            command([PYTHON,'-m','experiments.metaworld_transition_only_20261109.evaluate','--config',job['config'],
                '--step','2','--manifest',ROOT/'manifests/technical.json','--output',dest,
                '--horizon',horizon,'--execute',execute,'--budget','20'],out/(dest.name+'.log'),env,source,deadline,cap=1800)
            assert read(dest/'metrics.json')['passed']
        save(out/'PREFLIGHT.json',dict(passed=True,utc=now(),host=host,gpu=gpu,actual_new_updates=2,
            full_cpu_restore_fp32_backward=True,physical_episodes=2,physical_native_steps=40))
        while not (ROOT/'TRAINING_APPROVED.json').exists():
            if (ROOT/'STOP').exists():raise RuntimeError('Explicit campaign STOP')
            if (ROOT/'GLOBAL_GATE_ERROR.json').exists():raise RuntimeError('Global paired gate failed')
            if list(ROOT.glob('training/*/ERROR-*.json')):raise RuntimeError('Peer preflight failed')
            assert time.time()<deadline;global_gate();time.sleep(2)
        command([PYTHON,'-m','experiments.metaworld_transition_only_20261109.train','--config',job['config'],
            '--stop','10000','--resume',out/'checkpoint-0000002.pt'],out/'train-short10000.log',env,source,deadline)
        cpu_audit(job,10000,source,env,deadline)
        save(out/'TRAINING_COMPLETE.json',dict(passed=True,step=10000,utc=now()))
        while time.time()<deadline:
            if (ROOT/'STOP').exists():raise RuntimeError('Explicit campaign STOP')
            selected=claim_evaluation(host,gpu)
            if selected is None:
                if all((Path(e['output'])/'metrics.json').exists() or (ROOT/'evaluation-errors'/(e['name']+'.json')).exists() for e in plan['evaluations']):return
                time.sleep(10);continue
            e,parent=selected
            try:
                cpu_audit(parent,e['step'],source,env,deadline)
                command([PYTHON,'-m','experiments.metaworld_transition_only_20261109.evaluate','--config',e['config'],
                    '--step',e['step'],'--manifest',e['manifest'],'--output',e['output'],
                    '--horizon',e['horizon'],'--execute',e['execute']],Path(e['output']).with_suffix('.log'),env,source,deadline)
                assert read(Path(e['output'])/'metrics.json')['passed']
            except BaseException:
                save(ROOT/'evaluation-errors'/(e['name']+'.json'),dict(utc=now(),traceback=traceback.format_exc()));raise
    except BaseException:
        save(out/f'ERROR-worker-{os.getpid()}.json',dict(utc=now(),traceback=traceback.format_exc()));raise


def launch(host):
    plan=read(ROOT/'plan.json');source=Path(plan['source']);workers=[]
    for gpu in range(8):
        log=ROOT/f'worker-{host}-{gpu}.log'
        with log.open('x') as f:
            p=subprocess.Popen([PYTHON,'-m','experiments.metaworld_transition_only_20261109.campaign','work','--host',host,'--gpu',str(gpu)],
                cwd=source,env=environment(source,gpu),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(gpu=gpu,pid=p.pid))
    save(ROOT/f'launch-{host}.json',dict(host=host,workers=workers,utc=now(),source=str(source)))
    print(host,workers,flush=True)


def collect():
    plan=read(ROOT/'plan.json');cache={};jobs={j['name']:j for j in plan['jobs']}
    while time.time()<plan['deadline']:
        for e in plan['evaluations']:
            path=Path(e['output'])/'metrics.json'
            if path.exists() and e['name'] not in cache:
                d=read(path);assert d['passed'] and d['model_state_before']==d['model_state_after'] and d['model_updates']==0
                expected={(r['task'],r['seed']):r['sha256'] for r in read(e['manifest'])['rows']}
                assert {(r['task'],r['seed']):r['case_sha256'] for r in d['episodes']}==expected
                import numpy as np
                for row in d['episodes']:
                    assert sha(row['trace'])==row['trace_sha256'] and row['initial_rgb_replay']['passed']
                    with np.load(row['trace']) as z:
                        assert z['actions'].shape==(row['steps'],4) and len(z['successes'])==row['steps']<=500
                        assert np.isfinite(z['actions']).all() and np.max(np.abs(z['actions']))<=1
                        assert bool(z['successes'].any())==row['success']
                cache[e['name']]=dict(experiment=e,episodes=d['episodes'],weights_sha256=d['weights_sha256'])
        errors=[str(p) for p in ROOT.glob('training/*/ERROR-*.json')]+[str(p) for p in ROOT.glob('evaluation-errors/*.json')]
        groups=collections.defaultdict(list)
        for result in cache.values():
            e=result['experiment'];j=jobs[e['job']]
            key=f"trainH{j['horizon']}-inverse{int(j['inverse'])}-step{e['step']}-planH{e['horizon']}-execute{e['execute']}"
            groups[key].extend(result['episodes'])
        reports={}
        for name,rows in groups.items():
            task={t:dict(successes=sum(r['success'] for r in rows if r['task']==t),episodes=sum(r['task']==t for r in rows)) for t in TASKS}
            hard=[r for r in rows if r['task'] in HARD];hard4=[r for r in rows if r['task'] in HARD4]
            reports[name]=dict(episodes=len(rows),by_task=task,hard6_percent=100*sum(r['success'] for r in hard)/len(hard),
                hard4_percent=100*sum(r['success'] for r in hard4)/len(hard4),
                unique_physical_cases=len({(r['task'],r['seed']) for r in rows}))
        trained=sum((Path(j['output'])/'TRAINING_COMPLETE.json').exists() for j in plan['jobs'])
        save(ROOT/'results/summary.json',dict(utc=now(),trainings_complete=trained,trainings=24,evaluation_endpoints_complete=len(cache),
            evaluation_endpoints=len(plan['evaluations']),episodes=sum(len(r['episodes']) for r in cache.values()),errors=errors,groups=reports,
            scope='Paireddevelopmentalstudy;fourtrainedseedsrepeatthesamephysicalcases. Partialgroupsarenotbalancedconfirmedresults. '
                  'No publicSOTA/equalFLOPs/expert-free-data claim;onlyfactualtransition-supervisedgoalfreelearner.'))
        if trained==24 and len(cache)==len(plan['evaluations']):
            save(ROOT/'CLOSED.json',dict(utc=now(),passed=not errors,trainings=24,endpoints=len(cache),automatic_long_promotion=False));return
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch','work','collect']);p.add_argument('--source',type=Path)
    p.add_argument('--host');p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.source)
    elif a.mode=='launch':launch(a.host)
    elif a.mode=='work':work(a.host,a.gpu)
    else:collect()
