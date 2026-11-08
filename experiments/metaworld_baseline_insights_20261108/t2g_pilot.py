"""One-GPU finite T2G control diagnostic;borrow a native metadata worker.

All scaling jobs must already be claimed and this pair's scaling disposed.
The borrowed worker's current child finishes naturally;only its metadata
parent is paused. The exact parent resumes after all owned pilot children exit.
"""
import argparse
from pathlib import Path
import os
import shlex
import signal
import time
import traceback

import support_loan as loan
from experiments.metaworld_baselines_20261107.campaign import BASE,PYTHON,env,read,save,sha,now

ROOT=BASE/'runs/metaworld_t2g_control_20261108'
ORIGINAL=BASE/'runs/metaworld_baselines_native_20261108'
SCALING=BASE/'runs/metaworld_baseline_scaling_20261108'
NATIVE=BASE/'runs/metaworld_baselines_20261107/native-controller'
POSITIVE={'door-close-v3','coffee-button-v3'}


def prepare(helper,evaluator):
    ROOT.mkdir(exist_ok=False);cases=[];technical=[]
    for i in [0,1]:
        bank=read(ORIGINAL/f'manifests/intermediate-{i}.json')
        selected=[c for c in bank['rows'] if c['task'] in loan.HARD|POSITIVE]
        assert len(selected)==8;cases.extend(selected)
        technical.append(next(c for c in selected if c['task'] in loan.HARD))
    save(ROOT/'manifests/formal.json',dict(bank,rows=cases))
    save(ROOT/'manifests/technical.json',dict(bank,rows=technical))
    wanted={(c['task'],c['seed']):c['sha256'] for c in cases};jobs=[]
    for name in ['smwm-seed1','h-jepa-seed2']:
        j=next(j for j in read(ORIGINAL/'plan.json')['jobs'] if j['name']==name);step=j['epoch_steps']['100'];out=Path(j['output'])
        reference=[];weights=set()
        for i in [0,1]:
            d=read(out/f'evaluation-{step:07d}-{i}/metrics.json');assert d['passed'] and d['model_state_before']==d['model_state_after']
            weights.add(d['weights_sha256']);reference.extend(r for r in d['episodes'] if (r['task'],r['seed']) in wanted)
        assert len(reference)==16 and len(weights)==1
        assert {(r['task'],r['seed']):r['case_sha256'] for r in reference}==wanted
        checkpoint=out/f'model-step{step:07d}';assert sha(checkpoint/'weights.pt')==next(iter(weights))
        metric=BASE/f'runs/metaworld_t2g_adapted_20261108/{name}/metric-state.pt';proof=read(metric.with_name('summary.json'))
        assert proof['passed'] and proof['feature_proof']['weights_sha256']==next(iter(weights)) and proof['metric_state_sha256']==sha(metric)
        jobs.append(dict(name=name,checkpoint=str(checkpoint),weights_sha256=next(iter(weights)),data=read(j['config'])['data'],
            metric=str(metric),metric_sha256=sha(metric),metric_training_summary_sha256=sha(metric.with_name('summary.json')),reference_rows=reference))
    save(ROOT/'plan.json',dict(helper=str(helper),evaluator=str(evaluator),evaluator_sha256=sha(evaluator),jobs=jobs,
        formal_episodes=32,technical_episodes=8,unique_physical_cases=16,hard_cases=12,positive_cases=4,
        max_seconds=10800,deadline=read(SCALING/'plan.json')['deadline'],created_utc=now(),
        scope='T2G-adaptedpublicbaselinecost,2frozen100models x16samephysicalcases(12hard+4DoorClose/CoffeeButton). '
            'Standard100formalreferencesinheritedONLYafteractual20stepstandardactions/success/firstbankSHAparitygates. '
            'Onlyterminalcostchanges:last-stepMSE->publicT2Gscale*Euclideannormoffrozenown-RGBchronologyhead. '
            'CEM300x15/top10/H6/execute3/context2/f5/500anysuccessunchanged,extraheadcompute/1085CPUheadupdatesdisclosed. '
            'No externalweights/state/reward/task/actiontarget/strongteachers/goal-IDMpolicy. '
            'Nativebodybatchregs,NOTentirelysamplelevelcandidate/novelmethod/officialMetaWorld/freshconfirmation/SOTAclaim. '
            'BorrowONLYoneGPUafteroldnativechildnaturallyfinishes;other23unmodified. No oldchildkill/SGDrepeat/budgetextension.'))


def pilot(runner,gpu):
    plan=read(ROOT/'plan.json');assert sha(plan['evaluator'])==plan['evaluator_sha256'];helper=Path(plan['helper']);models=[]
    for j in plan['jobs']:
        assert sha(j['metric'])==j['metric_sha256'] and sha(Path(j['metric']).with_name('summary.json'))==j['metric_training_summary_sha256']
        for arm,label,budget in [('standard','technical',20),('t2g','technical',20),('t2g','formal',500)]:
            out=ROOT/j['name']/arm/label
            cmd=[PYTHON,plan['evaluator'],'--checkpoint',j['checkpoint'],'--data',j['data'],
                '--manifest',str(ROOT/f'manifests/{label}.json'),'--output',str(out),'--budget',str(budget)]
            if arm=='t2g':cmd+=['--t2g-state',j['metric']]
            runner.run(cmd,out.parent/f'{label}.log',env(helper,[gpu]),helper)
            report=read(out/'metrics.json');loan.verify(report,j,arm,label)
            expected={(r['task'],r['seed']):r for r in j['reference_rows']}
            assert all(r['first_bank_sha256']==expected[r['task'],r['seed']]['first_bank_sha256'] for r in report['episodes'])
            if arm=='t2g':
                assert report['t2g_metric_state_sha256']==j['metric_sha256']
                assert report['t2g_metric_model_state_before']==report['t2g_metric_model_state_after']
            if label=='formal':
                rows=report['episodes'];groups={}
                for task in sorted({r['task'] for r in rows}):
                    own=[r for r in rows if r['task']==task];old=[r for r in j['reference_rows'] if r['task']==task]
                    groups[task]=dict(episodes=len(own),t2g_successes=sum(r['success'] for r in own),reference_cem_successes=sum(r['success'] for r in old))
                result=dict(name=j['name'],groups=groups,hard_successes=sum(r['success'] for r in rows if r['task'] in loan.HARD),
                    reference_hard_successes=sum(r['success'] for r in j['reference_rows'] if r['task'] in loan.HARD),
                    total_successes=sum(r['success'] for r in rows),reference_total_successes=sum(r['success'] for r in j['reference_rows']),
                    hard_successful_tasks=sum(v['t2g_successes']>0 for t,v in groups.items() if t in loan.HARD))
                models.append(result);save(ROOT/'progress.json',dict(models=models,utc=now()))
    eligible=any(r['hard_successes']-r['reference_hard_successes']>=2 and r['hard_successful_tasks']>=2 and
                 r['total_successes']-r['reference_total_successes']>=2 for r in models)
    save(ROOT/'summary.json',dict(passed=True,models=models,formal_episodes=32,technical_episodes=8,
        unique_physical_cases=16,worldmodel_updates=0,head_updates_during_control=0,eligible_replication=eligible,utc=now(),scope=plan['scope']))


def work(host,gpu,parent):
    plan=read(ROOT/'plan.json');runner=loan.Runner(min(plan['deadline'],time.time()+plan['max_seconds']))
    scale=read(SCALING/'plan.json');assert all((SCALING/'claims'/f'{j["name"]}.json').exists() for j in scale['jobs'])
    owned=[]
    for j in scale['jobs']:
        c=read(SCALING/'claims'/f'{j["name"]}.json')
        if c['host']==host and c['pair']==gpu//2:
            assert (Path(j['output'])/'complete.json').exists(),'Do not borrow a pair still needed by scaling'
            owned.append(j['name'])
    registry=read(NATIVE/f'launch-scale-v1-{host}.json')
    assert next(w for w in registry['workers'] if w['gpu']==gpu)['pid']==parent
    record=loan.identity(parent);argv=shlex.split(record['command'])
    assert any(x.endswith('/native_queue.py') for x in argv) and argv[argv.index('--mode')+1]=='work'
    assert argv[argv.index('--gpu')+1]==str(gpu) and argv[argv.index('--host')+1]==host
    assert parent not in loan.gpu_pids([gpu]),'Only pause non-CUDA metadata'
    save(ROOT/'OWNERSHIP.json',dict(host=host,gpu=gpu,pid=os.getpid(),original_native_parent=record,
        completed_scaling_pair_jobs=owned,all_scaling_claims_exist=True,utc=now(),deadline=runner.deadline))
    def cancelled(signum,frame):runner.cancel.set()
    signal.signal(signal.SIGTERM,cancelled);signal.signal(signal.SIGINT,cancelled)
    try:
        with loan.paused_parent(record):
            try:
                while True:
                    live=[]
                    for child in loan.descendants(parent):
                        try:
                            if loan.identity(child)['state']!='Z':live.append(child)
                        except FileNotFoundError:pass
                    if not live and not loan.gpu_pids([gpu]):break
                    if runner.cancel.is_set() or time.time()>runner.deadline or (ROOT/'STOP').exists():raise TimeoutError('Native slot loan deadline/cancel')
                    time.sleep(5)
                save(ROOT/'GPU_LOAN_READY.json',dict(utc=now(),old_native_children_finished_naturally=True,sgd_repeated=0))
                pilot(runner,gpu)
            finally:runner.close()
    except BaseException:save(ROOT/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc()))
    finally:
        for _ in range(100):
            current=loan.identity(parent)
            if current['state'] not in ['T','t']:break
            time.sleep(.01)
        assert current['start']==record['start'] and current['state'] not in ['T','t']
        save(ROOT/'CLOSED.json',dict(utc=now(),original_native_parent_resumed=True,original_native_parent_pid=parent,
            pilot_passed=(ROOT/'summary.json').exists(),sgd_repeated=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','selftest','work'],required=True)
    p.add_argument('--helper',type=Path);p.add_argument('--evaluator',type=Path);p.add_argument('--host');p.add_argument('--gpu',type=int);p.add_argument('--parent',type=int)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.helper,a.evaluator)
    elif a.mode=='selftest':loan.ROOT=ROOT;loan.selftest()
    else:
        try:work(a.host,a.gpu,a.parent)
        except BaseException:
            save(ROOT/'LAUNCH_ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise
