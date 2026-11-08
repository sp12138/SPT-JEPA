"""Bounded frozen control experiment,borrowing one metadata controller's pair.

Native SGD/evaluation children finish naturally;their parent alone is paused.
Always resumes that exact parent after terminating any owned pilot children.
"""
import argparse
import concurrent.futures
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback

from experiments.metaworld_baselines_20261107.campaign import BASE,PYTHON,env,read,save,sha,now

ROOT=BASE/'runs/metaworld_control_support_20261108'
ORIGINAL=BASE/'runs/metaworld_baselines_native_20261108'
SCALING=BASE/'runs/metaworld_baseline_scaling_20261108'
HARD={'bin-picking-v3','door-open-v3','drawer-open-v3','peg-insert-side-v3','pick-place-v3','push-v3'}


def identity(pid):
    p=Path(f'/proc/{pid}');stat=p.joinpath('stat').read_text().split(') ',1)[1].split()
    return dict(pid=pid,start=stat[19],state=stat[0],command=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode())


def children(pid):
    found=set()
    for task in Path(f'/proc/{pid}/task').glob('*'):
        try:found.update(map(int,task.joinpath('children').read_text().split()))
        except FileNotFoundError:pass
    return found


def descendants(pid):
    found=set();pending=list(children(pid))
    while pending:
        p=pending.pop()
        if p in found:continue
        found.add(p);pending.extend(children(p))
    return found


@contextmanager
def paused_parent(record):
    current=identity(record['pid'])
    assert current['start']==record['start'] and current['command']==record['command']
    assert current['state'] not in ['T','t','Z'],'Never adopt an already-paused/dead controller'
    os.kill(record['pid'],signal.SIGSTOP)
    try:
        for _ in range(100):
            if identity(record['pid'])['state'] in ['T','t']:break
            time.sleep(.01)
        assert identity(record['pid'])['state'] in ['T','t']
        yield
    finally:
        current=identity(record['pid'])
        assert current['start']==record['start'] and current['command']==record['command']
        os.kill(record['pid'],signal.SIGCONT)


def gpu_pids(gpus):
    found=set()
    for gpu in gpus:
        text=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True)
        found.update(int(s) for s in text.splitlines() if s.strip().isdigit())
    return found


def selftest():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);child_done=root/'child';parent_done=root/'parent'
        program='import subprocess,sys; p=subprocess.Popen([sys.executable,"-c",'+repr('import time;from pathlib import Path;time.sleep(.8);Path('+repr(str(child_done))+').write_text("natural")')+']);p.wait();from pathlib import Path;Path('+repr(str(parent_done))+').write_text("resumed")'
        p=subprocess.Popen([sys.executable,'-c',program])
        try:
            deadline=time.time()+10
            while not children(p.pid) and time.time()<deadline:time.sleep(.01)
            record=identity(p.pid);assert children(p.pid)
            try:
                with paused_parent(record):
                    while not child_done.exists() and time.time()<deadline:time.sleep(.01)
                    assert child_done.read_text()=='natural' and not parent_done.exists()
                    raise RuntimeError('intentional pilot failure')
            except RuntimeError as error:assert str(error)=='intentional pilot failure'
            assert p.wait(timeout=10)==0 and parent_done.read_text()=='resumed'
        finally:
            if p.poll() is None:p.kill();p.wait()
    save(ROOT/'OS_HANDOFF_SELFTEST.json',dict(passed=True,actual_os_stop_continue=True,
        child_computation_completed_naturally=True,exception_resumes_exact_parent=True,gpu_contexts=0,utc=now()))


def prepare(helper,evaluator):
    ROOT.mkdir(exist_ok=False)
    jobs=[];cases=[]
    for i in [0,1]:
        bank=read(ORIGINAL/f'manifests/intermediate-{i}.json');selected=[c for c in bank['rows'] if c['task'] in HARD]
        assert len(selected)==6;cases.extend(selected)
        for label,rows in [('technical',selected[:1]),('formal',selected)]:save(ROOT/f'manifests/{label}-{i}.json',dict(bank,rows=rows))
    wanted={(c['task'],c['seed']):c['sha256'] for c in cases}
    for name in ['smwm-seed1','h-jepa-seed2']:
        j=next(j for j in read(ORIGINAL/'plan.json')['jobs'] if j['name']==name);step=j['epoch_steps']['100'];out=Path(j['output'])
        rows=[];weights=set()
        for i in [0,1]:
            d=read(out/f'evaluation-{step:07d}-{i}/metrics.json');assert d['passed'] and d['model_state_before']==d['model_state_after']
            weights.add(d['weights_sha256']);rows.extend(r for r in d['episodes'] if (r['task'],r['seed']) in wanted)
        assert len(rows)==12 and len(weights)==1
        assert {(r['task'],r['seed']):r['case_sha256'] for r in rows}==wanted
        checkpoint=out/f'model-step{step:07d}';assert sha(checkpoint/'weights.pt')==next(iter(weights))
        jobs.append(dict(name=name,checkpoint=str(checkpoint),data=read(j['config'])['data'],
            weights_sha256=next(iter(weights)),reference_rows=rows))
    save(ROOT/'plan.json',dict(helper=str(helper),evaluator=str(evaluator),evaluator_sha256=sha(evaluator),jobs=jobs,
        formal_episodes=48,technical_episodes=8,unique_physical_cases=12,max_seconds=7200,created_utc=now(),
        deadline=read(SCALING/'plan.json')['deadline'],
        scope='2frozen100epochmodels x2controlfamilies x12samecases. Standardmustreplayoriginalactions/success exactly. '
            'Heldprojectsfirstnative4Dcontrolacross5steps;identicalCEM300x15/H6/execute3/context2/f5/500anysuccess. '
            'Previouslyinspecteddevelopmentalcases,notfresh/full50/SOTA. Notnewmethod;old20epochheldnegativeexists. '
            'No newNNtraining/state/reward/taskID/actionlabels/teacher. OnlyRGBgoal+actualforwardcontrols. '
            'Borrow1pairafteroldchildfinishes;restoreexactoldcontroller. Other11pairsunchanged.'))


class Runner:
    def __init__(self,deadline):self.deadline=deadline;self.cancel=threading.Event();self.lock=threading.Lock();self.live={}
    def stop(self,p):
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    def run(self,cmd,log,environment,cwd):
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('x') as stream:
            p=subprocess.Popen(cmd,cwd=cwd,env=environment,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            with self.lock:self.live[p.pid]=p
            try:
                while p.poll() is None:
                    if self.cancel.is_set() or time.time()>self.deadline:raise TimeoutError('Bounded pilot cancelled/deadline')
                    time.sleep(.5)
                assert p.returncode==0,(cmd,p.returncode)
            finally:
                self.stop(p)
                with self.lock:self.live.pop(p.pid,None)
    def close(self):
        self.cancel.set()
        with self.lock:live=list(self.live.values())
        for p in live:self.stop(p)


def verify(report,j,arm,label):
    import numpy as np
    assert report['passed'] and report['model_updates']==0 and report['model_state_before']==report['model_state_after']
    assert report['weights_sha256']==j['weights_sha256']
    reference={(r['task'],r['seed']):r for r in j['reference_rows']}
    for r in report['episodes']:
        expected=reference[r['task'],r['seed']]
        assert r['case_sha256']==expected['case_sha256'] and r['initial_rgb_replay']['passed'] and sha(r['trace'])==r['trace_sha256']
        with np.load(r['trace']) as t:a,s=t['actions'],t['successes'];assert a.shape==(r['steps'],4) and bool(s.any())==r['success']
        assert np.isfinite(a).all() and np.max(np.abs(a))<=1
        if arm=='standard':
            with np.load(expected['trace']) as old:
                assert np.array_equal(a,old['actions'][:len(a)]) and np.array_equal(s,old['successes'][:len(s)])
            assert r['first_bank_sha256']==expected['first_bank_sha256']
        elif arm=='held':
            blocks=a[:len(a)//5*5].reshape(-1,5,4)
            assert np.max(np.abs(blocks-blocks[:,:1]))<1e-5


def pilot(runner,host,pair):
    plan=read(ROOT/'plan.json');helper=Path(plan['helper']);assert sha(plan['evaluator'])==plan['evaluator_sha256'];rows=[]
    for j in plan['jobs']:
        for arm in ['standard','held']:
            for label,budget in [('technical',20),('formal',500)]:
                def one(i,gpu):
                    out=ROOT/j['name']/arm/f'{label}-{i}'
                    cmd=[PYTHON,plan['evaluator'],'--checkpoint',j['checkpoint'],'--data',j['data'],
                        '--manifest',str(ROOT/f'manifests/{label}-{i}.json'),'--output',str(out),'--budget',str(budget)]
                    if arm=='held':cmd+=['--held']
                    runner.run(cmd,out.parent/f'{label}-{i}.log',env(helper,[gpu]),helper)
                    report=read(out/'metrics.json');verify(report,j,arm,label);return report
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    futures=[pool.submit(one,i,2*pair+i) for i in [0,1]]
                    reports=[f.result() for f in futures]
                if label=='formal':
                    rows.append(dict(name=j['name'],arm=arm,episodes=12,successes=sum(r['success'] for d in reports for r in d['episodes'])))
                    save(ROOT/'progress.json',dict(rows=rows,utc=now()))
    save(ROOT/'summary.json',dict(passed=True,rows=rows,formal_episodes=48,unique_physical_cases=12,
        model_updates=0,utc=now(),scope=plan['scope'],promote=all(next(r for r in rows if r['name']==j['name'] and r['arm']=='held')['successes']>=3 for j in plan['jobs'])))


def work(host,pair,parent,pilot_fn=None):
    plan=read(ROOT/'plan.json');runner=Runner(min(plan['deadline'],time.time()+plan['max_seconds']))
    record=identity(parent);assert 'runtime-scaling-v1' in record['command'] and '--mode work' in record['command'] and f'--pair {pair}' in record['command'] and f'--host {host}' in record['command']
    owners=[dict(read(f),name=f.stem) for f in (SCALING/'claims').glob('*.json')]
    owned=[r for r in owners if r['host']==host and r['pair']==pair and r['pid']==parent and not (SCALING/'training'/r['name']/'complete.json').exists()]
    assert len(owned)==1 and not (SCALING/'training'/owned[0]['name']/'error.json').exists()
    assert parent not in gpu_pids([2*pair,2*pair+1]),'Only pause a non-CUDA metadata controller'
    save(ROOT/'OWNERSHIP.json',dict(host=host,pair=pair,gpus=[2*pair,2*pair+1],pid=os.getpid(),original_parent=record,
        original_scaling_job=owned[0]['name'],utc=now(),deadline=runner.deadline))
    def cancelled(signum,frame):runner.cancel.set()
    signal.signal(signal.SIGTERM,cancelled);signal.signal(signal.SIGINT,cancelled)
    try:
        with paused_parent(record):
            try:
                while True:
                    live=[]
                    for p in descendants(parent):
                        try:
                            if identity(p)['state']!='Z':live.append(p)
                        except FileNotFoundError:pass
                    if not live and not gpu_pids([2*pair,2*pair+1]):break
                    if runner.cancel.is_set() or time.time()>runner.deadline or (ROOT/'STOP').exists():raise TimeoutError('No free loan pair before deadline/cancel')
                    time.sleep(5)
                save(ROOT/'GPU_LOAN_READY.json',dict(utc=now(),original_children_completed_naturally=True,original_parent_only_paused=True,sgd_repeated=0))
                (pilot if pilot_fn is None else pilot_fn)(runner,host,pair)
            finally:runner.close()
    except BaseException:save(ROOT/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc()))
    finally:
        for _ in range(100):
            current=identity(parent)
            if current['state'] not in ['T','t']:break
            time.sleep(.01)
        assert current['start']==record['start'] and current['state'] not in ['T','t']
        save(ROOT/'CLOSED.json',dict(utc=now(),original_parent_resumed=True,original_parent_pid=parent,
            pilot_passed=(ROOT/'summary.json').exists(),sgd_repeated=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','selftest','work'],required=True)
    p.add_argument('--helper',type=Path);p.add_argument('--evaluator',type=Path);p.add_argument('--host');p.add_argument('--pair',type=int);p.add_argument('--parent',type=int)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.helper,a.evaluator)
    elif a.mode=='selftest':selftest()
    else:work(a.host,a.pair,a.parent)
