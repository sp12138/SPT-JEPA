"""Retire only the optional frozen native-controller expansion by user request.

Primary 300-epoch training and common-CEM evaluation are never signalled.
Original plans, complete results and interrupted traces remain on disk.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import time

BASE = Path('/efs/prescottsun/SPT-JEPA')
ROOT = BASE/'runs/metaworld_baselines_20261107/native-controller'
OUT = ROOT/'closeout-20261108'
GOAL = BASE/'runs/metaworld_goal_rank_control_20261108'


def read(p):
    return json.loads(Path(p).read_text())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(tmp, p)


def proc(pid):
    try:
        p = Path(f'/proc/{pid}')
        s = (p/'stat').read_text()
        fields = s[s.rfind(')')+2:].split()
        return dict(pid=pid, ppid=int(fields[1]), state=fields[0], start=fields[19],
                    command=(p/'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except OSError:
        return None


def signal_exact(record, sig):
    current = proc(record['pid'])
    if current is None or current['state'] == 'Z':
        return
    assert current['start'] == record['start'] and current['command'] == record['command'], current
    os.kill(record['pid'], sig)


def prepare():
    OUT.mkdir(exist_ok=False)
    raw = (ROOT/'plan.json').read_bytes()
    (OUT/'original-plan.json').write_bytes(raw)
    (OUT/'original-summary.json').write_bytes((ROOT/'summary.json').read_bytes())
    plan = json.loads(raw)
    save(OUT/'REQUEST.json', dict(utc=now(), user='尽快把baselines收尾吧',
        scope='Stop optional native frozen GD/hierarchy expansion. Keep six already-started 300epoch primary continuations and their common-CEM final evaluations. No new baseline recipes/extensions.',
        original_plan_sha256=hashlib.sha256(raw).hexdigest(), planned_shards=len(plan['jobs']),
        scaling_plan_sha256=hashlib.sha256((BASE/'runs/metaworld_baseline_scaling_20261108/plan.json').read_bytes()).hexdigest(),
        preserve='All weights, checkpoints, logs, complete and partial traces; unfinished work is deferred/interrupted, never a model failure.'))
    plan.update(jobs=[], status='USER_CLOSED_OPTIONAL_EXPANSION', original_plan=str(OUT/'original-plan.json'),
                closeout_request=str(OUT/'REQUEST.json'), episodes=0)
    save(ROOT/'plan.json', plan)
    print('Native new claims disabled; original plan archived.')


def stop(host):
    assert read(ROOT/'plan.json')['jobs'] == []
    records = []
    for worker in read(ROOT/f'launch-scale-v1-{host}.json')['workers']:
        parent = proc(worker['pid'])
        if parent is None or parent['state'] == 'Z':
            records.append(dict(worker=worker, already_exited=True))
            continue
        assert 'native_queue.py --mode work --host '+host+' --gpu '+str(worker['gpu']) in parent['command']
        if host == 'dev6' and worker['pid'] == 20754 and not (GOAL/'CLOSED.json').exists():
            records.append(dict(worker=worker, protected_by_active_goal_loan=True, parent=parent))
            continue
        signal_exact(parent, signal.SIGSTOP)
        processes = [proc(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()]
        processes = [p for p in processes if p is not None]
        descendants = []
        pending = [parent['pid']]
        while pending:
            pid = pending.pop()
            children = [p for p in processes if p['ppid'] == pid]
            descendants.extend(children)
            pending.extend(p['pid'] for p in children)
        for child in descendants:
            assert child['state'] == 'Z' or ('native-hjepa-eval-v2.py' in child['command'] and str(ROOT/'results') in child['command']), child
        record = dict(worker=worker, parent=parent, descendants=descendants)
        records.append(record)
        save(OUT/f'stop-in-progress-{host}.json', dict(utc=now(), records=records))
        # Retire the dispatcher first, so deliberate child preemption cannot
        # become an algorithm/error outcome in its exception handler.
        signal_exact(parent, signal.SIGTERM)
        signal_exact(parent, signal.SIGCONT)
        for child in reversed(descendants):
            signal_exact(child, signal.SIGTERM)
    if host == 'dev13':
        collector = proc(read(ROOT/'collector-scaling-v1-launch.json')['pid'])
        if collector and collector['state'] != 'Z':
            assert 'native_queue.py --mode collect' in collector['command'], collector
            records.append(dict(collector=collector))
            signal_exact(collector, signal.SIGTERM)
    time.sleep(2)
    for record in records:
        if record.get('protected_by_active_goal_loan'):
            continue
        for p in record.get('descendants', [])+[record[k] for k in ['parent', 'collector'] if k in record]:
            current = proc(p['pid'])
            if current and current['state'] != 'Z':
                signal_exact(p, signal.SIGKILL)
    time.sleep(1)
    remaining = []
    for record in records:
        if record.get('protected_by_active_goal_loan'):
            continue
        for p in record.get('descendants', [])+[record[k] for k in ['parent', 'collector'] if k in record]:
            current = proc(p['pid'])
            if current and current['state'] != 'Z':
                remaining.append(current)
    save(OUT/f'stop-{host}-{time.time_ns()}.json', dict(utc=now(), host=host, records=records,
         remaining=remaining, passed=not remaining, primary_training_signals=0))
    print(json.dumps(dict(host=host, retired=sum('parent' in r and not r.get('protected_by_active_goal_loan') for r in records),
                         protected=sum(bool(r.get('protected_by_active_goal_loan')) for r in records), remaining=remaining)))
    assert not remaining


def audit():
    import numpy as np
    assert read(ROOT/'plan.json')['jobs'] == []
    plan = read(OUT/'original-plan.json')
    recipes = []
    models = {}
    common = {}
    hard = {'bin-picking-v3', 'door-open-v3', 'drawer-open-v3',
            'peg-insert-side-v3', 'pick-place-v3', 'push-v3'}
    for job in plan['jobs']:
        dest = ROOT/'results'/job['name']
        metric = dest/'metrics.json'
        claimed = ROOT/'claims'/(job['name']+'.json')
        traces = list(dest.glob('*.npz'))
        status = 'COMPLETE' if metric.exists() else ('INTERRUPTED_BY_USER_PRIORITY' if claimed.exists() else 'DEFERRED_NOT_STARTED')
        recipes.append(dict(name=job['name'], status=status, preserved_trace_files=len(traces)))
        if not metric.exists():
            continue
        result = read(metric)
        assert result['passed'] and result['model_updates'] == 0
        assert result['model_state_before'] == result['model_state_after']
        expected = {(r['task'], r['seed']):r['sha256'] for r in read(job['manifest'])['rows']}
        rows = result['episodes']
        assert {(r['task'], r['seed']):r['case_sha256'] for r in rows} == expected
        parent = read(claimed)['parent']
        cache_key = parent['output']
        if cache_key not in common:
            step = read(parent['config'])['steps']
            reports = [read(Path(parent['output'])/f'evaluation-{step:07d}-{i}/metrics.json') for i in [0, 1]]
            assert all(r['passed'] and r['model_state_before'] == r['model_state_after'] for r in reports)
            assert {r['weights_sha256'] for r in reports} == {result['weights_sha256']}
            common[cache_key] = {(r['task'],r['seed']):r for report in reports for r in report['episodes']}
        key = f"{job['method']}-seed{job['seed']}-epoch{job.get('training_epochs',100)}"
        model = models.setdefault(key, dict(episodes=0, native_successes=0, matched_cem_successes=0,
            hard_episodes=0, native_hard_successes=0, matched_cem_hard_successes=0,
            native_only=0, cem_only=0, unique_cases=[], complete=False))
        for row in rows:
            trace = Path(row['trace'])
            assert hashlib.sha256(trace.read_bytes()).hexdigest() == row['trace_sha256']
            with np.load(trace) as z:
                actions, successes = z['actions'], z['successes']
                assert actions.shape == (row['steps'],4) and len(successes) == row['steps'] <= 500
                assert np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1
                assert bool(successes.any()) == row['success'] and row['initial_rgb_replay']['passed']
            other = common[cache_key][row['task'], row['seed']]
            assert row['case_sha256'] == other['case_sha256']
            native, cem = bool(row['success']), bool(other['success'])
            model['episodes'] += 1
            model['native_successes'] += native
            model['matched_cem_successes'] += cem
            model['native_only'] += native and not cem
            model['cem_only'] += cem and not native
            model['unique_cases'].append([row['task'],row['seed']])
            if row['task'] in hard:
                model['hard_episodes'] += 1
                model['native_hard_successes'] += native
                model['matched_cem_hard_successes'] += cem
    for model in models.values():
        assert len({tuple(c) for c in model['unique_cases']}) == model['episodes']
        model['complete'] = model['episodes'] == 1000
        model['macro50_percent'] = None
        model['native_partial_percent'] = 100*model['native_successes']/model['episodes']
        model['matched_cem_partial_percent'] = 100*model['matched_cem_successes']/model['episodes']
    report = dict(utc=now(), status='CLOSED_PARTIAL_BY_USER', passed=True, planned_shards=len(recipes),
        completed_shards=sum(r['status']=='COMPLETE' for r in recipes),
        interrupted_shards=sum(r['status']=='INTERRUPTED_BY_USER_PRIORITY' for r in recipes),
        deferred_shards=sum(r['status']=='DEFERRED_NOT_STARTED' for r in recipes),
        episodes=sum(m['episodes'] for m in models.values()), models=models, recipes=recipes,
        preserved_trace_files=sum(r['preserved_trace_files'] for r in recipes),
        uncertified_partial_traces=sum(r['preserved_trace_files'] for r in recipes if r['status']!='COMPLETE'),
        errors=[str(p) for p in (ROOT/'errors').glob('*.json')],
        scope='Only complete shards have final frozen-NN/RGB/actualtrace audits and enter the paired subset comparison. '
              'Interrupted partial traces retained but excluded. No model has the prespecified complete1000case score. '
              'NativeGD and commonCEM differ in compute/horizon/feedback/prior;this is a controller-bundle comparison,not a single-factor causal test.',
        model_updates=0, neural_models_loaded=0, render_calls=0, new_physical_episodes=0)
    save(OUT/'summary.json', report)
    save(ROOT/'summary.json', report)
    save(ROOT/'CLOSED.json', {k:report[k] for k in ['utc','status','completed_shards','interrupted_shards','deferred_shards','errors']})
    print(json.dumps({k:report[k] for k in ['status','completed_shards','interrupted_shards','deferred_shards','episodes','uncertified_partial_traces','errors']}))
    print(json.dumps(models))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'stop', 'audit'])
    parser.add_argument('--host')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'stop':
        stop(args.host)
    else:
        audit()
