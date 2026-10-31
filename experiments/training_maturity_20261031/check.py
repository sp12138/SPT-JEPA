"""One read-only maturity check. No loop, model call, planning scores or fixes.

Local: python3 check.py --all --label 123026
Remote: python -B CHECKER --host dev2 --label 123026 --expected-sha SHA
The only writes are new timestamped JSON evidence under the fixed run root.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import time

B = Path('/efs/prescottsun/SPT-JEPA')
ROOT = B/'runs/cube_maturity23342_20261031'
REMOTE = B/'reports/training-maturity-20261031/checker-v3/check.py'
PY = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'
HOSTS = ('dev2', 'dev13', 'dev15')


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value): return str(value)
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    return value


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: print(json.dumps(clean(data), indent=2, allow_nan=False), file=f)


def gradient_evidence(status, records, family):
    """Finished summaries omit gradients; retain the actual training timestamp."""
    row, source = status, 'status'
    if status['state'] == 'finished' and not any(k.startswith('grad_norm_') for k in status):
        row = next((r for r in reversed(records) if r.get('state') == 'running'
                    and any(k.startswith('grad_norm_') for k in r)), {})
        source = 'last_logged_running_record' if row else 'missing'
    grads = {k: v for k, v in row.items() if k.startswith('grad_norm_')}
    active = {k: v for k, v in grads.items()
              if not (family == 'local' and k in ['grad_norm_encoder', 'grad_norm_pool'])}
    positive = bool(active) and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                                    for v in active.values())
    return dict(gradients=grads, active_gradient_positive=positive,
                gradient_source=source, gradient_evidence_step=row.get('step'),
                gradient_evidence_updated_at=row.get('updated_at'))


def check(host, label):
    plan, protocol = read(ROOT/'plan.json'), read(ROOT/'protocol.json')
    jobs = [j for j in plan['jobs'] if j['host'] == host]
    assert len(plan['jobs']) == 68 and len(jobs) == {'dev2': 20, 'dev13': 22, 'dev15': 26}[host]
    assert {j['gpu'] for j in jobs} == {0, 1}
    worker, launch = read(ROOT/'workers'/(host+'.json')), read(ROOT/('launch-'+host+'.json'))
    errors, rows, children, archives = [], [], [], []
    # Completed monitors legitimately exit; live dispatcher/monitor existence is
    # recorded here, without restarting or deciding to stop any process.
    process_exists = {k: Path('/proc', str(launch[k])).exists() for k in ['dispatcher_pid', 'monitor_pid']}
    for a in protocol['runs']:
        if a['host'] != host: continue
        run = ROOT/a['id']; status = read(run/'status.json')
        records = [json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines() if x]
        finite = all(math.isfinite(v) for row in records for k, v in row.items()
                     if isinstance(v, (int, float)) and (k in ['loss', 'fm', 'rate', 'pred', 'sigreg', 'feature_std'] or k.startswith('grad_norm_')))
        evidence = gradient_evidence(status, records, a['family'])
        if not finite or not evidence['active_gradient_positive'] or (run/'error.txt').exists(): errors.append(a['id']+': unhealthy trace or error.txt')
        if status['step'] > 23342: errors.append(a['id']+': configured cap exceeded')
        if sha(run/'config.json') != a['config_sha256']: errors.append(a['id']+': config hash changed')
        recent = [x['seconds_per_step'] for x in records if x['state'] == 'running'][-3:]
        rate = statistics.median(recent) if recent else status['seconds_per_step']
        rows.append(dict(id=a['id'], family=a['family'], state=status['state'], step=status['step'], loss=status['loss'],
            parts={k: status[k] for k in ['fm', 'rate', 'pred', 'sigreg', 'feature_std'] if k in status},
            **evidence, all_trace_finite=finite,
            recent_seconds_per_step=rate, remaining_training_hours=max(0, 23342-status['step'])*rate/3600,
            updated_at=status['updated_at'], trace_records=len(records),
            error=(run/'error.txt').read_text() if (run/'error.txt').exists() else None))
    for gpu, item in worker['running'].items():
        job = next(j for j in jobs if j['id'] == item['id']); proc = Path('/proc')/str(item['pid'])
        child = dict(gpu=int(gpu), job=job['id'], phase=job['phase'], pid=item['pid'], exists=proc.exists())
        try:
            command = [x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
            cwd = str((proc/'cwd').resolve())
            env = dict(x.split('=', 1) for x in (proc/'environ').read_bytes().decode(errors='replace').split(chr(0)) if '=' in x)
            child.update(command=command, cwd=cwd, original_source_environment_exact=command == job['command'] and cwd == job['source']['source']
                         and env['PYTHONPATH'] == job['source']['pythonpath'] and env['CUDA_VISIBLE_DEVICES'] == gpu)
            if not child['original_source_environment_exact']: errors.append(job['id']+': process/source mismatch')
        except (FileNotFoundError, ProcessLookupError):
            child['snapshot_race_possible'] = True; errors.append(job['id']+': recorded child absent during one snapshot')
        children.append(child)
    exits = {j['id']: read(ROOT/'exits'/(j['id']+'.json')) for j in jobs if (ROOT/'exits'/(j['id']+'.json')).exists()}
    failed = {k: v for k, v in exits.items() if v['returncode'] != 0}
    for job in jobs:
        if not job['phase'].startswith('archive') or job['id'] not in exits: continue
        path = Path(job['output'])/'fullstate.json'; proof = read(path) if path.exists() else {}
        exists = bool(proof.get('archive')) and Path(proof['archive']).is_file()
        passed = exits[job['id']]['returncode'] == 0 and proof.get('passed') and all(proof.get('checks', {}).values()) and exists
        archives.append(dict(job=job['id'], passed=bool(passed), step=proof.get('step'), archive_exists=exists,
                             checkpoint_sha256=proof.get('checkpoint_sha256'), path=str(path)))
        if not passed: errors.append(job['id']+': archive proof failed/missing')
    complete = len(exits) == len(jobs)
    stale = time.time()-worker['updated_at'] > 180
    if not complete and (stale or not all(process_exists.values())): errors.append('incomplete host: stale/missing dispatcher or monitor')
    gpu = subprocess.check_output(['nvidia-smi', '-i', '0,1', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    apps = subprocess.check_output(['nvidia-smi', '-i', '0,1', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    result = dict(at=time.time(), host=host, checker_sha256=sha(__file__), rows=rows, children=children,
        process_exists=process_exists, worker_pid=worker['pid'], worker_stale=stale,
        planned=len(jobs), completed=len(exits), complete=complete, failed=failed, archives=archives,
        owned_gpus=gpu, owned_apps=apps, errors=errors,
        scope='One read only. No planning outcome files read; no model/fit/simulation, control change, signal, retry or training. Archive hash is the existing audit record, not a repeated checkpoint rehash.')
    write(ROOT/('review'+label)/(host+'.json'), result)
    return dict(host=host, steps=[r['step'] for r in rows], completed=len(exits), failed=len(failed), errors=errors)


def aggregate(label):
    folder = ROOT/('review'+label)
    hosts = [read(folder/(h+'.json')) for h in HOSTS]
    # Accept the already-saved12:00 one-shot records, avoiding a second GPU read.
    for host in hosts:
        for row in host['rows']:
            if 'remaining_training_hours' not in row:
                row['remaining_training_hours'] = max(0, 23342-row['step'])*row['seconds_per_step']/3600
    result = dict(label=label, at=time.time(), checker_sha256=sha(__file__), hosts=hosts,
        planned=68, completed=sum(h['completed'] for h in hosts), failed=sum(len(h['failed']) for h in hosts),
        errors=[dict(host=h['host'], message=e) for h in hosts for e in h['errors']],
        rows=[r for h in hosts for r in h['rows']], archives=[a for h in hosts for a in h['archives']],
        scope='All six fixed slots/root, one snapshot per host. No partial planning scores. ETA is recent throughput only, excluding archive/restart/evaluation overhead.')
    write(folder/'summary.json', result)
    return dict(path=str(folder/'summary.json'), completed=result['completed'], failed=result['failed'],
        errors=result['errors'], archive_count=len(result['archives']),
        rows=[dict(id=r['id'], step=r['step'], loss=r['loss'], finite=r['all_trace_finite'], active_gradients=r['active_gradient_positive'],
                   remaining_training_hours=round(r['remaining_training_hours'], 2)
                       if isinstance(r['remaining_training_hours'], (int, float)) else r['remaining_training_hours']) for r in result['rows']])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--label', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true'); group.add_argument('--host', choices=HOSTS); group.add_argument('--aggregate', action='store_true')
    p.add_argument('--expected-sha'); args = p.parse_args()
    assert re.fullmatch('[a-z0-9_-]{1,40}', args.label)
    if args.expected_sha: assert sha(__file__) == args.expected_sha
    if args.all:
        # Run this orchestrator locally: three independent one-shot SSH reads,
        # followed by aggregation of the already-saved shared-EFS artifacts.
        def remote(host, mode):
            command = shlex.join([PY, '-B', str(REMOTE), *mode, '--label', args.label, '--expected-sha', sha(__file__)])
            result = subprocess.run(['ssh', host, command], text=True, capture_output=True)
            if result.returncode: raise RuntimeError(f'{host} read failed ({result.returncode}):\n{result.stdout}\n{result.stderr}')
            return result.stdout
        with ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(lambda h: remote(h, ['--host', h]), HOSTS))
        print(remote('dev13', ['--aggregate']), end='')
    else: print(json.dumps(check(args.host, args.label) if args.host else aggregate(args.label), allow_nan=False))


if __name__ == '__main__': main()
