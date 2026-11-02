"""Finite, dependency-safe recovery; old workers finish and old failure evidence stays immutable."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import time
from common import PY, read, write, sha, source_check


def alive(record):
    path = Path('/proc') / str(record['pid']) / 'cmdline'
    try:return [v.decode() for v in path.read_bytes().split(b'\0') if v] == record['command']
    except OSError:return False


def resume_partition(root, rows):
    """Recorded workers are adopted, never launched twice; all original terminals stay intact."""
    pending, adopted, results = [], [], []
    for row in rows:
        terminal = root / 'terminals' / (row['id'] + '.json')
        process = root / 'processes' / (row['id'] + '.json')
        if terminal.exists():
            results.append(dict(id=row['id'], gpu=None, returncode=0 if read(terminal).get('passed') else 1))
        elif process.exists():adopted.append((row, read(process)))
        else:pending.append(row)
    return pending, adopted, results


def main(root, host, resume_owned=False):
    root = Path(root); p = read(root / 'protocol.json'); source_check(p['source']); source_check(p['helper'])
    parent = Path(p['repair_parent_root']); all_old = [r['old_process'] for r in p['jobs'] if r.get('old_process', {}).get('host') == host]
    pending = [r for r in p['jobs'] if r['host'] == host]; running = {}; results = []; adopted = []; started = time.monotonic()
    remaining_seconds = 259200
    if resume_owned:
        original = read(root / 'dispatchers' / (host + '-start.json'))
        elapsed = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(original['started_utc'])).total_seconds()
        remaining_seconds -= elapsed
        assert remaining_seconds > 0
        all_old += [read(path) for path in (root / 'processes').glob('*.json') if read(path)['host'] == host]
        pending, adopted, results = resume_partition(root, pending)
    pending.sort(key=lambda row: (row['recovery_mode'] != 'reuse_finished_training', row.get('tier', 0), row['id']))
    (root / 'logs').mkdir(exist_ok=True)
    start_name = host + ('-save-io-v8-start.json' if resume_owned else '-start.json')
    write(root / 'dispatchers' / start_name, dict(pid=os.getpid(), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), jobs=[r['id'] for r in pending], adopted=[r['id'] for r, _ in adopted], remaining_seconds=remaining_seconds))
    while pending or running or adopted:
        if time.monotonic() - started > remaining_seconds:raise TimeoutError('Finite72h recovery cap; retain evidence and workers for explicit review.')
        for row, record in list(adopted):
            terminal = root / 'terminals' / (row['id'] + '.json')
            if alive(record):continue
            if not terminal.exists():
                write(terminal, dict(id=row['id'], passed=False, outcome='adopted_worker_exited_without_terminal', nominal_budget_complete=False))
            results.append(dict(id=row['id'], gpu=record['gpu'], returncode=0 if read(terminal).get('passed') else 1))
            adopted.remove((row, record))
        for gpu, (child, log, row) in list(running.items()):
            code = child.poll()
            if code is not None:log.close();results.append(dict(id=row['id'], gpu=gpu, returncode=code));del running[gpu]
        if (root / 'STOP').exists():
            for row in pending:
                write(root / 'terminals' / (row['id'] + '.json'), dict(id=row['id'], passed=False, outcome='cancelled_before_start', nominal_budget_complete=False))
                results.append(dict(id=row['id'], gpu=None, returncode=130))
            pending.clear()
            for _, _, row in running.values():
                marker = Path(row['run']) / 'STOP'
                if not marker.exists():write(marker, dict(reason='Cooperative recovery stop; preserve checkpoint.'))
            for row, _ in adopted:
                marker = Path(row['run']) / 'STOP'
                if not marker.exists():write(marker, dict(reason='Cooperative adopted-worker stop; preserve checkpoint.'))
        busy = {r['gpu'] for r in all_old if alive(r)}
        # Import completed Cube results without a new model/GPU/simulator call.
        for row in list(pending):
            if row['recovery_mode'] != 'inherit_completed_parent':continue
            terminal = Path(row['repair_parent_terminal'])
            if not terminal.exists():continue
            try:value = read(terminal)
            except ValueError:continue
            original = dict(path=str(terminal), sha256=sha(terminal), terminal=value)
            passed = value.get('passed') and value.get('nominal_budget_complete')
            if passed:
                target = root / 'evaluations' / row['id']
                target.symlink_to(parent / 'evaluations' / row['id'], target_is_directory=True)
            write(root / 'terminals' / (row['id'] + '.json'), dict(id=row['id'], passed=bool(passed),
                outcome='inherited_nominal_budget_complete' if passed else 'parent_requires_technical_review',
                nominal_budget_complete=bool(passed), parent=original, new_training_updates=0, new_simulations=0))
            results.append(dict(id=row['id'], gpu=None, returncode=0 if passed else 1));pending.remove(row)
        for gpu in p['slots'][host]:
            if gpu in busy or gpu in running:continue
            ready = None
            for row in pending:
                if row['recovery_mode'] == 'fresh_after_technical_release':
                    if not (root / 'FRESH_DISPATCH_RELEASE.json').exists():continue
                    dependency = row.get('calibration_dependency')
                    if dependency and not Path(dependency).exists():continue
                    ready = row;break
                if row['recovery_mode'] != 'reuse_finished_training':continue
                terminal = Path(row['repair_parent_terminal'])
                if not terminal.exists():continue
                try:value = read(terminal)
                except ValueError:continue
                expected = value.get('outcome') == 'execution_error' and 'recent_minimum' in str(value.get('error'))
                if not expected:
                    write(root / 'terminals' / (row['id'] + '.json'), dict(id=row['id'], passed=False, outcome='unexpected_parent_failure_requires_review', nominal_budget_complete=False, parent=str(terminal)))
                    results.append(dict(id=row['id'], gpu=None, returncode=1));pending.remove(row);break
                row['repair_parent_terminal_sha256'] = sha(terminal);ready = row;break
            if ready is None:continue
            memory = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', str(gpu)], text=True).strip())
            if memory > 1:continue
            # Runtime binding lives separately; the immutable protocol never changes.
            if ready['recovery_mode'] == 'reuse_finished_training':
                write(root / 'bindings' / (ready['id'] + '.json'), dict(parent_terminal_sha256=ready['repair_parent_terminal_sha256']))
            worker = str(Path(__file__).with_name('full_worker.py')) if resume_owned else p['helper']['source'] + '/experiments/lingjing_fullspec_20261102/full_worker.py'
            command = [PY, '-B', '-u', worker, '--root', str(root), '--id', ready['id'], '--gpu', str(gpu)]
            env = dict(os.environ, PYTHONPATH=p['source']['pythonpath'], PYTHONDONTWRITEBYTECODE='1')
            log = (root / 'logs' / (ready['id'] + '-worker.log')).open('x')
            child = subprocess.Popen(command, cwd=p['source']['source'], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(root / 'processes' / (ready['id'] + '.json'), dict(pid=child.pid, host=host, gpu=gpu, command=command))
            running[gpu] = (child, log, ready);pending.remove(ready)
        if pending or running or adopted:time.sleep(15)
    write(root / 'dispatchers' / (host + '-finished.json'), dict(passed=all(r['returncode'] == 0 for r in results), results=results, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--host', required=True); p.add_argument('--resume-owned', action='store_true')
    a = p.parse_args(); main(a.root, a.host, a.resume_owned)
