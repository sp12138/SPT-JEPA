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


def main(root, host):
    root = Path(root); p = read(root / 'protocol.json'); source_check(p['source']); source_check(p['helper'])
    parent = Path(p['repair_parent_root']); all_old = [r['old_process'] for r in p['jobs'] if r.get('old_process', {}).get('host') == host]
    pending = [r for r in p['jobs'] if r['host'] == host]; running = {}; results = []; started = time.monotonic()
    pending.sort(key=lambda row: (row['recovery_mode'] != 'reuse_finished_training', row.get('tier', 0), row['id']))
    (root / 'logs').mkdir(exist_ok=True)
    write(root / 'dispatchers' / (host + '-start.json'), dict(pid=os.getpid(), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), jobs=[r['id'] for r in pending]))
    while pending or running:
        if time.monotonic() - started > 259200:raise TimeoutError('Finite72h recovery cap; retain evidence and workers for explicit review.')
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
            command = [PY, '-B', '-u', p['helper']['source'] + '/experiments/lingjing_fullspec_20261102/full_worker.py', '--root', str(root), '--id', ready['id'], '--gpu', str(gpu)]
            env = dict(os.environ, PYTHONPATH=p['source']['pythonpath'], PYTHONDONTWRITEBYTECODE='1')
            log = (root / 'logs' / (ready['id'] + '-worker.log')).open('x')
            child = subprocess.Popen(command, cwd=p['source']['source'], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(root / 'processes' / (ready['id'] + '.json'), dict(pid=child.pid, host=host, gpu=gpu, command=command))
            running[gpu] = (child, log, ready);pending.remove(ready)
        if pending or running:time.sleep(15)
    write(root / 'dispatchers' / (host + '-finished.json'), dict(passed=all(r['returncode'] == 0 for r in results), results=results, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--host', required=True)
    a = p.parse_args(); main(a.root, a.host)
