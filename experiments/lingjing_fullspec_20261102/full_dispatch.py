"""Finite host queue sharing only completed slots of the owned short-stage graph."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import time
from common import B, PY, read, write, source_check


def live(pid, token):
    path = Path('/proc') / str(pid) / 'cmdline'
    try:return token.encode() in path.read_bytes()
    except OSError:return False


def main(root, host):
    root = Path(root); p = read(root / 'protocol.json'); source_check(p['source']); source_check(p['helper'])
    pending = [row for row in p['jobs'] if row['host'] == host]; running = {}; results = []; started = time.monotonic()
    hard = B / 'runs/lingjing_fullspec_hard2000_20261102'; parent = read(hard / 'protocol.json')
    old_jobs = [row for row in parent['jobs'] if row['host'] == host]
    assert len(old_jobs) == 12
    (root / 'logs').mkdir(exist_ok=True)
    write(root / 'dispatchers' / (host + '-start.json'), dict(pid=os.getpid(), slots=p['slots'][host], jobs=[row['id'] for row in pending], started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    while pending or running:
        if time.monotonic() - started > 259200:raise TimeoutError('Finite72h queue cap reached; preserve running jobs and source for explicit recovery.')
        for gpu, (child, log, row) in list(running.items()):
            code = child.poll()
            if code is not None:
                log.close(); results.append(dict(id=row['id'], gpu=gpu, returncode=code)); del running[gpu]
        stopped = (root / 'STOP').exists() or (root / 'HOLD').exists() or (hard / 'STOP').exists()
        if stopped:
            for row in pending:
                write(root / 'terminals' / (row['id'] + '.json'), dict(passed=False, id=row['id'], outcome='cancelled_before_start', nominal_budget_complete=False))
                results.append(dict(id=row['id'], gpu=None, returncode=130))
            pending.clear()
            for child, log, row in running.values():
                marker = Path(row['run']) / 'STOP'
                if not marker.exists():write(marker, dict(reason='Cooperative campaign stop; retain checkpoint.'))
        old_records = [hard / 'processes' / (row['id'] + '.json') for row in old_jobs]
        # No slot may be taken before the old finite queue has dispatched its entire immutable host list.
        all_dispatched = all(path.exists() for path in old_records)
        busy = set()
        if all_dispatched:
            for path in old_records:
                record = read(path)
                if live(record['pid'], str(hard)):busy.add(record['gpu'])
        for gpu in p['slots'][host]:
            if not pending or gpu in running or not all_dispatched or gpu in busy:continue
            selected = None
            for i, row in enumerate(pending):
                dependency = row.get('calibration_dependency')
                if not dependency or Path(dependency).exists():selected = i;break
                parent_id = f'round2-P3-a-s{row["seed"]}'; terminal = root / 'terminals' / (parent_id + '.json')
                if terminal.exists() and read(terminal).get('outcome') in ['execution_error', 'cancelled_before_start']:
                    write(root / 'terminals' / (row['id'] + '.json'), dict(passed=False, id=row['id'], outcome='blocked_by_failed_calibration_input', nominal_budget_complete=False))
                    results.append(dict(id=row['id'], gpu=None, returncode=125)); selected = -i-1;break
            if selected is None:continue
            if selected < 0:pending.pop(-selected-1);continue
            memory = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', str(gpu)], text=True).strip())
            if memory > 1:continue
            row = pending.pop(selected); env = dict(os.environ, PYTHONPATH=p['source']['pythonpath'], PYTHONDONTWRITEBYTECODE='1')
            command = [PY, '-B', '-u', p['helper']['source'] + '/experiments/lingjing_fullspec_20261102/full_worker.py', '--root', str(root), '--id', row['id'], '--gpu', str(gpu)]
            log = (root / 'logs' / (row['id'] + '-worker.log')).open('x')
            child = subprocess.Popen(command, cwd=p['source']['source'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(root / 'processes' / (row['id'] + '.json'), dict(pid=child.pid, host=host, gpu=gpu, command=command))
            running[gpu] = (child, log, row)
        if pending or running:time.sleep(15)
    write(root / 'dispatchers' / (host + '-finished.json'), dict(passed=all(row['returncode'] == 0 for row in results), results=results, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--host', required=True)
    a = p.parse_args(); main(a.root, a.host)
