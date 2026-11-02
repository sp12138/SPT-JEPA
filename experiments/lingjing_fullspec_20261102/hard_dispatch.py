"""Finite36-job hard-task queue, all24slots available only after old owned queues exit."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import time
from common import PY, read, write, source_check


def main(root, host, slots=tuple(range(8))):
    root = Path(root); p = read(root / 'protocol.json'); source_check(p['source']); source_check(p['helper'])
    pending = [r for r in p['jobs'] if r['host'] == host]; running = {}; results = []
    (root / 'logs').mkdir(exist_ok=True)
    assert slots and len(set(slots)) == len(slots) and all(0 <= gpu < 8 for gpu in slots)
    write(root / 'dispatchers' / (host + '-start.json'), dict(pid=os.getpid(), slots=list(slots), jobs=[r['id'] for r in pending], started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    while pending or running:
        for gpu, (child, log, job) in list(running.items()):
            code = child.poll()
            if code is not None:
                log.close(); results.append(dict(id=job['id'], returncode=code, gpu=gpu)); del running[gpu]
        if (root / 'HOLD').exists() or (root / 'STOP').exists():pending.clear()
        for gpu in slots:
            if not pending or gpu in running:continue
            memory = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', str(gpu)], text=True).strip())
            assert memory <= 1, (gpu, memory)
            row = pending.pop(0); env = dict(os.environ, PYTHONPATH=p['source']['pythonpath'], PYTHONDONTWRITEBYTECODE='1')
            command = [PY, '-B', '-u', p['helper']['source'] + '/experiments/lingjing_fullspec_20261102/hard_worker.py', '--root', str(root), '--id', row['id'], '--gpu', str(gpu)]
            log = (root / 'logs' / (row['id'] + '-worker.log')).open('x')
            child = subprocess.Popen(command, cwd=p['source']['source'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(root / 'processes' / (row['id'] + '.json'), dict(pid=child.pid, gpu=gpu, host=host, command=command))
            running[gpu] = (child, log, row)
        if pending or running:time.sleep(5)
    write(root / 'dispatchers' / (host + '-finished.json'), dict(passed=all(r['returncode'] == 0 for r in results), results=results, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--host', required=True); p.add_argument('--slots', type=int, nargs='+', default=list(range(8)))
    a = p.parse_args(); main(a.root, a.host, a.slots)
