"""Finite smoke queue; no recurring monitor and no implicit long continuation."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import time

from common import PY, read, write, source_check


def main(root, host):
    root = Path(root)
    p = read(root / 'protocol.json')
    source_check(p['source'])
    pending = [r for r in p['jobs'] if r['host'] == host]
    running = {}
    results = []
    write(root / 'dispatchers' / f'{host}-start.json', dict(pid=os.getpid(), jobs=[r['id'] for r in pending],
          started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    (root / 'logs').mkdir(exist_ok=True)
    while pending or running:
        for gpu, (child, log, row) in list(running.items()):
            code = child.poll()
            if code is not None:
                log.close()
                results.append(dict(id=row['id'], gpu=gpu, returncode=code))
                del running[gpu]
        if (root / 'STOP').exists() or (root / 'HOLD').exists():
            pending.clear()
        for gpu in range(8):
            if not pending or gpu in running:
                continue
            memory = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', str(gpu)], text=True).strip())
            assert memory <= 1, (gpu, memory)
            row = pending.pop(0)
            env = dict(os.environ, PYTHONPATH=p['source']['pythonpath'], CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1')
            command = [PY, '-B', '-u', p['source']['source'] + '/experiments/lingjing_fullspec_20261102/smoke_worker.py', '--root', str(root), '--id', row['id']]
            log = (root / 'logs' / (row['id'] + '-worker.log')).open('x')
            child = subprocess.Popen(command, cwd=p['source']['source'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(root / 'processes' / (row['id'] + '.json'), dict(pid=child.pid, gpu=gpu, host=host, command=command))
            running[gpu] = (child, log, row)
        if pending or running:
            time.sleep(5)
    write(root / 'dispatchers' / f'{host}-finished.json', dict(passed=all(r['returncode'] == 0 for r in results), results=results,
          finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--host', required=True)
    a = p.parse_args()
    main(a.root, a.host)
