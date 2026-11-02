"""Finite frozen diagnostic jobs on slots disjoint from the200-update smokes."""
import argparse
import datetime
import os
from pathlib import Path
import signal
import subprocess
import time

from common import read, write, sha, source_check


def main(root, host):
    root = Path(root); p = read(root / 'protocol.json'); source_check(p['source']); source_check(p['helper'])
    pending = [j for j in p['jobs'] if j['host'] == host]; running = {}; results = []
    (root / 'logs').mkdir(exist_ok=True)
    write(root / 'dispatchers' / f'{host}-start.json', dict(pid=os.getpid(), jobs=len(pending), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    while pending or running:
        for gpu, (child, log, job, started) in list(running.items()):
            code = child.poll()
            if code is None and time.monotonic() - started > 1800:
                os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=30); code = 124
            if code is not None:
                log.close(); proof = None
                if code == 0:
                    proof = read(Path(job['output']) / 'state-check.json')
                    assert proof['state_before'] == proof['state_after'] and proof['model_updates'] == 0
                    assert sha(Path(job['checkpoint']) / 'weights.pt') == job['weights_sha256']
                row = dict(id=job['id'], returncode=code, gpu=gpu, proof=proof)
                write(root / 'terminals' / (job['id'] + '.json'), row)
                results.append(dict(id=job['id'], returncode=code, gpu=gpu)); del running[gpu]
        if (root / 'STOP').exists() or (root / 'HOLD').exists(): pending.clear()
        for gpu in p['reserved_slots'][host]:
            if not pending or gpu in running:continue
            memory = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', str(gpu)], text=True).strip())
            assert memory <= 1, (gpu, memory)
            job = pending.pop(0)
            assert sha(job['config']) == job['config_sha256']
            env = dict(os.environ, PYTHONPATH=p['source']['pythonpath'], CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1')
            log = (root / 'logs' / (job['id'] + '.log')).open('x')
            child = subprocess.Popen(job['command'], cwd=p['source']['source'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(root / 'processes' / (job['id'] + '.json'), dict(pid=child.pid, gpu=gpu, host=host, command=job['command']))
            running[gpu] = (child, log, job, time.monotonic())
        if pending or running:time.sleep(5)
    write(root / 'dispatchers' / f'{host}-finished.json', dict(passed=all(r['returncode'] == 0 for r in results), results=results,
          finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--host', required=True)
    a = p.parse_args(); main(a.root, a.host)
