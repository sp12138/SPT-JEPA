"""Finite dependency: seed0 now; seeds1/2 after the owned hard-task queue exits. No LLM monitor."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import time
from common import B, PY, read, write, sha, source_check


def main(root):
    root = Path(root); p = read(root / 'protocol.json'); source_check(p['source']); source_check(p['helper'])
    jobs = {r['seed']: r for r in p['jobs']}; running = {}; started = time.monotonic(); launched_pair = False
    (root / 'logs').mkdir(exist_ok=True)
    def launch(seed):
        row = jobs[seed]; gpu = row['gpu']
        memory = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', str(gpu)], text=True).strip())
        assert memory <= 1, (seed, gpu, memory)
        env = dict(os.environ, PYTHONPATH=p['source']['pythonpath'], PYTHONDONTWRITEBYTECODE='1')
        command = [PY, '-B', '-u', p['helper']['source'] + '/experiments/lingjing_fullspec_20261102/p1_worker.py', '--root', str(root), '--id', row['id']]
        log = (root / 'logs' / (row['id'] + '-worker.log')).open('x')
        child = subprocess.Popen(command, cwd=p['source']['source'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(root / 'launches' / (row['id'] + '.json'), dict(pid=child.pid, gpu=gpu, memory_before=memory, command=command, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        running[seed] = (child, log)
    assert read(B / 'reports/lingjing_fullspec_20261102/p1-interface-v1/proof.json')['passed']
    launch(0)
    exits = {}
    while running or not launched_pair:
        if time.monotonic() - started > 172800:raise TimeoutError('Finite48h coordinator cap reached; preserve all workers/checkpoints for explicit recovery.')
        for seed, (child, log) in list(running.items()):
            code = child.poll()
            if code is not None:log.close(); exits[str(seed)] = code; del running[seed]
        if not launched_pair:
            hard = B / 'runs/lingjing_fullspec_hard2000_20261102'
            done = [hard / 'dispatchers' / (host + '-finished.json') for host in ['dev2', 'dev13', 'dev15']]
            if all(x.exists() for x in done):
                assert all(len(read(x)['results']) == 12 for x in done)
                # Independent P1 controls remain useful even if a hard-task technical chain failed.
                launch(1); launch(2); launched_pair = True
        if running or not launched_pair:time.sleep(30)
    terminals = [read(root / 'terminals' / (r['id'] + '.json')) for r in p['jobs']]
    report = dict(passed=all(r['passed'] for r in terminals), training_complete=all(r['training_complete'] for r in terminals),
                  terminals=terminals, exits=exits, protocol_sha256=sha(root / 'protocol.json'),
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), automatic_extension=False)
    write(root / 'results/summary.json', report)
    write(root / 'HOLD', dict(reason='Finite matched L0 budget ended; no extra epoch selected.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    main(p.parse_args().root)
