"""One finite queue on a reserved GPU; unchanged historical scoring runner."""
import argparse
import datetime
import os
from pathlib import Path
import signal
import subprocess
import time

from common import read, sha, write


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True);p.add_argument('--host', required=True)
    p.add_argument('--gpu', required=True, type=int);a = p.parse_args()
    root = Path(a.root);protocol = read(root / 'protocol.json')
    assert protocol['passed'] and len(protocol['jobs']) == 8
    jobs = [x for x in protocol['jobs'] if (x['host'], x['gpu']) == (a.host, a.gpu)]
    assert len(jobs) in [1, 2]
    memory = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', str(a.gpu)], text=True).strip())
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader', '-i', str(a.gpu)], text=True).strip()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).splitlines()
    assert memory <= 1 and uuid not in {x.strip() for x in apps}
    write(root / 'workers' / f'{a.host}-gpu{a.gpu}-start.json', dict(pid=os.getpid(),
          utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), jobs=[x['id'] for x in jobs],
          protocol_sha256=sha(root / 'protocol.json'), worker_sha256=sha(__file__), initial_memory_mib=memory, no_compute_process=True))
    for job in jobs:
        assert sha(job['spec']) == job['spec_sha256']
        spec = read(job['spec']);out = Path(job['output']);out.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, PYTHONPATH=spec['model_source']['pythonpath'], CUDA_VISIBLE_DEVICES=str(a.gpu),
                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        started = time.monotonic()
        log = root / 'logs' / (job['id'] + '.log');log.parent.mkdir(exist_ok=True)
        with log.open('x') as f:
            child = subprocess.Popen(job['command'], cwd=spec['model_source']['source'], env=env,
                    stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            write(root / 'processes' / (job['id'] + '.json'), dict(pid=child.pid, command=job['command'], gpu=a.gpu, host=a.host))
            try:code = child.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid, signal.SIGKILL);child.wait()
                code = 124
        write(root / 'exits' / (job['id'] + '.json'), dict(passed=code == 0, returncode=code,
              seconds=time.monotonic()-started, spec_sha256=sha(job['spec']), log=str(log),
              utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        assert code == 0, str(log)
    write(root / 'workers' / f'{a.host}-gpu{a.gpu}.json', dict(passed=True, jobs=[x['id'] for x in jobs], monitor=False))
    print('Finite counterfactual queue complete', a.host, a.gpu)


if __name__ == '__main__':main()
