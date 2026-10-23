"""Run one frozen 2048-case FVD evaluation with the reference evaluator."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('report', type=Path)
args = parser.parse_args()
root = args.report.resolve()
plan = json.loads((root/'plan.json').read_text())
jobs = root/'jobs'
output = Path(plan['output'])
cache = Path(plan['cache_root'])
expected_protocol = plan.get('protocol_sha256', plan.get('reference_protocol_sha256'))
lock = (root/'worker.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
active = None

def status(state, **fields):
    record = dict(status=state, time=time.time(), host=socket.gethostname(), pid=os.getpid(),
                  checkpoint=plan['checkpoint'], step=plan['step'], output=str(output), **fields)
    temp = root/'status.json.tmp'
    temp.write_text(json.dumps(record,indent=2)+'\n')
    temp.replace(root/'status.json')
    print(json.dumps(record),flush=True)

def stop(signum, _frame):
    if active is not None and active.poll() is None:
        os.killpg(active.pid, signal.SIGTERM)
        try:
            active.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(active.pid, signal.SIGKILL)
            active.wait()
    raise KeyboardInterrupt('signal '+str(signum))

signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)
try:
    assert plan.get('allowed_host', 'dev8') in socket.gethostname(), 'Wrong evaluation host'
    for name,digest in plan['source_hashes'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    sys.path[:0] = [str(jobs),plan['runtime']]
    from common import verify_runtime
    from fvd_coordination import protocol_hash, valid_score
    from vforge.training.checkpoint import read_manifest
    verify_runtime()
    manifest = read_manifest(plan['checkpoint'])
    assert manifest.completed_steps==plan['step'] and manifest.state=='full'
    assert all(manifest.components_present[k] for k in ['weights','optimizer','trainer'])
    assert protocol_hash()==expected_protocol
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total',
                                   '--format=csv,noheader'],text=True).splitlines()
    assert len(gpu)==8 and all('H200' in g for g in gpu)
    busy = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                    '--format=csv,noheader'],text=True).strip()
    assert not busy,'GPUs occupied: '+busy
    env = dict(os.environ,PYTHONPATH=plan['runtime'],PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
               HYDRA_FULL_ERROR='1',FVD_EVALUATION_HARDWARE='H200',
               FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED='1',
               FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=str(cache/'scaling_fa4_h200/fa4'),
               CUTE_DSL_CACHE_DIR=str(cache/'scaling_fa4_h200/cute'),
               TRITON_CACHE_DIR=str(cache/'triton'),
               TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
               CUDA_CACHE_PATH=str(cache/'cuda'),XDG_CACHE_HOME=str(cache))
    command = [sys.executable,'-u','-m','torch.distributed.run','--standalone',
               '--nproc-per-node=8','--max-restarts=0','--monitor-interval=3',
               str(jobs/'eval_entry.py'),plan['name'],plan['checkpoint'],str(output)]
    status('running',command=command,cases_completed=0,cases_total=2048)
    started = time.time()
    with (root/'evaluation.log').open('a') as handle:
        active = subprocess.Popen(command,cwd=plan['runtime'],env=env,stdout=handle,
                                  stderr=subprocess.STDOUT,start_new_session=True)
        while active.poll() is None:
            completed = len(list(output.glob('fvd_*.npy.json')))
            status('running',child_pid=active.pid,cases_completed=completed,cases_total=2048,
                   elapsed_seconds=time.time()-started)
            try:
                active.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        code = active.returncode
    assert code==0,'Evaluation exited '+str(code)+'; see evaluation.log'
    assert valid_score(output,expected_protocol),'Final score validation failed'
    score = json.loads((output/'score.json').read_text())
    status('completed',cases_completed=2048,cases_total=2048,fvd=score['metric']['fvd'],
           elapsed_seconds=time.time()-started,score=str(output/'score.json'))
except BaseException as exc:
    status('failed',error=repr(exc))
    raise
