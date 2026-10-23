"""Finite 8-H200 pretraining and sequential tuning/FVD branch workers."""
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
import traceback
import uuid

ROOT = Path(os.environ['SPT_RUN_ROOT']).resolve()
PLAN = json.loads((ROOT / 'plan.json').read_text())
JOBS = Path(PLAN['jobs'])
RUNTIME = Path(PLAN['runtime'])
CACHE = Path(PLAN.get('cache_root', ROOT.parents[1] / 'cache'))
ACTIVE = None
MODE = sys.argv[1]
WORKER = ROOT / 'workers' / MODE


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources():
    for name, expected in json.loads((JOBS / 'runtime_hashes.json').read_text()).items():
        if digest(RUNTIME / name) != expected:
            raise RuntimeError('Runtime changed: ' + name)
    for name, expected in json.loads((JOBS / 'job_hashes.json').read_text()).items():
        if digest(ROOT / name) != expected:
            raise RuntimeError('Frozen job changed: ' + name)


def progress(**values):
    atomic(WORKER / 'status.json', dict(time=time.time(), pid=os.getpid(), host=socket.gethostname(),
                                      mode=MODE, **values))


def environment():
    return dict(os.environ, PYTHONPATH=str(RUNTIME), PYTHONUNBUFFERED='1',
                PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                MKL_NUM_THREADS='2', HYDRA_FULL_ERROR='1',
                TRITON_CACHE_DIR=str(CACHE / 'triton'),
                TORCHINDUCTOR_CACHE_DIR=str(CACHE / 'inductor'),
                CUDA_CACHE_PATH=str(CACHE / 'cuda'),
                XDG_CACHE_HOME=str(CACHE),
                FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED='1',
                FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=str(CACHE / 'scaling_fa4_h200/fa4'),
                CUTE_DSL_CACHE_DIR=str(CACHE / 'scaling_fa4_h200/cute'))


def check_hardware():
    gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total',
                                    '--format=csv,noheader'], text=True).splitlines()
    if len(gpus) != 8 or any('H200' not in g for g in gpus):
        raise RuntimeError('Worker requires 8 H200 GPUs')
    return gpus


def wait_idle():
    while True:
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader'], text=True).strip()
        if not busy:
            return
        progress(status='waiting_for_idle_gpus', busy_pids=busy.splitlines())
        time.sleep(60)


def checkpoint(output, step):
    from vforge.training.checkpoint import read_manifest
    path = Path(output) / 'checkpoints' / f'{step:05d}.dcp'
    if not (path / 'manifest.json').exists():
        return None
    m = read_manifest(str(path))
    if (m.completed_steps != step or m.state != 'full' or m.world_size != 8
            or not all(m.components_present.get(k) for k in ['weights', 'optimizer', 'trainer'])):
        raise RuntimeError('Incomplete full checkpoint: ' + str(path))
    return path


def run(command, log_path, phase, **identity):
    global ACTIVE
    progress(status='running', phase=phase, command=command, **identity)
    with log_path.open('a') as log:
        ACTIVE = subprocess.Popen(command, cwd=RUNTIME, env=environment(), stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        progress(status='running', phase=phase, child_pid=ACTIVE.pid, **identity)
        code = ACTIVE.wait()
        ACTIVE = None
    if code:
        raise RuntimeError(f'{phase} exited {code}; log: {log_path}')


def train(config_path, output, endpoint, label):
    from omegaconf import OmegaConf
    from vforge.training.checkpoint import discover_checkpoints
    if checkpoint(output, endpoint):
        return
    verify_sources()
    cfg = OmegaConf.load(config_path)
    existing = discover_checkpoints(str(Path(output) / 'checkpoints'))
    if existing:
        cfg.resume.path = 'auto'
        # Branch continuation has unchanged semantics and can be strict.
        cfg.resume.strict = True
    effective = WORKER / (label + '_effective.yaml')
    effective.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    wait_idle()
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc-per-node=8', '--max-restarts=0', '--monitor-interval=3',
               str(JOBS / 'train_entry.py'), str(effective)]
    run(command, WORKER / (label + '.log'), label, output=str(output), target_step=endpoint)
    if not checkpoint(output, endpoint):
        raise RuntimeError('Training returned without endpoint checkpoint')
    atomic(WORKER / (label + '_PASS.json'), dict(step=endpoint, output=str(output), time=time.time()))


def stop(signum, _frame):
    if ACTIVE is not None and ACTIVE.poll() is None:
        os.killpg(ACTIVE.pid, signal.SIGTERM)
        try:
            ACTIVE.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(ACTIVE.pid, signal.SIGKILL)
            ACTIVE.wait()
    raise KeyboardInterrupt(f'signal {signum}')


def main():
    if MODE != 'pretrain':
        raise ValueError(MODE)
    WORKER.mkdir(parents=True, exist_ok=True)
    lock = (WORKER / 'worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(RUNTIME))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        verify_sources()
        atomic(WORKER / 'node.json', dict(gpus=check_hardware(), host=socket.gethostname(),
                                         pid=os.getpid(), time=time.time()))
        (WORKER / 'FAIL.json').unlink(missing_ok=True)
        train(PLAN['pretrain_config'], PLAN['pretrain_output'], PLAN['max_steps'], 'pretrain')
        progress(status='completed')
        atomic(WORKER / 'PASS.json', dict(mode=MODE, time=time.time(), pid=os.getpid()))
    except BaseException as exc:
        atomic(WORKER / 'FAIL.json', dict(error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
        progress(status='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
