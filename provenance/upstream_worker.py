"""Finite 8-H200 pretraining and sequential tuning/FVD branch workers."""
from copy import deepcopy
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

ROOT = Path('/efs/stormyxie/reports/rgb16_scale_budget_20261010')
PLAN = json.loads((ROOT / 'plan.json').read_text())
JOBS = Path(PLAN['jobs'])
RUNTIME = Path(PLAN['runtime'])
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
                FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED='1',
                FLASH_ATTENTION_CUTE_DSL_CACHE_DIR='/efs/stormyxie/cache/scaling_fa4_h200/fa4',
                CUTE_DSL_CACHE_DIR='/efs/stormyxie/cache/scaling_fa4_h200/cute')


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


def audit_branch(case, source):
    import torch
    from omegaconf import OmegaConf
    from vforge.training.checkpoint import read_manifest
    m = read_manifest(str(source))
    saved = m.fingerprints.get('semantic_config')
    if not isinstance(saved, dict):
        raise RuntimeError('Missing source semantic config')
    pre = OmegaConf.to_container(OmegaConf.load(PLAN['pretrain_config']), resolve=True)
    cfg = OmegaConf.to_container(OmegaConf.load(case['config']), resolve=True)
    if {k: pre[k] for k in saved} != saved:
        raise RuntimeError('Source checkpoint does not match campaign pretrain')
    permitted = deepcopy(saved)
    permitted['model']['extras']['conditioner']['context_mask_ratio'] = 0.0
    permitted['model']['extras']['flow']['distant_target_prob'] = 0.0
    if {k: cfg[k] for k in saved} != permitted:
        raise RuntimeError('Unexpected tuning semantic changes')
    if m.torch_version != str(torch.__version__) or m.python_version != sys.version.split()[0]:
        raise RuntimeError('Resume runtime mismatch')
    assert all(cfg['resume'][k] for k in ['weights', 'optimizer', 'training_state'])
    atomic(WORKER / f'p{case["pretrain_steps"]}_boundary_audit.json',
           dict(status='PASS', source=str(source), completed_steps=m.completed_steps,
                components=m.components_present, allowed_changes=['mask_ratio', 'distant_target_prob']))


def evaluate(case):
    fj = Path(PLAN['evaluation']['report']) / 'jobs'
    sys.path.insert(0, str(fj))
    from fvd_coordination import valid_score, protocol_hash
    out = Path(PLAN['evaluation']['report']) / 'fvd/results' / case['name'] / f'{case["completed_steps"]:05d}'
    if valid_score(out, protocol_hash()):
        return
    verify_sources()
    wait_idle()
    ckpt = checkpoint(case['output'], case['completed_steps'])
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc-per-node=8', '--max-restarts=0', '--monitor-interval=3',
               str(fj / 'eval_entry.py'), case['name'], str(ckpt), str(out)]
    run(command, WORKER / f'p{case["pretrain_steps"]}_fvd.log', 'fvd',
        pretrain_steps=case['pretrain_steps'], checkpoint=str(ckpt), output=str(out))
    if not valid_score(out, protocol_hash()):
        raise RuntimeError('FVD returned without valid full score')


def branch_worker():
    summary = []
    for case in PLAN['cases']:
        step = case['pretrain_steps']
        deadline = time.monotonic() + 30 * 86400
        while True:
            source = checkpoint(PLAN['pretrain_output'], step)
            if source:
                break
            if (ROOT / 'workers/pretrain/FAIL.json').exists():
                raise RuntimeError('Main pretraining failed before required checkpoint')
            if time.monotonic() > deadline:
                raise TimeoutError('No pretraining milestone after 30 days')
            progress(status='waiting_for_checkpoint', pretrain_steps=step)
            time.sleep(60)
        audit_branch(case, source)
        train(case['config'], case['output'], case['completed_steps'], f'p{step}_tune')
        evaluate(case)
        score_path = Path(PLAN['evaluation']['report']) / 'fvd/results' / case['name'] / f'{case["completed_steps"]:05d}' / 'score.json'
        summary.append(dict(pretrain_steps=step, tuning_steps=2000,
                            score=str(score_path), fvd=json.loads(score_path.read_text())['metric']['fvd']))
        atomic(ROOT / 'comparison.json', dict(rows=summary, complete=len(summary) == 5))


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
    if MODE not in ['pretrain', 'branches']:
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
        if MODE == 'pretrain':
            train(PLAN['pretrain_config'], PLAN['pretrain_output'], 50000, 'pretrain')
        else:
            branch_worker()
        progress(status='completed')
        atomic(WORKER / 'PASS.json', dict(mode=MODE, time=time.time(), pid=os.getpid()))
    except BaseException as exc:
        atomic(WORKER / 'FAIL.json', dict(error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
        progress(status='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
