"""Exact native 200-update smoke; preserve resumable state, check pairing and interfaces."""
import argparse
import datetime
import os
from pathlib import Path
import signal
import subprocess
import time

from common import B, PY, read, write, sha, source_check


def audit(root, row, source):
    import torch
    from as_jepa_lewm.train import TrainModule
    from as_jepa_lewm.factory import gradient_expectations
    from experiments.unified_three_20261101.common import same
    from experiments.ideas_validate_20261031.preflight import checkpoint_report, finite
    torch.set_num_threads(2)
    run = Path(row['run'])
    cfg = read(run / 'config.json')
    state = torch.load(run / 'last.ckpt', map_location='cpu', weights_only=False)
    weights = torch.load(run / 'model-step0000200/weights.pt', map_location='cpu', weights_only=True)
    health = checkpoint_report(state, weights, cfg, 200)
    assert all(health['checks'].values()), health
    module = TrainModule(cfg)
    module.load_state_dict(state['state_dict'], strict=True)
    configured = module.configure_optimizers()
    optimizer = configured['optimizer']
    scheduler = configured['lr_scheduler']['scheduler']
    optimizer.load_state_dict(state['optimizer_states'][0])
    scheduler.load_state_dict(state['lr_schedulers'][0])
    assert same(module.state_dict(), state['state_dict'])
    assert same(optimizer.state_dict(), state['optimizer_states'][0])
    assert same(scheduler.state_dict(), state['lr_schedulers'][0])
    trace = [__import__('json').loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert all(finite(r) for r in trace)
    gradient = {}
    for prefix, expectation in gradient_expectations(module.model, cfg).items():
        maximum = max(float(r.get('grad_norm_' + prefix, 0)) for r in trace)
        if expectation == 'active':
            assert maximum > 0, (prefix, trace)
        if expectation == 'frozen':
            assert maximum == 0, prefix
        gradient[prefix] = dict(expectation=expectation, maximum=maximum)
    pairing = None
    if row['id'] in ['round2-P2-F7-s1', 'round2-P2-F7-s2']:
        initial = torch.load(run / 'initial-model/weights.pt', map_location='cpu', weights_only=True)
        reference = torch.load(Path(cfg['initial_encoder_checkpoint']) / 'weights.pt', map_location='cpu', weights_only=True)
        mismatch = [k for k in initial if k not in reference or not torch.equal(initial[k], reference[k])]
        assert set(initial) == set(reference) and not mismatch, mismatch
        pairing = dict(passed=True, tensors=len(initial), reference=cfg['initial_encoder_checkpoint'], reference_sha256=sha(Path(cfg['initial_encoder_checkpoint']) / 'weights.pt'))
    runtime = read(run / 'stage-runtime-source-0000200.json')
    assert runtime['cwd'] == source['source'] and runtime['PYTHONPATH'] == source['pythonpath']
    for key, path in runtime['paths'].items():
        assert Path(path).resolve().is_relative_to(Path(source['source']).resolve())
        assert sha(path) == runtime['sha256'][key]
    archive = run / 'full-archives/step0000200.ckpt'
    archive.parent.mkdir(exist_ok=True)
    os.link(run / 'last.ckpt', archive)
    report = dict(passed=True, id=row['id'], health=health, gradient=gradient, initial_pairing=pairing,
                  native_cpu_model_optimizer_scheduler_restore=True, runtime=runtime,
                  full_checkpoint_sha256=sha(archive), portable_sha256=sha(run / 'model-step0000200/weights.pt'),
                  trace=trace, initial_scale=read(run / 'initial_scale.json') if (run / 'initial_scale.json').exists() else None,
                  scope='Technical smoke only. 200 updates are before warmup ends; no task-performance or convergence verdict.')
    write(root / 'audits' / f'{row["id"]}-200.json', report)


def main(root, ident):
    root = Path(root)
    protocol = read(root / 'protocol.json')
    source = protocol['source']
    source_check(source)
    row = next(r for r in protocol['jobs'] if r['id'] == ident)
    cfg = read(Path(row['run']) / 'config.json')
    assert sha(Path(row['run']) / 'config.json') == row['config_sha256']
    assert not (root / 'HOLD').exists() and not (root / 'STOP').exists()
    env = dict(os.environ, **protocol['environment'], PYTHONPATH=source['pythonpath'], PYTHONDONTWRITEBYTECODE='1')
    env['TMPDIR'] = '/tmp/spt-lingjing-fullspec-20261102'
    Path(env['TMPDIR']).mkdir(exist_ok=True, parents=True)
    env['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    started = time.monotonic()
    commands = []
    def execute(name, command, timeout):
        log = root / 'logs' / f'{ident}-{name}.log'
        with log.open('x') as handle:
            child = subprocess.Popen(command, cwd=source['source'], env=env, stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            write(root / 'children' / f'{ident}-{name}.json', dict(pid=child.pid, command=command, cwd=source['source'],
                  CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'], host=row['host']))
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                code = 124
        commands.append(dict(name=name, command=command, returncode=code, log=str(log)))
        assert code == 0, str(log)
    passed = False
    error = None
    try:
        execute('train200', [PY, '-B', '-u', '-m', 'as_jepa_tools.train_stage', '--config',
                str(Path(row['run']) / 'config.json'), '--stop-step', '200'], 2400)
        assert read(Path(row['run']) / 'status.json')['step'] == 200
        audit(root, row, source)
        if ident in ['round2-P2-F7-s1', 'round2-K-K1-s1']:
            command = [PY, '-B', '-u', '-m', 'as_jepa_tools.rollout_gain', '--checkpoint',
                       str(Path(row['run']) / 'model-step0000200'), '--data', cfg.get('evaluation_data', cfg['data']),
                       '--output', str(root / 'interfaces' / ident / 'rollout'), '--cases', '16', '--fp32']
            if row['task'] != 'pusht':
                command[command.index('--data') + 1] = cfg['data']
                command += ['--dataset', 'benchmark']
            execute('rollout-interface', command, 900)
        if ident == 'round2-T-T4-s1':
            execute('benchmark-interface', [PY, '-B', '-u', '-m', 'as_jepa_tools.benchmark_evaluate', '--checkpoint',
                    str(Path(row['run']) / 'model-step0000200'), '--manifest', protocol['manifests']['tworoom']['primary'],
                    '--data', cfg['data'], '--output', str(root / 'interfaces' / ident / 'planning'), '--limit', '2'], 1200)
        passed = True
    except Exception as exc:
        error = repr(exc)
    write(root / 'smoke-results' / f'{ident}.json', dict(passed=passed, id=ident, commands=commands, error=error,
          seconds=time.monotonic() - started, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    if not passed:
        raise RuntimeError(error)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--id', required=True)
    a = p.parse_args()
    main(a.root, a.id)
