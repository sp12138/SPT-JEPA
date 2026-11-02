"""One finite native2000 chain with full-state restoration and frozen content/use diagnostics."""
import argparse
import datetime
import os
from pathlib import Path
import signal
import subprocess
import time

from common import PY, read, write, sha, source_check


def native_audit(root, row, source):
    import torch
    from as_jepa_lewm.train import TrainModule
    from as_jepa_lewm.factory import gradient_expectations
    from experiments.ideas_validate_20261031.preflight import checkpoint_report, finite
    from experiments.unified_three_20261101.common import same
    torch.set_num_threads(2)
    run = Path(row['run']); cfg = read(run / 'config.json')
    state = torch.load(run / 'last.ckpt', map_location='cpu', weights_only=False)
    weights = torch.load(run / 'model-step0002000/weights.pt', map_location='cpu', weights_only=True)
    report = checkpoint_report(state, weights, cfg, 2000); assert all(report['checks'].values()), report
    model = TrainModule(cfg); model.load_state_dict(state['state_dict'], strict=True)
    opt = model.configure_optimizers(); optimizer = opt['optimizer']; scheduler = opt['lr_scheduler']['scheduler']
    optimizer.load_state_dict(state['optimizer_states'][0]); scheduler.load_state_dict(state['lr_schedulers'][0])
    assert same(model.state_dict(), state['state_dict'])
    assert same(optimizer.state_dict(), state['optimizer_states'][0]) and same(scheduler.state_dict(), state['lr_schedulers'][0])
    trace = [__import__('json').loads(x) for x in (run / 'metrics.jsonl').read_text().splitlines()]; assert all(finite(x) for x in trace)
    gradients = {}
    for prefix, expectation in gradient_expectations(model.model, cfg).items():
        value = max(r.get('grad_norm_' + prefix, 0) for r in trace)
        if expectation == 'active': assert value > 0, prefix
        if expectation == 'frozen': assert value == 0, prefix
        gradients[prefix] = dict(expectation=expectation, maximum=value)
    initial = torch.load(run / 'initial-model/weights.pt', map_location='cpu', weights_only=True)
    reference = torch.load(Path(row['reference_initial']) / 'weights.pt', map_location='cpu', weights_only=True)
    common = [k for k in initial if k in reference and not k.startswith('pool.head.')]
    mismatch = [k for k in common if not torch.equal(initial[k], reference[k])]
    assert not mismatch, mismatch
    runtime = read(run / 'stage-runtime-source-0002000.json')
    assert runtime['cwd'] == source['source'] and runtime['PYTHONPATH'] == source['pythonpath']
    for key, path in runtime['paths'].items(): assert sha(path) == runtime['sha256'][key]
    archive = run / 'full-archives/step0002000.ckpt'; archive.parent.mkdir(exist_ok=True); os.link(run / 'last.ckpt', archive)
    result = dict(passed=True, checkpoint=report, gradients=gradients, native_cpu_model_optimizer_scheduler_restore=True,
                  initial_common_non_head_tensors_exact=len(common), initial_head_scope='Head follows the declared one-time calibration, including uncalibrated and sigma-resolution controls.',
                  runtime=runtime, full_checkpoint_sha256=sha(archive), portable_sha256=sha(run / 'model-step0002000/weights.pt'),
                  trace=trace, initial_scale=read(run / 'initial_scale.json') if (run / 'initial_scale.json').exists() else None)
    write(root / 'audits' / (row['id'] + '.json'), result)


def main(root, ident, gpu):
    root = Path(root); p = read(root / 'protocol.json'); source_check(p['source']); source_check(p['helper'])
    row = next(r for r in p['jobs'] if r['id'] == ident); run = Path(row['run']); cfg = read(run / 'config.json')
    assert sha(run / 'config.json') == row['config_sha256']
    assert not (root / 'STOP').exists() and not (root / 'HOLD').exists()
    env = dict(os.environ, **p['environment'], PYTHONPATH=p['source']['pythonpath'], CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), PYTHONDONTWRITEBYTECODE='1')
    env['TMPDIR'] = '/tmp/spt-lingjing-hard-wave'; Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)
    results = []; started = time.monotonic(); passed = False; error = None
    def execute(label, command, timeout):
        log = root / 'logs' / (ident + '-' + label + '.log')
        with log.open('x') as handle:
            child = subprocess.Popen(command, cwd=p['source']['source'], env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            write(root / 'children' / (ident + '-' + label + '.json'), dict(pid=child.pid, command=command, host=row['host'], gpu=gpu))
            try: code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                code = 124
        results.append(dict(label=label, returncode=code, log=str(log))); assert code == 0, str(log)
    try:
        execute('train2000', [PY, '-B', '-u', '-m', 'as_jepa_tools.train_stage', '--config', str(run / 'config.json'), '--stop-step', '2000'], 14400)
        native_audit(root, row, p['source'])
        cp = str(run / 'model-step0002000'); out = root / 'evaluations' / ident
        tool = 'df_state_check' if row['model_class'] == 'decoder_free' else 'round2_state_check'
        execute('state', [PY, '-B', '-u', p['helper']['source'] + f'/as_jepa_tools/{tool}.py', '--checkpoint', cp, '--config', str(run / 'config.json'), '--output', str(out / 'state')], 1800)
        if row['model_class'] != 'decoder_free':
            execute('content', [PY, '-B', '-u', p['helper']['source'] + '/experiments/lingjing_fullspec_20261102/hard_content.py', '--checkpoint', cp, '--config', str(run / 'config.json'), '--output', str(out / 'content')], 1800)
        passed = True
    except Exception as exc:error = repr(exc)
    write(root / 'terminals' / (ident + '.json'), dict(passed=passed, id=ident, results=results, error=error, seconds=time.monotonic() - started,
          finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), further_training='HOLD pending paired scientific review'))
    if not passed:raise RuntimeError(error)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--id', required=True); p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args(); main(a.root, a.id, a.gpu)
