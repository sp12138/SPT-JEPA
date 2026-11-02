"""Finite matched-baseline training, full restores, raw/precise-BN planning and probes."""
import argparse
import datetime
import os
from pathlib import Path
import signal
import subprocess
import time
from common import B, PY, read, write, sha, source_check
from hard_worker import native_audit


def main(root, ident):
    root = Path(root); p = read(root / 'protocol.json'); source_check(p['source']); source_check(p['helper'])
    row = next(r for r in p['jobs'] if r['id'] == ident); run = Path(row['run']); cfg = read(run / 'config.json')
    assert sha(run / 'config.json') == row['config_sha256']
    assert not (root / 'HOLD').exists() and not (root / 'STOP').exists()
    env = dict(os.environ, **p['environment'], PYTHONPATH=p['source']['pythonpath'], CUDA_VISIBLE_DEVICES=str(row['gpu']),
               MUJOCO_EGL_DEVICE_ID=str(row['gpu']), PYTHONDONTWRITEBYTECODE='1')
    env['TMPDIR'] = '/tmp/spt-lingjing-p1'; Path(env['TMPDIR']).mkdir(exist_ok=True, parents=True)
    (root / 'logs').mkdir(exist_ok=True)
    results = []; started = time.monotonic(); error = None; trained = False
    def execute(name, command, timeout, required=True):
        log = root / 'logs' / (ident + '-' + name + '.log')
        with log.open('x') as f:
            child = subprocess.Popen(command, cwd=p['source']['source'], env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            write(root / 'children' / (ident + '-' + name + '.json'), dict(pid=child.pid, command=command, host=row['host'], gpu=row['gpu']))
            try: code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid, signal.SIGKILL);child.wait()
                code = 124
        results.append(dict(name=name, command=command, returncode=code, log=str(log)))
        if required:assert code == 0, str(log)
        return code
    try:
        for index, step in enumerate(p['stages']):
            command = [PY, '-B', '-u', '-m', 'as_jepa_tools.train_stage', '--config', str(run / 'config.json'), '--stop-step', str(step)]
            if index:command.append('--resume')
            execute('train' + str(step), command, 108000)
            native_audit(root, row, p['source'], step=step)
            if step not in p['evaluation_stages']:continue
            cp = run / f'model-step{step:07d}'; before = sha(cp / 'weights.pt'); out = root / 'evaluations' / ident / str(step)
            execute('bn' + str(step), [PY, '-B', '-u', p['helper']['source'] + '/experiments/lingjing_fullspec_20261102/l0_bn.py',
                    '--run', str(run), '--step', str(step), '--output', str(out / 'bn')], 1800)
            for variant, checkpoint in [('raw', cp), ('bn', out / 'bn/model')]:
                execute(f'probe-{step}-{variant}', [PY, '-B', '-u', '-m', 'as_jepa_tools.probes', '--checkpoint', str(checkpoint),
                        '--data', cfg['evaluation_data'], '--output', str(out / variant / 'probes'), '--train', '20000', '--test', '5000', '--fp32'], 3600, required=False)
                for seed in p['plan_seeds']:
                    execute(f'plan-{step}-{variant}-{seed}', [PY, '-B', '-u', '-m', 'as_jepa_tools.planning', '--checkpoint', str(checkpoint),
                            '--output', str(out / variant / ('plan' + str(seed))), '--upstream', str(B / 'sources/stable-worldmodel'),
                            '--data-root', str(B / 'data/hardtasks-v1/planning'), '--test-episodes', str(Path(cfg['evaluation_data']) / 'test_episodes.npy'),
                            '--seed', str(seed), '--episodes', '50', '--split', 'heldout'], 3600, required=False)
            assert before == sha(cp / 'weights.pt')
        trained = True
    except Exception as exc:error = repr(exc)
    write(root / 'terminals' / (ident + '.json'), dict(passed=trained and all(r['returncode'] == 0 for r in results), training_complete=trained,
          id=ident, results=results, error=error, seconds=time.monotonic() - started, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          scope='Planning/probe failures are retained and do not silently discard a healthy matched training continuation. No further training beyond152650.'))
    if not trained:raise RuntimeError(error)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--id', required=True)
    a = p.parse_args(); main(a.root, a.id)
