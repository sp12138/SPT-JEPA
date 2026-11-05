"""Finite training/evaluation using the original Lightning trainer and safe IO."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.metaworld_contact_20261106.diagnose import save, digest
from experiments.metaworld_contact_20261106.launch import BASE, PYTHON, environment


TASKS = ['bin-picking-v3', 'pick-place-v3', 'push-v3', 'drawer-open-v3', 'door-open-v3',
         'peg-insert-side-v3', 'reach-v3', 'reach-wall-v3']
ARMS = [('metric', 'global', 'metric'), ('global-mse', 'global', 'mse'),
        ('local-mse', 'local', 'mse'), ('local-flow', 'local', 'flow')]


def command(cmd, log, env, source, deadline):
    seconds = max(1, int(deadline-time.time()))
    with Path(log).open('x') as stream:
        result = subprocess.run(['/usr/bin/timeout', '--signal=TERM', '--kill-after=30', str(seconds)]+cmd,
                                cwd=source, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    assert result.returncode == 0, (result.returncode, str(log))


def prepare(root, source):
    cache = root/'cache'; views = root/'cache-views'; views.mkdir()
    for arm in ['tf-control', 'native-control']:
        for seed in [1, 2]:
            view = views/f'{arm}-seed{seed}'; view.mkdir()
            for shard in range(4):
                src = cache/f'{arm}-seed{seed}-shard{shard}'
                assert json.loads((src/'manifest.json').read_text())['passed']
                (view/str(shard)).symlink_to(src)
    original = json.loads((BASE/'runs/metaworld_proposal_20261104_v3/plan.json').read_text())
    recipes = []
    for host, seed in [('dev13', 1), ('dev15', 2)]:
        parent = next(r['checkpoint'] for r in original['recipes'] if r['arm'] == 'tf-control' and r['seed'] == seed)
        for gpu, (arm, visual, objective) in enumerate(ARMS):
            name = f'{arm}-seed{seed}'; run = root/'bridge'/name; run.mkdir(parents=True)
            config = dict(seed=seed, output=str(run), dataset_kind='trajectory_pairs',
                          data=original['data'], trajectory_cache='/tmp/prescottsun-mw-pairs-20261106',
                          feature_cache=str(views/f'tf-control-seed{seed}'), model_class='temporal_bridge',
                          model=dict(visual_source=visual, objective=objective, chunk=6, window=2, width=192, action_dim=20),
                          batch_size=128, workers=4, max_steps=40000, checkpoint_every=1000, cpu_threads=2,
                          max_seconds=20*3600, precision='bf16-mixed', record_initial_weights=True, trace_batches=True,
                          record_runtime_source=True,
                          retain_steps=[2, 600, 3000, 10000, 40000], float32_matmul_precision='highest')
            save(run/'config.json', config)
            recipes.append(dict(name=name, arm=arm, gpu=gpu, host=host, seed=seed, parent=parent, run=str(run), config_sha256=digest(run/'config.json')))
    plan = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), recipes=recipes,
                source=str(source), manifest=original['case_manifest'], data=original['data'], tasks=TASKS,
                stages=[2, 600, 3000, 10000], deadline_unix=time.time()+24*3600,
                scope='Eight scratch auxiliary learners;global arms also use own same-data20epoch TF encoder whose cost must count. '
                      'All50tasks/phase0 only;factual actions and temporal offsets,no optimal-action labels.Candidate sample-level. '
                      'Native Lightning SGD/checkpointing retained.Finite10000-update screen,not automatic long training or monitor.')
    save(root/'bridge-plan.json', plan)


def stage_command(source, config, step, resume):
    helper = source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'
    assert helper.is_file(), helper
    cmd = [PYTHON, '-B', '-u', str(helper), '--config', str(config), '--stop-step', str(step)]
    if resume:
        cmd.append('--resume')
    return cmd


def work(root, source, host, gpu):
    assert host in ['dev13', 'dev15']
    pointer = root/'active-bridge-plan.json'
    path = Path(json.loads(pointer.read_text())['path']) if pointer.exists() else root/'bridge-plan.json'
    plan = json.loads(path.read_text())
    recipe = next(r for r in plan['recipes'] if r['host'] == host and r['gpu'] == gpu)
    run = Path(recipe['run']); env = environment(source, gpu)
    local = Path('/tmp/prescottsun-mw-pairs-20261106')
    while not (local/'COPIED.json').exists():
        assert not (local/'COPY_ERROR.json').exists(), 'Local trajectory cache copy failed'
        assert time.time() < plan['deadline_unix']
        time.sleep(5)
    assert digest(run/'config.json') == recipe['config_sha256']
    try:
        for step in plan['stages']:
            assert not (root/'STOP').exists() and not (run/'STOP').exists()
            existing = run/'last.ckpt'
            completed = 0
            if existing.exists():
                import torch
                completed = int(torch.load(existing, map_location='cpu', weights_only=False)['global_step'])
            if completed == step:
                save(run/f'REUSED-step{step}.json', dict(step=step, native_sha256=digest(existing),
                     scope='Reuse actually completed native SGD;only failed independent audit repeats.No training updates replayed.'))
            else:
                assert completed < step
                command(stage_command(source, run/'config.json', step, completed > 0),
                        run/f'train-{step}.log', env, source, plan['deadline_unix'])
            command([PYTHON, '-B', '-u', str(source/'experiments/metaworld_contact_20261106/audit_bridge.py'),
                     '--run', str(run), '--step', str(step), '--output', str(run/f'audit-{step}.json')],
                    run/f'audit-{step}.log', env, source, plan['deadline_unix'])
            assert json.loads((run/f'audit-{step}.json').read_text())['passed']
            archive = run/f'archive-step{step:07d}'; archive.mkdir()
            shutil.copyfile(run/'last.ckpt', archive/'native.ckpt')
            save(archive/'manifest.json', dict(step=step, native_sha256=digest(archive/'native.ckpt')))
            seeds = [42] if step == 2 else [42, 1042] if step == 600 else [42, 43, 1042, 1043] if step == 3000 else [42, 43, 44, 45, 1042, 1043, 1044, 1045]
            tasks = ['reach-v3'] if step == 2 else TASKS
            out = run/f'eval-step{step:07d}'
            cmd = [PYTHON, '-B', '-u', str(source/'experiments/metaworld_contact_20261106/evaluate.py'),
                   '--mode', 'metric' if recipe['arm'] == 'metric' else 'bridge', '--run', str(run), '--step', str(step),
                   '--parent', recipe['parent'], '--data', plan['data'], '--manifest', plan['manifest'],
                   '--output', str(out), '--budget', '20' if step == 2 else '500', '--seeds']+list(map(str, seeds))+['--tasks']+tasks
            command(cmd, run/f'eval-{step}.log', env, source, plan['deadline_unix'])
            assert json.loads((out/'summary.json').read_text())['passed']
            save(run/'progress.json', dict(step=step, state='stage-complete', updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        save(run/'complete.json', dict(passed=True, step=10000, scope='Finite screen complete;do not infer successful manipulation without actual evaluation rows.'))
    except BaseException as error:
        save(run/'execution-error.json', dict(error=repr(error), updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['prepare', 'work'], required=True)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--host'); p.add_argument('--gpu', type=int)
    a = p.parse_args()
    if a.mode == 'prepare':
        prepare(a.root, a.source)
    else:
        work(a.root, a.source, a.host, a.gpu)
