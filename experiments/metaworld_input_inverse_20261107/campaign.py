"""Finite four-host candidate exploration; published baselines are NOT retrained."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

BASE = Path('/efs/prescottsun/SPT-JEPA')
PYTHON = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'
HOSTS = ['dev13', 'dev15', 'dev6', 'dev6-2c']
TASKS = ['bin-picking-v3', 'door-open-v3', 'drawer-open-v3',
         'peg-insert-side-v3', 'pick-place-v3', 'push-v3', 'reach-v3', 'reach-wall-v3']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def read(path):
    for attempt in range(3):
        try:
            return json.loads(Path(path).read_text())
        except OSError:
            if attempt == 2:
                raise
            time.sleep(.2)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def environment(source, gpu):
    paths = [source, BASE/'reports/metaworld_improve_20261104/helper-698ae41',
             BASE/'reports/metaworld_improve_20261104/runtime-7b78f5d',
             BASE/'environments/metaworld-sanity-20261103/lib/python3.12/site-packages',
             BASE/'environments/completion-overlay', BASE/'environments/overlay',
             BASE/'runs/mechanism_cubeaudit_20261030/source-v1/runtime',
             BASE/'sources/stable-worldmodel', BASE/'reports/metaworld_eval_20261103/source-b574b4f']
    env = os.environ.copy()
    env.update(PYTHONPATH=':'.join(map(str, paths)), CUDA_VISIBLE_DEVICES=str(gpu),
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=str(gpu),
               LD_LIBRARY_PATH=str(BASE/'environments/egl-lib/usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH', ''),
               __EGL_VENDOR_LIBRARY_FILENAMES=str(BASE/'environments/egl-lib/10_nvidia.json'))
    return env


def execute(cmd, log, env, source, deadline, cap=86400):
    remaining = min(cap, int(deadline-time.time()))
    if remaining <= 0:
        raise TimeoutError('Finite experiment deadline reached')
    with Path(log).open('x') as stream:
        subprocess.run(['/usr/bin/timeout', '--signal=TERM', '--kill-after=60', str(remaining), *cmd],
                       env=env, cwd=source, stdin=subprocess.DEVNULL,
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=remaining+120)


def variants():
    result = []
    # Two global weights, shared across all tasks. Structural changes are crossed
    # with both state widths; no task-specific selection or policy target.
    for family in ['observed', 'highnoise', 'generative', 'predictive']:
        for slots in [4, 16]:
            for weight in [10., 100.]:
                name = f'{family}-s{slots}-w{int(weight)}'
                result.append((name, dict(state_slots=slots, inverse_weight=weight,
                    inverse_kind='flow' if family == 'generative' else 'mse',
                    inverse_predicted=family == 'predictive',
                    flow_logit_shift=math.log(4.) if family == 'highnoise' else 0.)))
    return result


def prepare(root, source):
    root.mkdir(parents=True, exist_ok=False)
    recipes = []
    for seed, hosts in [(1, HOSTS[:2]), (2, HOSTS[2:])]:
        for index, (arm, changes) in enumerate(variants()):
            host, gpu = hosts[index//8], index % 8
            run = root/'training'/f'{arm}-seed{seed}'
            run.mkdir(parents=True)
            model = dict(mode='L2', beta=.25, sigma0=.1, window=8, anchor_gap=0,
                         renderer_width=384, renderer_depth=12, renderer_heads=6,
                         action_dim=20, rho=.8, forecast_fraction=.5,
                         forecast_seed=20261029+seed*100003,
                         inverse_seed=20261107+seed*100003, **changes)
            cfg = dict(output=str(run), data=str(BASE/'data/metaworld-lerobot-b46a1c18-packed-v1'),
                evaluation_data=str(BASE/'data/metaworld-lerobot-b46a1c18-packed-v1'),
                dataset_kind='benchmark_hdf5', model_class='executed_inverse', model=model,
                supervision_contract='executed_forward_input_inverse_auxiliary_only',
                seed=seed, batch_size=128, workers=2, cpu_threads=2, max_steps=41200,
                max_seconds=5*86400, checkpoint_every=1000,
                retain_steps=[2, 1000, 5000, 20000, 41200], precision='bf16-mixed',
                record_initial_weights=True, record_runtime_source=True)
            save(run/'config.json', cfg)
            save(run/'ORIGIN.json', dict(all_weights_random=True, external_pretraining=False,
                old_policy_ancestry=False, actions='Same actual forward input is reconstructed only as a training auxiliary.',
                deployment='Encoder + forward predictor + CEM; inverse head never queried.',
                candidate_only=True, no_baseline_training=True))
            recipes.append(dict(name=run.name, arm=arm, host=host, gpu=gpu, seed=seed,
                                run=str(run), config_sha256=digest(run/'config.json')))
    old_manifest = BASE/'runs/metaworld_eval_20261103/cases-v2/manifest.json'
    old = read(old_manifest)
    # Screening cases are previously inspected; confirmation is kept separate.
    chosen = [x for x in old['rows'] if x['seed'] in [42, 43, 1042, 1043]]
    assert len(chosen) == 32
    save(root/'screen-cases.json', dict(rows=chosen, source=str(old_manifest),
        source_sha256=digest(old_manifest), scope='Previously inspected development cases, not an untouched confirmation set.'))
    save(root/'plan.json', dict(created_utc=now(), source=str(source), recipes=recipes,
        hosts=HOSTS, stages=[2, 1000, 5000], max_steps=41200,
        screen_manifest=str(root/'screen-cases.json'), full_manifest=str(old_manifest), tasks=TASKS,
        deadline_unix=time.time()+5*86400, public_baselines='Published numbers ONLY; zero baseline SGD or new baseline evaluation.',
        scope='Original LeRobot expert data/internal image-goal protocol. Not equivalent to gated JEPA-WMs data or public SOTA.',
        long_gate=dict(max_selected=4, minimum_manipulation_percent=35., minimum_hard4_percent=20.,
                       require_both_seeds=True, long_steps=[20000, 41200],
                       scope='Resource-allocation criterion, not a matched-budget superiority test.')))
    save(root/'RECOVERY.json', dict(status='PREPARED_ENTRY_GATES_PENDING', created_utc=now(),
        source=str(source), plan='plan.json', old_goals_and_monitors='remain off',
        immutable_old_campaigns=True, high_performance_achieved=False))


def launch(root, source, host):
    if host not in HOSTS:
        raise ValueError('Host not authorized by latest user')
    records = []
    plan = read(root/'plan.json')
    for recipe in plan['recipes']:
        if recipe['host'] != host:
            continue
        run = Path(recipe['run'])
        cmd = [PYTHON, '-B', '-u', str(source/'experiments/metaworld_input_inverse_20261107/campaign.py'),
               '--mode', 'work', '--root', str(root), '--source', str(source), '--name', recipe['name']]
        with (run/'worker.log').open('x') as stream:
            process = subprocess.Popen(cmd, env=environment(source, recipe['gpu']), cwd=source,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        records.append(dict(pid=process.pid, started_utc=now(), command=cmd, **recipe))
    save(root/f'launch-{host}.json', dict(host=host, workers=records, created_utc=now()))
    print(json.dumps(dict(host=host, workers=[dict(gpu=r['gpu'], pid=r['pid'], name=r['name']) for r in records])), flush=True)


def evaluate(root, source, run, step, manifest, destination, env, deadline, technical=False):
    execute([PYTHON, '-B', '-u', str(source/'experiments/metaworld_input_inverse_20261107/evaluate.py'),
             '--checkpoint', str(run/f'model-step{step:07d}'), '--data', read(run/'config.json')['evaluation_data'],
             '--manifest', str(manifest), '--output', str(destination),
             '--budget', '20' if technical else '500'] + (['--technical'] if technical else []),
            run/(destination.name+'.log'), env, source, deadline)
    report = read(destination/'summary.json')
    assert report['passed'] and report['model_unchanged']
    return report


def work(root, source, name):
    plan = read(root/'plan.json')
    recipe = next(x for x in plan['recipes'] if x['name'] == name)
    run = Path(recipe['run']); deadline = plan['deadline_unix']
    env = environment(source, recipe['gpu'])
    try:
        from experiments.metaworld_eval_20261103.worker import free_gpu
        free_gpu(recipe['gpu'])
        assert digest(run/'config.json') == recipe['config_sha256']
        for step in plan['stages']:
            if (root/'STOP').exists() or (run/'STOP').exists():
                return
            execute([PYTHON, '-B', '-u', str(source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'),
                     '--config', str(run/'config.json'), '--stop-step', str(step)]
                    + (['--resume'] if step > 2 else []), run/f'train-{step}.log', env, source, deadline)
            execute([PYTHON, '-B', '-u', str(source/'experiments/metaworld_input_inverse_20261107/audit.py'),
                     '--run', str(run), '--step', str(step), '--output', str(run/f'audit-{step}.json')],
                    run/f'audit-{step}.log', env, source, deadline, cap=1800)
            assert read(run/f'audit-{step}.json')['passed']
            archive = run/f'archive-step{step:07d}'
            archive.mkdir()
            os.link(run/'last.ckpt', archive/'native.ckpt')
            save(archive/'manifest.json', dict(step=step, sha256=digest(archive/'native.ckpt')))
            if step == 2:
                evaluate(root, source, run, step, Path(plan['screen_manifest']), run/'entry20', env, deadline, technical=True)
            if step == 5000:
                evaluate(root, source, run, step, Path(plan['screen_manifest']), run/'screen5000', env, deadline)
            save(run/'progress.json', dict(step=step, state='stage_complete', updated_utc=now()))
        save(run/'screen-complete.json', dict(passed=True, step=5000, ended_utc=now()))
        # Only the finite collector may select continuations, after both seeds.
        while not (root/'selection.json').exists():
            if time.time() > deadline or (root/'STOP').exists():
                return
            time.sleep(15)
        selection = read(root/'selection.json')
        if recipe['arm'] not in selection['selected']:
            save(run/'HOLD', dict(reason='Not selected for long training', selection=str(root/'selection.json')))
            return
        for step in [20000, 41200]:
            if (root/'STOP').exists() or (run/'STOP').exists():
                return
            execute([PYTHON, '-B', '-u', str(source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'),
                     '--config', str(run/'config.json'), '--stop-step', str(step), '--resume'],
                    run/f'train-{step}.log', env, source, deadline)
            execute([PYTHON, '-B', '-u', str(source/'experiments/metaworld_input_inverse_20261107/audit.py'),
                     '--run', str(run), '--step', str(step), '--output', str(run/f'audit-{step}.json')],
                    run/f'audit-{step}.log', env, source, deadline)
            assert read(run/f'audit-{step}.json')['passed']
            evaluate(root, source, run, step, Path(plan['full_manifest']), run/f'eval-{step}', env, deadline)
            save(run/'progress.json', dict(step=step, state='stage_complete', updated_utc=now()))
        save(run/'complete.json', dict(passed=True, ended_utc=now(), step=41200))
    except BaseException as error:
        save(run/'execution-error.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=now()))
        raise


def collect(root):
    plan = read(root/'plan.json')
    while time.time() < plan['deadline_unix']:
        if (root/'STOP').exists():
            return
        recipes = plan['recipes']
        settled = all((Path(r['run'])/'screen-complete.json').exists()
                      or (Path(r['run'])/'execution-error.json').exists() for r in recipes)
        if settled:
            break
        time.sleep(30)
    else:
        return
    grouped = {}
    errors = []
    for r in recipes:
        run = Path(r['run'])
        if (run/'execution-error.json').exists():
            errors.append(dict(name=r['name'], **read(run/'execution-error.json')))
            continue
        rows = read(run/'screen5000/summary.json')['rows']
        manipulation = [x for x in rows if x['task'] not in ['reach-v3', 'reach-wall-v3']]
        hard = [x for x in manipulation if x['task'] not in ['door-open-v3', 'drawer-open-v3']]
        score = dict(seed=r['seed'], manipulation=100*sum(x['success'] for x in manipulation)/len(manipulation),
                     hard4=100*sum(x['success'] for x in hard)/len(hard),
                     overall=100*sum(x['success'] for x in rows)/len(rows))
        grouped.setdefault(r['arm'], []).append(score)
    eligible = []
    for arm, scores in grouped.items():
        if len(scores) != 2:
            continue
        if all(s['manipulation'] >= 35 and s['hard4'] >= 20 for s in scores):
            eligible.append((sum(s['manipulation'] for s in scores)/2, arm))
    selected = [arm for _, arm in sorted(eligible, reverse=True)[:4]]
    save(root/'selection.json', dict(selected=selected, scores=grouped, execution_errors=errors,
         created_utc=now(), scope='Development-screen compute allocation only; no public-baseline superiority established.'))
    save(root/'results/screen-summary.json', dict(passed=not errors, selected=selected, scores=grouped,
        execution_errors=errors, baselines_trained=0, source=plan['source']))
    if not selected:
        save(root/'HOLD', dict(reason='No candidate passed long-training gate', utc=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['prepare', 'launch', 'work', 'collect'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--host')
    parser.add_argument('--name')
    args = parser.parse_args()
    if args.mode == 'prepare': prepare(args.root, args.source)
    elif args.mode == 'launch': launch(args.root, args.source, args.host)
    elif args.mode == 'collect': collect(args.root)
    else: work(args.root, args.source, args.name)
