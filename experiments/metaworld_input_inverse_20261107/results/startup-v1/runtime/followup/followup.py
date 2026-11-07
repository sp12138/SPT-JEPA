"""Finite candidate-only replication and 50-task evaluation after the screen.

Never starts a baseline. Slots wait for their exact current owner to exit. The
learning code is the same immutable source used in the initial 32 trainings.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
from statistics import mean
import time
import traceback

from experiments.metaworld_input_inverse_20261107.campaign import (
    PYTHON, digest, environment, evaluate, execute, now, read, save)


def alive(pid, marker):
    try:
        return marker.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:
        return False


def prepare(root, helper):
    plan = read(root/'plan.json')
    out = root/'followup'; out.mkdir(exist_ok=False)
    policy = Path('/efs/prescottsun/SPT-JEPA/sources/lerobot-mw-f5c9bc02')
    config = policy/'src/lerobot/envs/metaworld_config.json'
    tasks = sorted(read(config)['TASK_POLICY_MAPPING'])
    assert len(tasks) == 50
    save(out/'plan.json', dict(source=plan['source'], helper=str(helper),
        cache_helper='/efs/prescottsun/SPT-JEPA/reports/metaworld_input_inverse_20261107/helper-io-v2',
        deadline_unix=plan['deadline_unix'], policy_source=str(policy),
        policy_config_sha256=digest(config), tasks=tasks,
        seeds=list(range(6042, 6052))+list(range(7042, 7052)),
        new_training_seeds=list(range(3, 9)),
        rules='Only configurations selected by the original BOTH-SEED gate. Six fresh scratch replicas per selected configuration. '
              'All seeds retained; no second screening of replicas. All 50 tasks at 41200, including expert goal failures. '
              'No published-baseline training/evaluation or old disqualified policy ancestry.',
        scope='New internal image-goal cohort, not a published MT50/JEPA-WMs score.'))
    print(json.dumps(dict(prepared=str(out), tasks=50, physical_cases=1000,
                         extra_scratch_trainings_max=24)))


def cases(root, index):
    follow = root/'followup'; p = read(follow/'plan.json')
    source = Path(p['source']); env = environment(source, 7)
    command = [PYTHON, '-B', '-u', str(source/'experiments/metaworld_eval_20261103/prepare_cases.py'),
               '--source', p['policy_source'], '--output', str(follow/f'cases-part{index}'),
               '--tasks', *p['tasks'][index::4], '--seeds', *map(str, p['seeds'])]
    try:
        execute(command, follow/f'cases-part{index}.log', env, source, p['deadline_unix'], cap=7200)
    except BaseException as error:
        save(follow/f'cases-part{index}-error.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=now()))
        raise


def merge_cases(root):
    import numpy as np
    follow = root/'followup'; p = read(follow/'plan.json'); rows = []
    if (follow/'cases/manifest.json').exists():
        return True
    for index in range(4):
        f = follow/f'cases-part{index}/manifest.json'
        if not f.exists():
            return False
        part = read(f); assert part['passed']; rows.extend(part['rows'])
    assert len(rows) == 1000 and len({(r['task'], r['seed']) for r in rows}) == 1000
    assert {r['task'] for r in rows} == set(p['tasks'])
    old = read(read(root/'plan.json')['full_manifest'])['rows']
    previous = {}
    for r in old:
        previous.setdefault(r['task'], []).append(np.load(r['path'])['rand_vec'])
    for r in rows:
        assert digest(r['path']) == r['sha256']
        with np.load(r['path']) as archive:
            assert not any(np.array_equal(archive['rand_vec'], v) for v in previous.get(r['task'], []))
    rows.sort(key=lambda r: (r['task'], r['seed']))
    manifest = dict(passed=True, rows=rows, tasks=p['tasks'], seeds=p['seeds'],
        expert_goal_failures=sum(not r['expert_any_success'] for r in rows),
        disjoint_original160_task_vectors=True, new_policy_updates=0,
        scope='Fresh physical cases fixed before candidate outcomes. Scripted expert constructs ONLY terminal RGB goals. '
              'No expert action, reward, state or task ID supplied to the learner/planner; failures retained. '
              'Independent LeRobot camera,500-native any-success,not public JEPA-WMs/canonical MT50.')
    save(follow/'cases/manifest.json', manifest)
    for shard in range(10):
        selected = set(p['tasks'][shard*5:(shard+1)*5])
        save(follow/f'cases/shard{shard:02d}.json', dict(manifest, rows=[r for r in rows if r['task'] in selected]))
    return True


def create_jobs(root):
    follow = root/'followup'; p = read(follow/'plan.json'); parent = read(root/'plan.json')
    selection = read(root/'selection.json'); jobs = []
    for arm in selection['selected']:
        template = next(r for r in parent['recipes'] if r['arm'] == arm and r['seed'] == 1)
        runs = {r['seed']: r['run'] for r in parent['recipes'] if r['arm'] == arm}
        for seed in p['new_training_seeds']:
            name = f'{arm}-seed{seed}'; taskroot = follow/'replicas'/name
            run = taskroot/'training'/name; run.mkdir(parents=True, exist_ok=False)
            cfg = copy.deepcopy(read(Path(template['run'])/'config.json'))
            cfg.update(output=str(run), seed=seed)
            cfg['model'].update(forecast_seed=20261029+seed*100003, inverse_seed=20261107+seed*100003)
            save(run/'config.json', cfg)
            save(run/'ORIGIN.json', dict(all_weights_random=True, old_policy_ancestry=False,
                parent_config_only=template['run'], inherited_tensors=0,
                selected_gate_sha256=digest(root/'selection.json'), independent_seed=seed,
                inverse_target='Exactly the executed forward inputs; training-only auxiliary.'))
            recipe = dict(name=name, arm=arm, seed=seed, run=str(run), host=None, gpu=None,
                          config_sha256=digest(run/'config.json'))
            save(taskroot/'plan.json', dict(parent, recipes=[recipe],
                scope='Fresh independent scratch replica of an already-selected candidate; no baseline.'))
            save(taskroot/'selection.json', selection)
            save(taskroot/'adoption.json', {name: dict(pid=0, children=[])})
            jobs.append(dict(id=f'train-{name}', kind='train', arm=arm, seed=seed,
                             taskroot=str(taskroot), run=str(run)))
            runs[seed] = str(run)
        for seed, run in sorted(runs.items()):
            for shard in range(10):
                jobs.append(dict(id=f'eval50-{arm}-seed{seed}-part{shard:02d}', kind='eval50',
                    arm=arm, seed=seed, run=run, shard=shard,
                    manifest=str(follow/f'cases/shard{shard:02d}.json'),
                    output=str(follow/'evaluation50'/f'{arm}-seed{seed}'/f'part{shard:02d}')))
    save(follow/'jobs.json', dict(jobs=jobs, selected=selection['selected'],
        created_utc=now(), selection_sha256=digest(root/'selection.json')))


def ready(job):
    if job['kind'] == 'train':
        return True
    run = Path(job['run'])
    # Do not expose an export while native saving/auditing is incomplete.
    audit = run/'audit-io-v2-41200.json'
    return audit.exists() and read(audit).get('passed') and (run/'model-step0041200/weights.pt').exists()


def slot(root, host, gpu):
    follow = root/'followup'; p = read(follow/'plan.json'); source = Path(p['source'])
    owner = next(r for r in read(root/f'launch-io-v2-{host}.json')['workers'] if r['gpu'] == gpu)
    logdir = follow/'slots'/f'{host}-gpu{gpu}'; logdir.mkdir(parents=True, exist_ok=False)
    try:
        while alive(owner['pid'], str(root)):
            if time.time() >= p['deadline_unix'] or (root/'STOP').exists():
                return
            time.sleep(15)
        while time.time() < p['deadline_unix'] and not (root/'STOP').exists():
            if not (follow/'jobs.json').exists():
                time.sleep(15); continue
            jobs = read(follow/'jobs.json')['jobs']
            if not jobs or (follow/'COMPLETE.json').exists():
                return
            acquired = None
            # Native replica jobs come before evaluations, to keep learning ahead.
            for job in jobs:
                state = follow/'jobs'/job['id']
                if state.exists() or not ready(job):
                    continue
                state.parent.mkdir(exist_ok=True)
                try:
                    state.mkdir()
                except FileExistsError:
                    continue
                acquired = job, state; break
            if acquired is None:
                time.sleep(15); continue
            job, state = acquired
            save(state/'owner.json', dict(host=host, gpu=gpu, pid=os.getpid(), utc=now(), job=job))
            env = environment(source, gpu)
            env['SPT_INPUT_INVERSE_NN_SOURCE'] = str(source)
            try:
                from experiments.metaworld_eval_20261103.worker import free_gpu
                free_gpu(gpu)
                if job['kind'] == 'train':
                    taskroot = Path(job['taskroot']); plan = read(taskroot/'plan.json')
                    plan['recipes'][0].update(host=host, gpu=gpu); save(taskroot/'plan.json', plan)
                    execute([PYTHON, '-B', '-u', str(Path(p['cache_helper'])/'io_worker.py'),
                        '--root', str(taskroot), '--source', str(source), '--helper', p['cache_helper'],
                        '--name', Path(job['run']).name, '--adoption', str(taskroot/'adoption.json')],
                        state/'execution.log', env, source, p['deadline_unix'], cap=4*86400)
                    assert read(Path(job['run'])/'complete.json')['passed']
                else:
                    result = evaluate(root, source, Path(job['run']), 41200, Path(job['manifest']),
                                      Path(job['output']), env, p['deadline_unix'])
                    assert len(result['rows']) == 100
                save(state/'complete.json', dict(passed=True, ended_utc=now()))
            except BaseException as error:
                save(state/'execution-error.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=now()))
    except BaseException as error:
        save(logdir/'execution-error.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=now()))
        raise


def aggregate(root, jobs):
    follow = root/'followup'; groups = {}; errors = []
    for job in jobs:
        state = follow/'jobs'/job['id']
        if (state/'execution-error.json').exists():
            errors.append(dict(job=job, **read(state/'execution-error.json'))); continue
        if job['kind'] != 'eval50':
            continue
        report = read(Path(job['output'])/'summary.json')
        assert report['passed'] and report['model_unchanged'] and not report['inverse_deployed']
        groups.setdefault(job['arm'], {}).setdefault(job['seed'], []).extend(report['rows'])
    results = {}
    for arm, models in groups.items():
        scores = {}
        incomplete = {}
        for seed, rows in models.items():
            assert len({(r['task'], r['seed']) for r in rows}) == len(rows)
            if len(rows) != 1000:
                incomplete[seed] = dict(episodes=len(rows), expected=1000)
                continue
            task_scores = {task: 100*mean([r['success'] for r in rows if r['task'] == task])
                           for task in sorted({r['task'] for r in rows})}
            scores[seed] = dict(overall=float(mean(list(task_scores.values()))), tasks=task_scores,
                goal_failures=sum(not r['expert_goal_success'] for r in rows),
                episodes=1000, model_seed=seed)
        results[arm] = dict(models=scores, incomplete_models=incomplete,
            all_seed_mean=float(mean([r['overall'] for r in scores.values()])) if scores else None,
            selected_seeds_mean=(float(mean([r['overall'] for s, r in scores.items() if s in [1, 2]]))
                                 if any(s in [1, 2] for s in scores) else None),
            fresh_replica_mean=(float(mean([r['overall'] for s, r in scores.items() if s >= 3]))
                                if any(s >= 3 for s in scores) else None))
    save(follow/'results/summary.json', dict(passed=not errors, results=results, execution_errors=errors,
        baselines_trained=0, baseline_evaluations=0, utc=now(),
        scope='Full50 INTERNAL RGB image-goal cohort,500-native any-success. Published scores remain context only. '
              'Initial two selected learners and six fresh scratch replicas reported separately; all failures retained.'))


def controller(root):
    follow = root/'followup'; p = read(follow/'plan.json')
    while time.time() < p['deadline_unix'] and not (root/'STOP').exists():
        case_errors = list(follow.glob('cases-part*-error.json'))
        if case_errors:
            save(follow/'execution-error.json', dict(error='Case construction failed; no dependent jobs released.',
                sources=[str(f) for f in case_errors], utc=now()))
            return
        if not merge_cases(root):
            time.sleep(15); continue
        if not (root/'selection.json').exists():
            time.sleep(15); continue
        if not (follow/'jobs.json').exists():
            create_jobs(root)
        jobs = read(follow/'jobs.json')['jobs']
        if not jobs:
            save(follow/'COMPLETE.json', dict(passed=True, reason='No selected candidates; no extra SGD.', utc=now()))
            return
        # Failed replica training makes its dependent evaluations explicitly blocked.
        for job in jobs:
            if job['kind'] != 'eval50':
                continue
            train = follow/'jobs'/f"train-{job['arm']}-seed{job['seed']}"/'execution-error.json'
            original = Path(job['run'])/'execution-error-io-v2.json'
            blocked = train if train.exists() else original if original.exists() else None
            state = follow/'jobs'/job['id']
            if blocked and not state.exists():
                state.mkdir(parents=True)
                save(state/'execution-error.json', dict(error='Upstream candidate training failed', upstream=str(blocked)))
        settled = all(any((follow/'jobs'/j['id']/f).exists() for f in ['complete.json', 'execution-error.json']) for j in jobs)
        if settled:
            aggregate(root, jobs)
            save(follow/'COMPLETE.json', dict(passed=not any((follow/'jobs'/j['id']/'execution-error.json').exists() for j in jobs), utc=now()))
            return
        time.sleep(30)
    save(follow/'DEADLINE.json', dict(reason='Finite campaign stopped or deadline reached', utc=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['prepare', 'cases', 'controller', 'slot'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--helper', type=Path)
    parser.add_argument('--host'); parser.add_argument('--gpu', type=int); parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare': prepare(args.root, args.helper)
    elif args.mode == 'cases': cases(args.root, args.index)
    elif args.mode == 'controller': controller(args.root)
    else: slot(args.root, args.host, args.gpu)
