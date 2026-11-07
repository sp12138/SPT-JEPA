"""CPU-only replay attribution. Privileged diagnostics never enter a learner.

Uses only existing executed traces; zero neural inference, rendering, or SGD.
Physical contacts are descriptive, not a task-independent success criterion.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

BASE = Path('/efs/prescottsun/SPT-JEPA')
ROOT = BASE / 'runs/metaworld_baseline_insights_20261108'
FORMAL = BASE / 'runs/metaworld_baselines_native_20261108'
LEGACY = BASE / 'runs/metaworld_baselines_20261107'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare():
    cases = [c for i in [0, 1] for c in read(FORMAL / f'manifests/intermediate-{i}.json')['rows']]
    bank = {(c['task'], c['seed']): c for c in cases}; assert len(bank) == 100
    jobs = []

    def add(source, report, field, selection=None):
        for row in report[field]:
            key = row['task'], row['seed']
            if key not in bank or (selection is not None and row['policy'] != selection): continue
            assert row['case_sha256'] == bank[key]['sha256']
            jobs.append(dict(source=source, case=bank[key], trace=row['trace'],
                             trace_sha256=row['trace_sha256'], steps=row['steps'], success=row['success']))

    for job in read(LEGACY / 'reference/plan.json')['jobs']:
        add(f'reference-lewm20-seed{job["seed"]}', read(LEGACY / 'reference/workers' / job['name'] / 'formal/metrics.json'), 'episodes')
    for shard in range(16):
        report = read(LEGACY / f'floors/shard{shard:02}/summary.json')
        for policy in ['zero', 'uniform', 'train-mean']: add('floor-' + policy, report, 'rows', policy)
    plan = read(FORMAL / 'plan.json')
    for job in plan['jobs']:
        if job['method'] not in ['lewm-idm', 'h-jepa'] or job['seed'] not in [2, 3]: continue
        for shard in [0, 1]:
            report = read(Path(job['output']) / f'evaluation-0008240-{shard}/metrics.json')
            assert report['passed'] and report['model_state_before'] == report['model_state_after']
            add(job['name'] + '-epoch20', report, 'episodes')
    groups = {name: sum(j['source'] == name for j in jobs) for name in {j['source'] for j in jobs}}
    assert len(groups) == 9 and set(groups.values()) == {100} and len(jobs) == 900
    ROOT.mkdir(parents=True, exist_ok=False)
    save(ROOT / 'plan.json', dict(jobs=jobs, shards=6, groups=groups, created_utc=now(), deadline=time.time()+7200,
        scope='900 replayed traces from9controllers on100 shared cases,50tasks. No independent900contexts. '
              'Historical native20epoch reference and new native20epoch transfers remain distinct. '
              'No new policy,NN,RGB rendering,SGD,or trainingdata. State/contact used only for attribution.'))
    print(json.dumps(dict(jobs=len(jobs), groups=groups)))


class ContactReader:
    def __init__(self, env):
        import mujoco
        self.mujoco = mujoco
        m = env.model
        roots = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ['leftclaw', 'rightclaw']]
        assert min(roots) >= 0

        def top(body):
            while m.body_parentid[body]: body = int(m.body_parentid[body])
            return body

        robot = top(roots[0]); assert robot == top(roots[1])
        self.finger = {}; self.dynamic = {}
        for geom in range(m.ngeom):
            body = int(m.geom_bodyid[geom]); root = top(body); side = None; mobile = False
            while body:
                if body in roots: side = roots.index(body)
                mobile = mobile or bool(m.body_jntnum[body]); body = int(m.body_parentid[body])
            self.finger[geom] = side
            self.dynamic[geom] = root if mobile and root != robot else None

    def __call__(self, env):
        touched = [set(), set()]
        for i, c in enumerate(env.data.contact):
            for a, b in [(int(c.geom1), int(c.geom2)), (int(c.geom2), int(c.geom1))]:
                side = self.finger[a]; other = self.dynamic[b]
                if side is None or other is None: continue
                force = np.zeros(6); self.mujoco.mj_contactForce(env.model, env.data, i, force)
                if force[0] > 0: touched[side].add(other)
        return bool(touched[0] or touched[1]), bool(touched[0] & touched[1])


def replay(job):
    from experiments.metaworld_eval_20261103.environment import make_environment, physics_state, close_environment
    case = job['case']; assert sha(case['path']) == case['sha256'] and sha(job['trace']) == job['trace_sha256']
    with np.load(case['path']) as cached, np.load(job['trace']) as trace:
        actions = trace['actions'].copy(); expected = trace['successes'].copy()
        assert actions.shape == (job['steps'], 4) and len(expected) == len(actions)
        assert np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1
        env, state, _ = make_environment(case['task'], case['seed'], cached['rand_vec'])
        try:
            assert np.array_equal(state, cached['initial_state'])
            assert all(np.array_equal(v, cached['initial_physics_'+k]) for k, v in physics_state(env).items())
            contact = ContactReader(env); obj0 = np.asarray(env._get_pos_objects()).reshape(-1, 3).copy()
            tcp0 = env.tcp_center.copy(); target = np.asarray(env._target_pos).copy()
            initial_distance = float(np.linalg.norm(obj0-tcp0, axis=1).min())
            minimum = initial_distance; obj_delta = 0.; lift = 0.; any_count = dual_count = dual_run = longest_dual = 0
            first_contact = None; first_dual = None; actual = []
            for i, action in enumerate(actions):
                _, _, _, _, info = env.step(action.astype(np.float32)); actual.append(bool(info['success']))
                obj = np.asarray(env._get_pos_objects()).reshape(-1, 3)
                assert obj.shape == obj0.shape
                minimum = min(minimum, float(np.linalg.norm(obj-env.tcp_center, axis=1).min()))
                obj_delta = max(obj_delta, float(np.linalg.norm(obj-obj0, axis=1).max()))
                lift = max(lift, float((obj-obj0)[:, 2].max()))
                any_contact, dual = contact(env); any_count += any_contact; dual_count += dual
                dual_run = dual_run+1 if dual else 0; longest_dual = max(longest_dual, dual_run)
                if any_contact and first_contact is None: first_contact = i+1
                if dual and first_dual is None: first_dual = i+1
            assert np.array_equal(np.asarray(actual, bool), expected), (job['source'], case['task'], case['seed'])
            assert bool(expected.any()) == job['success']
            return dict(source=job['source'], task=case['task'], case_seed=case['seed'], case_sha256=case['sha256'],
                source_trace_sha256=job['trace_sha256'], steps=len(actions), success=bool(expected.any()),
                first100_success=bool(expected[:100].any()), expert_goal_success=case['expert_any_success'],
                initial_tcp_nearest_object_distance=initial_distance, min_tcp_nearest_object_distance=minimum,
                max_reported_object_displacement=obj_delta, max_reported_object_lift=lift,
                finger_external_movable_contact_steps=any_count, dual_finger_same_tree_contact_steps=dual_count,
                longest_dual_contact_steps=longest_dual, first_contact_step=first_contact, first_dual_contact_step=first_dual,
                initial_tcp_target_distance=float(np.linalg.norm(tcp0-target)),
                final_tcp_target_distance=float(np.linalg.norm(env.tcp_center-target)),
                initial_nearest_object_target_distance=float(np.linalg.norm(obj0-target, axis=1).min()),
                final_nearest_object_target_distance=float(np.linalg.norm(obj-target, axis=1).min()),
                action_mean=actions.mean(0).tolist(), action_std=actions.std(0).tolist(),
                action_saturated_fraction=(np.abs(actions) > .99).mean(0).tolist(), replay_success_sequence_exact=True)
        finally: close_environment(env)


def work(shard):
    plan = read(ROOT / 'plan.json'); rows = []; started = time.monotonic()
    try:
        for job in plan['jobs'][shard::plan['shards']]:
            assert time.time() < plan['deadline'], 'Finite CPU attribution deadline'
            rows.append(replay(job))
            if len(rows) % 10 == 0:
                save(ROOT / f'shard{shard}/progress.json', dict(rows=rows, utc=now()))
                print(json.dumps(dict(shard=shard, completed=len(rows), seconds=time.monotonic()-started)), flush=True)
        save(ROOT / f'shard{shard}/summary.json', dict(passed=True, rows=rows, utc=now(), seconds=time.monotonic()-started,
             model_updates=0, neural_models_loaded=0, rendering_calls=0))
    except BaseException:
        import traceback
        save(ROOT / f'shard{shard}/error.json', dict(utc=now(), traceback=traceback.format_exc())); raise


def collect():
    plan = read(ROOT / 'plan.json'); rows = []
    for shard in range(plan['shards']):
        d = read(ROOT / f'shard{shard}/summary.json'); assert d['passed']; rows.extend(d['rows'])
    assert len(rows) == len(plan['jobs'])
    metrics = ['success', 'first100_success', 'min_tcp_nearest_object_distance', 'max_reported_object_displacement',
               'max_reported_object_lift', 'finger_external_movable_contact_steps', 'dual_finger_same_tree_contact_steps',
               'longest_dual_contact_steps']
    groups = {}
    for source in sorted({r['source'] for r in rows}):
        selected = [r for r in rows if r['source'] == source]; assert len(selected) == 100
        assert len({(r['task'], r['case_seed']) for r in selected}) == 100
        groups[source] = dict(episodes=100, macro50_percent=100*np.mean([r['success'] for r in selected]), tasks={})
        for task in sorted({r['task'] for r in selected}):
            group = [r for r in selected if r['task'] == task]; assert len(group) == 2
            groups[source]['tasks'][task] = {k:float(np.mean([r[k] for r in group])) for k in metrics}
    save(ROOT / 'summary.json', dict(passed=True, episodes=len(rows), unique_physical_cases=len({(r['task'], r['case_seed']) for r in rows}), groups=groups, rows=rows,
        all_success_sequences_exact=True, model_updates=0, neural_models_loaded=0, rendering_calls=0, utc=now(),
        scope=plan['scope'], interpretation='Contact/distance are descriptive. Nearest reported object need not be the task target; '
            'some tasks need no bilateral grasp. Neither contact nor lift alone proves success or a unique failure cause.'))
    print(json.dumps(dict(passed=True, episodes=len(rows), scores={k:v['macro50_percent'] for k,v in groups.items()})))


def wait():
    plan = read(ROOT / 'plan.json')
    while time.time() < plan['deadline']:
        errors = [str(ROOT / f'shard{i}/error.json') for i in range(plan['shards'])
                  if (ROOT / f'shard{i}/error.json').exists()]
        if errors:
            save(ROOT / 'ERRORS.json', dict(errors=errors, utc=now())); return
        if all((ROOT / f'shard{i}/summary.json').exists() for i in range(plan['shards'])):
            collect(); return
        time.sleep(15)
    save(ROOT / 'DEADLINE.json', dict(utc=now(), unfinished=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['prepare', 'work', 'collect', 'wait'], required=True)
    parser.add_argument('--shard', type=int); parser.add_argument('--root', type=Path); args = parser.parse_args()
    if args.root is not None: ROOT = args.root
    if args.mode == 'prepare': prepare()
    elif args.mode == 'work': work(args.shard)
    elif args.mode == 'collect': collect()
    else: wait()
