"""Read-only audit and comparison of existing per-clip evaluation records."""
import argparse
import json
from pathlib import Path
import time


def main(root, output):
    import numpy as np
    plan = json.loads((root / 'plan.json').read_text())
    groups, missing, ids = {}, [], set()
    for job in plan['evaluation_jobs']:
        path = root / 'evaluations' / job['name'] / 'result.json'
        try:
            result = json.loads(path.read_text())
        except FileNotFoundError:
            missing.append(job['name'])
            continue
        assert result['passed'] and result['state_unchanged']
        assert all(result[key] == job[key] for key in ['arm', 'model_seed', 'planner_seed', 'clip'])
        assert job['name'] not in ids
        ids.add(job['name'])
        key = (job['category'], job['arm'], job.get('epoch', 'public'))
        groups.setdefault(key, []).append(result)
    summary = {}
    for key, rows in groups.items():
        seeds = sorted({r['model_seed'] for r in rows})
        assert all(len({(r['planner_seed'], r['clip']) for r in rows if r['model_seed'] == seed}) == 48
                   for seed in seeds)
        models = {seed: {r['model_sha256'] for r in rows if r['model_seed'] == seed} for seed in seeds}
        assert all(len(v) == 1 for v in models.values())
        summary['/'.join(map(str, key))] = dict(
            episodes=len(rows), model_seeds=seeds, physical_clips=16,
            fidelity_percent=float(np.mean([r['metrics']['frechet/skill'] for r in rows]) * 100),
            ate_xyz_m=float(np.mean([r['metrics']['ate/end_distance_xyz'] for r in rows])),
            per_seed_fidelity_percent={str(s): float(np.mean([r['metrics']['frechet/skill'] for r in rows
                                                            if r['model_seed'] == s]) * 100) for s in seeds},
            model_state_hashes={str(s): next(iter(v)) for s, v in models.items()})
    differences = {}
    control = groups[('trained', 'lewm', 100)]
    controls = {(r['model_seed'], r['planner_seed'], r['clip']): r['metrics']['frechet/skill'] for r in control}
    rng = np.random.default_rng(20261105)
    for arm in ['temporal-forecast', 'innovation']:
        rows = groups[('trained', arm, 100)]
        deltas = {k: [] for k in range(16)}
        for r in rows:
            key = (r['model_seed'], r['planner_seed'], r['clip'])
            deltas[r['clip']].append(r['metrics']['frechet/skill'] - controls[key])
        by_case = np.array([np.mean(deltas[k]) for k in range(16)])
        bootstrap = by_case[rng.integers(0, 16, size=(10000, 16))].mean(1)
        differences[arm + '-minus-lewm'] = dict(mean_pp=float(by_case.mean() * 100),
            conditional_case_cluster_ci95_pp=(np.quantile(bootstrap, [.025, .975]) * 100).tolist(),
            interpretation='Case-cluster uncertainty conditional on these three trained learners; not 144 independent physical trials.')
    training = {}
    for recipe in plan['recipes']:
        run = root / 'training' / recipe['name']
        completed = run / 'training-complete.json'
        health = json.loads((run / 'health-rank0.json').read_text())
        training[recipe['name']] = dict(completed=completed.exists(), step=health['step'], loss=health['loss'])
    value = dict(updated_unix=time.time(), verified_existing_endpoints=len(ids), expected_endpoints=1584,
                 missing=missing, groups=summary, differences=differences, training=training,
                 score_is_offline_fidelity_not_robot_success=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(verified=len(ids), missing=len(missing), groups=summary, differences=differences)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(Path(args.root), Path(args.output))
