"""Exercise gated replication, checkpoint readiness and incomplete-eval handling."""
import json
from pathlib import Path
import tempfile

from experiments.metaworld_input_inverse_20261107.campaign import read, save
from experiments.metaworld_input_inverse_20261107.followup import create_jobs, ready, aggregate


def main():
    with tempfile.TemporaryDirectory(prefix='input-inverse-followup-test-') as temp:
        root = Path(temp); follow = root/'followup'
        recipes = []
        for arm in ['selected', 'rejected']:
            for seed in [1, 2]:
                run = root/f'{arm}-seed{seed}'
                save(run/'config.json', dict(output=str(run), seed=seed, model_class='executed_inverse',
                    model=dict(inverse_weight=10, forecast_seed=0, inverse_seed=0)))
                recipes.append(dict(arm=arm, seed=seed, run=str(run)))
        save(root/'plan.json', dict(recipes=recipes, deadline_unix=9999999999))
        save(root/'selection.json', dict(selected=['selected']))
        save(follow/'plan.json', dict(new_training_seeds=list(range(3, 9))))
        create_jobs(root)
        jobs = read(follow/'jobs.json')['jobs']
        trains = [j for j in jobs if j['kind'] == 'train']
        evaluations = [j for j in jobs if j['kind'] == 'eval50']
        assert len(trains) == 6 and len(evaluations) == 80
        assert {j['seed'] for j in trains} == set(range(3, 9))
        assert {j['arm'] for j in jobs} == {'selected'}
        assert len({j['id'] for j in jobs}) == 86
        for job in trains:
            cfg = read(Path(job['run'])/'config.json')
            assert cfg['seed'] == job['seed']
            assert cfg['model']['forecast_seed'] == 20261029+job['seed']*100003
            assert read(Path(job['run'])/'ORIGIN.json')['inherited_tensors'] == 0
        candidate = evaluations[0]; run = Path(candidate['run'])
        assert not ready(candidate)
        (run/'model-step0041200').mkdir()
        (run/'model-step0041200/weights.pt').write_bytes(b'test export')
        assert not ready(candidate)  # Export alone must NOT race native audit.
        save(run/'audit-io-v2-41200.json', dict(passed=False))
        assert not ready(candidate)
        save(run/'audit-io-v2-41200.json', dict(passed=True))
        assert ready(candidate)
        # Only complete 50-task models contribute to an overall mean.
        for job in evaluations:
            state = follow/'jobs'/job['id']
            if job['seed'] == 8 and job['shard'] == 9:
                save(state/'execution-error.json', dict(error='Synthetic failed evaluation'))
                continue
            rows = [dict(task=f'task{job["shard"]*5+t:02d}', seed=s,
                         success=job['seed'] <= 2, expert_goal_success=True)
                    for t in range(5) for s in range(20)]
            save(Path(job['output'])/'summary.json', dict(passed=True, model_unchanged=True,
                inverse_deployed=False, rows=rows))
            save(state/'complete.json', dict(passed=True))
        aggregate(root, evaluations)
        result = read(follow/'results/summary.json')
        arm = result['results']['selected']
        assert not result['passed'] and len(result['execution_errors']) == 1
        assert arm['selected_seeds_mean'] == 100 and arm['fresh_replica_mean'] == 0
        assert set(arm['models']) == set(map(str, range(1, 8)))
        assert arm['incomplete_models']['8']['episodes'] == 900
        assert result['baselines_trained'] == result['baseline_evaluations'] == 0
    print(json.dumps(dict(passed=True, checks=['selected_only', 'six_independent_scratch_seeds',
        'all50_evaluation_jobs', 'native_audit_before_export_read', 'partial_results_excluded',
        'selected_and_fresh_seeds_separated', 'no_baseline_execution'])))


if __name__ == '__main__':
    main()
