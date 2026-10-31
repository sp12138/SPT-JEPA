"""Collect completed Cube controls using JSON only; no model/GPU/worker calls."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import statistics


def collect(root):
    inputs = {}

    def read(path):
        path = Path(path)
        data = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    plan = read(root / 'plan.json')
    protocol = read(root / 'protocol.json')
    assert len(plan['jobs']) == 68 and len(protocol['runs']) == 6
    exits = [read(root / 'exits' / (j['id'] + '.json')) for j in plan['jobs']]
    assert all(e['returncode'] == 0 for e in exits), 'Require every endpoint before reading outcomes'
    runs, audits, planning, manifests, settings = [], [], [], {}, None
    for run in protocol['runs']:
        directory = root / run['id']
        status = read(directory / 'status.json')
        config = read(directory / 'config.json')
        assert status['state'] == 'finished' and status['step'] == 23342
        assert not (directory / 'error.txt').exists()
        assert inputs[str(directory / 'config.json')] == run['config_sha256']
        assert config['batch_size'] == 128 and config['max_steps'] == 124910
        runs.append(dict(id=run['id'], family=run['family'], seed=run['training_seed'],
                         step=status['step'], model_source=run['source']['source_commit'],
                         config_sha256=run['config_sha256']))
    for job in plan['jobs']:
        phase, out = job['phase'], Path(job['output'])
        if phase.startswith('archive'):
            proof = read(out / 'fullstate.json')
            assert proof['passed'] and all(proof['checks'].values())
            assert proof['step'] == int(phase.removeprefix('archive'))
            assert Path(proof['archive']).is_file()
            audits.append(dict(id=job['id'], kind='archive', passed=True, step=proof['step'],
                               archive=proof['archive'], recorded_checkpoint_sha256=proof['checkpoint_sha256']))
        elif phase == 'fullstate':
            proof = read(out / 'preflight.json')
            assert proof['passed'] and proof['checkpoint']['step'] == 23342
            assert proof['loss_finite'] and proof['metrics_finite'] and proof['all_parameter_gradients_finite']
            audits.append(dict(id=job['id'], kind='fullstate', passed=True,
                               recorded_checkpoint_sha256=proof['checkpoint_sha256']))
        elif phase in ['health', 'transition']:
            path = out / ('health.json' if phase == 'health' else 'state-audit.json')
            proof = read(path)
            if phase == 'transition':
                assert proof['step'] == 23342 and proof['cases'] == 64
            audits.append(dict(id=job['id'], kind=phase, path=str(path), sha256=inputs[str(path)],
                               verification='Existing endpoint exited0; JSON parsed. No new model audit or threshold claim.'))
        elif phase.startswith(('original-', 'correlated-')):
            path = out / 'metrics.json'
            metric = read(path)
            successes = metric['metrics']['episode_successes']
            assert len(successes) == len(metric['pairs']) == 50
            assert all(v in [0, 1, False, True] for v in successes)
            fraction = sum(successes) / len(successes)
            # The native evaluator stores success_rate in percentage points.
            assert abs(metric['metrics']['success_rate'] - 100 * fraction) < 1e-6
            assert metric['checkpoint_audit']['strict_equal']
            assert metric['checkpoint'] == str(root / job['id'].removeprefix(phase + '-') / 'model-step0023342')
            controller, case = phase.split('-', 1)
            binding = dict(sha256=metric['manifest_sha256'], pairs=metric['pairs'])
            if case in manifests:
                assert manifests[case] == binding
            else:
                manifests[case] = binding
                read(metric['manifest'])
                assert inputs[metric['manifest']] == binding['sha256']
            common = {k: metric[k] for k in ['solver', 'plan', 'normalization', 'protocol', 'split']}
            if settings is None:
                settings = common
            else:
                assert settings == common
            if controller == 'correlated':
                diagnostic = metric['controller_diagnostic']
                assert diagnostic['checkpoint_files_unchanged']
                assert diagnostic['matched_cases_checkpoint_solver_plan_normalization']
            planning.append(dict(id=job['id'], family=job['family'], seed=job['training_seed'],
                                 controller=controller, case_set=case, percent=100 * fraction,
                                 successes=successes, metrics_path=str(path), metrics_sha256=inputs[str(path)]))
    counts = {kind: sum(a['kind'] == kind for a in audits) for kind in ['archive', 'fullstate', 'health', 'transition']}
    assert counts == dict(archive=14, fullstate=6, health=4, transition=6)
    assert len(planning) == 24
    means = []
    for family in ['native', 'tf', 'local']:
        for controller in ['original', 'correlated']:
            for case in ['primary', 'fresh']:
                rows = sorted((r for r in planning if (r['family'], r['controller'], r['case_set']) ==
                               (family, controller, case)), key=lambda r: r['seed'])
                assert [r['seed'] for r in rows] == [1, 2]
                means.append(dict(family=family, controller=controller, case_set=case,
                                  seed_percent=[r['percent'] for r in rows],
                                  mean_percent=statistics.mean(r['percent'] for r in rows)))
    stop_proofs = [read(root / 'user-stop-monitoring-20261031' / (host + '.json'))
                   for host in ['dev2', 'dev13', 'dev15']]
    assert all(p['inactive'] and p['training_untouched'] for p in stop_proofs)
    return dict(root=str(root), completed=68, failed=0,
                finished_beijing=datetime.fromtimestamp(max(e['finished_at'] for e in exits), ZoneInfo('Asia/Shanghai')).isoformat(),
                runs=runs, endpoint_exits=exits, audits=audits, audit_counts=counts,
                planning=planning, means=means, manifests=manifests, common_evaluation=settings,
                monitor_stop_proofs=stop_proofs, goal_status='paused_by_user',
                next_review=None, automatic_extension=False, input_json_sha256=inputs,
                collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limitations='Two fixed training seeds and two 50-case validation sets; not canonical CLEAR or a final independent test. Equal updates/batch do not equalize model size or compute. No convergence, significance, novelty or broad-stability claim. Existing JSON audit proofs checked; checkpoint hashes are recorded audit hashes, not newly rehashed weights. No new model forward, training, simulation, GPU query, process change or monitoring restart.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(completed=result['completed'], audits=result['audit_counts'], means=result['means'])))
