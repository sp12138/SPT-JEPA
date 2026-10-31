"""Native unchanged task evaluation under one predeclared candidate intervention."""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys
from common import PY, read, sha, source_check, state_hash, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True)
    p.add_argument('--inner', action='store_true')
    a = p.parse_args()
    spec = read(a.spec)
    if not a.inner:
        env = dict(os.environ, PYTHONPATH=spec['model_source']['pythonpath'], PYTHONDONTWRITEBYTECODE='1')
        os.chdir(spec['model_source']['source'])
        os.execve(PY, [PY, '-B', '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--inner'], env)
    import numpy as np
    from adapter import install
    from as_jepa_tools import benchmark_evaluate
    for path, expected in spec['immutable_inputs'].items():
        assert sha(path) == expected, path
    source_check(spec['model_source'])
    source_check(spec['helper_source'])
    bank = read(spec['bank_report'])
    assert bank['passed'] and bank['no_validation_test_overlap'] and bank['count'] == 1024
    assert sha(bank['bank']) == bank['bank_sha256']
    assert Path(benchmark_evaluate.__file__).resolve() == Path(spec['native_evaluator']).resolve()
    with np.load(bank['bank']) as arrays:
        features, actions = arrays['features'].copy(), arrays['actions'].copy()
    out = Path(spec['output'])
    assert not out.exists()
    out.mkdir(parents=True)
    argv = [benchmark_evaluate.__file__]
    for key in ('manifest', 'checkpoint', 'data', 'output'):
        argv.extend(['--' + key, spec[key]])
    if spec.get('limit'):
        argv.extend(['--limit', str(spec['limit'])])
    saved = sys.argv
    try:
        sys.argv = argv
        with install(spec['mode'], features, actions, bank, spec['model_source']['source']) as records:
            benchmark_evaluate.main()
    finally:
        sys.argv = saved
    assert len(records) == 1
    state = records[0]
    after = state_hash(state.model)
    assert after == state.state_before
    assert state.predict_calls == 5 * state.calls
    assert len(state.injections) * 30 == state.calls
    for path, expected in spec['immutable_inputs'].items():
        assert sha(path) == expected, path
    report = read(out / 'metrics.json')
    reference = read(spec['reference_metrics'])
    assert report['manifest_sha256'] == sha(spec['manifest']) and report['checkpoint'] == spec['checkpoint']
    for key in ('solver', 'plan', 'normalization', 'split', 'protocol', 'checkpoint_audit'):
        assert report[key] == reference[key], key
    assert report['pairs'] == read(spec['manifest'])['pairs'][:spec.get('limit') or 50]
    same_reference = report['manifest_sha256'] == reference['manifest_sha256'] and report['checkpoint'] == reference['checkpoint']
    diagnostic = dict(passed=True, mode=spec['mode'], spec_sha256=sha(a.spec), bank_report_sha256=sha(spec['bank_report']),
            state_before=state.state_before, state_after=after, model_source=state.model_source,
            model_source_sha256=sha(state.model_source), cost_calls=state.calls, predict_calls=state.predict_calls,
            encoder_calls=state.encoder_calls, solves=state.solves, injections=state.injections,
            candidates_per_cost=300, first_iteration_replacements=0 if spec['mode'] == 'original' else 64,
            action_execution='Unmodified native CEM elite update/return and WorldModelPolicy inverse scaler/unblocking.',
            training_steps=0, optimal_action_labels=0, extra_model_calls=0,
            historical_original_success_vector_equal=None if spec['mode'] != 'original' or not same_reference else
                report['metrics']['episode_successes'] == reference['metrics']['episode_successes'][:spec.get('limit') or 50],
            limitation='Controller control using recorded training segments. No learned-method or novelty claim. Historical simulator repeats need not match bitwise.')
    write(out / 'trajectory-search.json', diagnostic)
    print(json.dumps(dict(completed=True, id=spec['id'], mode=spec['mode'], cases=len(report['pairs']),
                         cost_calls=state.calls, success=report['metrics']['success_rate'])), flush=True)


if __name__ == '__main__':
    main()
