"""Read completed same-coordinate caches; no fitting, model import, or GPU."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def collect(path):
    report = json.loads(path.read_text())
    rows, references = [], {}
    for row in report['rows']:
        cfg_path = Path(row['artifacts']['config']['path'])
        assert sha(cfg_path) == row['artifacts']['config']['sha256']
        cfg = json.loads(cfg_path.read_text())
        assert cfg['model'].get('spatial_chart', 'none') == 'none'
        assert cfg['model'].get('spatial_basis', 'identity') == 'identity'
        assert cfg['model']['spatial_mode'] == 'local'
        assert len(set(cfg['model']['spatial_scale'])) == 1
        calibration = cfg['spatial_calibration']
        assert sha(calibration['path']) == calibration['sha256']
        assert json.loads(Path(calibration['path']).read_text())['normalization'] == 'global_rms'
        state_path = Path(row['artifacts']['state']['path'])
        assert sha(state_path) == row['artifacts']['state']['sha256']
        state = json.loads(state_path.read_text())
        cache_path = state_path.with_suffix('.npz')
        with np.load(cache_path) as cache:
            observed, actions = cache['observed'], cache['actions']
            clean = cache['clean_rollout'].astype(np.float64)
        assert observed.shape == (64, 8, 1024) and clean.shape == (64, 7, 1024)
        key = (row['task'], row['seed'])
        signature = (cfg['model']['spatial_mean'], cfg['model']['spatial_scale'], state['manifest_sha256'])
        if key in references:
            obs0, act0, sig0 = references[key]
            assert np.array_equal(observed, obs0) and np.array_equal(actions, act0) and signature == sig0
        else:
            references[key] = observed.copy(), actions.copy(), signature
        target = observed[:, 1:].astype(np.float64)
        errors = dict(zero_prediction=np.square(target),
                      initial_carry=np.square(target - observed[:, :1]),
                      clean_rollout=np.square(target - clean))
        mse = {name: float(value.mean()) for name, value in errors.items()}
        for name, value in mse.items():
            assert np.isclose(value, state['conditions'][name]['mse'], rtol=1e-10, atol=1e-12)
        energy = np.square(clean).mean((0, 2))
        alignment = (2 * clean * target).mean((0, 2))
        zero_horizon = errors['zero_prediction'].mean((0, 2))
        clean_horizon = errors['clean_rollout'].mean((0, 2))
        residual = clean_horizon - (zero_horizon + energy - alignment)
        assert np.max(np.abs(residual)) < 1e-12
        rows.append(dict(task=row['task'], seed=row['seed'], arm=row['arm'], step=state['step'],
                         **mse, clean_over_zero=mse['clean_rollout'] / mse['zero_prediction'],
                         carry_over_zero=mse['initial_carry'] / mse['zero_prediction'],
                         prediction_energy=float(energy.mean()),
                         twice_prediction_target_alignment=float(alignment.mean()),
                         mse_identity_max_absolute_residual=float(np.max(np.abs(residual))),
                         per_horizon_clean_over_zero=(clean_horizon / zero_horizon).tolist(),
                         per_horizon_prediction_energy=energy.tolist(),
                         per_horizon_twice_prediction_target_alignment=alignment.tolist(),
                         per_horizon_mse={name: value.mean((0, 2)).tolist() for name, value in errors.items()},
                         cache=dict(path=str(cache_path), sha256=sha(cache_path)),
                         state=dict(path=str(state_path), sha256=sha(state_path))))
    assert len(rows) == 12 and len(references) == 4
    return dict(rows=rows, same_observed_actions_calibration_windows_exact=True,
                existing_zero_prediction_metric_reproduced=True,
                parent_report=dict(path=str(path), sha256=sha(path)), helper_sha256=sha(__file__),
                scope='Existing completed 600-step caches only. Constant zero in the fixed train-centered global-RMS local1024 coordinates; not physical zero. MSE = target energy + prediction energy - twice prediction/target inner product, averaged per coordinate. No validation-mean or alpha fitting, model execution, new cohort or GPU. Ratios are descriptive, not causal or cross-coordinate comparisons.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps([{k: row[k] for k in ['task', 'seed', 'arm', 'clean_over_zero', 'prediction_energy', 'twice_prediction_target_alignment', 'mse_identity_max_absolute_residual']} for row in result['rows']], indent=2))
