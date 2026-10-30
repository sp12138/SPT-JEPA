"""One fixed CPU RBF readout on completed identity-carrier600 embeddings.

No encoder/model calls, no active experiment results, no validation tuning.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

B = Path('/efs/prescottsun/SPT-JEPA')


def squared_distances(left, right):
    return np.maximum(np.square(left).sum(1)[:, None] + np.square(right).sum(1)[None]
                      - 2 * left @ right.T, 0.)


def kernel_fit_predict(train_x, train_y, validation_x):
    center = train_x.mean(0)
    scale = np.maximum(train_x.std(0), 1e-12)
    x, tx = (train_x-center)/scale, (validation_x-center)/scale
    distances = squared_distances(x, x)
    bandwidth = float(np.median(np.sqrt(distances[np.triu_indices(len(x), k=1)])))
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Nonpositive predeclared median bandwidth')
    gram = np.exp(-distances / (2 * bandwidth**2))
    test_gram = np.exp(-squared_distances(tx, x) / (2 * bandwidth**2))
    # Train-only centering gives the unpenalized intercept of the linear probe.
    columns, grand = gram.mean(0), gram.mean()
    centered = gram-gram.mean(1, keepdims=True)-columns[None]+grand
    test_centered = test_gram-test_gram.mean(1, keepdims=True)-columns[None]+grand
    mean_y = train_y.mean(0)
    alpha = np.linalg.solve(centered + len(x)*.001*np.eye(len(x)), train_y-mean_y)
    return centered@alpha+mean_y, test_centered@alpha+mean_y, dict(
        bandwidth=bandwidth, median_distance_pairs=len(x)*(len(x)-1)//2,
        ridge_mean_loss_units=.001, gram_regularizer=len(x)*.001,
        kernel='exp(-squared_distance/(2*bandwidth**2))', kernel_centered=True,
        bandwidth_selection='All off-diagonal training pair distances; no validation access')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orientation-results', type=Path,
                        default=B/'reports/carrier-orientation600-20261031/results.json')
    parser.add_argument('--helper', type=Path,
                        default=B/'code/experiments/carrier_progress_20261031/orientation_probe.py')
    parser.add_argument('--output', type=Path,
                        default=B/'reports/carrier-orientation-rbf600-20261031')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'results.json').exists():
        raise FileExistsError('Preserve the one predeclared readout result')
    spec = importlib.util.spec_from_file_location('completed_orientation_probe', args.helper)
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    old = helper.read(args.orientation_results)
    if helper.digest(args.helper) != old['source_sha256']:
        raise AssertionError('Use the unchanged completed orientation helper')
    rows = []
    for row in old['rows']:
        if row['arm'] != 'identity':
            continue
        assert row['step'] == 600
        cal = Path(row['provenance']['calibration'])
        meta = helper.read(cal/'calibration.json')
        assert helper.digest(cal/'calibration.json') == row['provenance']['calibration_sha256']
        assert meta['weights_sha256'] == row['provenance']['weights_sha256']
        train_manifest = helper.read(meta['train_manifest'])
        audit = B/'runs/ideas_validate600_20261031'/('transition-'+row['id'])
        val_manifest = helper.read(audit/'windows.json')
        assert helper.digest(audit/'windows.json') == row['provenance']['validation_manifest_sha256']
        assert helper.digest(meta['train_manifest']) == row['provenance']['training_manifest_sha256']
        assert helper.digest(cal/'training-embeddings.npz') == meta['training_embeddings_sha256']
        assert helper.digest(audit/'state-audit.npz') == row['provenance']['state_arrays_sha256']
        assert not {r['episode'] for r in train_manifest['windows']}.intersection(r['episode'] for r in val_manifest['windows'])
        card = helper.read(Path(meta['data'])/'dataset_card.json')
        with h5py.File(card['h5_path'], 'r', swmr=True) as h5:
            train_q = helper.load_quaternions(h5, train_manifest).reshape(-1, 4)
            val_q = helper.load_quaternions(h5, val_manifest)[:, 1:].reshape(-1, 4)
        train_y, val_y = helper.invariant(train_q), helper.invariant(val_q)
        with np.load(cal/'training-embeddings.npz') as store:
            train_x = store['clips'].reshape(-1, 64).astype(np.float64)
        with np.load(audit/'state-audit.npz') as store:
            val_x = store['observed_future'].reshape(-1, 64).astype(np.float64)
        assert train_x.shape == (1024, 64) and val_x.shape == (448, 64)
        mean_y = train_y.mean(0)

        def score(prediction, target, quaternions, windows):
            decoded, _ = helper.decode_codebook(prediction, train_y, train_q)
            return dict(features=helper.feature_metrics(prediction, target, mean_y),
                        angles=helper.angle_metrics(helper.cube_symmetry_angle_deg(decoded, quaternions), windows))

        matrix, intercept, _ = helper.ridge_fit(train_x, train_y)
        linear_train = score(train_x@matrix+intercept, train_y, train_q, 128)
        linear = score(val_x@matrix+intercept, val_y, val_q, 64)
        original = row['scores']['cube24_tensor81']['observed_future']
        for key in ('variance_weighted_r2','mean_r2'):
            np.testing.assert_allclose(linear['features'][key], original['features'][key], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(linear['angles']['per_frame_deg'], original['angles']['per_frame_deg'], rtol=1e-10, atol=1e-12)
        train_pred, val_pred, cfg = kernel_fit_predict(train_x, train_y, val_x)
        rows.append(dict(id=row['id'], seed=row['seed'], config=cfg,
            train=dict(linear=linear_train, rbf=score(train_pred, train_y, train_q, 128)),
            validation=dict(linear=linear, rbf=score(val_pred, val_y, val_q, 64)),
            codebook_references=row['codebook_references'], provenance=row['provenance'],
            linear_reproduces_completed_orientation_probe=True))
    assert len(rows) == 2
    result=dict(rows=rows, source_sha256=helper.digest(__file__),
        completed_orientation_report=str(args.orientation_results), completed_orientation_report_sha256=helper.digest(args.orientation_results),
        helper_sha256=helper.digest(args.helper), task='cube', carrier='identity at600, seeds1/2',
        training_clips=128, training_frames=1024, validation_clips=64, validation_future_frames=448,
        no_gpu_encoder_or_model_calls=True, no_validation_tuning=True,
        interpretation='Fixed nonlinear accessibility test, not total-information proof. Within-clip frames are correlated;1024 frames are128 episodes. Physical labels fit only probes; no proposed training objective changes. Failed RBF does not rule out other nonlinear readouts or finite-sample limitations.')
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# One fixed RBF orientation probe','',
        '| Seed | Readout | Train invariant R² | Validation invariant R² | Validation angle | Within15° |',
        '|---:|---|---:|---:|---:|---:|']
    for row in rows:
        for kind in ['linear','rbf']:
            train=row['train'][kind];val=row['validation'][kind]
            lines.append(f"| {row['seed']} | {kind} | {train['features']['variance_weighted_r2']:.5f} | {val['features']['variance_weighted_r2']:.5f} | {val['angles']['mean_deg']:.3f}° | {100*val['angles']['fraction_within15deg']:.2f}% |")
    lines += ['',result['interpretation'],'',
        'Train-only coordinate standardization and centered RBF kernel; bandwidth is the median off-diagonal training distance. Fixed ridge .001 in mean-loss units, equivalent to n*.001 in the unnormalized Gram system. Same cube24 invariant labels, training-only quaternion codebook and pinned angular metric as the completed linear diagnostic. The linear probe reproduces its original results before this comparison.','',
        'No bandwidth/regularization/seed search. No active result was read. No encoder, GPU, world-model or planner update occurred.']
    (args.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
