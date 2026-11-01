"""One nested measurement:retain all192 channels in each existing2x2 mean."""
import argparse
from pathlib import Path

import numpy as np
import torch

from common import B, read, sha, write
from experiments.carrier_progress_20261031 import carrier_spatial_probe as old
from experiments.carrier_progress_20261031 import local_carrier_probe as local
from experiments.carrier_progress_20261031 import orientation_probe as pose


def paired(new, previous):
    delta = np.asarray(new).reshape(64, 7).mean(1) - np.asarray(previous).reshape(64, 7).mean(1)
    indices = np.random.default_rng(20261101).integers(64, size=(10000, 64))
    return dict(delta=float(delta.mean()), ci95=np.quantile(delta[indices].mean(1), [.025, .975]).tolist())


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--output', required=True)
    args = parser.parse_args();torch.set_num_threads(2)
    import h5py, hdf5plugin
    rows, proofs, differences = [], [], []
    previous = read(B / 'reports/carrier-local1024-20261031/results.json')
    projection = local.channel_projection() / np.float32(4.)
    for seed in [1, 2]:
        directory = B / f'reports/carrier-spatial600-20261031/seed{seed}-features'
        metadata = read(directory / 'extraction.json');manifests = metadata['manifests']
        parent = B / f'runs/unified_spatial_checkpoint600_20261101/cube-field-seed{seed}'
        cached = torch.load(Path(metadata['run']) / 'model-step0000600/weights.pt', map_location='cpu', weights_only=True, mmap=True)
        current = torch.load(parent / 'model-step0000600/weights.pt', map_location='cpu', weights_only=True, mmap=True)
        names = [k for k in cached if k.startswith('encoder.')]
        assert len(names) == 201 and all(torch.equal(cached[k], current[k]) for k in names)
        assert np.array_equal(current['spatial_projection'].numpy(), projection)
        del cached, current
        inputs = {};bindings = {};nesting_error = 0.
        for split in ['train', 'validation']:
            old.assert_metadata(Path(metadata['data']), manifests[split], split)
            path = directory / (split + '__fullpatch__full.npy')
            assert sha(path) == metadata['files'][path.name]['sha256']
            raw = np.load(path, mmap_mode='r')
            full = np.asarray(raw).reshape(-1, 8, 2, 8, 2, 192).mean(axis=(2, 4)).reshape(-1, 64, 192)
            projected = local.local_project(raw.reshape(-1, 49152), projection, 2)
            nesting_error = max(nesting_error, float(np.abs((full @ projection).reshape(-1, 1024) - projected).max()))
            inputs[split] = {'local16': projected, 'uncompressed192': full.reshape(-1, 12288)}
            if split == 'validation':
                inputs[split] = {k: v.reshape(64, 8, -1)[:, 1:].reshape(448, -1) for k, v in inputs[split].items()}
            bindings[str(path)] = sha(path)
        assert nesting_error < 3e-5
        # Fixed measurements and their provenance are complete before labels.
        card = read(Path(metadata['data']) / 'dataset_card.json')
        with h5py.File(card['h5_path'], 'r', swmr=True) as handle:
            tq = pose.load_quaternions(handle, manifests['train']).reshape(-1, 4)
            vq = pose.load_quaternions(handle, manifests['validation'])[:, 1:].reshape(-1, 4)
            ty = np.concatenate([pose.invariant(tq), old.load_xyz(handle, manifests['train']).reshape(-1, 3)], 1)
            vy = np.concatenate([pose.invariant(vq), old.load_xyz(handle, manifests['validation'])[:, 1:].reshape(-1, 3)], 1)
        for measurement in ['local16', 'uncompressed192']:
            x, vx = [inputs[split][measurement].astype(np.float64) for split in ['train', 'validation']]
            gram, cross, tn, vn, std = old.standardized_grams(x, vx)
            for kind, (tp, vp, configuration) in old.readouts_from_grams(gram, cross, tn, vn, ty, True).items():
                score = old.score(vp, vy, vq, ty, tq, 64)
                if measurement == 'local16':
                    reference = next(x for x in previous['rows'] if (x['seed'], x['representation'], x['readout']) == (seed, 'cell8_shared16', kind))
                    assert abs(score['xyz_euclidean']['mean_cm'] - reference['validation']['xyz_euclidean']['mean_cm']) < 2e-5
                rows.append(dict(seed=seed, measurement=measurement, dimensions=x.shape[-1], readout=kind,
                                 configuration=configuration, validation=score))
        for kind in ['linear', 'rbf']:
            prior, full = [next(r['validation'] for r in rows if (r['seed'], r['measurement'], r['readout']) == (seed, name, kind))
                           for name in ['local16', 'uncompressed192']]
            differences.append(dict(seed=seed, readout=kind,
                    xyz_cm=paired(full['xyz_euclidean']['per_frame_cm'], prior['xyz_euclidean']['per_frame_cm']),
                    angle_deg=paired(full['orientation_angles']['per_frame_deg'], prior['orientation_angles']['per_frame_deg'])))
        proofs.append(dict(seed=seed, all201_encoder_tensors_exact=True, input_bindings=bindings,
                           fixed_projection_recovery_max_abs_error=nesting_error))
    assert not torch.cuda.is_initialized()
    write(args.output, dict(passed=True, rows=rows, differences=differences, proofs=proofs, source_sha256=sha(__file__),
          calls=dict(model=0, encoder=0, renderer=0, simulator=0, GPU=0),
          scope='Single fixed nested measurement,not a width/matrix search. Same2x2 means;only remove192→16 projection. '
                'Original local16 RBF/linear readouts reproduced. Same original128 train clips/64 validation clips, '
                'fixed train-only standardization/ridge/RBF bandwidth/codebook;labels fit diagnostics only. '
                'Unequal feature dimensions and finite-sample bias;information nesting does not guarantee better L2 geometry, '
                'learning or planning. Not a new algorithm/qualified original idea.'))
    print([(r['seed'], r['measurement'], r['readout'], r['validation']['xyz_euclidean']['mean_cm'],
            r['validation']['orientation_angles']['mean_deg']) for r in rows])
    print(differences)


if __name__ == '__main__':main()
