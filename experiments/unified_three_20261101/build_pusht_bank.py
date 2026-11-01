"""PushT counterpart of the fixed1024 train-trajectory candidate library."""
import argparse
from pathlib import Path

from common import B, read, sha, write
from experiments.trajectory_search_20261101.common import SEED, LIBRARY, array_sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    import ast
    import inspect
    import types
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    import torch
    import stable_pretraining as spt
    from sklearn.preprocessing import StandardScaler
    from torch.nn import functional as F
    from torchvision.transforms import v2 as transforms
    from clear_lewm.runner import _image_transform

    torch.set_num_threads(2)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    data = B / 'data/pusht-lance-full-v1'
    card = read(data / 'dataset_card.json')
    offsets, lengths, heldout = [np.load(data / (k + '.npy')) for k in ['offsets', 'lengths', 'test_episodes']]
    stored = np.load(data / 'actions.npy', mmap_mode='r')
    assert len(offsets) == len(lengths) == 18685 and card['frameskip'] == 5
    candidates = np.flatnonzero((lengths > 35) & ~np.isin(np.arange(len(lengths)), heldout))
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(candidates, LIBRARY, replace=False)
    starts = [int(rng.integers(int(lengths[ep]) - 35)) for ep in chosen]
    assert len(set(chosen.tolist())) == LIBRARY and not np.isin(chosen, heldout).any()
    # Execute only the pinned upstream's pure image-transform function. This
    # avoids constructing an environment/model while checking actual parity.
    planner = B / 'sources/stable-worldmodel/scripts/plan/eval_wm.py'
    definition = next(x for x in ast.parse(planner.read_text()).body if isinstance(x, ast.FunctionDef) and x.name == 'img_transform')
    scope = dict(torch=torch, transforms=transforms, spt=spt)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(planner), 'exec'), scope)
    native = scope['img_transform'](types.SimpleNamespace(eval=types.SimpleNamespace(img_size=224)), torch.float32)
    common_transform = _image_transform(224)
    h5path = B / 'data/hardtasks-v1/planning/datasets/pusht_expert_train.h5'
    features, actions, raw_actions, provenance = [], [], [], []
    largest_alignment_error = 0.
    with h5py.File(h5path, 'r', rdcc_nbytes=64 << 20) as h5, torch.inference_mode():
        assert np.array_equal(offsets, h5['ep_offset'][:]) and np.array_equal(lengths, h5['ep_len'][:])
        raw_all = h5['action'][:]
        finite = ~np.isnan(raw_all).any(axis=1)
        assert np.isfinite(raw_all[finite]).all()
        scaler = StandardScaler().fit(raw_all[finite])
        normalization_rows = int(finite.sum())
        del raw_all, finite
        for ep, start in zip(chosen, starts):
            row = int(offsets[ep]) + start
            images = np.stack([h5['pixels'][row], h5['pixels'][row + 25]])
            raw = np.asarray(h5['action'][row:row + 25], dtype=np.float32)
            assert raw.shape == (25, 2) and np.isfinite(raw).all()
            assert np.all(h5['episode_idx'][row:row + 26] == ep)
            old = raw.copy()
            old -= np.asarray(card['action_mean'])
            old /= np.asarray(card['action_std'])
            aligned = np.asarray(stored[row + np.arange(5) * 5])
            error = float(np.abs(aligned - old.reshape(5, 10)).max())
            largest_alignment_error = max(largest_alignment_error, error)
            np.testing.assert_allclose(aligned, old.reshape(5, 10), rtol=2e-6, atol=1e-6)
            tensor = torch.from_numpy(images.copy()).permute(0, 3, 1, 2)
            pixels = native(tensor)
            assert torch.equal(pixels, common_transform(tensor))
            features.append(F.adaptive_avg_pool2d(pixels, (8, 8)).reshape(384).numpy())
            # Same full-dataset finite-row scaler as the original P0 evaluator;
            # do not substitute the similar but distinct Lance training scaler.
            normalized = scaler.transform(raw.copy()).astype(np.float32).reshape(5, 10)
            actions.append(normalized)
            raw_actions.append(raw)
            provenance.append(dict(episode=int(ep), start=start, rows=[row, row + 25],
                                   rgb_sha256=array_sha(images), raw_action_sha256=array_sha(raw)))
    destination = out / 'bank.npz'
    np.savez_compressed(destination, features=np.stack(features), actions=np.stack(actions), raw_actions=np.stack(raw_actions))
    with np.load(destination) as f:
        assert f['features'].shape == (1024, 384) and f['actions'].shape == (1024, 5, 10)
        assert all(np.isfinite(f[k]).all() for k in f.files)
    write(out / 'bank.json', dict(passed=True, task='pusht', count=LIBRARY, seed=SEED, split='train',
          script_sha256=sha(__file__), bank=str(destination), bank_sha256=sha(destination),
          dataset_card_sha256=sha(data / 'dataset_card.json'), h5_path=str(h5path),
          metadata_sha256={k: sha(data / (k + '.npy')) for k in ['offsets', 'lengths', 'test_episodes']},
          action_mean=scaler.mean_.tolist(), action_std=scaler.scale_.tolist(),
          scaler_scope='Unchanged native P0 finite-row full-H5 evaluation scaler;not an optimal-action label or train-only scaler.',
          normalization_rows=normalization_rows, action_alignment_max_error=largest_alignment_error,
          native_transform_sha256=sha(planner), shared_transform_source_sha256=sha(inspect.getfile(_image_transform)),
          all2048_transform_outputs_exact=True, no_validation_test_overlap=True,
          episodes=provenance, rgb_reads=2048, candidate_action_rows=25600,
          fields_read=['pixels', 'action', 'episode_idx', 'ep_offset', 'ep_len'],
          model_calls=0, simulator_calls=0, fit_steps=0,
          feature='Native transformed RGB,area-pool8x8;start+endpoint25;unchanged1024/K64 rule.',
          limitation='Executed train trajectories are not assumed optimal. Candidate library/control only,not a new sample-level learner.'))
    print('PushT1024 train-only candidate library complete;zero model/simulator calls.', flush=True)


if __name__ == '__main__':
    main()
