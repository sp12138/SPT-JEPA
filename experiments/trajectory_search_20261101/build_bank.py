"""Read exactly 1024 train episodes; retain RGB summaries and actual actions."""
import argparse
from pathlib import Path
from common import B, SEED, LIBRARY, array_sha, read, sha, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    import h5py
    import hdf5plugin  # noqa: F401; read-only decompression registration
    import numpy as np
    import torch
    from torch.nn import functional as F
    from clear_lewm.runner import _image_transform
    import inspect

    torch.set_num_threads(2)
    data, out = Path(a.data), Path(a.output)
    assert not out.exists()
    out.mkdir(parents=True)
    card = read(data / 'dataset_card.json')
    assert card['task'] in ('cube', 'tworoom') and card['frameskip'] == 5
    arrays = {name: np.load(data / (name + '.npy')) for name in
              ('episode_ids', 'offsets', 'lengths', 'train_episodes', 'validation_episodes', 'test_episodes')}
    selected = np.flatnonzero(np.isin(arrays['episode_ids'], arrays['train_episodes']) & (arrays['lengths'] > 35))
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(selected, LIBRARY, replace=False)
    starts = [int(rng.integers(int(arrays['lengths'][ep]) - 35)) for ep in chosen]
    assert not np.isin(arrays['episode_ids'][chosen], np.r_[arrays['validation_episodes'], arrays['test_episodes']]).any()
    mean, std = np.asarray(card['action_mean']), np.asarray(card['action_std'])
    transform = _image_transform(224)
    features, actions, raw_actions, provenance = [], [], [], []
    with h5py.File(card['h5_path'], 'r', rdcc_nbytes=64 << 20) as h5, torch.inference_mode():
        for ep, start in zip(chosen, starts):
            row = int(arrays['offsets'][ep]) + start
            # Exactly two allowed RGB rows and the 25 executed action rows.
            images = np.stack([h5['pixels'][row], h5['pixels'][row + 25]])
            raw = np.asarray(h5['action'][row:row + 25], dtype=np.float32)
            assert images.ndim == 4 and images.shape[0] == 2 and images.shape[-1] == 3
            assert raw.shape == (25, len(mean))
            assert np.all(np.asarray(h5['ep_idx'][row:row + 26]) == arrays['episode_ids'][ep])
            assert np.isfinite(raw).all()
            normalized = raw.copy()
            normalized -= mean
            normalized /= std
            pixels = transform(torch.from_numpy(images.copy()).permute(0, 3, 1, 2))
            feature = F.adaptive_avg_pool2d(pixels, (8, 8)).reshape(384).numpy()
            features.append(feature)
            actions.append(normalized.reshape(5, 5*len(mean)))
            raw_actions.append(raw)
            provenance.append(dict(episode=int(arrays['episode_ids'][ep]), start=start, rows=[row, row + 25],
                                   rgb_sha256=array_sha(images), raw_action_sha256=array_sha(raw)))
    destination = out / 'bank.npz'
    np.savez_compressed(destination, features=np.stack(features), actions=np.stack(actions), raw_actions=np.stack(raw_actions))
    with np.load(destination) as bank:
        assert bank['features'].shape == (LIBRARY, 384) and bank['actions'].shape == (LIBRARY, 5, 5*len(mean))
        assert all(np.isfinite(bank[k]).all() for k in bank.files)
    write(out / 'bank.json', dict(passed=True, task=card['task'], count=LIBRARY, seed=SEED, split='train',
          bank=str(destination), bank_sha256=sha(destination), dataset_card_sha256=sha(data / 'dataset_card.json'),
          metadata_sha256={name: sha(data / (name + '.npy')) for name in arrays},
          action_mean=mean.tolist(), action_std=std.tolist(), image_transform_source=inspect.getfile(_image_transform),
          image_transform_source_sha256=sha(inspect.getfile(_image_transform)), episodes=provenance,
          rgb_reads=2 * LIBRARY, action_rows=25 * LIBRARY, model_calls=0, simulator_calls=0, fit_steps=0,
          no_validation_test_overlap=True, fields_read=['pixels', 'action', 'ep_idx'],
          feature='Native transformed RGB, area-pool8x8; concatenate start then endpoint at25.',
          limitation='Recorded behavior is not assumed optimal. This is a train-only candidate library, not an action-supervised policy.'))
    print('Train-only bank complete:1024 episodes,2048 RGB reads,zero models/simulation.', flush=True)


if __name__ == '__main__':
    main()
