"""Small shared data/provenance helpers; all training remains native."""
import json
from pathlib import Path

import numpy as np

from experiments.trajectory_search_20261101.common import B, PY, read, write, sha, source_check, state_hash


def dataset(cfg, split):
    if cfg.get('dataset_kind') == 'benchmark_hdf5':
        from as_jepa_tools.benchmark_data import BenchmarkWindows
        return BenchmarkWindows(cfg['data'], window=cfg['model']['window'], seed=cfg['seed'], split=split)
    from as_jepa_tools.data import FullPushTWindows
    assert cfg['all_phase_data']
    return FullPushTWindows(cfg['data'], window=cfg['model']['window'], gap=cfg.get('gap', 8),
                            seed=cfg['seed'], split='train' if split == 'train' else 'test')


def windows(ds, count=64, seed=1042):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(ds.episodes), count, replace=False)
    result = []
    for position in chosen:
        begin = int(ds.ends[position - 1]) if position else 0
        index = begin + int(rng.integers(ds.counts[position]))
        ep, start = ds.locate(index)
        episode = int(ds.episode_ids[ep]) if hasattr(ds, 'episode_ids') else int(ep)
        result.append(dict(index=index, episode=episode, start=start))
    assert len({r['episode'] for r in result}) == count
    return result


def same(a, b):
    import torch
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def reference(task, seed, arm='flat'):
    if task == 'pusht':
        return B / f'runs/promising_20261029/temporal-forecast-seed{seed}-selected-epoch3/config.json'
    if arm == 'tf':
        return B / f'runs/carrier_learn6000_20261031/{task}-temporal-forecast-seed{seed}/config.json'
    return B / f'runs/local_isotropic600_20261031/{task}-spatial-local-seed{seed}/config.json'


def allocations():
    result = []
    for task, host in [('pusht', 'dev2'), ('tworoom', 'dev13'), ('cube', 'dev15')]:
        for seed in (1, 2):
            for j, arm in enumerate(('flat', 'field', 'tf')):
                result.append(dict(id=f'{task}-{arm}-seed{seed}', task=task, host=host,
                                   gpu=(seed-1)*3+j, seed=seed, arm=arm))
    return result
