"""Only factual RGB histories, their executed inputs, and observed outcomes.

All native starts have at least one real future observation. Every arm sees
the same starts and the same maximum-length arrays; only its loss horizon
changes. No desired goal, expert-path query, task/state/reward input exists.
"""
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from .common import read


class FactualWindows(Dataset):
    def __init__(self, directory, split='train', history=8, maximum_horizon=15):
        self.history = history; self.maximum_horizon = maximum_horizon
        self.shards = []; self.records = []
        for shard in range(12):
            p = Path(directory)/str(shard); manifest = read(p/'manifest.json')
            assert manifest['passed'] and manifest['physical_stride'] == 1
            assert manifest['all_phases_grouped_split'] and manifest['terminal_macro_padding_recovered']
            self.shards.append(dict(pixels=np.load(p/'pixels.npy', mmap_mode='r'),
                                    inputs=np.load(p/'known_input.npy', mmap_mode='r')))
            self.records.extend(dict(shard=shard, **r) for r in manifest['records'] if r['split'] == split)
        assert all(r['length'] >= 2 for r in self.records)
        self.ends = np.cumsum([r['length']-1 for r in self.records])
        self.natural_length = int(self.ends[-1])
        self.episode_ids = sorted(r['episode'] for r in self.records)

    def __len__(self):
        return self.natural_length

    def locate(self, index):
        index = int(index) % self.natural_length
        episode = int(np.searchsorted(self.ends, index, side='right'))
        record = self.records[episode]
        t = index-(int(self.ends[episode-1]) if episode else 0)
        return record, t

    def __getitem__(self, index):
        record, t = self.locate(index); data = self.shards[record['shard']]
        offset = record['offset']; length = record['length']
        def image(j):
            return torch.from_numpy(np.asarray(data['pixels'][offset+j]).copy()).permute(2,0,1)
        def action(j):
            return torch.from_numpy(np.asarray(data['inputs'][offset+j]).copy()).float()
        starts = [t-self.history+1+j for j in range(self.history)]
        history = torch.stack([image(max(0,j)) for j in starts])
        past = torch.stack([action(j) if j >= 0 else torch.zeros(4) for j in starts[:-1]])
        valid = min(self.maximum_horizon, length-1-t)
        futures = torch.stack([image(t+min(j+1,valid)) for j in range(self.maximum_horizon)])
        controls = torch.stack([action(t+j) if j < valid else torch.zeros(4) for j in range(self.maximum_horizon)])
        return dict(history_pixels=history, history_actions=past,
            history_valid=torch.tensor([j >= 0 for j in starts]),
            future_pixels=futures, actions=controls,
            future_valid=torch.arange(self.maximum_horizon) < valid,
            episode=record['episode'], start=t)


class WindowSampler(Sampler):
    def __init__(self, dataset, start, seed, batch_size):
        self.dataset=dataset; self.start=start; self.seed=seed
        self.length=(len(dataset)//batch_size)*batch_size

    def __len__(self):return self.length

    def __iter__(self):
        n=len(self.dataset); epoch=-1
        for position in range(self.start,self.start+self.length):
            e=position//n
            if e != epoch:
                epoch=e; rng=np.random.default_rng(self.seed+e*1000003)
                multiplier=int(rng.integers(1,n)); shift=int(rng.integers(0,n))
                while math.gcd(multiplier,n) != 1:multiplier=(multiplier+1)%n or 1
            yield (multiplier*(position%n)+shift)%n
