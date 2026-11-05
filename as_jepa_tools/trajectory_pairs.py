"""Hindsight endpoints and factual actions; no assumption of action optimality."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryPairs(Dataset):
    def __init__(self, config, split='train'):
        self.root = Path(config['trajectory_cache'])
        self.seed = config['seed']; self.chunk = config['model'].get('chunk', 6)
        self.source = config['model'].get('visual_source', 'local')
        self.metric = config['model'].get('objective') == 'metric'
        self.encode_epoch_index = True
        self.shards = []
        self.records = []
        feature_root = Path(config.get('feature_cache', self.root))
        for shard in range(4):
            directory = self.root/str(shard)
            manifest = json.loads((directory/'manifest.json').read_text())
            assert manifest['passed'] and manifest['model_unchanged']
            feature = (torch.load(feature_root/str(shard)/'features.pt', map_location='cpu', weights_only=True)['embedding'].numpy()
                       if self.source in ['global', 'hybrid'] else None)
            actions = np.load(directory/'actions.npy', mmap_mode='r')
            self.shards.append(dict(directory=directory, feature=feature, actions=actions))
            if self.source == 'patch':
                self.shards[-1]['patches'] = np.load(feature_root/str(shard)/'patches.npy', mmap_mode='r')
            for r in manifest['records']:
                if r['split'] == split:
                    self.records.append(dict(shard=shard, **r))
        self.ends = np.cumsum([r['length']-1 for r in self.records])
        self.natural_length = int(self.ends[-1])
        self.images = {}
        card = json.loads((Path(config['data'])/'dataset_card.json').read_text())
        self.mean = np.asarray(card['action_mean'], np.float32)
        self.std = np.asarray(card['action_std'], np.float32)

    def __len__(self):
        return self.natural_length

    def locate(self, index):
        normalized = int(index) % self.natural_length
        ep = int(np.searchsorted(self.ends, normalized, side='right'))
        start = normalized-(int(self.ends[ep-1]) if ep else 0)
        return self.records[ep], start

    def __getitem__(self, index):
        record, start = self.locate(index)
        # Every endpoint is later in the SAME logged trajectory. Terminal
        # frames carry no success annotation; failed episodes remain eligible.
        rng = np.random.default_rng(self.seed+int(index)*1009)
        end = record['length']-1 if rng.random() < .5 else int(rng.integers(start+1, record['length']))
        if self.metric and rng.random() < .1:
            end = start
        shard = record['shard']; data = self.shards[shard]
        offset = record['offset']; rows = np.asarray([offset+start, offset+end])
        actions = np.zeros((self.chunk, 20), np.float32)
        mask = np.zeros(self.chunk, np.float32)
        n = min(self.chunk, end-start)
        actions[:n] = (data['actions'][offset+start:offset+start+n]-self.mean)/self.std
        mask[:n] = 1
        value = dict(action=torch.from_numpy(actions),
                     action_mask=torch.from_numpy(mask), horizon=np.int64(end-start),
                     episode=record['episode'], start=start, rows=torch.from_numpy(rows))
        if self.source in ['global', 'hybrid']:
            value['features'] = torch.from_numpy(data['feature'][rows].copy())
        if self.source == 'patch':
            value['patches'] = torch.from_numpy(data['patches'][rows].copy())
        if self.source in ['local', 'hybrid']:
            if shard not in self.images:
                self.images[shard] = np.load(data['directory']/'pixels.npy', mmap_mode='r')
            value['pixels'] = torch.from_numpy(self.images[shard][rows].copy()).permute(0, 3, 1, 2)
        return value
