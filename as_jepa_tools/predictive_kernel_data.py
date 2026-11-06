"""Causal next-observation targets; never an action as prediction target."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class KernelTransitions(Dataset):
    def __init__(self,cfg,split='train'):
        root=Path(cfg['feature_cache']);self.seed=cfg['seed'];self.boundary=cfg['model']['boundary']
        self.encode_epoch_index=True;self.shards=[];self.records=[]
        for shard in range(4):
            folder=root/str(shard);manifest=json.loads((folder/'manifest.json').read_text())
            assert manifest['passed']and manifest['feature_seed']==cfg['model']['feature_seed']
            self.shards.append(dict(features=np.load(folder/'features.npy',mmap_mode='r'),
                                    inputs=np.load(folder/'executed_input.npy',mmap_mode='r')))
            self.records.extend(dict(shard=shard,**r)for r in manifest['records']if r['split']==split)
        self.ends=np.cumsum([r['length']-1 for r in self.records]);self.natural_length=int(self.ends[-1])

    def __len__(self):return self.natural_length

    def locate(self,index):
        row=int(index)%self.natural_length;episode=int(np.searchsorted(self.ends,row,side='right'))
        return self.records[episode],row-(int(self.ends[episode-1])if episode else 0)

    def __getitem__(self,index):
        r,t=self.locate(index);data=self.shards[r['shard']];offset=r['offset'];rng=np.random.default_rng(self.seed+int(index)*1009)
        # In the boundary query the target frame is strictly BETWEEN its two
        # observed endpoints: it is never present among the inputs.
        boundary=self.boundary and t+2<r['length']and rng.random()<.5
        end=(r['length']-1 if rng.random()<.5 else int(rng.integers(t+2,r['length'])))if boundary else t
        z=data['features'];control=np.zeros(4,np.float32)
        if not boundary:control=np.asarray(data['inputs'][offset+t]).copy()
        def feature(i):return torch.from_numpy(np.asarray(z[offset+i]).copy())
        return dict(previous_z=feature(max(0,t-1)),current_z=feature(t),next_z=feature(t+1),
            goal_z=feature(end)if boundary else torch.zeros_like(feature(t)),
            executed_control=torch.from_numpy(control),boundary=np.float32(boundary),history_valid=np.float32(t>0),
            episode=r['episode'],start=t,target_row=offset+t+1,goal_row=offset+end)
