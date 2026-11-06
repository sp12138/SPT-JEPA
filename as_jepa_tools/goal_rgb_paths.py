"""Observation-only ordered paths;never even opens a command array."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class GoalRGBPaths(Dataset):
    def __init__(self,cfg,split='train'):
        self.seed=cfg['seed'];self.horizon=cfg['model']['horizon'];self.encode_epoch_index=True;self.shards=[];self.records=[]
        for shard in range(12):
            path=Path(cfg['native_rgb_cache'])/str(shard);manifest=json.loads((path/'manifest.json').read_text())
            assert manifest['passed']and manifest['physical_stride']==1
            self.shards.append(np.load(path/'pixels.npy',mmap_mode='r'))
            self.records.extend(dict(shard=shard,**r)for r in manifest['records']if r['split']==split)
        self.ends=np.cumsum([r['length']-2 for r in self.records]);self.natural_length=int(self.ends[-1])

    def __len__(self):return self.natural_length

    def __getitem__(self,index):
        row=int(index)%self.natural_length;ep=int(np.searchsorted(self.ends,row,side='right'))
        r=self.records[ep];t=row-(int(self.ends[ep-1])if ep else 0);k=min(self.horizon,r['length']-t-2)
        rng=np.random.default_rng(self.seed+int(index)*1009)
        goal=r['length']-1 if rng.random()<.5 else int(rng.integers(t+k+1,r['length']))
        def image(j):return torch.from_numpy(np.asarray(self.shards[r['shard']][r['offset']+j]).copy()).permute(2,0,1)
        return dict(current_pixels=image(t),next_pixels=image(t+k),goal_pixels=image(goal),forecast_steps=np.int64(k),
            episode=r['episode'],start=t,target_row=r['offset']+t+k,goal_row=r['offset']+goal,
            boundary=np.float32(1),executed_control=torch.zeros(4),previous_control=torch.zeros(4),history_valid=np.float32(0))
