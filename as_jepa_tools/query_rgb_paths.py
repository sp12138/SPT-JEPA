"""Matched five-frame causal/goal queries with RGB-only supervised targets."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class QueryRGBPaths(Dataset):
    def __init__(self,cfg,split='train'):
        self.seed=cfg['seed'];self.encode_epoch_index=True;self.shards=[];self.records=[];self.horizon=int(cfg['model'].get('forecast_horizon',5))
        for shard in range(12):
            path=Path(cfg['native_rgb_cache'])/str(shard);manifest=json.loads((path/'manifest.json').read_text())
            assert manifest['passed']and manifest['physical_stride']==1 and manifest['terminal_macro_padding_recovered']
            self.shards.append(dict(pixels=np.load(path/'pixels.npy',mmap_mode='r'),inputs=np.load(path/'known_input.npy',mmap_mode='r')))
            self.records.extend(dict(shard=shard,**r)for r in manifest['records']if r['split']==split)
        assert all(r['length']>self.horizon+1 for r in self.records)
        self.ends=np.cumsum([r['length']-self.horizon-1 for r in self.records]);self.natural_length=int(self.ends[-1])

    def __len__(self):return self.natural_length

    def __getitem__(self,index):
        row=int(index)%self.natural_length;ep=int(np.searchsorted(self.ends,row,side='right'));r=self.records[ep]
        t=row-(int(self.ends[ep-1])if ep else 0);rng=np.random.default_rng(self.seed+int(index)*1009);goal_query=rng.random()<.5
        horizon=self.horizon;goal=r['length']-1 if rng.random()<.5 else int(rng.integers(t+horizon+1,r['length']));data=self.shards[r['shard']];offset=r['offset']
        def image(j):return torch.from_numpy(np.asarray(data['pixels'][offset+j]).copy()).permute(2,0,1)
        current=torch.stack([image(t+j)for j in range(horizon)])
        previous=torch.stack([image(t+j-5)if t+j>=5 else image(t+j)for j in range(horizon)])
        future=torch.stack([image(t+j+1)for j in range(horizon)])
        controls=torch.zeros(horizon,4)if goal_query else torch.from_numpy(np.asarray(data['inputs'][offset+t:offset+t+horizon]).copy())
        history=torch.stack([torch.zeros(4)if goal_query or t+j==0 else torch.from_numpy(np.asarray(data['inputs'][offset+t+j-1]).copy())for j in range(horizon)])
        return dict(current_pixels=current[0],previous_pixels=previous[0],next_pixels=future[-1],future_pixels=future,
            current_sequence_pixels=current,previous_sequence_pixels=previous,previous_control_sequence=history,
            history_valid_sequence=torch.tensor([t+j>=5 for j in range(horizon)],dtype=torch.float32),
            goal_pixels=image(goal)if goal_query else torch.zeros_like(current[0]),executed_sequence=controls,executed_control=controls[0],
            previous_control=history[0],history_valid=np.float32(t>=5),boundary=np.float32(goal_query),
            episode=r['episode'],start=t,target_row=offset+t+horizon,goal_row=offset+goal)
