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
        assert all(r['length']>6 for r in self.records)
        # Keep exactly the original H5 starts,including short trajectories.
        # H15 uses only genuinely observed future frames;its padded tail is
        # masked out of the likelihood and never treated as an observation.
        self.ends=np.cumsum([r['length']-6 for r in self.records]);self.natural_length=int(self.ends[-1])

    def __len__(self):return self.natural_length

    def __getitem__(self,index):
        row=int(index)%self.natural_length;ep=int(np.searchsorted(self.ends,row,side='right'));r=self.records[ep]
        t=row-(int(self.ends[ep-1])if ep else 0);rng=np.random.default_rng(self.seed+int(index)*1009);goal_query=rng.random()<.5
        horizon=self.horizon;valid_horizon=min(horizon,r['length']-t-2)
        goal=r['length']-1 if rng.random()<.5 else int(rng.integers(t+valid_horizon+1,r['length']));data=self.shards[r['shard']];offset=r['offset']
        def image(j):return torch.from_numpy(np.asarray(data['pixels'][offset+j]).copy()).permute(2,0,1)
        actual=[min(j,valid_horizon-1)for j in range(horizon)]
        current=torch.stack([image(t+j)for j in actual])
        previous=torch.stack([image(t+j-5)if t+j>=5 else image(t+j)for j in actual])
        future=torch.stack([image(t+j+1)for j in actual])
        controls=torch.stack([torch.zeros(4)if goal_query or j>=valid_horizon else torch.from_numpy(np.asarray(data['inputs'][offset+t+j]).copy())for j in range(horizon)])
        history=torch.stack([torch.zeros(4)if goal_query or t+j==0 or j>=valid_horizon else torch.from_numpy(np.asarray(data['inputs'][offset+t+j-1]).copy())for j in range(horizon)])
        result=dict(current_pixels=current[0],previous_pixels=previous[0],next_pixels=future[valid_horizon-1],future_pixels=future,
            current_sequence_pixels=current,previous_sequence_pixels=previous,previous_control_sequence=history,
            history_valid_sequence=torch.tensor([t+j>=5 for j in actual],dtype=torch.float32),
            goal_pixels=image(goal)if goal_query else torch.zeros_like(current[0]),executed_sequence=controls,executed_control=controls[0],
            previous_control=history[0],history_valid=np.float32(t>=5),boundary=np.float32(goal_query),
            episode=r['episode'],start=t,target_row=offset+t+valid_horizon,goal_row=offset+goal)
        if horizon>5:result['future_valid']=torch.tensor([j<valid_horizon for j in range(horizon)],dtype=torch.float32)
        return result
