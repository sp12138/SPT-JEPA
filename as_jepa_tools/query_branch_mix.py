"""Original windows plus only the newly intervened future windows.

The union uses each eligible start once;there is no tuned corpus mixing weight.
Controls are forward inputs.The goal-query branch reads no controls.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from as_jepa_tools.query_rgb_paths import QueryRGBPaths


class BranchRGBPaths(Dataset):
    def __init__(self,cfg,split='train'):
        assert cfg['model'].get('forecast_horizon',5)==5
        self.seed=cfg['seed'];self.goal_queries=cfg.get('intervention_goal_queries',True);self.encode_epoch_index=True;self.shards=[];self.records=[]
        wanted='train'if split=='train'else'validation'
        for tile in range(4):
            path=Path(cfg['intervention_rgb_cache'])/str(tile);manifest=json.loads((path/'manifest.json').read_text());assert manifest['passed']
            self.shards.append({name:np.load(path/(name+'.npy'),mmap_mode='r')for name in['pixels','inputs','prefix_pixels','prefix_controls']})
            self.records.extend(dict(shard=tile,**r)for r in manifest['records']if r['split']==wanted)
        self.ends=np.cumsum([r['length']-4 for r in self.records]);self.natural_length=int(self.ends[-1]);assert self.natural_length>0

    def __len__(self):return self.natural_length

    def __getitem__(self,index):
        row=int(index)%self.natural_length;episode=int(np.searchsorted(self.ends,row,side='right'));r=self.records[episode]
        start=row-(int(self.ends[episode-1])if episode else 0);data=self.shards[r['shard']];offset=r['offset'];context=r['context'];length=r['length']
        rng=np.random.default_rng(self.seed+int(index)*1009);goal_query=self.goal_queries and start+5<length and rng.random()<.5
        goal=length if not goal_query or rng.random()<.5 else int(rng.integers(start+6,length+1))
        def image(t):
            assert -5<=t<=length
            value=data['prefix_pixels'][context,t+5]if t<=0 else data['pixels'][offset+t-1]
            return torch.from_numpy(np.asarray(value).copy()).permute(2,0,1)
        def control(t):
            assert -5<=t<length
            value=data['prefix_controls'][context,t+5]if t<0 else data['inputs'][offset+t]
            return torch.from_numpy(np.asarray(value).copy())
        current=torch.stack([image(start+j)for j in range(5)]);previous=torch.stack([image(start+j-5)for j in range(5)])
        future=torch.stack([image(start+j+1)for j in range(5)])
        controls=torch.zeros(5,4)if goal_query else torch.stack([control(start+j)for j in range(5)])
        history=torch.zeros(5,4)if goal_query else torch.stack([control(start+j-1)for j in range(5)])
        return dict(current_pixels=current[0],previous_pixels=previous[0],next_pixels=future[-1],future_pixels=future,
            current_sequence_pixels=current,previous_sequence_pixels=previous,previous_control_sequence=history,
            history_valid_sequence=torch.ones(5),goal_pixels=image(goal)if goal_query else torch.zeros_like(current[0]),
            executed_sequence=controls,executed_control=controls[0],previous_control=history[0],history_valid=np.float32(1),boundary=np.float32(goal_query),
            episode=1000000+episode,start=start,target_row=episode*100+start+5,goal_row=episode*100+goal)


class QueryBranchMix(Dataset):
    def __init__(self,cfg,split='train'):
        self.original=QueryRGBPaths(cfg,split);self.branches=BranchRGBPaths(cfg,split)
        self.natural_length=len(self.original)+len(self.branches);self.encode_epoch_index=True
    def __len__(self):return self.natural_length
    def __getitem__(self,index):
        epoch,row=divmod(int(index),self.natural_length)
        if row<len(self.original):return self.original[row+epoch*len(self.original)]
        return self.branches[row-len(self.original)+epoch*len(self.branches)]
