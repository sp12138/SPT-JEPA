"""Only completed, factual five-step primitives whose input is exactly a0."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from experiments.metaworld_transition_only_20261109.common import read


class HeldPrimitives(Dataset):
    def __init__(self,old,new,split='train'):
        self.shards=[];self.records=[];self.rows=[];self.pool_counts={}
        for corpus,directory in [('ordinary_expert',old),('random_intervention',new)]:
            count=0
            for shard in range(12):
                p=Path(directory)/str(shard);m=read(p/'manifest.json');assert m['passed'] and m['physical_stride']==1
                index=len(self.shards);arrays=dict(pixels=np.load(p/'pixels.npy',mmap_mode='r'),inputs=np.load(p/'known_input.npy',mmap_mode='r'));self.shards.append(arrays)
                for r in m['records']:
                    if r['split']!=split or r['length']<6:continue
                    record=dict(r,shard=index);ri=len(self.records);self.records.append(record);o=r['offset'];length=r['length']
                    if corpus=='random_intervention':times=range(7,length-5,5)
                    else:
                        a=np.asarray(arrays['inputs'][o:o+length-1]);same=(a[1:]==a[:-1]).all(1).astype(np.int32)
                        times=np.flatnonzero(np.convolve(same,np.ones(4,np.int32),mode='valid')==4)
                    for t in times:
                        self.rows.append((ri,int(t)));count+=1
            self.pool_counts[corpus]=count
        self.natural_length=len(self.rows);self.episode_ids=sorted({r['episode'] for r in self.records});assert self.rows

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        ri,t=self.rows[int(index)%len(self.rows)];r=self.records[ri];s=self.shards[r['shard']];o=r['offset'];a0=np.asarray(s['inputs'][o+t]).copy()
        assert np.array_equal(np.asarray(s['inputs'][o+t:o+t+5]),np.broadcast_to(a0,(5,4)))
        def image(i):return torch.from_numpy(np.asarray(s['pixels'][o+i]).copy()).permute(2,0,1)
        times=list(range(t-7,t+1))
        return dict(history_pixels=torch.stack([image(max(0,i)) for i in times]),
            history_actions=torch.stack([torch.from_numpy(np.asarray(s['inputs'][o+i]).copy()).float() if i>=0 else torch.zeros(4) for i in times[:-1]]),
            history_valid=torch.tensor([i>=0 for i in times]),future_pixels=torch.stack([image(t+i) for i in range(1,6)]),
            known_input_a0=torch.from_numpy(a0).float(),episode=r['episode'],start=t)
