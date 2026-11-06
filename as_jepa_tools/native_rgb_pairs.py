"""Native-frame future prediction,with no inverse-action targets."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NativeRGBPairs(Dataset):
    def __init__(self,cfg,split='train'):
        self.seed=cfg['seed'];self.encode_epoch_index=True;self.shards=[];self.records=[]
        for shard in range(12):
            path=Path(cfg['native_rgb_cache'])/str(shard);manifest=json.loads((path/'manifest.json').read_text())
            assert manifest['passed']and manifest['physical_stride']==1 and manifest['terminal_macro_padding_recovered']
            self.shards.append(dict(pixels=np.load(path/'pixels.npy',mmap_mode='r'),inputs=np.load(path/'known_input.npy',mmap_mode='r')))
            self.records.extend(dict(shard=shard,**r)for r in manifest['records']if r['split']==split)
        self.ends=np.cumsum([r['length']-1 for r in self.records]);self.natural_length=int(self.ends[-1])

    def __len__(self):return self.natural_length

    def __getitem__(self,index):
        row=int(index)%self.natural_length;episode=int(np.searchsorted(self.ends,row,side='right'))
        r=self.records[episode];t=row-(int(self.ends[episode-1])if episode else 0);offset=r['offset'];data=self.shards[r['shard']]
        rng=np.random.default_rng(self.seed+int(index)*1009)
        boundary=t+2<r['length']and rng.random()<.5
        end=(r['length']-1 if rng.random()<.5 else int(rng.integers(t+2,r['length'])))if boundary else t
        def image(i):return torch.from_numpy(np.asarray(data['pixels'][offset+i]).copy()).permute(2,0,1)
        return dict(current_pixels=image(t),next_pixels=image(t+1),goal_pixels=image(end)if boundary else torch.zeros_like(image(t)),
            executed_control=torch.zeros(4)if boundary else torch.from_numpy(np.asarray(data['inputs'][offset+t]).copy()),
            previous_control=torch.from_numpy(np.asarray(data['inputs'][offset+t-1]).copy())if t else torch.zeros(4),
            boundary=np.float32(boundary),history_valid=np.float32(t>0),episode=r['episode'],start=t,target_row=offset+t+1,goal_row=offset+end)
