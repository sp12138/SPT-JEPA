"""Known current local controls ground a goal controller;one inverse loss only."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from as_jepa_tools.native_inverse import NativeInversePairs


class LocalInversePairs(Dataset):
    def __init__(self,cfg,split='train'):
        self.seed=cfg['seed'];self.shards=[];self.records=[];self.encode_epoch_index=True
        card=json.loads((Path(cfg['data'])/'dataset_card.json').read_text());self.mean=np.asarray(card['action_mean'],np.float32);self.std=np.asarray(card['action_std'],np.float32)
        wanted='train'if split=='train'else'validation'
        for tile in range(4):
            path=Path(cfg['local_inverse_cache'])/str(tile);manifest=json.loads((path/'manifest.json').read_text());assert manifest['passed']and manifest['resolution']==224
            self.shards.append({k:np.load(path/(k+'.npy'),mmap_mode='r')for k in['pixels','inputs','prefix_pixels']})
            self.records.extend(dict(shard=tile,episode=10000000+tile*100000+index,**row)for index,row in enumerate(manifest['records'])if row['split']==wanted)
        self.ends=np.cumsum([r['length']-4 for r in self.records]);self.natural_length=int(self.ends[-1])
    def __len__(self):return self.natural_length
    def __getitem__(self,index):
        row=int(index)%self.natural_length;ep=int(np.searchsorted(self.ends,row,side='right'));r=self.records[ep];t=row-(int(self.ends[ep-1])if ep else 0);data=self.shards[r['shard']];off=r['offset']
        current=data['prefix_pixels'][r['context'],5]if t==0 else data['pixels'][off+t-1];goal=data['pixels'][off+t+4]
        target=(np.asarray(data['inputs'][off+t:off+t+5]).reshape(20).copy()-self.mean)/self.std
        return dict(pixels=torch.from_numpy(np.stack([current,goal]).copy()).permute(0,3,1,2),action=torch.from_numpy(target[None]),action_mask=torch.ones(1),
            horizon=np.int64(0),episode=r['episode']*5,start=t,rows=torch.tensor([off+t,off+t+5]),current_interval_start=off+t,current_interval_stop=off+t+5)


class LocalInverseMix(Dataset):
    def __init__(self,cfg,split='train'):
        self.original=NativeInversePairs(cfg,split);self.local=LocalInversePairs(cfg,split);self.records=self.original.records+self.local.records
        self.natural_length=len(self.original)+len(self.local);self.encode_epoch_index=True
    def __len__(self):return self.natural_length
    def __getitem__(self,index):
        epoch,row=divmod(int(index),self.natural_length)
        if row<len(self.original):return self.original[row+epoch*len(self.original)]
        return self.local[row-len(self.original)+epoch*len(self.local)]
