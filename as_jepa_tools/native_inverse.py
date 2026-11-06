"""One CURRENT executed five-native interval;no subsequent action labels.

The later RGB is a hindsight goal,not a future-command reference.All input
images are original lossless224RGB.Episode balancing never reads task IDs.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class NativeInversePairs(Dataset):
    def __init__(self,cfg,split='train'):
        assert cfg['model']['chunk']==1 and cfg['model']['action_dim']==20
        assert cfg['model']['visual_source']=='local'and not cfg['model']['temporal_aux']and not cfg['model']['history_context']and cfg['model']['condition_horizon']=='none'
        self.seed=cfg['seed'];self.phase=cfg.get('inverse_anchors','all');self.balance=cfg.get('inverse_balance','frames')
        assert self.phase in['all','phase0']and self.balance in['frames','episodes']
        assert self.balance!='episodes'or self.phase=='all'
        self.encode_epoch_index=True;self.records=[];self.shards=[]
        card=json.loads((Path(cfg['data'])/'dataset_card.json').read_text());self.mean=np.asarray(card['action_mean'],np.float32);self.std=np.asarray(card['action_std'],np.float32)
        for shard in range(12):
            path=Path(cfg['native_inverse_cache'])/str(shard);manifest=json.loads((path/'manifest.json').read_text())
            assert manifest['passed']and manifest['resolution']==224 and manifest['no_model_or_feature_file_opened']
            self.shards.append(dict(pixels=np.load(path/'pixels.npy',mmap_mode='r'),controls=np.load(path/'known_input.npy',mmap_mode='r')))
            self.records.extend(dict(shard=shard,**r)for r in manifest['records']if r['split']==split)
        self.counts=[r['length']-5 if self.phase=='all'else(r['length']-6)//5+1 for r in self.records]
        assert min(self.counts)>0;self.ends=np.cumsum(self.counts);self.natural_length=int(self.ends[-1])

    def __len__(self):return self.natural_length

    def __getitem__(self,index):
        index=int(index);row=index%self.natural_length;rng=np.random.default_rng(self.seed+index*1009)
        if self.balance=='episodes':
            episode=row%len(self.records);t=int(rng.integers(self.records[episode]['length']-5))
        else:
            episode=int(np.searchsorted(self.ends,row,side='right'));t=row-(int(self.ends[episode-1])if episode else 0)
            if self.phase=='phase0':t*=5
        record=self.records[episode];available=(record['length']-1-t)//5;assert available>=1
        delta=available if rng.random()<.5 else int(rng.integers(1,available+1));goal=t+delta*5
        data=self.shards[record['shard']];offset=record['offset'];rows=np.array([offset+t,offset+goal])
        image=torch.from_numpy(np.asarray(data['pixels'][rows]).copy()).permute(0,3,1,2)
        # The ONLY supervised command is the already-executedCURRENT interval.
        # No command at t+5 or any later time is indexed here.
        current=np.asarray(data['controls'][offset+t:offset+t+5]).reshape(20).copy()
        action=(current-self.mean)/self.std
        return dict(pixels=image,action=torch.from_numpy(action[None]),action_mask=torch.ones(1),horizon=np.int64(0),
            episode=record['episode']*5,start=t,rows=torch.from_numpy(rows),current_interval_start=offset+t,current_interval_stop=offset+t+5)
