"""Common episode split, windows, actions and anchor selection for every arm."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class PushTWindows(Dataset):
    def __init__(self, root, window=8, gap=8, split="train", length=None, seed=0):
        self.root=Path(root); self.window=window; self.gap=gap; self.seed=seed
        card=json.loads((self.root/"dataset_card.json").read_text())
        assert card["frameskip"]==5
        self.offsets=np.load(self.root/"offsets.npy"); self.lengths=np.load(self.root/"lengths.npy")
        test=set(np.load(self.root/"test_episodes.npy").tolist())
        episodes=[i for i,n in enumerate(self.lengths) if n>=window and ((i in test)==(split=="test"))]
        self.episodes=np.asarray(episodes,dtype=np.int32)
        self.counts=np.asarray([self.lengths[i]-window+1 for i in episodes],dtype=np.int64)
        self.ends=np.cumsum(self.counts); self.natural_length=int(self.ends[-1]); self.length=length or self.natural_length
        self.p224=self.p64=self.actions=self.states=None
        self.raw_rows=np.load(self.root/"raw_rows.npy")
    def open(self):
        if self.p224 is None:
            self.p224=np.load(self.root/"pixels224.npy",mmap_mode="r")
            self.p64=np.load(self.root/"pixels64.npy",mmap_mode="r")
            self.actions=np.load(self.root/"actions.npy",mmap_mode="r")
            self.states=np.load(self.root/"states.npy",mmap_mode="r")
    def __len__(self):return self.length
    def locate(self,index):
        index=int(index)%self.natural_length; i=int(np.searchsorted(self.ends,index,side="right"))
        return int(self.episodes[i]), index-(int(self.ends[i-1]) if i else 0)
    def __getitem__(self,index):
        self.open(); ep,start=self.locate(index); offset=int(self.offsets[ep]); n=int(self.lengths[ep]); w=self.window
        eligible=np.r_[np.arange(max(0,start-self.gap+1)),np.arange(start+w-1+self.gap,n)] if self.gap else np.asarray([start])
        # Stateless sampling is identical across paired runs and workers.
        anchor=int(eligible[hash((ep,start,self.seed))%len(eligible)]) if len(eligible) else (0 if start>n-(start+w) else n-1)
        rows=np.arange(offset+start,offset+start+w)
        return dict(pixels=torch.from_numpy(np.array(self.p224[rows])).permute(0,3,1,2),
                    render=torch.from_numpy(np.array(self.p64[rows])).permute(0,3,1,2),
                    anchor=torch.from_numpy(np.array(self.p64[offset+anchor])).permute(2,0,1),
                    anchor224=torch.from_numpy(np.array(self.p224[offset+anchor])).permute(2,0,1),
                    action=torch.from_numpy(np.array(self.actions[rows])),state=torch.from_numpy(np.array(self.states[rows])),
                    episode=ep,start=start,rows=torch.from_numpy(rows.copy()),anchor_row=offset+anchor,
                    anchor_fallback=not bool(len(eligible)))


def seed_worker(worker):
    np.random.seed(torch.initial_seed()%2**32)
