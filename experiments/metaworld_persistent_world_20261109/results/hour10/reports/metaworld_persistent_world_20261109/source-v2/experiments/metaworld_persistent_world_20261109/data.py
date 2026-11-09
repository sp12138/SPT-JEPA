"""A natural-length union of ordinary transition pools; no goal/path examples."""
import numpy as np
from experiments.metaworld_joint_flow_20261109.data import EndpointWindows,add_endpoint


class PooledWindows(EndpointWindows):
    def __init__(self,directories,split='train',history=8,maximum_horizon=60):
        self.history=history;self.maximum_horizon=maximum_horizon;self.shards=[];self.records=[]
        for directory in directories:
            part=EndpointWindows(directory,split,history,maximum_horizon);offset=len(self.shards)
            self.shards.extend(part.shards);self.records.extend(dict(r,shard=r['shard']+offset) for r in part.records)
        self.ends=np.cumsum([r['length']-1 for r in self.records]);self.natural_length=int(self.ends[-1])
        self.episode_ids=sorted(r['episode'] for r in self.records);assert len(set(self.episode_ids))==len(self.episode_ids)
