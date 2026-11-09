"""Load only the selected factual endpoint; preserve every native start."""
import numpy as np
import torch
from experiments.metaworld_transition_only_20261109.data import FactualWindows


class EndpointWindows(FactualWindows):
    def __getitem__(self,index):
        record,t=self.locate(index);data=self.shards[record['shard']];offset=record['offset'];length=record['length']
        def image(j):return torch.from_numpy(np.asarray(data['pixels'][offset+j]).copy()).permute(2,0,1)
        def action(j):return torch.from_numpy(np.asarray(data['inputs'][offset+j]).copy()).float()
        starts=[t-self.history+1+j for j in range(self.history)];valid=min(self.maximum_horizon,length-1-t)
        return dict(history_pixels=torch.stack([image(max(0,j)) for j in starts]),
            history_actions=torch.stack([action(j) if j>=0 else torch.zeros(4) for j in starts[:-1]]),
            history_valid=torch.tensor([j>=0 for j in starts]),
            actions=torch.stack([action(t+j) if j<valid else torch.zeros(4) for j in range(self.maximum_horizon)]),
            future_valid=torch.arange(self.maximum_horizon)<valid,episode=record['episode'],start=t,index=int(index))

    def endpoint(self,index,duration):
        record,t=self.locate(index);assert 1<=duration<=min(self.maximum_horizon,record['length']-1-t)
        image=self.shards[record['shard']]['pixels'][record['offset']+t+duration]
        return torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)


def add_endpoint(data,batch,durations,device):
    batch['endpoint_pixels']=torch.stack([data.endpoint(int(i),int(d)) for i,d in zip(batch['index'].cpu(),durations.cpu())]).to(device)
    return batch
