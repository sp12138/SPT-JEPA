"""Read observed endpoints and their reverse-time histories, never goal queries."""
import numpy as np
import torch
from experiments.metaworld_joint_flow_20261109.data import EndpointWindows


def add_targets(data,batch,durations,device):
    endpoints=[];reverse_history=[];reverse_past=[];reverse_valid=[];reverse_actions=[];original=[]
    for index,duration in zip(batch['index'].cpu(),durations.cpu()):
        r,t=data.locate(int(index));d=int(duration);s=data.shards[r['shard']];o=r['offset'];length=r['length'];end=t+d
        def image(j):return torch.from_numpy(np.asarray(s['pixels'][o+j]).copy()).permute(2,0,1)
        def action(j):return torch.from_numpy(np.asarray(s['inputs'][o+j]).copy()).float()
        starts=[end+data.history-1-j for j in range(data.history)]
        endpoints.append(image(end));original.append(image(t))
        reverse_history.append(torch.stack([image(min(j,length-1)) for j in starts]))
        reverse_past.append(torch.stack([action(j-1) if j<length else torch.zeros(4) for j in starts[:-1]]))
        reverse_valid.append(torch.tensor([j<length for j in starts]))
        reverse_actions.append(torch.stack([action(end-1-j) if j<d else torch.zeros(4) for j in range(data.maximum_horizon)]))
    for key,value in dict(endpoint_pixels=endpoints,reverse_endpoint_pixels=original,reverse_history_pixels=reverse_history,
        reverse_history_actions=reverse_past,reverse_history_valid=reverse_valid,reverse_actions=reverse_actions).items():
        batch[key]=torch.stack(value).to(device)
    return batch
