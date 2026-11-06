"""Future observation pairs with own past command;no current action field."""
import numpy as np
import torch

from as_jepa_tools.predictive_kernel_data import KernelTransitions


class AttainablePairs(KernelTransitions):
    def __getitem__(self,index):
        r,t=self.locate(index);data=self.shards[r['shard']];offset=r['offset']
        rng=np.random.default_rng(self.seed+int(index)*1009)
        valid=t+2<r['length']
        end=(r['length']-1 if rng.random()<.5 else int(rng.integers(t+2,r['length'])))if valid else t
        def feature(i):return torch.from_numpy(np.asarray(data['features'][offset+i]).copy())
        return dict(previous_z=feature(max(0,t-1)),current_z=feature(t),next_z=feature(t+1),
            goal_z=feature(end)if valid else torch.zeros_like(feature(t)),goal_valid=np.float32(valid),
            previous_control=torch.from_numpy(np.asarray(data['inputs'][offset+t-1]).copy())if t else torch.zeros(4),
            history_valid=np.float32(t>0),episode=r['episode'],start=t,target_row=offset+t+1,goal_row=offset+end)
