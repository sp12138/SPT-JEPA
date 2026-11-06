"""One next-observation target,queried with three information sets."""
import numpy as np
import torch

from as_jepa_tools.predictive_kernel_data import KernelTransitions


class DensityTransitions(KernelTransitions):
    def __getitem__(self, index):
        record, t = self.locate(index)
        data, offset = self.shards[record['shard']], record['offset']
        rng = np.random.default_rng(self.seed+int(index)*1009)
        goal_valid = t+2 < record['length']
        end = (record['length']-1 if rng.random() < .5 else int(rng.integers(t+2, record['length']))) if goal_valid else t
        def feature(i):
            return torch.from_numpy(np.asarray(data['features'][offset+i]).copy())
        return dict(previous_z=feature(max(0, t-1)), current_z=feature(t), next_z=feature(t+1),
                    goal_z=feature(end) if goal_valid else torch.zeros_like(feature(t)),
                    executed_control=torch.from_numpy(np.asarray(data['inputs'][offset+t]).copy()),
                    previous_control=torch.from_numpy(np.asarray(data['inputs'][offset+t-1]).copy())if t>0 else torch.zeros(4),
                    history_valid=np.float32(t > 0), goal_valid=np.float32(goal_valid),
                    episode=record['episode'], start=t, target_row=offset+t+1, goal_row=offset+end)
