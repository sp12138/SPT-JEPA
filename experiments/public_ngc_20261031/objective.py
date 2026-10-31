"""Only the existing GoalMSE seam changes; the public NGC code is unmodified."""
from contextlib import contextmanager
import inspect
from pathlib import Path
import numpy as np
import torch
from torch import nn
from common import read, write, sha, state_hash
from vendor.cost import NGCCost


def digest_tensor(t):
    import hashlib
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


class Recorder:
    def __init__(self, output): self.output, self.costs, self.rows = Path(output), [], []
    def record(self, info, native, ngc):
        values = torch.stack([native,ngc]).detach().cpu().numpy(); self.costs.append(values)
        self.rows.append(dict(call=len(self.rows), shape=list(values.shape),
            actions_sha256=digest_tensor(info['action_candidates']), goal_sha256=digest_tensor(info['goal_emb']),
            predicted_sha256=digest_tensor(info['predicted_emb'][...,-1,:]),
            selected_indices=np.argmin(values,axis=-1).tolist()))
    def finish(self):
        assert self.rows
        np.savez_compressed(self.output/'costs.npz', costs=np.concatenate(self.costs,axis=1),
                            call_offsets=np.r_[0,np.cumsum([x.shape[1] for x in self.costs])])
        write(self.output/'banks.json',self.rows)
        return dict(calls=len(self.rows), metric_order=['l2','ngc'], costs_sha256=sha(self.output/'costs.npz'),
                    banks_sha256=sha(self.output/'banks.json'), extra_model_calls=0)


class Goal(nn.Module):
    def __init__(self, native, nodes, distances, mode, recorder=None):
        super().__init__(); assert mode in ['l2','ngc']
        self.native,self.ngc,self.mode,self.recorder = native,NGCCost(nodes,distances,15,4),mode,recorder
    def forward(self, info):
        native = self.native(info)
        assert self.native.pred_key == 'predicted_emb' and self.native.goal_key == 'goal_emb'
        delta = info['predicted_emb'][...,-1,:]-info['goal_emb'][:,None,-1,:].detach()
        manual = delta.square().sum(-1) if self.native.reduction == 'sum' else delta.square().mean(-1)
        assert torch.equal(native,manual), 'Native GoalMSE algebra mismatch'
        ngc = self.ngc(info)
        assert torch.isfinite(native).all() and torch.isfinite(ngc).all()
        if self.recorder: self.recorder.record(info,native,ngc)
        return native if self.mode == 'l2' else ngc


@contextmanager
def install(nodes, distances, mode, recorder=None, spec=None):
    from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
    from stable_worldmodel.policy import WorldModelPolicy
    old_init,old_env = ShootingCostEvaluator.__init__,WorldModelPolicy.set_env
    records = []
    def initialize(self, model, objective, *args, **kwargs):
        assert type(objective) is GoalMSE
        source = Path(inspect.getsourcefile(type(model))).resolve()
        if spec: assert source.is_relative_to(Path(spec['model_source']['source']))
        record = dict(model=model,model_source=str(source),model_source_sha256=sha(source),
                      state_before=state_hash(model),goal_config=dict(pred_key=objective.pred_key,
                      goal_key=objective.goal_key,reduction=objective.reduction))
        records.append(record)
        adapted = Goal(objective,nodes,distances,mode,recorder).to(next(model.parameters()).device)
        result = old_init(self,model,adapted,*args,**kwargs)
        return result
    def set_env(self, env):
        result = old_env(self,env)
        if spec:
            card = read(Path(spec['data'])/'dataset_card.json')
            np.testing.assert_array_equal(self.process['action'].mean_,card['action_mean'])
            np.testing.assert_array_equal(self.process['action'].scale_,card['action_std'])
            cfg = self.cfg
            assert (cfg.horizon,cfg.receding_horizon,cfg.history_len,cfg.action_block) == (5,5,3,5)
            records[0]['actual_scaler_checked'] = True
        return result
    ShootingCostEvaluator.__init__,WorldModelPolicy.set_env = initialize,set_env
    try: yield records
    finally: ShootingCostEvaluator.__init__,WorldModelPolicy.set_env = old_init,old_env
