"""Sample-level consequence identification without a pixel reconstruction loss."""
import types

import torch
from torch.nn import functional as F

from experiments.robocasa_refine.analytic_model import trajectory_categorical_nll


def current_next_nll(current,future,query,sigma):
    """Binary consequence likelihood relative to the observed current frame.

    For query=current+u and displacement d=future-current, the loss is
    softplus((||u-d||^2-||u||^2)/(2*sigma^2)). With fixed nonzero u its
    unique displacement optimum is d=u; an unchanged encoding gives log(2).
    No unobserved zero-action outcome or target action is supplied.
    """
    positive=(query.float()-future.float()).flatten(-2).square().sum(-1)
    reference=(query.float()-current.float()).flatten(-2).square().sum(-1)
    return F.softplus((positive-reference)/(2*sigma**2)),(positive<reference).float().mean()


def identification_loss(self,batch,step=0,ramp_steps=0):
    pixels,_,_=self.prepare(batch)
    info=self.encode(dict(pixels=pixels,action=batch['action']))
    flat=info['emb'];b,w=flat.shape[:2]
    observed=flat.reshape(b,w,self.state_slots,self.state_width)
    mode=self.consequence_likelihood_mode
    if mode=='rollout':
        predicted=self.mean_trajectory(flat[:,0],info['act_emb'][:,:-1]).reshape_as(observed)
    else:
        # Each query uses its own observed current frame and the input of that
        # factual transition. A discriminant query is never fed back as state.
        current=flat[:,:-1].reshape(b*(w-1),1,-1)
        actions=info['act_emb'][:,:-1].reshape(b*(w-1),1,-1)
        query=self.predict(current,actions)[:,0].reshape(b,w-1,self.state_slots,self.state_width)
        predicted=torch.cat((observed[:,:1],query),dim=1)
    if mode=='factual-pair':
        nll,accuracy=current_next_nll(observed[:,:-1],observed[:,1:],predicted[:,1:],self.sigma0)
    else:
        nll,accuracy=trajectory_categorical_nll(observed,predicted,self.sigma0)
    loss=nll.mean()
    metrics=dict(loss=loss,categorical_consequence_nll=loss,
        feature_std=observed.float().std((0,1)).mean(),likelihood_only=loss.new_tensor(1.))
    metrics['next_preferred_to_current' if mode=='factual-pair' else 'trajectory_retrieval_accuracy']=accuracy
    metrics['clean_rollout_mse' if mode=='rollout' else 'factual_prediction_mse']=(
        observed[:,1:].float()-predicted[:,1:].float()).square().mean()
    return loss,metrics


def install(model,cfg):
    assert cfg.get('forecast_rate_geometry')=='trajectory-categorical'
    mode=cfg.get('consequence_likelihood_mode','rollout')
    assert mode in ('rollout','factual-trajectory','factual-pair')
    assert not cfg.get('ema_future_targets')
    if mode in ('rollout','factual-pair'):
        assert cfg.get('relative_state_coding') and cfg.get('isometric_actions')
        assert cfg.get('pure_action_transport')
    model.consequence_likelihood_mode=mode
    # Preserve checkpoint construction; this random decoder is never evaluated
    # or optimized. The RGB encoder remains the sole visual learner.
    model.renderer.requires_grad_(False)
    model.loss=types.MethodType(identification_loss,model)
