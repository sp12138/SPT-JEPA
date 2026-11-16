"""Sample-level consequence identification without a pixel reconstruction loss."""
import types

from experiments.robocasa_refine.analytic_model import trajectory_categorical_nll


def identification_loss(self,batch,step=0,ramp_steps=0):
    pixels,_,_=self.prepare(batch)
    info=self.encode(dict(pixels=pixels,action=batch['action']))
    flat=info['emb'];b,w=flat.shape[:2]
    observed=flat.reshape(b,w,self.state_slots,self.state_width)
    predicted=self.mean_trajectory(flat[:,0],info['act_emb'][:,:-1]).reshape_as(observed)
    nll,accuracy=trajectory_categorical_nll(observed,predicted,self.sigma0)
    loss=nll.mean()
    return loss,dict(loss=loss,categorical_consequence_nll=loss,
        trajectory_retrieval_accuracy=accuracy,
        feature_std=observed.float().std((0,1)).mean(),
        clean_rollout_mse=(observed[:,1:].float()-predicted[:,1:].float()).square().mean(),
        likelihood_only=loss.new_tensor(1.))


def install(model,cfg):
    assert cfg.get('forecast_rate_geometry')=='trajectory-categorical'
    assert cfg.get('relative_state_coding') and cfg.get('isometric_actions')
    assert cfg.get('pure_action_transport') and not cfg.get('ema_future_targets')
    # Preserve checkpoint construction; this random decoder is never evaluated
    # or optimized. The RGB encoder remains the sole visual learner.
    model.renderer.requires_grad_(False)
    model.loss=types.MethodType(identification_loss,model)
