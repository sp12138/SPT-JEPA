"""Identify the input of the first factual transition, never a future action.

The observed inverse reads q(x0),q(x1) and targets the already executed a0.
The paired variant uses the same inverse for F(q(x0),a0), sharing that same a0
target. It tests whether agreement between observed and predicted action effects
helps control. This is an auxiliary loss, not a new ELBO or an optimal policy.
Independent worlds never exchange representations or statistics.
"""
import types

import torch
from torch import nn


def inverse_error(model,initial,following,current_action):
    inputs=torch.cat((initial,following-initial),-1)
    prediction=model.inverse_head(inputs)
    return .5*(prediction.float()-current_action.float()).square().mean()


def _encode(self,info):
    result=self._inverse_encode(info)
    self._inverse_codes=result["emb"][:,:2]
    return result


def _loss(self,batch,step=0,ramp_steps=0):
    try:
        loss,metrics=self._inverse_loss(batch,step,ramp_steps)
        if self.current_inverse=="control":return loss,metrics
        initial,following=self._inverse_codes.unbind(1)
        current=batch["action"][:,0]
        observed=inverse_error(self,initial,following,current)
        extra=observed
        metrics["inverse_observed_a0"]=observed
        if self.current_inverse=="paired":
            # The original loss clears its context. Rebuild it from actual x0;
            # no observed future image is an input to this prediction.
            if hasattr(self,"context_bridge"):
                self.context_tokens=self.feature_adapter(batch["features"][:,0].float())
            prediction=self.predict(initial[:,None],self.action_encoder(current[:,None]))[:,0]
            predicted=inverse_error(self,initial,prediction,current)
            extra=.5*(observed+predicted)
            metrics["inverse_predicted_a0"]=predicted
        # Fixed unit-variance Gaussian NLL per normalized input coordinate.
        # Reuse the existing beta, without a task-specific inverse coefficient.
        loss=loss+self.beta*extra
        metrics.update(loss=loss,current_input_inverse=extra)
        return loss,metrics
    finally:
        self._inverse_codes=None
        if hasattr(self,"context_bridge"):self.context_tokens=None


def attach(model,kind,seed):
    assert kind in ("control","observed","paired")
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(820331+seed).get_state())
        model.inverse_head=nn.Sequential(nn.LayerNorm(2*model.state_dim),
            nn.Linear(2*model.state_dim,256),nn.GELU(),nn.Linear(256,14))
    model.current_inverse=kind
    model._inverse_codes=None
    model._inverse_encode=model.encode;model.encode=types.MethodType(_encode,model)
    model._inverse_loss=model.loss;model.loss=types.MethodType(_loss,model)
    return model
