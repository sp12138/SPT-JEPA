"""Calibrate the exact clean, deterministic operator used by the planner."""
import torch
from as_jepa_lewm.robust import RobustLeWM


class RefinedSPT(RobustLeWM):
    def __init__(self, clean_calibration=False, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(clean_calibration,bool)
        self.clean_calibration=clean_calibration
        self._capture_clean=False
        self._clean_features=None

    def encode(self, info):
        result=super().encode(info)
        if self._capture_clean:
            self._clean_features=(result["emb"],result["act_emb"])
        return result

    def loss(self,batch,step=0,ramp_steps=0):
        self._capture_clean=self.clean_calibration
        try:
            loss,metrics=super().loss(batch,step,ramp_steps)
            if self.clean_calibration:
                features,actions=self._clean_features
                training=self.predictor.training
                # The calibrated operator is the planner's deterministic mean.
                # The original noisy/dropout training branch remains unchanged.
                self.predictor.eval()
                try:
                    # Treat observed codes as fixed system-identification data;
                    # the original SPT objective still learns the encoder/pool.
                    history=[features[:,0].detach()]
                    for t in range(features.shape[1]-1):
                        history.append(self.predict(torch.stack(history,1),actions[:,:t+1])[:,-1])
                    prediction=torch.stack(history[1:],1).float()
                finally:
                    self.predictor.train(training)
                # Observed future RGB codes are targets, never rollout inputs.
                # All control inputs precede the predicted transition endpoint;
                # no action is a supervised prediction target.
                target=features[:,1:].detach().float()
                calibration=.5*(prediction-target).square().mean()/(self.sigma0**2)
                loss=loss+calibration
                metrics.update(loss=loss,clean_rollout_gaussian_nll=calibration,
                               clean_rollout_mse=(prediction-target).square().mean())
            return loss,metrics
        finally:
            self._capture_clean=False
            self._clean_features=None
