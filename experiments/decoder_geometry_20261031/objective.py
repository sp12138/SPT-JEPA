"""Fixed decoder-response factors at the existing native objective seam."""
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn

METRICS = ("identity", "decoder", "rotated")


def validate_matrices(matrices):
    required = {"decoder_gram_raw", "decoder_gram", "rotated_gram", "decoder_factor", "rotated_factor", "rotation"}
    assert set(matrices) == required
    for value in matrices.values():
        assert value.shape == (64, 64) and np.isfinite(value).all()
    raw, norm, q = (matrices[k] for k in ["decoder_gram_raw", "decoder_gram", "rotation"])
    assert np.trace(raw) > 0
    np.testing.assert_allclose(raw, raw.T, rtol=1e-12, atol=1e-14)
    eigen = np.linalg.eigvalsh(raw)
    assert eigen[0] >= -64*np.finfo(np.float64).eps*eigen[-1]
    np.testing.assert_allclose(norm, raw*64/np.trace(raw), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(q.T@q, np.eye(64), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(matrices["rotated_gram"], q.T@norm@q, rtol=1e-10, atol=1e-12)
    for name in METRICS[1:]:
        factor = matrices[name+"_factor"]
        np.testing.assert_allclose(factor@factor.T, matrices[name+"_gram"], rtol=1e-10, atol=1e-12)


class DecoderGoal(nn.Module):
    def __init__(self, native, matrices, selected="identity", recorder=None):
        super().__init__()
        if selected not in METRICS: raise ValueError(selected)
        validate_matrices(matrices)
        self.native, self.selected, self.recorder = native, selected, recorder
        for name in METRICS[1:]:
            self.register_buffer(name+"_factor", torch.tensor(matrices[name+"_factor"], dtype=torch.float32))

    def forward(self, info):
        native_cost = self.native(info)
        pred = info[self.native.pred_key]
        goal = info[self.native.goal_key][:, None, -1:].expand_as(pred)
        delta = (pred[:, :, -1:]-goal[:, :, -1:].detach()).flatten(2)
        assert delta.shape[-1] == 64
        manual = delta.square().sum(-1) if self.native.reduction == "sum" else delta.square().mean(-1)
        if not torch.equal(manual, native_cost):
            raise AssertionError("Native identity algebra mismatch; no metric evaluation is valid")
        scores = {"identity":native_cost}
        with torch.autocast(device_type=delta.device.type, enabled=False):
            for name in METRICS[1:]:
                squared = (delta.float() @ getattr(self, name+"_factor")).square()
                scores[name] = squared.sum(-1) if self.native.reduction == "sum" else squared.mean(-1)
        if not all(torch.isfinite(v).all() for v in scores.values()):
            raise ValueError("Nonfinite candidate cost")
        if self.recorder:
            self.recorder.record(info, scores, delta, (manual-native_cost).abs().max())
        # Identity returns this exact original tensor, without re-encoding or casting.
        return native_cost if self.selected == "identity" else scores[self.selected]


@contextmanager
def install_objective(matrices, selected, recorder=None):
    from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
    original = ShootingCostEvaluator.__init__; records = []
    def initialize(self, model, objective, *args, **kwargs):
        if type(objective) is not GoalMSE: raise TypeError("Expected the unchanged native GoalMSE")
        adapted = DecoderGoal(objective, matrices, selected, recorder).to(next(model.parameters()).device)
        records.append(dict(class_name="GoalMSE", pred_key=objective.pred_key, goal_key=objective.goal_key,
            reduction=objective.reduction, prediction_time="last", goal_time="last"))
        original(self, model, adapted, *args, **kwargs)
    ShootingCostEvaluator.__init__ = initialize
    try: yield records
    finally: ShootingCostEvaluator.__init__ = original
