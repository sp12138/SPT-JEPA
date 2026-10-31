"""Common Box-bounded controller: candidate scoring AND returned raw commands.

Uses the policy's actual fitted sklearn scaler and the attached environment's
single_action_space. No changes to CEM sampling, moments, selected normalized
plans, warm-start state, learned models or training. CPU sklearn conversion is
intentional: it follows the exact installed execution scaler, including casts.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import numpy as np
import torch


def array_sha(value):
    x = np.ascontiguousarray(value)
    return hashlib.sha256(str(x.shape).encode()+str(x.dtype).encode()+x.tobytes()).hexdigest()


class BoxProjection:
    def __init__(self, space, scaler):
        from gymnasium.spaces import Box
        from sklearn.preprocessing import StandardScaler
        assert isinstance(space, Box) and len(space.shape) == 1
        assert isinstance(scaler, StandardScaler) and scaler.with_mean and scaler.with_std
        self.low, self.high = space.low.copy(), space.high.copy()
        assert np.isfinite(self.low).all() and np.isfinite(self.high).all()
        assert np.all(self.low < self.high)
        self.dim = space.shape[0]
        assert scaler.mean_.shape == scaler.scale_.shape == (self.dim,)
        assert np.isfinite(scaler.mean_).all() and np.all(scaler.scale_ > 0)
        self.scaler = scaler
        self.stats = dict(score_calls=0, score_coordinates=0, score_clipped_coordinates=0,
                          normalized_change_squared_sum=0., execution_calls=0,
                          execution_finite_coordinates=0, execution_clipped_coordinates=0,
                          returned_command_violations=0)

    def numpy_project(self, value):
        assert value.dtype in (np.float32, np.float64) and value.shape[-1] % self.dim == 0
        raw = self.scaler.inverse_transform(value.reshape(-1, self.dim).copy())
        clipped = np.clip(raw, self.low, self.high)
        normalized = self.scaler.transform(clipped.copy()).reshape(value.shape)
        return normalized, raw, clipped

    def candidates(self, value):
        assert not value.requires_grad, 'This adapter is only for the native CEM evaluator'
        original = value.detach().cpu().numpy()
        assert np.isfinite(original).all()
        projected, raw, clipped = self.numpy_project(original)
        s = self.stats
        s['score_calls'] += 1; s['score_coordinates'] += raw.size
        s['score_clipped_coordinates'] += int(np.count_nonzero(raw != clipped))
        s['normalized_change_squared_sum'] += float(np.square(original.astype(np.float64)-projected).sum())
        if s['score_calls'] == 1:
            s['first_score'] = dict(shape=list(value.shape), input_sha256=array_sha(original),
                projected_sha256=array_sha(projected), physical_projected_sha256=array_sha(clipped))
        return torch.from_numpy(projected).to(device=value.device, dtype=value.dtype)

    def execution(self, raw):
        # Native policy represents inactive envs by NaN; preserve those entries.
        assert isinstance(raw, np.ndarray) and raw.shape[-1] == self.dim
        assert not np.isinf(raw).any(), 'Infinite command is an error, not an inactive environment'
        assert np.array_equal(np.isnan(raw).any(-1), np.isnan(raw).all(-1)), 'Only whole inactive rows may be NaN'
        clipped = np.clip(raw, self.low, self.high)
        finite = np.isfinite(raw)
        s = self.stats; s['execution_calls'] += 1
        s['execution_finite_coordinates'] += int(finite.sum())
        s['execution_clipped_coordinates'] += int((finite & (raw != clipped)).sum())
        violations = finite & ((clipped < self.low) | (clipped > self.high))
        s['returned_command_violations'] += int(violations.sum())
        assert not violations.any() and np.array_equal(np.isnan(raw), np.isnan(clipped))
        if s['execution_calls'] == 1:
            s['first_execution'] = dict(input_sha256=array_sha(raw), returned_sha256=array_sha(clipped))
        return clipped

    def report(self):
        return dict(physical_lower=self.low.tolist(), physical_upper=self.high.tolist(),
                    action_mean=self.scaler.mean_.tolist(), action_scale=self.scaler.scale_.tolist(),
                    **self.stats)


@contextmanager
def install(mode):
    """Restore native methods even on failure; 'original' passes objects unchanged."""
    assert mode in ('original', 'bounded')
    from stable_worldmodel.policy import WorldModelPolicy
    from stable_worldmodel.planning import ShootingCostEvaluator
    original_set = WorldModelPolicy.set_env
    original_get = WorldModelPolicy.get_action
    original_rollout = ShootingCostEvaluator._rollout
    projections = []

    def set_env(policy, env):
        result = original_set(policy, env)
        assert isinstance(policy.solver.cost, ShootingCostEvaluator)
        projection = BoxProjection(env.single_action_space, policy.process['action'])
        assert not hasattr(policy, '_box_projection') and not hasattr(policy.solver.cost, '_box_projection')
        policy._box_projection = projection
        policy.solver.cost._box_projection = projection
        projections.append(projection)
        return result

    def rollout(cost, info, candidates):
        projection = cost._box_projection
        effective = projection.candidates(candidates) if mode == 'bounded' else candidates
        return original_rollout(cost, info, effective)

    def get_action(policy, info, **kwargs):
        command = original_get(policy, info, **kwargs)
        return policy._box_projection.execution(command) if mode == 'bounded' else command

    try:
        WorldModelPolicy.set_env = set_env
        WorldModelPolicy.get_action = get_action
        ShootingCostEvaluator._rollout = rollout
        yield projections
    finally:
        WorldModelPolicy.set_env = original_set
        WorldModelPolicy.get_action = original_get
        ShootingCostEvaluator._rollout = original_rollout
