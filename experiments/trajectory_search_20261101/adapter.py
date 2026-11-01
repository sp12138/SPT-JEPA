"""Replace 64 first-iteration candidates in-place inside unmodified native CEM."""
from contextlib import contextmanager
from pathlib import Path
import inspect
import numpy as np
import torch
from torch.nn import functional as F
from experiments.trajectory_search_20261101.common import SEED, REPLACE, array_sha, state_hash


class Injection:
    def __init__(self, mode, features, actions):
        assert mode in ('original', 'random', 'retrieval')
        self.mode = mode
        self.features = np.asarray(features, dtype=np.float64)
        self.actions = np.asarray(actions, dtype=np.float32)
        assert self.features.shape == (len(actions), 384) and self.actions.shape[1] == 5
        assert self.actions.shape[-1] in (10, 25)
        self.rng = np.random.default_rng(SEED)
        self.calls = 0
        self.solves = 0
        self.within_solve = 0
        self.injections = []

    def begin(self):
        self.within_solve = 0
        self.solves += 1

    def apply(self, info, candidates):
        assert candidates.shape == (1, 300, 5, self.actions.shape[-1]) and not candidates.requires_grad
        first = self.within_solve % 30 == 0
        self.within_solve += 1
        self.calls += 1
        if not first:
            return
        before = array_sha(candidates.detach().cpu().numpy())
        untouched = candidates[:, 65:].clone()
        mean = candidates[:, 0].clone()
        ids = None
        query_hash = None
        if self.mode != 'original':
            if self.mode == 'random':
                ids = self.rng.choice(len(self.actions), REPLACE, replace=False)
            else:
                # Native transformed current and goal RGB only; no physical fields.
                pixels = torch.stack([info['pixels'][0, 0, -1], info['goal'][0, 0, -1]])
                query = F.adaptive_avg_pool2d(pixels.float(), (8, 8)).reshape(384).cpu().numpy()
                distance = np.square(self.features - query.astype(np.float64)).sum(1)
                ids = np.argsort(distance, kind='stable')[:REPLACE]
                query_hash = array_sha(query)
            replacement = torch.from_numpy(self.actions[ids].copy()).to(candidates)
            candidates[:, 1:65].copy_(replacement[None])
            assert torch.equal(candidates[0, 1:65], replacement)
        assert torch.equal(candidates[:, 0], mean) and torch.equal(candidates[:, 65:], untouched)
        self.injections.append(dict(solve=self.solves, batch=(self.within_solve - 1) // 30,
             ids=None if ids is None else ids.tolist(), query_sha256=query_hash, before_sha256=before,
             after_sha256=array_sha(candidates.detach().cpu().numpy()), replaced=0 if ids is None else REPLACE,
             preserved_mean_and_other235=True))


@contextmanager
def install(mode, features, actions, expected_scaler, model_root):
    from stable_worldmodel.policy import WorldModelPolicy
    from stable_worldmodel.planning import ShootingCostEvaluator
    from stable_worldmodel.planning.solver.cem import CEMSolver
    old_set, old_solve, old_rollout = WorldModelPolicy.set_env, CEMSolver.solve, ShootingCostEvaluator._rollout
    records = []

    def set_env(policy, env):
        result = old_set(policy, env)
        solver = policy.solver
        assert isinstance(solver, CEMSolver) and isinstance(solver.cost, ShootingCostEvaluator)
        assert (solver.num_samples, solver.n_steps, solver.topk, solver.var_scale) == (300, 30, 30, 1)
        assert not solver.callbacks and solver.batch_size == 1
        assert (policy.cfg.horizon, policy.cfg.receding_horizon, policy.cfg.history_len, policy.cfg.action_block) == (5, 5, 3, 5)
        np.testing.assert_array_equal(policy.process['action'].mean_, expected_scaler['action_mean'])
        np.testing.assert_array_equal(policy.process['action'].scale_, expected_scaler['action_std'])
        model = solver.cost.model
        assert Path(inspect.getfile(type(model))).is_relative_to(model_root)
        state = Injection(mode, features, actions)
        state.model = model
        state.state_before = state_hash(model)
        state.model_source = inspect.getfile(type(model))
        state.predict_calls = 0
        state.encoder_calls = 0
        state.original_predict = model.predict
        def predict(*args, **kwargs):
            state.predict_calls += 1
            return state.original_predict(*args, **kwargs)
        model.predict = predict
        def encoded(*args):
            state.encoder_calls += 1
        state.hooks = [model.encoder.register_forward_pre_hook(encoded)]
        if getattr(model, 'renderer', None) is not None:
            def forbidden(*args):
                raise AssertionError('No renderer calls in this planning control')
            state.hooks.append(model.renderer.register_forward_pre_hook(forbidden))
        solver._trajectory_injection = state
        solver.cost._trajectory_injection = state
        records.append(state)
        return result

    def solve(solver, info_dict, init_action=None):
        state = solver._trajectory_injection
        state.begin()
        assert init_action is None or init_action.shape[1] == 0
        answer = old_solve(solver, info_dict, init_action=init_action)
        assert state.within_solve == 30 * len(next(iter(info_dict.values())))
        assert torch.isfinite(answer['actions']).all() and np.isfinite(answer['costs']).all()
        return answer

    def rollout(cost, info, candidates):
        cost._trajectory_injection.apply(info, candidates)
        return old_rollout(cost, info, candidates)

    try:
        WorldModelPolicy.set_env, CEMSolver.solve, ShootingCostEvaluator._rollout = set_env, solve, rollout
        yield records
    finally:
        WorldModelPolicy.set_env, CEMSolver.solve, ShootingCostEvaluator._rollout = old_set, old_solve, old_rollout
        for state in records:
            state.model.predict = state.original_predict
            for hook in state.hooks:
                hook.remove()
