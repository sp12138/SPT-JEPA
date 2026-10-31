"""Known-bank and native-CEM CPU contracts; no datasets or checkpoint evaluation."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch import nn
from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
from stable_worldmodel.planning.solver.cem import CEMSolver
from experiments.carrier_progress_20261031.metric_core import MetricRecorder
from experiments.decoder_geometry_20261031.calibrate import finalize
from experiments.decoder_geometry_20261031.objective import DecoderGoal, METRICS, install_objective, validate_matrices


def matrices():
    diagonal = torch.ones(64, dtype=torch.float64); diagonal[:3] = torch.tensor([1., 4., 9.])
    vectors = torch.diag(diagonal.sqrt()).repeat(16, 1).reshape(128, 8, 64)
    return finalize(vectors)[0]


class SpyGoal(GoalMSE):
    def forward(self, info):
        self.last = super().forward(info)
        return self.last


class ToyDynamics(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.ones(())); self.calls = []
    def rollout(self, info, actions):
        self.calls.append((info["pixels"].clone(), actions.clone()))
        result = dict(info)
        result["predicted_emb"] = info["pixels"] + torch.nn.functional.pad(actions.cumsum(-2), (0, 61))*self.weight
        return result


class Contracts(unittest.TestCase):
    def test_native_tensor_identity_and_readonly(self):
        torch.manual_seed(11); info = dict(predicted_emb=torch.randn(2, 5, 4, 64), goal_emb=torch.randn(2, 3, 64))
        before = {k:v.clone() for k,v in info.items()}; rng = torch.get_rng_state().clone()
        for reduction in ["sum", "mean"]:
            native = SpyGoal(reduction=reduction)
            value = DecoderGoal(native, matrices(), "identity")(info)
            self.assertIs(value, native.last)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key in info: self.assertTrue(torch.equal(info[key], before[key]))

    def test_known_bank_quadratics_orders_and_labels(self):
        pred = torch.zeros(1, 3, 1, 64); pred[0, 0, 0, 0] = 2; pred[0, 1, 0, 1] = 1.4; pred[0, 2, 0, 2] = .5
        info = dict(predicted_emb=pred, goal_emb=torch.zeros(1, 1, 64), action_candidates=torch.zeros(1, 3, 2, 3))
        arrays = matrices(); delta = pred[:, :, 0]
        with tempfile.TemporaryDirectory() as directory:
            recorder = MetricRecorder(directory, metric_names=METRICS)
            outputs = []
            for name in METRICS:
                value = DecoderGoal(GoalMSE(), arrays, name, recorder)(info); outputs.append(value)
                gram = torch.eye(64) if name == "identity" else torch.tensor(arrays[name+"_gram"], dtype=torch.float32)
                expected = torch.einsum("bsi,ij,bsj->bs", delta, gram, delta)
                torch.testing.assert_close(value, expected)
                self.assertTrue(torch.equal(value.argsort(), expected.argsort()))
            self.assertFalse(torch.equal(outputs[0].argsort(), outputs[1].argsort()))
            audit = recorder.finish(); self.assertEqual(audit["metric_order"], list(METRICS))
            with np.load(Path(directory)/"metric-costs.npz") as f:
                self.assertEqual(f["costs"].shape, (3, 3, 3))
                for i in range(3): np.testing.assert_array_equal(f["costs"][:, 0], f["costs"][:, i])
            rows = json.loads((Path(directory)/"metric-candidate-banks.json").read_text())
            self.assertEqual(set(rows[0]["selected_indices"]), set(METRICS))
            self.assertEqual(len({r["action_candidates_sha256"] for r in rows}), 1)
            self.assertEqual(audit["manual_identity_max_abs_error"], 0)

    def test_nonfinite_invalid_matrix_and_identity_mismatch_block(self):
        broken = matrices(); broken["decoder_factor"][0, 0] = np.nan
        with self.assertRaises(AssertionError): validate_matrices(broken)
        class BadGoal(GoalMSE):
            def forward(self, info): return super().forward(info)+1
        info = dict(predicted_emb=torch.zeros(1, 2, 1, 64), goal_emb=torch.zeros(1, 1, 64))
        with self.assertRaises(AssertionError): DecoderGoal(BadGoal(), matrices())(info)

    def test_raw_actions_and_rollout_call_budget_unchanged(self):
        actions = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3)/10
        info = dict(pixels=torch.ones(1, 2, 4, 64), goal_emb=torch.zeros(1, 1, 64))
        reference = ToyDynamics(); native = ShootingCostEvaluator(reference, GoalMSE(), encode_goal=None)
        expected = native.get_cost(copy.deepcopy(info), actions)
        for name in METRICS:
            model = ToyDynamics()
            with install_objective(matrices(), name): cost = ShootingCostEvaluator(model, GoalMSE(), encode_goal=None)
            value = cost.get_cost(copy.deepcopy(info), actions)
            self.assertEqual(len(model.calls), 1)
            for x,y in zip(model.calls[0], reference.calls[0]): self.assertTrue(torch.equal(x,y))
            if name == "identity": self.assertTrue(torch.equal(value, expected))

    def test_native_cem_identity_actions_and_candidate_banks_exact(self):
        import gymnasium as gym
        info = dict(pixels=torch.ones(2, 4, 64), goal=torch.zeros(2, 1, 64)); outputs, calls = [], []
        for adapted in [False, True]:
            model = ToyDynamics(); objective = DecoderGoal(GoalMSE(), matrices()) if adapted else GoalMSE()
            cost = ShootingCostEvaluator(model, objective, encode_goal=lambda m,i:i["goal"][:, 0])
            solver = CEMSolver(cost, batch_size=1, num_samples=12, n_steps=3, topk=3, device="cpu", seed=1234)
            solver.configure(action_space=gym.spaces.Box(-1, 1, shape=(2, 3)), n_envs=2,
                config=SimpleNamespace(action_block=1, horizon=4))
            outputs.append(solver.solve(copy.deepcopy(info))); calls.append(model.calls)
        self.assertTrue(torch.equal(outputs[0]["actions"], outputs[1]["actions"]))
        self.assertEqual(outputs[0]["costs"], outputs[1]["costs"])
        self.assertEqual(len(calls[0]), len(calls[1]))
        for a,b in zip(calls[0], calls[1]):
            for x,y in zip(a,b): self.assertTrue(torch.equal(x,y))


if __name__ == "__main__":
    torch.set_num_threads(1); unittest.main()
