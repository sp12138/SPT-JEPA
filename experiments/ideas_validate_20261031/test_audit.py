"""CPU contract tests for alignment, observation access and frozen gradients."""
import unittest

import torch

from experiments.ideas_validate_20261031.audit import (
    evaluate_latents, expected_gradients, intervention_actions, path_query_diagnostic,
)
from experiments.ideas_validate_20261031.preflight import (
    checkpoint_report, compare_tensors, gradient_report,
)


class AdditiveModel(torch.nn.Module):
    def __init__(self, carry_only=False, leaky=False):
        super().__init__()
        self.action_encoder = torch.nn.Identity()
        self.carry_only, self.leaky = carry_only, leaky

    def predict(self, states, actions):
        if self.leaky:
            return states + actions + states.mean(1, keepdim=True)
        return states if self.carry_only else states + actions


def exact_trajectories():
    initial = torch.tensor([[2., -1.], [-3., 4.], [1., 2.]])
    actions = torch.tensor([[[1., .2], [2., -.1], [-1., .8], [99., 99.]],
                            [[-.5, 1.], [.2, 2.], [.8, -1.], [98., 98.]],
                            [[.3, -.2], [.4, -.3], [.5, .8], [97., 97.]]])
    observed = torch.cat([initial[:, None], initial[:, None] + actions[:, :-1].cumsum(1)], 1)
    return observed, actions


class AuditContracts(unittest.TestCase):
    def test_causal_alignment_and_normalization(self):
        observed, actions = exact_trajectories()
        result, states = evaluate_latents(AdditiveModel(), observed, actions, batch_size=2)
        self.assertEqual(states["clean_rollout"].shape, observed[:, 1:].shape)
        self.assertLess(result["conditions"]["clean_rollout"]["mse"], 1e-12)
        self.assertLess(result["conditions"]["teacher_forced_prefix"]["mse"], 1e-12)
        self.assertAlmostEqual(result["conditions"]["initial_carry"]["normalized_by_initial_variation"], 1)
        self.assertAlmostEqual(result["conditions"]["shifted_observed_carry"]["normalized_by_local_variation"], 1)
        self.assertGreater(result["conditions"]["other_initial"]["mse"], 1)
        self.assertGreater(result["conditions"]["wrong_action_order"]["mse"], 0)
        self.assertEqual(result["other_initial_case_indices"], [2, 0, 1])
        # Global intervention pairing is unchanged by chunk size, including a
        # final chunk of size one; there is never an accidental self-swap.
        other, _ = evaluate_latents(AdditiveModel(), observed, actions, batch_size=3)
        self.assertEqual(result["conditions"], other["conditions"])

    def test_padding_never_becomes_a_real_action(self):
        _, actions = exact_trajectories()
        for value in intervention_actions(actions).values():
            self.assertTrue(torch.equal(value[:, -1], actions[:, -1]))
            self.assertLess(float(value[:, :-1].abs().max()), 3)

    def test_shifted_observed_carry_is_not_free_rollout(self):
        observed, actions = exact_trajectories()
        result, states = evaluate_latents(AdditiveModel(carry_only=True), observed, actions)
        self.assertTrue(torch.equal(states["teacher_forced_prefix"], states["shifted_observed_carry"]))
        self.assertTrue(torch.equal(states["clean_rollout"], states["initial_carry"]))
        self.assertNotEqual(result["conditions"]["teacher_forced_prefix"]["mse"],
                            result["conditions"]["clean_rollout"]["mse"])

    def test_vectorized_teacher_future_leak_is_detected(self):
        observed, actions = exact_trajectories()
        result, states = evaluate_latents(AdditiveModel(leaky=True), observed, actions)
        self.assertGreater(result["vectorized_teacher_vs_prefix_mse"], .01)
        expected_first = 2 * observed[:, 0] + actions[:, 0]
        self.assertTrue(torch.equal(states["teacher_forced_prefix"][:, 0], expected_first))

    def test_exact_collapsed_variation_is_not_floored(self):
        observed = torch.zeros(3, 4, 2)
        actions = torch.zeros_like(observed)
        result, _ = evaluate_latents(AdditiveModel(carry_only=True), observed, actions)
        self.assertIsNone(result["conditions"]["clean_rollout"]["normalized_by_initial_variation"])
        self.assertEqual(result["actual_temporal_variation"]["exact_zero_initial_windows"], 3)

    def test_frozen_active_optional_are_different_contracts(self):
        model = torch.nn.Module()
        model.frozen = torch.nn.Linear(2, 2)
        model.active = torch.nn.Linear(2, 2)
        model.optional = torch.nn.Linear(2, 2)
        model.frozen.requires_grad_(False)
        model.active(torch.ones(1, 2)).sum().backward()
        spec = {"frozen": "frozen", "active": "active", "optional": "optional"}
        rows = gradient_report(model, spec)
        self.assertTrue(all(row["passed"] for row in rows.values()))
        model.active.zero_grad(set_to_none=True)
        self.assertFalse(gradient_report(model, spec)["active"]["passed"])
        model.audit_expected_gradients = lambda: {"frozen": "frozen", "active": "trainable_active"}
        self.assertEqual(expected_gradients(model, {"audit_expectations": {"gradients": {"active": "optional"}}}),
                          {"frozen": "frozen", "active": "optional"})

    def test_checkpoint_and_initial_checks_find_real_corruption(self):
        portable = {"frozen.weight": torch.ones(2), "active.weight": torch.zeros(2)}
        native = {"model." + k: v.clone() for k, v in portable.items()}
        checkpoint = dict(global_step=2, state_dict=native, optimizer_states=[{"x": torch.ones(1)}],
                          lr_schedulers=[{"last_epoch": 2}], campaign_samples=8,
                          campaign_rng={k: 1 for k in ["python", "numpy", "torch", "cuda"]})
        self.assertTrue(all(checkpoint_report(checkpoint, portable, {"batch_size": 4}, 2)["checks"].values()))
        checkpoint["state_dict"]["model.active.weight"][0] = 1
        self.assertFalse(checkpoint_report(checkpoint, portable, {"batch_size": 4}, 2)["checks"]["portable_equal"])
        changed = {k: v.clone() for k, v in portable.items()}
        changed["frozen.weight"][0] = 2
        self.assertFalse(compare_tensors(portable, changed, ["frozen"])["equal"])

    def test_path_query_intervention_keeps_targets_but_changes_initial(self):
        observed, actions = exact_trajectories()
        _, states = evaluate_latents(AdditiveModel(), observed, actions)

        class QueryHead:
            path_query_mode = "symbol"

            def __init__(self):
                self.inputs = []

            def path_query_targets(self, render, step=0):
                b, w = render.shape[:2]
                return dict(features=torch.zeros(b, w, 192),
                    prototypes=torch.arange(b)[:, None, None, None].expand(b, 2, 3, 192).float(),
                    symbols=torch.zeros(b, 2, w, dtype=torch.long),
                    transforms=torch.zeros(b, 2, 3, dtype=torch.long),
                    completion=torch.zeros(b, 2, 3, dtype=torch.long),
                    masks=torch.zeros(w+3, w), query_indices=torch.tensor([[0, 1, 2], [1, 2, 3]]))

            def path_query_logits_from_states(self, value, targets):
                self.inputs.append((value.clone(), targets["prototypes"].clone()))
                return torch.zeros(value.shape[0], 2, value.shape[1]+3, 5)

        model = QueryHead()
        result = path_query_diagnostic(model, observed, {"other_initial": states["other_initial"]},
                                       torch.zeros(3, 4, 3, 2, 2), batch_size=3)
        self.assertTrue(torch.equal(model.inputs[0][0][:, 0], observed.roll(1, 0)[:, 0]))
        self.assertTrue(all(torch.equal(prototypes, model.inputs[0][1]) for _, prototypes in model.inputs))
        self.assertAlmostEqual(result["conditions"]["other_initial"]["symbol_future_only"]["relative_to_uniform_ce"], 1., places=6)


if __name__ == "__main__":
    unittest.main()
