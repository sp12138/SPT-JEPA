"""Compatibility of explicit variants with historical training/export dispatch."""
import unittest
import torch
from as_jepa_lewm.factory import build_model, model_target, gradient_expectations
from as_jepa_lewm.robust import RobustLeWM
from vforge.ops import OpsConfig, initialize_ops


class ModelFactoryContracts(unittest.TestCase):
    def test_historical_precedence_and_explicit_override(self):
        for cfg, suffix in [({}, "model.ASJepaLeWM"),
                            ({"research_variant": True}, "research.ResearchLeWM"),
                            ({"research_variant": True, "robust_variant": True}, "robust.RobustLeWM"),
                            ({"frontier_variant": True, "robust_variant": True}, "frontier.FrontierLeWM"),
                            ({"model_class": "carrier", "robust_variant": True}, "carrier.CarrierLeWM"),
                            ({"model_class": "pathquery"}, "pathquery.PathQueryLeWM")]:
            self.assertEqual(model_target(cfg), "as_jepa_lewm." + suffix)
        with self.assertRaises(ValueError):
            model_target({"model_class": "unregistered.module"})

    def test_original_robust_tensors_rng_and_configuration_unchanged(self):
        torch.set_num_threads(2)
        initialize_ops(OpsConfig(backend="sdpa_math", allow_fallback=False),
                       device=torch.device("cpu"), parallel_state=None)
        kwargs = dict(mode="L2", beta=.25, window=3, encoder_depth=1,
                      renderer_width=48, renderer_depth=2, renderer_heads=3,
                      rho=.8, forecast_fraction=.5)
        torch.manual_seed(19)
        old = RobustLeWM(**kwargs)
        old_rng = torch.get_rng_state().clone()
        torch.manual_seed(19)
        cfg = {"robust_variant": True, "model": kwargs.copy()}
        new = build_model(cfg)
        self.assertIs(type(new), type(old))
        self.assertEqual(cfg["model"], kwargs)
        torch.testing.assert_close(torch.get_rng_state(), old_rng, rtol=0, atol=0)
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        for key, tensor in old.state_dict().items():
            torch.testing.assert_close(tensor, new.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(gradient_expectations(new),
                         {"encoder": "active", "predictor": "active", "renderer": "active"})

    def test_frozen_contract_is_explicit_and_validated(self):
        class Frozen:
            def audit_expected_gradients(self):
                return {"encoder": "frozen", "predictor": "active", "chart": "optional"}
        self.assertEqual(gradient_expectations(Frozen())["encoder"], "frozen")
        self.assertEqual(gradient_expectations(Frozen(),
            {"audit_expectations": {"gradients": {"chart": "active"}}})["chart"], "active")
        class Invalid:
            def audit_expected_gradients(self):
                return {"encoder": "ignore_failure"}
        with self.assertRaises(ValueError):
            gradient_expectations(Invalid())


if __name__ == "__main__":
    unittest.main()
