"""CPU contracts for matched sample-local path-query controls."""
import itertools
import math
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from as_jepa_lewm.pathquery import PathQueryLeWM
from as_jepa_lewm.robust import RobustLeWM
from vforge.ops import OpsConfig, initialize_ops


MODES = ("symbol", "completion", "transform")
MAPS = ((0, 1, 2), (1, 1, 2), (0, 2, 2), (2, 2, 2), (1, 2, 2))


def oracle(word):
    result = []
    for start in range(3):
        progress = start
        for symbol in word:
            if progress == 0 and symbol == 0:
                progress = 1
            elif progress == 1 and symbol == 1:
                progress = 2
        result.append(progress)
    return MAPS.index(tuple(result))


class PathQueryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        initialize_ops(OpsConfig(backend="sdpa_math", allow_fallback=False),
                       device=torch.device("cpu"), parallel_state=None)

    def model(self, mode="none", window=8, cls=PathQueryLeWM, **extra):
        torch.manual_seed(11)
        kwargs = dict(path_query_mode=mode) if cls is PathQueryLeWM else {}
        model = cls(mode="L2", beta=.25, window=window, encoder_depth=1,
                    renderer_width=48, renderer_depth=2, renderer_heads=3,
                    forecast_fraction=1., **kwargs, **extra)
        torch.nn.init.normal_(model.renderer.final_layer.linear.weight, std=.01)
        return model.eval()

    @staticmethod
    def batch(window=3):
        torch.manual_seed(18)
        return dict(pixels=torch.randint(0, 256, (2, window, 3, 224, 224), dtype=torch.uint8),
                    render=torch.randint(0, 256, (2, window, 3, 64, 64), dtype=torch.uint8),
                    anchor=torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8),
                    anchor224=torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8),
                    action=torch.randn(2, window, 10))

    def test_none_exact_baseline_loss_metrics_gradients_rng(self):
        old, new = self.model(cls=RobustLeWM, window=3), self.model(window=3)
        old.train(); new.train()
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
        self.assertFalse(hasattr(new, "path_query_head"))
        batch = self.batch()
        torch.manual_seed(29)
        first, fm = old.loss(batch, step=7)
        first.backward(); rng = torch.get_rng_state()
        torch.manual_seed(29)
        second, sm = new.loss(batch, step=7)
        second.backward()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(set(fm), set(sm))
        for name, value in fm.items():
            torch.testing.assert_close(value, sm[name], rtol=0, atol=0)
        for (name, p), (other, q) in zip(old.named_parameters(), new.named_parameters()):
            self.assertEqual(name, other)
            self.assertEqual(p.grad is None, q.grad is None)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        new.load_state_dict(old.state_dict(), strict=True)

    def test_equal_old_tensors_head_initialization_budget_rng_and_reload(self):
        baseline = self.model(); rng = torch.get_rng_state()
        head = None
        for mode in MODES:
            model = self.model(mode)
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
            self.assertEqual(sum(p.numel() for p in model.path_query_head.parameters()), 70533)
            for name, value in baseline.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
            if head is None:
                head = {k: v.clone() for k, v in model.path_query_head.state_dict().items()}
            for name, value in head.items():
                torch.testing.assert_close(value, model.path_query_head.state_dict()[name], rtol=0, atol=0)
            replica = self.model(mode)
            replica.load_state_dict(model.state_dict(), strict=True)
            self.assertEqual(model.audit_expected_gradients()["path_query_head"], "active")

    def test_exhaustive_five_class_algebra_and_composition(self):
        checked = 0
        for length in range(8):
            words = list(itertools.product(range(3), repeat=length))
            labels = PathQueryLeWM.path_transform_classes(torch.tensor(words, dtype=torch.long).reshape(len(words), length))
            for word, label in zip(words, labels.tolist()):
                self.assertEqual(label, oracle(word))
                cut = length // 2
                left, right = MAPS[oracle(word[:cut])], MAPS[oracle(word[cut:])]
                self.assertEqual(MAPS[label], tuple(right[left[i]] for i in range(3)))
                self.assertEqual(label == 3, MAPS[label][0] == 2)
                checked += 1
        self.assertEqual(checked, 3280)
        # Same negative completion at progress zero, different response to an A prefix.
        self.assertEqual(MAPS[0][0], MAPS[2][0])
        self.assertNotEqual(MAPS[0][1], MAPS[2][1])

    def test_targets_independent_oracle_halves_ties_rng_and_sample_locality(self):
        model = self.model("transform")
        torch.manual_seed(37)
        render = torch.randn(2, 8, 3, 64, 64, requires_grad=True)
        rng = torch.get_rng_state()
        targets = model.path_query_targets(render, step=7)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        # Explicit 8x8 block means independent of adaptive pooling implementation.
        features = render.detach().reshape(2, 8, 3, 8, 8, 8, 8).mean((4, 6)).reshape(2, 8, 192)
        torch.testing.assert_close(features, targets["features"], rtol=1e-5, atol=1e-7)
        for b in range(2):
            one = model.path_query_targets(render[b:b+1], step=7)
            for key in ("features", "prototypes", "symbols", "transforms", "completion"):
                torch.testing.assert_close(targets[key][b:b+1], one[key], rtol=0, atol=0)
                self.assertFalse(targets[key].requires_grad)
            for q, indices in enumerate(targets["query_indices"].tolist()):
                self.assertEqual(len(set(indices)), 3)
                word = [min(range(3), key=lambda k: float((features[b, t]-features[b, indices[k]]).square().sum())) for t in range(8)]
                self.assertEqual(targets["symbols"][b, q].tolist(), word)
                self.assertEqual(targets["transforms"][b, q].tolist(), [oracle(word), oracle(word[:4]), oracle(word[4:])])
        ties = model.path_query_targets(torch.zeros_like(render), step=7)
        self.assertEqual(int(ties["symbols"].sum()), 0)
        self.assertTrue(bool((ties["transforms"] == 1).all()))

    def test_causal_rollout_fixed_query_interventions_and_call_budget(self):
        model = self.model("transform")
        torch.manual_seed(41)
        flat = torch.randn(2, 8, 64, requires_grad=True)
        act = torch.randn(2, 8, 64, requires_grad=True)
        render = torch.randn(2, 8, 3, 64, 64, requires_grad=True)
        targets = model.path_query_targets(render, step=9)
        def predict(history, actions):
            return history + .05 * actions
        with patch.object(model, "predict", side_effect=predict) as mocked:
            rng = torch.get_rng_state()
            out = model.path_query_outputs(flat, act, targets=targets, step=9)
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
            self.assertEqual(mocked.call_count, 7)
            for t, call in enumerate(mocked.call_args_list):
                self.assertEqual(call.args[0].shape[1], t+1)
                self.assertEqual(call.args[1].shape[1], t+1)
            expected = torch.cat([flat[:, :1], flat[:, :1]+.05*act[:, :-1].cumsum(1)], 1)
            torch.testing.assert_close(out["forecast"], expected)
            altered = flat.detach().clone(); altered[:, 1:] += 100
            future = model.path_query_outputs(altered, act, targets=targets, step=9)
            torch.testing.assert_close(out["logits"], future["logits"], rtol=0, atol=0)
            new_queries = model.path_query_outputs(flat, act, render=render.flip(1), step=9)
            torch.testing.assert_close(out["forecast"], new_queries["forecast"], rtol=0, atol=0)
            self.assertFalse(torch.equal(out["logits"], new_queries["logits"]))
            out["logits"].square().mean().backward()
            self.assertGreater(float(flat.grad[:, 0].abs().sum()), 0)
            self.assertEqual(float(flat.grad[:, 1:].abs().sum()), 0)
            self.assertGreater(float(act.grad[:, :-1].abs().sum()), 0)
            self.assertEqual(float(act.grad[:, -1:].abs().sum()), 0)
            self.assertIsNone(render.grad)

    def test_same_forward_and_independent_ce_formula_all_arms(self):
        reference_logits = None
        for mode, k in zip(MODES, (3, 2, 5)):
            model = self.model(mode)
            torch.manual_seed(43)
            flat, act = torch.randn(2, 8, 64), torch.randn(2, 8, 64)
            render = torch.randn(2, 8, 3, 64, 64)
            out = model.path_query_outputs(flat, act, render, step=11)
            self.assertEqual(out["logits"].shape, (2, 8, 11, 5))
            if reference_logits is None:
                reference_logits = out["logits"].detach()
            torch.testing.assert_close(reference_logits, out["logits"], rtol=0, atol=0)
            loss, metrics = model.path_query_loss(flat, act, render, step=11)
            chosen = out["logits"][:, :, :8, :3] if mode == "symbol" else out["logits"][:, :, 8:, :k]
            labels = out["targets"][dict(symbol="symbols", completion="completion", transform="transforms")[mode]]
            logp = chosen - chosen.logsumexp(-1, keepdim=True)
            expected = -logp.gather(-1, labels[..., None]).mean() / math.log(k)
            torch.testing.assert_close(loss, expected)
            torch.testing.assert_close(loss, metrics["path_query_"+mode+"_relative_ce"])
            loss.backward()
            final = model.path_query_head[-1]
            self.assertGreater(float(final.weight.grad[:k].abs().sum()), 0)
            self.assertEqual(float(final.weight.grad[k:].abs().sum()), 0)
            self.assertEqual(float(final.bias.grad[k:].abs().sum()), 0)
            model.zero_grad(set_to_none=True)
            split = sum(model.path_query_loss(flat[i:i+1], act[i:i+1], render[i:i+1], step=11)[0] for i in range(2)) / 2
            torch.testing.assert_close(loss, split, rtol=1e-6, atol=1e-6)

    def test_native_loss_one_encoding_gradients_cache_rng_all_arms(self):
        baseline = self.model(window=3); baseline.train()
        batch = self.batch()
        torch.manual_seed(53)
        _, old_metrics = baseline.loss(batch, step=7)
        old_rng = torch.get_rng_state()
        for mode in MODES:
            model = self.model(mode, window=3); model.train()
            torch.manual_seed(53)
            with patch.object(model, "encode", wraps=model.encode) as encode:
                loss, metrics = model.loss(batch, step=7)
                self.assertEqual(encode.call_count, 1)
            self.assertIsNone(model._path_query_capture)
            torch.testing.assert_close(old_rng, torch.get_rng_state(), rtol=0, atol=0)
            for key in set(old_metrics)-{"loss"}:
                torch.testing.assert_close(old_metrics[key], metrics[key], rtol=0, atol=0)
            torch.testing.assert_close(loss, old_metrics["loss"]+metrics["path_query_aux"], rtol=0, atol=0)
            loss.backward()
            for prefix in ("encoder", "pool", "predictor", "renderer", "path_query_head"):
                parameters = list(getattr(model, prefix).parameters())
                self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in parameters), prefix)
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
            # Native output projection AND AdaLN-zero conditioning start zero.
            # Two ordinary updates open the two barriers without changing init.
            optimizer = torch.optim.SGD(model.parameters(), lr=.001)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            loss, _ = model.loss(batch, step=8); loss.backward()
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            loss, _ = model.loss(batch, step=9); loss.backward()
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.action_encoder.parameters()))
        model = self.model("transform", window=3)
        with patch.object(RobustLeWM, "loss", side_effect=RuntimeError("test")):
            with self.assertRaises(RuntimeError):
                model.loss(batch)
        self.assertIsNone(model._path_query_capture)

    def test_invalid_modes_and_unmodified_public_encode(self):
        for kwargs in (dict(path_query_weight=float("nan")), dict(path_query_weight=-1),
                       dict(response_query_mode="direct")):
            with self.assertRaises(ValueError):
                self.model("transform", **kwargs)
        model = self.model("transform", window=3)
        pixels, _, _ = model.prepare(self.batch())
        info = model.encode(dict(pixels=pixels))
        self.assertEqual(info["emb"].shape, (2, 3, 64))
        self.assertNotIn("act_emb", info)


if __name__ == "__main__":
    unittest.main()
