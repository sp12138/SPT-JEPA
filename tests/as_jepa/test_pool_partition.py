"""Same-parameter spatial/shuffled pooling diagnostics and unchanged full path."""
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import torch

from as_jepa_lewm.model import PartitionedQueryPool
from as_jepa_lewm.robust import RobustLeWM
from vforge.ops import OpsConfig, initialize_ops


class PoolPartitionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        initialize_ops(OpsConfig(backend="sdpa_math", allow_fallback=False),
                       device=torch.device("cpu"), parallel_state=None)

    def model(self, cls=RobustLeWM, **extra):
        torch.manual_seed(11)
        model = cls(mode="L2", beta=.25, window=3, encoder_depth=1,
                    renderer_width=48, renderer_depth=2, renderer_heads=3,
                    rho=.8, forecast_fraction=.5, **extra)
        torch.nn.init.normal_(model.renderer.final_layer.linear.weight, std=.01)
        return model.eval()

    def batch(self):
        torch.manual_seed(18)
        return dict(pixels=torch.randint(0, 256, (2, 3, 3, 224, 224), dtype=torch.uint8),
                    render=torch.randint(0, 256, (2, 3, 3, 64, 64), dtype=torch.uint8),
                    anchor=torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8),
                    anchor224=torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8),
                    action=torch.randn(2, 3, 10))

    def reference_class(self):
        root = Path(__file__).resolve().parents[2]
        def read(name):
            override = os.environ.get(f"AS_JEPA_{name.upper()}_REFERENCE")
            return Path(override).read_text() if override else subprocess.check_output(
                ["git", "show", f"421523f:as_jepa_lewm/{name}.py"], cwd=root, text=True)
        model = types.ModuleType("pool_reference_model")
        exec(compile(read("model"), "<model_reference_421523f>", "exec"), model.__dict__)
        robust = types.ModuleType("pool_reference_robust")
        with patch.dict(sys.modules, {"as_jepa_lewm.model": model}):
            exec(compile(read("robust"), "<robust_reference_421523f>", "exec"), robust.__dict__)
        return robust.RobustLeWM

    def test_full_exact_reference_loss_all_gradients_and_rng(self):
        old, new = self.model(self.reference_class()), self.model(pool_partition="full")
        old.train(); new.train()
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
        batch = self.batch()
        torch.manual_seed(29)
        first, first_metrics = old.loss(batch, step=7)
        first.backward()
        old_rng = torch.get_rng_state()
        torch.manual_seed(29)
        second, second_metrics = new.loss(batch, step=7)
        second.backward()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(set(first_metrics), set(second_metrics))
        for name in first_metrics:
            torch.testing.assert_close(first_metrics[name], second_metrics[name], rtol=0, atol=0)
        torch.testing.assert_close(old_rng, torch.get_rng_state(), rtol=0, atol=0)
        checked = 0
        for (nx, x), (ny, y) in zip(old.named_parameters(), new.named_parameters()):
            self.assertEqual(nx, ny)
            self.assertEqual(x.grad is None, y.grad is None)
            if x.grad is not None:
                checked += 1
                torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
        self.assertGreater(checked, 100)

    def test_same_parameter_keys_initial_tensors_and_rng(self):
        baseline = self.model(pool_partition="full")
        baseline_rng = torch.get_rng_state()
        for mode in ["quadrants", "shuffled"]:
            candidate = self.model(pool_partition=mode)
            torch.testing.assert_close(baseline_rng, torch.get_rng_state(), rtol=0, atol=0)
            self.assertEqual(set(baseline.state_dict()), set(candidate.state_dict()))
            self.assertEqual(sum(p.numel() for p in baseline.parameters()),
                             sum(p.numel() for p in candidate.parameters()))
            for key, value in baseline.state_dict().items():
                torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)

    def test_balanced_coverage_shared_cls_locality_and_isolated_rng(self):
        masks = {}
        for mode in ["quadrants", "shuffled"]:
            pool = PartitionedQueryPool(12, 3, 4, 16, mode)
            before = torch.get_rng_state()
            mask = pool.attention_mask(257, torch.device("cpu"))
            torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
            self.assertEqual(mask.shape, (4, 257))
            self.assertFalse(bool(mask[:, 0].any()))
            torch.testing.assert_close((~mask[:, 1:]).sum(1), torch.full((4,), 64))
            torch.testing.assert_close((~mask[:, 1:]).sum(0), torch.ones(256, dtype=torch.long))
            torch.manual_seed(123)
            torch.testing.assert_close(mask, pool.attention_mask(257, torch.device("cpu")))
            masks[mode] = mask
        quad = (~masks["quadrants"][:, 1:]).reshape(4, 16, 16)
        for slot in range(4):
            expected = torch.zeros((16, 16), dtype=torch.bool)
            row, column = slot // 2, slot % 2
            expected[row*8:(row+1)*8, column*8:(column+1)*8] = True
            torch.testing.assert_close(quad[slot], expected)
        self.assertFalse(torch.equal(masks["quadrants"], masks["shuffled"]))

    def test_mask_blocks_only_direct_pool_reads(self):
        torch.manual_seed(41)
        pool = PartitionedQueryPool(12, 3, 4, 16, "quadrants").eval()
        pool.initialize()
        tokens = torch.randn(2, 17, 12)
        mask = pool.attention_mask(17, tokens.device)
        changed = tokens.clone()
        changed[:, mask[0]] += torch.randn_like(changed[:, mask[0]]) * 5
        original, perturbed = pool(tokens), pool(changed)
        torch.testing.assert_close(original[:, 0], perturbed[:, 0], rtol=0, atol=0)
        self.assertGreater(float((original[:, 1:]-perturbed[:, 1:]).detach().abs().max()), 1e-6)

    def test_partitioned_loss_is_sample_level_and_gradients_are_functional(self):
        batch = self.batch()
        fixed_noise = lambda value, *a, **k: torch.full_like(value, .3)
        fixed_time = lambda count, device, shift: torch.full((count, 1), .5, device=device)
        fixed_choose = lambda *shape, **kw: torch.full(shape, .3, **kw)
        for mode in ["quadrants", "shuffled"]:
            with self.subTest(mode=mode):
                model = self.model(pool_partition=mode)
                with patch("torch.randn_like", fixed_noise), patch("torch.rand", fixed_choose), \
                        patch("vforge.models.as_jepa.flow.draw_timesteps", fixed_time):
                    whole, _ = model.loss(batch, step=7)
                    separate = [model.loss({k: v[i:i+1] for k, v in batch.items()}, step=7)[0]
                                for i in range(2)]
                torch.testing.assert_close(whole, torch.stack(separate).mean(), rtol=5e-5, atol=5e-6)
                whole.backward()
                for module in [model.encoder, model.pool, model.predictor, model.renderer]:
                    gradients = [p.grad for p in module.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
                    self.assertTrue(any(bool(g.abs().sum()>0) for g in gradients))


if __name__ == "__main__":
    unittest.main()
