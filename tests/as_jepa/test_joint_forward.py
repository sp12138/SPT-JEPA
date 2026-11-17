"""Physical grouping contracts for joint RGB prediction."""
import unittest

import torch
from torch import nn

from experiments.robocasa_stable_scaling.joint_forward import JointPool, synchronize_tokens


class MeanPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(8, 4))

    def forward(self, tokens):
        return tokens.mean(1)[:, None].expand(-1, 8, -1)


class JointForwardContracts(unittest.TestCase):
    def test_pairing_keeps_physical_sample_and_time(self):
        tokens = torch.tensor([100*b+10*c+t for b in range(2)
                               for c in range(2) for t in range(3)]).reshape(12, 1, 1)
        expected = [100*b+10*c+t for b in range(2)
                    for t in range(3) for c in range(2)]
        self.assertEqual(synchronize_tokens(tokens, 2, 3).flatten().tolist(), expected)

    def test_two_cache_rows_are_halves_of_the_joint_state(self):
        pool = JointPool(MeanPool())
        tokens = torch.arange(4*3*4).float().reshape(4, 3, 4)
        actual = pool(tokens).reshape(2, 8, 4)
        expected = tokens.reshape(2, 6, 4).mean(1)[:, None].expand(-1, 8, -1)
        torch.testing.assert_close(actual, expected)

    def test_other_physical_samples_do_not_change_output_or_gradient(self):
        pool = JointPool(MeanPool())
        tokens = torch.randn(4, 3, 4, requires_grad=True)
        before = pool(tokens)
        grad = torch.autograd.grad(before[:2].sum(), tokens)[0]
        self.assertTrue(torch.equal(grad[2:], torch.zeros_like(grad[2:])))
        changed = tokens.detach().clone(); changed[2:] += 100
        self.assertTrue(torch.equal(before[:2], pool(changed)[:2]))

    def test_camera_embeddings_stay_inside_the_physical_pair(self):
        pool = JointPool(MeanPool())
        with torch.no_grad():
            pool.view_embedding[0] = 2
            pool.view_embedding[1] = 4
        result = pool(torch.zeros(2, 3, 4))
        torch.testing.assert_close(result, torch.full_like(result, 3.))


if __name__ == '__main__':
    unittest.main()
