"""Meaningful CPU contracts against the actual installed native CEM solver."""
import argparse
from types import SimpleNamespace
import numpy as np
import torch
from common import read, sha, write
from adapter import Injection


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    from stable_worldmodel.planning.solver.cem import CEMSolver
    from gymnasium.spaces import Box
    import inspect

    torch.set_num_threads(2)
    features = np.zeros((1024, 384), np.float32)
    actions = np.ones((1024, 5, 25), np.float32) * .25
    image = torch.zeros(1, 1, 3, 8, 8)
    info = dict(pixels=image, goal=image.clone())
    class StopAfterUpdate(Exception):
        pass
    class Cost(torch.nn.Module):
        def __init__(self, mode, stop=False):
            super().__init__()
            self.map = None if mode is None else Injection(mode, features, actions)
            self.banks = []
            self.stop = stop
        def get_cost(self, expanded, candidates):
            if self.map is not None:
                self.map.apply(expanded, candidates)
            self.banks.append(candidates.clone())
            if self.stop and len(self.banks) == 2:
                raise StopAfterUpdate()
            return (candidates - .25).square().sum((-1, -2))
    def solver(cost):
        s = CEMSolver(cost, batch_size=1, num_samples=300, n_steps=30, topk=30, seed=42, device='cpu')
        s.configure(action_space=Box(-1., 1., shape=(1, 5), dtype=np.float32), n_envs=1,
                    config=SimpleNamespace(horizon=5, action_block=5))
        if cost.map is not None:
            cost.map.begin()
        return s
    native = Cost(None)
    control = Cost('original')
    s0, s1 = solver(native), solver(control)
    x, y = s0.solve(info), s1.solve(info)
    assert torch.equal(x['actions'], y['actions']) and x['costs'] == y['costs']
    assert all(torch.equal(a, b) for a, b in zip(native.banks, control.banks))
    assert torch.equal(s0.torch_gen.get_state(), s1.torch_gen.get_state())
    for mode in ('random', 'retrieval'):
        cost = Cost(mode, stop=True)
        s = solver(cost)
        try:
            s.solve(info)
        except StopAfterUpdate:
            pass
        else:
            raise AssertionError('Expected the second-iteration observation')
        first, second = cost.banks
        assert torch.equal(first[:, 0], native.banks[0][:, 0])
        assert torch.equal(first[:, 65:], native.banks[0][:, 65:])
        assert torch.equal(first[0, 1:65], torch.from_numpy(actions[:64]))
        # The native solver must update from the injected candidates, not from
        # an untransformed shadow bank. Its next mean is exactly the .25 elite.
        assert torch.equal(second[:, 0], torch.full((1, 5, 25), .25))
        assert len(cost.map.injections) == 1
        if mode == 'retrieval':
            assert cost.map.injections[0]['ids'] == list(range(64))
    # Retrieval must depend on the query. Two query clusters give disjoint IDs.
    f = features.copy()
    f[512:] = 1
    m = Injection('retrieval', f, actions)
    expanded0 = {k: v[:, None].expand(1, 300, *v.shape[1:]) for k, v in info.items()}
    expanded1 = {k: torch.ones_like(v) for k, v in expanded0.items()}
    m.begin(); m.apply(expanded0, torch.zeros(1, 300, 5, 25)); ids0 = m.injections[-1]['ids']
    m.begin(); m.apply(expanded1, torch.zeros(1, 300, 5, 25)); ids1 = m.injections[-1]['ids']
    assert ids0 == list(range(64)) and ids1 == list(range(512, 576))
    files = {str(f): sha(f) for f in [__file__, inspect.getfile(Injection)]}
    write(a.output, dict(passed=True, native_original_bitwise=True, all30_banks_and_returned_actions_exact=True,
          original_CEM_RNG_exact=True, injected_candidates_drive_actual_native_elite_update=True,
          mean_and_remaining235_candidates_exact=True, stable_ties=True, query_dependence=True,
          models_loaded=0, simulator_calls=0, native_cem=inspect.getfile(CEMSolver),
          native_cem_sha256=sha(inspect.getfile(CEMSolver)), helper_files=files))
    print('Native CPU candidate/elite/return/RNG contracts pass.', flush=True)


if __name__ == '__main__':
    main()
