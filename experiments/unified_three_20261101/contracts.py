"""CPU contracts for one shared local-output model, not a performance test."""
import argparse
import json
from pathlib import Path

import torch

from as_jepa_lewm.local_readout import LocalReadoutLeWM, LocalReadoutPredictor
from as_jepa_lewm.spatial_carrier import SpatialCarrierLeWM
from vforge.ops import OpsConfig, initialize_ops


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); a = p.parse_args()
    torch.set_num_threads(2)
    initialize_ops(OpsConfig(backend='sdpa_math', allow_fallback=False), device=torch.device('cpu'), parallel_state=None)
    kwargs = dict(spatial_mean=[0.] * 1024, spatial_scale=[1.] * 1024, window=8,
                  encoder_depth=1, renderer_width=48, renderer_depth=1, renderer_heads=3,
                  rho=.8, forecast_fraction=.5, beta=.25)
    torch.manual_seed(1); flat = SpatialCarrierLeWM(**kwargs).eval()
    expected_rng = torch.get_rng_state()
    torch.manual_seed(1); local = LocalReadoutLeWM(local_readout_seed=1, **kwargs).eval()
    assert torch.equal(expected_rng, torch.get_rng_state())
    fs, ls = flat.state_dict(), local.state_dict()
    common = [k for k in fs if not k.startswith('predictor.transformer.output_proj.')]
    assert all(torch.equal(fs[k], ls[k]) for k in common)
    old_count = sum(x.numel() for x in flat.predictor.parameters())
    new_count = sum(x.numel() for x in local.predictor.parameters())
    assert old_count - new_count == 16880
    state = torch.randn(2, 4, 1024); action = torch.randn_like(state)
    with torch.no_grad():
        assert torch.equal(flat.predict(state, action), state)
        assert torch.equal(local.predict(state, action), state)
    grid = torch.arange(1024.).reshape(1, 1, 1024)
    neighbors = LocalReadoutPredictor.neighborhoods(grid)
    for row, col in [(0, 0), (3, 4), (7, 7)]:
        expected = torch.zeros(16, 3, 3)
        raw = grid.reshape(8, 8, 16)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                y, x = row + dy, col + dx
                if 0 <= y < 8 and 0 <= x < 8:
                    expected[:, dy + 1, dx + 1] = raw[y, x]
        torch.testing.assert_close(neighbors[0, 0, row * 8 + col], expected.flatten(), rtol=0, atol=0)
    # A nonzero readout exposes any causal/batch leakage hidden by zero init.
    torch.nn.init.normal_(local.predictor.readout[-1].weight, std=.01)
    changed = state.clone(); changed[:, 2:] += 3
    changed_action = action.clone(); changed_action[:, 2:] -= 2
    with torch.no_grad():
        original = local.predict(state, action)
        future = local.predict(changed, changed_action)
        torch.testing.assert_close(original[:, :2], future[:, :2], rtol=0, atol=0)
        singleton = local.predict(state[:1], action[:1])
        torch.testing.assert_close(original[:1], singleton, rtol=2e-6, atol=2e-6)
    optimizer = torch.optim.AdamW(local.predictor.parameters(), lr=1e-4)
    target = torch.randn_like(state)
    for _ in range(2):
        optimizer.zero_grad(); loss = (local.predict(state, action) - target).square().mean()
        loss.backward(); optimizer.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in local.predictor.parameters())
    assert all(sum(float(p.grad.double().square().sum()) for p in m.parameters()) > 0
               for m in local.predictor.transformer.layers)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(passed=True, CUDA_initialized=torch.cuda.is_initialized(),
                  common_initial_tensors=len(common), outer_RNG_exact=True,
                  initial_carry_exact=True, causal_prefix_exact=True, batch_independent=True,
                  neighborhoods_exact=True, native_body_preserved=True,
                  flat_predictor_parameters=old_count, local_predictor_parameters=new_count,
                  new_readout_parameters=sum(p.numel() for p in local.predictor.readout.parameters()),
                  parameter_difference=new_count-old_count,
                  scope='CPU contracts only; no task-learning, generalization or novelty claim.')
    assert not result['CUDA_initialized']
    with out.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
