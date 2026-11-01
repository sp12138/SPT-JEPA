"""Structural CPU contracts for full-capacity spatial communication."""
import argparse
import copy
import json
from pathlib import Path

import torch

from as_jepa_lewm.spatial_carrier import SpatialCarrierLeWM
from as_jepa_lewm.spatial_native import SpatialNativeLeWM, SpatialNativePredictor, SpatialBlock
from as_jepa_lewm.patch_carrier import frame_mask
from vforge.ops import OpsConfig, initialize_ops


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); a = p.parse_args()
    torch.set_num_threads(2)
    initialize_ops(OpsConfig(backend='sdpa_math', allow_fallback=False), device=torch.device('cpu'), parallel_state=None)
    kwargs = dict(spatial_mean=[0.] * 1024, spatial_scale=[1.] * 1024, window=8,
                  encoder_depth=1, renderer_width=48, renderer_depth=1, renderer_heads=3,
                  rho=.8, forecast_fraction=.5, beta=.25)
    torch.manual_seed(1); flat = SpatialCarrierLeWM(**kwargs).eval(); expected = torch.get_rng_state()
    torch.manual_seed(1); spatial = SpatialNativeLeWM(spatial_predictor_seed=1, **kwargs).eval()
    assert torch.equal(expected, torch.get_rng_state())
    left, right = flat.state_dict(), spatial.state_dict()
    assert left.keys() == right.keys()
    replaced = [k for k in left if k.startswith(('predictor.transformer.input_proj.', 'predictor.transformer.output_proj.'))]
    shared = [k for k in left if k not in replaced]
    assert len(replaced) == 4 and all(torch.equal(left[k], right[k]) for k in shared)
    old, new = [sum(p.numel() for p in m.predictor.parameters()) for m in [flat, spatial]]
    assert old == 11389184 and new == 11001104
    state, action = torch.randn(2, 3, 1024), torch.randn(2, 3, 1024)
    with torch.no_grad():assert torch.equal(spatial.predict(state, action), state)
    mask = frame_mask(3)
    assert mask.shape == (192, 192)
    for i in range(3):
        for j in range(3):assert bool(mask[i*64:(i+1)*64, j*64:(j+1)*64].all()) == (j <= i)
    # Wake the copied native gates/output only for these structural probes.
    with torch.no_grad():
        for block in spatial.predictor.transformer.layers:block.adaLN_modulation[-1].bias.fill_(.05)
        torch.nn.init.normal_(spatial.predictor.transformer.output_proj.weight, std=.01)
    changed, changed_action = state.clone(), action.clone()
    changed[:, 2:] += 3; changed_action[:, 2:] -= 2
    with torch.no_grad():
        answer = spatial.predict(state, action)
        future = spatial.predict(changed, changed_action)
        torch.testing.assert_close(answer[:, :2], future[:, :2], rtol=0, atol=0)
        torch.testing.assert_close(answer[:1], spatial.predict(state[:1], action[:1]), rtol=2e-6, atol=2e-6)
    probe = state[:1, :1].clone().requires_grad_(True)
    value = spatial.predictor(probe, action[:1, :1])[0, 0, :16].sum()
    derivative = torch.autograd.grad(value, probe)[0].reshape(64, 16)
    remote_response = float(derivative[-1].double().norm())
    assert remote_response > 1e-12
    # With one spatial slot, copied native blocks must agree with the actual
    # original block, including normalization/modulation, not just mask shape.
    native = copy.deepcopy(flat.predictor.transformer.layers[0]).eval()
    with torch.no_grad():native.adaLN_modulation[-1].bias.fill_(.05)
    copied = SpatialBlock(copy.deepcopy(native)).eval()
    x, c = torch.randn(2, 3, 192), torch.randn(2, 3, 192)
    with torch.no_grad():
        torch.testing.assert_close(native(x, c), copied(x[:, :, None], c, torch.ones(3, 3, dtype=torch.bool).tril())[:, :, 0], rtol=2e-6, atol=2e-6)
    # A single token can use all16 channels after a nonzero output update;
    # the64-token map therefore has no fixed global191-D output restriction.
    tr = spatial.predictor.transformer
    jac = torch.autograd.functional.jacobian(lambda z: tr.output_proj(tr.norm(tr.input_proj(z))), torch.randn(16))
    rank = int(torch.linalg.matrix_rank(jac))
    assert rank == 16
    # Match native stochastic teacher-forced + private-stream autoregressive
    # forecasts. Recompute happens after the private RNG context has exited.
    plain, checked = copy.deepcopy(spatial.predictor).train(), copy.deepcopy(spatial.predictor).train()
    plain.checkpoint_blocks, checked.checkpoint_blocks = False, True
    outer = torch.Generator(device='cpu').manual_seed(1042).get_state()
    target = torch.randn_like(state)
    results = []
    for model in [plain, checked]:
        torch.set_rng_state(outer)
        teacher = model(state, action)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(17).get_state())
            history = [state[:, 0]]
            for t in range(state.shape[1]-1):
                history.append(history[-1] + model(torch.stack(history, 1), action[:, :t+1])[:, -1])
            forecast = torch.stack(history, 1)
        loss = (teacher-target).square().mean() + (forecast-target).square().mean()
        loss.backward()
        results.append((loss.detach(), torch.get_rng_state(), {k:v.grad.clone() for k,v in model.named_parameters()}))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    assert torch.equal(results[0][1], results[1][1])
    max_gradient_error = 0.
    for key in results[0][2]:
        left_grad, right_grad = results[0][2][key], results[1][2][key]
        torch.testing.assert_close(left_grad, right_grad, rtol=1e-5, atol=1e-6)
        max_gradient_error = max(max_gradient_error, float((left_grad-right_grad).abs().max()))
    original_seed_all = torch.cuda.manual_seed_all
    def forbidden(*args, **kwargs):raise AssertionError('CPU projection initialization must not reseed CUDA')
    try:
        torch.cuda.manual_seed_all = forbidden
        isolated = SpatialNativePredictor(copy.deepcopy(flat.predictor), 123)
    finally:torch.cuda.manual_seed_all = original_seed_all
    assert not torch.cuda.is_initialized()
    result = dict(passed=True, CUDA_initialized=False, shared_initial_tensors=len(shared),
        replaced_projection_tensors=replaced, all_six_native_blocks_and_condition_tensors_exact=True,
        outer_CPU_RNG_exact=True, no_CUDA_reseed=True, native_block_forward_parity=True,
        initial_carry_exact=True, causal_prefix_exact=True, batch_independent=True,
        all_same_frame_slots_visible=True, actual_remote_slot_gradient_norm=remote_response,
        per_token_output_rank_after_probe=rank, old_predictor_parameters=old,
        new_predictor_parameters=new, parameter_difference=new-old,
        checkpoint_loss_exact=True, checkpoint_outer_RNG_after_backward_exact=True,
        checkpoint_gradients_match=True, checkpoint_gradient_max_abs_error=max_gradient_error,
        scope='CPU structural probes,not learned task success or novelty. Uses64 spatial tokens with full native6-layer/192hidden/16head/64headwidth/2048MLP body;compute and training dropout streams differ from flat.')
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':main()
