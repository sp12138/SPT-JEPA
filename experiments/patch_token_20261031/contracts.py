"""Bounded CPU-only construction/mask/budget audit. No learned data or optimizer."""
import argparse
import copy
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from prototype import PatchTokenPredictor, frame_mask, forward_macs, sequence_budget


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def count(model):
    return sum(p.numel() for p in model.parameters())


def open_fixture(model):
    """Nonzero synthetic weights only: zero-init would make causality vacuous."""
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(731)
        nn.init.normal_(model.output_proj.weight, std=.02)
        for block in model.layers:
            nn.init.normal_(block.adaLN_modulation[-1].weight, std=.01)
            nn.init.normal_(block.adaLN_modulation[-1].bias, std=.1)
    return model.eval()


def measured_linear_macs(model, x, a):
    total = [0]
    handles = []
    def observe(layer, inputs, output):
        total[0] += output.numel()*layer.in_features
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            handles.append(layer.register_forward_hook(observe))
    try:
        with torch.no_grad():model(x, a)
    finally:
        for handle in handles:handle.remove()
    return total[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--native-source', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    started = time.monotonic()
    native_root = Path(args.native_source)
    module_path = native_root/'stable_worldmodel/wm/lewm/module.py'
    rollout_path = native_root/'stable_worldmodel/wm/lewm/lewm.py'
    native = load_file('audited_native_modules', module_path)
    native_lewm = load_file('audited_native_lewm', rollout_path)
    torch.manual_seed(1042)
    before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    patch = PatchTokenPredictor().eval()
    mixed = PatchTokenPredictor(basis='orthogonal').eval()
    assert torch.equal(before, torch.get_rng_state())
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(numpy_before, after))
    for name, value in patch.named_parameters():
        assert torch.equal(value, dict(mixed.named_parameters())[name])
    baseline = native.Predictor(num_frames=7, depth=6, heads=16, mlp_dim=2048,
        input_dim=1024, hidden_dim=192, output_dim=1024, dim_head=64, dropout=.1,
        emb_dropout=0.).eval()
    nn.init.zeros_(baseline.transformer.output_proj.weight)
    nn.init.zeros_(baseline.transformer.output_proj.bias)
    x = torch.randn(2, 7, 1024);a = torch.randn_like(x)
    for model in (baseline, patch, mixed):
        with torch.no_grad():assert torch.equal(x+model(x,a), x)
    linear_counts = {}
    for name, model in [('native',baseline),('patch',patch)]:
        actual = measured_linear_macs(model, x, a)
        expected = forward_macs(7, batch=2, kind=name)['linear']
        assert actual == expected, (name, actual, expected)
        linear_counts[name] = actual
    matrix = mixed.coordinate_basis.double()
    orthogonal_error = float((matrix.T@matrix-torch.eye(1024,dtype=torch.float64)).abs().max())
    assert orthogonal_error == 0.
    raw_distance = ((x[:,0].double()-x[:,1].double())**2).mean(-1)
    transformed_distance = (((x[:,0].double()-x[:,1].double())@matrix)**2).mean(-1)
    metric_error = float((raw_distance-transformed_distance).abs().max())
    assert metric_error < 1e-12

    probe = open_fixture(copy.deepcopy(patch))
    xx = x[:1,:3].clone();aa = a[:1,:3].clone()
    mask = frame_mask(3)
    frame = torch.arange(3).repeat_interleave(64)
    assert torch.equal(mask,frame[None,:] <= frame[:,None])
    assert bool(mask[:64,:64].all()) and not bool(mask[:64,64:].any())
    with torch.no_grad():
        y = probe(xx,aa)
        future_x = xx.clone();future_x[:,2] += 3
        future_a = aa.clone();future_a[:,2] -= 7
        err_x = float((probe(future_x,aa)[:,:2]-y[:,:2]).abs().max())
        err_a = float((probe(xx,future_a)[:,:2]-y[:,:2]).abs().max())
        prefix_error = float((probe(xx[:,:2],aa[:,:2])-y[:,:2]).abs().max())
        sameframe = xx.clone();sameframe[:,0,-16:] += .5
        spatial_effect = float((probe(sameframe,aa)[:,0,:16]-y[:,0,:16]).abs().max())
        now_a = aa.clone();now_a[:,0] += .5
        action_effect = float((probe(xx,now_a)[:,0]-y[:,0]).abs().max())
    assert max(err_x,err_a,prefix_error) < 2e-6
    assert spatial_effect > 1e-9 and action_effect > 1e-7
    grad_x = xx.clone().requires_grad_();grad_a = aa.clone().requires_grad_()
    probe(grad_x,grad_a)[:,0].square().sum().backward()
    assert not bool(grad_x.grad[:,1:].any()) and not bool(grad_a.grad[:,1:].any())
    rejected_padding = False
    try:probe(xx,torch.cat([aa,aa[:,:1]],1))
    except ValueError:rejected_padding = True
    assert rejected_padding

    class CachedModel(native_lewm.LeWM):
        def __init__(self):
            super().__init__(nn.Identity(),copy.deepcopy(probe),
                native.Embedder(input_dim=10,smoothed_dim=10,emb_dim=1024,mlp_scale=4))
            self.calls = []
        def encode(self, info):
            raise AssertionError('No encoder call allowed in cache-contract fixture')
        def predict(self, emb, act_emb):
            self.calls.append((emb.detach().clone(),act_emb.detach().clone()))
            return emb+self.predictor(emb,act_emb)

    rollout_tests = []
    for H in (1,3):
        model = CachedModel().eval()
        observed = torch.randn(1,2,H,1024)
        past = torch.randn(1,2,H-1,10)
        candidates = torch.randn(1,2,5,10)
        info = dict(pixels=torch.zeros(1,2,H,1,1,1),emb=observed,action_history=past)
        with torch.no_grad():result = model.rollout(copy.deepcopy(info),candidates)
        lengths = [H+t for t in range(5)]
        assert [call[0].shape[1] for call in model.calls] == lengths
        assert result['predicted_emb'].shape == (1,2,H+5,1024)
        assert torch.equal(result['predicted_emb'][:,:,:H],observed)
        assert torch.equal(result['action'],torch.cat([past,candidates[:,:,:1]],2))
        with torch.no_grad():all_actions = model.action_encoder(torch.cat([past,candidates],2).flatten(0,1))
        for t,call in enumerate(model.calls):
            assert torch.equal(call[1],all_actions[:,:H+t])
        changed = candidates.clone();changed[:,:,4] += 20
        model.calls.clear()
        with torch.no_grad():other = model.rollout(copy.deepcopy(info),changed)['predicted_emb']
        error = float((other[:,:,:H+4]-result['predicted_emb'][:,:,:H+4]).abs().max())
        assert error == 0.
        rollout_tests.append(dict(initial_history=H,predict_lengths=lengths,output_shape=list(other.shape),
             future_action_prefix_error=error,executed_action_history_exact=True,encoder_calls=0))

    # Final LayerNorm removes at most one tangent direction per64-D token.
    # A16-row output can still be full row rank; block diagonal repetition then
    # has rank64*16. This is output-stage capacity, NOT fitted model/data rank.
    hidden = torch.randn(64,requires_grad=True)
    jac = torch.autograd.functional.jacobian(lambda v:probe.output_proj(probe.norm(v)),hidden)
    local_rank = int(torch.linalg.matrix_rank(jac.double()))
    assert local_rank == 16

    timings = []
    for name, model in [('native',baseline),('patch',probe)]:
        model.train();model.zero_grad(set_to_none=True)
        tx = torch.randn(1,7,1024,requires_grad=True);ta = torch.randn_like(tx)
        start = time.monotonic();prediction = model(tx,ta);loss = prediction.square().mean();loss.backward()
        seconds = time.monotonic()-start
        assert torch.isfinite(prediction).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        timings.append(dict(model=name,batch=1,frames=7,seconds=seconds,
            train_mode=True,forward_backward_calls=1,note='Single CPU timing; not GPU throughput estimate'))

    budgets = {}
    for label,lengths,batch in [('training_TF',[7,1,2,3,4,5,6,7],128),
                               ('CEM_H1',[1,2,3,4,5],300),('CEM_H3',[3,4,5,6,7],300)]:
        row = {kind:sequence_budget(lengths,batch,kind=kind) for kind in ('native','patch')}
        row['patch_orthogonal'] = sequence_budget(lengths,batch,kind='patch',orthogonal=True)
        row.update(batch=batch,predict_lengths=lengths,patch_over_native=row['patch']['total']/row['native']['total'],
                   orthogonal_over_patch=row['patch_orthogonal']['total']/row['patch']['total'])
        budgets[label] = row
    score_elements = 128*4*64**2*sum(t*t for t in [7,1,2,3,4,5,6,7])*6
    result = dict(passed=True,CPU_only=True,GPU_initialized=torch.cuda.is_initialized(),
        elapsed_seconds=time.monotonic()-started,torch_version=torch.__version__,CPU_threads=2,
        sources={str(path):sha(path) for path in [Path(__file__),Path(__file__).with_name('prototype.py'),module_path,rollout_path]},
        parameter_counts=dict(native_predictor=count(baseline),patch_predictor=count(patch),
            orthogonal_predictor=count(mixed),orthogonal_fixed_buffer_elements=matrix.numel(),
            retained_action_encoders={str(d):count(native.Embedder(input_dim=d,smoothed_dim=d,emb_dim=1024,mlp_scale=4)) for d in (10,25)}),
        exact_linear_hook_counts_B2_T7=linear_counts,private_RNG_unchanged=True,paired_parameters_exact=True,
        initial_raw_carry_exact=True,orthogonality_max_abs=orthogonal_error,goal_metric_float64_max_abs=metric_error,
        mask=dict(shape=list(mask.shape),allowed=int(mask.sum()),future_state_error=err_x,future_action_error=err_a,
            prefix_error=prefix_error,sameframe_last_to_first_slot_effect=spatial_effect,
            current_action_effect=action_effect,future_gradients_exact_zero=True,padded_action_rejected=True),
        native_cached_rollout=rollout_tests,shared_output_local_jacobian_rank=local_rank,
        output_stage_possible_global_rank=64*local_rank,output_rank_is_capacity_not_empirical=True,
        budgets=budgets,naive_training_attention_score_bf16_GiB=score_elements*2/2**30,
        naive_training_attention_scope='One attention-score tensor per layer/call summed across retained TF graph; excludes softmax/backward/other activations. Fused SDPA may not materialize it.',
        CPU_smoke_timings=timings,no_optimizer_steps=True,no_data_or_labels_read=True,
        not_included_in_MACs=['encoder','action_encoder','DiT','backward','nonlinearities','normalization','softmax','dropout','elementwise','memory traffic'])
    assert not result['GPU_initialized']
    output = Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,parameters=result['parameter_counts'],seconds=result['elapsed_seconds'],
                         ratios={k:v['patch_over_native'] for k,v in budgets.items()})))


if __name__ == '__main__':main()
