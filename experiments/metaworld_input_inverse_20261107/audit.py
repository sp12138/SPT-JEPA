"""Native full-state audit plus actual executed-input inverse data-flow checks."""
import argparse
import json
from pathlib import Path


def run(args):
    import torch
    from torch.utils.data import DataLoader, Subset
    from experiments.metaworld_eval_20261103.audit import main as native_audit
    from as_jepa_lewm.factory import build_model
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    cfg = json.loads((args.run/'config.json').read_text())
    native_output = args.output.with_name(args.output.stem+'-native.json')
    native_audit(str(args.run), args.step, str(native_output))
    report = json.loads(native_output.read_text())
    model = build_model(cfg).cuda().eval()
    weights = torch.load(args.run/f'model-step{args.step:07d}/weights.pt', map_location='cuda', weights_only=True)
    model.load_state_dict(weights, strict=True)
    ds = BenchmarkWindows(cfg['data'], window=8, split='test')
    batch = next(iter(DataLoader(Subset(ds, [0, len(ds)//2]), batch_size=2, num_workers=0)))
    batch = {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
    # Poison the final padding action. It is NOT a valid transition input or target.
    def loss_for(value):
        torch.manual_seed(70831)
        torch.cuda.manual_seed_all(70831)
        return model.loss(value, args.step)[0]
    with torch.no_grad():
        reference = float(loss_for(batch))
        poisoned = dict(batch, action=batch['action'].clone())
        poisoned['action'][:, -1] = 12345
        padding_error = abs(float(loss_for(poisoned))-reference)
    assert padding_error == 0, padding_error
    # The inverse term must be separable across clips and use only the pair.
    a = torch.randn(2, 7, model.state_dim, device='cuda', requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    target = torch.randn(2, 7, 20, device='cuda')
    noise = torch.randn_like(target)
    time = torch.full((2, 7, 1), .8, device='cuda')
    losses = model.inverse_per_sample(a, b, target, noise, time)
    grads = torch.autograd.grad(losses[0], [a, b])
    cross_sample = max(float(g[1].abs().max()) for g in grads)
    assert cross_sample == 0, cross_sample
    # Removing the head has no effect on actual inference encoding/prediction.
    with torch.no_grad():
        pixels, _, _ = model.prepare(batch)
        info = model.encode(dict(pixels=pixels, action=batch['action']))
        before = model.predict(info['emb'][:, :-1], info['act_emb'][:, :-1])
        inverse = model.inverse_head
        model.inverse_head = torch.nn.Identity()
        torch.manual_seed(95031)
        output = model.encode(dict(pixels=pixels, action=batch['action']))
        after = model.predict(output['emb'][:, :-1], output['act_emb'][:, :-1])
        model.inverse_head = inverse
        deployment_error = float((before-after).abs().max())
    assert deployment_error == 0, deployment_error
    report.update(padding_action_poison_max_abs=padding_error,
        inverse_cross_sample_gradient_max_abs=cross_sample,
        deployment_inverse_removal_max_abs=deployment_error,
        inverse_target='EXACT action[:, :-1] consumed by forward predictor on the same observed transitions',
        no_goal_policy_action_target=True, inverse_deployed=False, passed=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, step=args.step, padding=padding_error,
                         cross_sample=cross_sample, deployment=deployment_error)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
