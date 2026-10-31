"""Score only actually executed crossed outcomes with frozen native world models."""
import argparse
import os
from pathlib import Path
import sys
from common import PY, array_sha, read, sha, source_check, state_hash, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-spec', required=True)
    p.add_argument('--acquisition', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--inner', action='store_true')
    a = p.parse_args()
    spec = read(a.model_spec)
    if not a.inner:
        env = dict(os.environ, PYTHONPATH=spec['model_source']['pythonpath'], PYTHONDONTWRITEBYTECODE='1')
        os.chdir(spec['model_source']['source'])
        os.execve(PY, [PY, '-B', '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--inner'], env)
    import inspect
    import numpy as np
    import torch
    from torch.nn import functional as F
    from stable_worldmodel.wm.utils import load_pretrained
    from clear_lewm.runner import _image_transform
    from vforge.ops import OpsConfig, initialize_ops
    source_check(spec['model_source'])
    acquisition = read(a.acquisition)
    assert acquisition['passed'] and acquisition['all_training_episodes']
    out = Path(a.output)
    assert not out.exists()
    out.mkdir(parents=True)
    checkpoint = Path(spec['checkpoint'])
    # Portable config contains the actual factory kwargs, including native model class.
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=torch.device('cuda'), parallel_state=None)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    model = load_pretrained(str(checkpoint)).cuda().eval().requires_grad_(False)
    weights = torch.load(checkpoint / 'weights.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    assert Path(inspect.getfile(type(model))).is_relative_to(spec['model_source']['source'])
    initial_hash = state_hash(model)
    card = read(Path(spec['data']) / 'dataset_card.json')
    mean, std = np.asarray(card['action_mean']), np.asarray(card['action_std'])
    transform = _image_transform(224)
    temperature = 2 * float(getattr(model, 'sigma0', 1.)) ** 2
    assert temperature > 0
    counts = dict(rgb=0, encoder=0, predict=0, action_encoder=0, renderer=0)
    original = model.predict
    def predict(*args, **kwargs):
        counts['predict'] += 1
        return original(*args, **kwargs)
    model.predict = predict
    def encoded(module, args):
        counts['encoder'] += 1
        counts['rgb'] += len(args[0])
    hooks = [model.encoder.register_forward_pre_hook(encoded)]
    def acted(*args):
        counts['action_encoder'] += 1
    hooks.append(model.action_encoder.register_forward_pre_hook(acted))
    if getattr(model, 'renderer', None) is not None:
        def forbidden(*args):
            raise AssertionError('No renderer in crossed diagnostic')
        hooks.append(model.renderer.register_forward_pre_hook(forbidden))
    rows = []
    with torch.inference_mode():
        for entry in acquisition['files']:
            assert sha(entry['path']) == entry['sha256']
            with np.load(entry['path']) as data:
                pixels = data['pixels'].copy()
                actions = data['raw_actions'].copy()
                valid = bool(data['observed_step_mask'].all())
            assert pixels.shape == (4, 6, 224, 224, 3) and actions.shape == (4, 25, 5)
            x = torch.from_numpy(pixels).reshape(24, 224, 224, 3).permute(0, 3, 1, 2)
            emb = model.encode(dict(pixels=transform(x).cuda()[:, None]))['emb'][:, 0].reshape(4, 6, -1)
            actions -= mean
            actions /= std
            commands = torch.from_numpy(actions.reshape(4, 5, 25)).cuda()
            info = dict(pixels=emb.new_empty(1, 4, 1, 0), emb=emb[None, :, :1])
            pred = model.rollout(info, commands[None])['predicted_emb'][0]
            assert pred.shape == emb.shape and torch.isfinite(pred).all()
            # Candidate branch x query outcome; identity labels are shuffled in
            # training if this diagnostic earns a later learning experiment.
            costs = (pred[:, None, -1] - emb[None, :, -1]).square().sum(-1)
            true_costs = (emb[:, None, -1] - emb[None, :, -1]).square().sum(-1)
            def scores(cost):
                full = -cost.T / temperature
                action_logits = torch.stack([full[g, [(g//2)*2, (g//2)*2+1]] for g in range(4)])
                state_logits = torch.stack([full[g, [g%2, 2+g%2]] for g in range(4)])
                return dict(full_ce=float(F.cross_entropy(full, torch.arange(4, device='cuda'))),
                      full_accuracy=float((full.argmax(-1) == torch.arange(4, device='cuda')).float().mean()),
                      action_ce=float(F.cross_entropy(action_logits, torch.tensor([0, 1, 0, 1], device='cuda'))),
                      action_accuracy=float((action_logits.argmax(-1) == torch.tensor([0, 1, 0, 1], device='cuda')).float().mean()),
                      state_ce=float(F.cross_entropy(state_logits, torch.tensor([0, 0, 1, 1], device='cuda'))),
                      state_accuracy=float((state_logits.argmax(-1) == torch.tensor([0, 0, 1, 1], device='cuda')).float().mean()))
            target = out / f'quad{entry["quad"]:04d}.npz'
            np.savez_compressed(target, predicted=pred.cpu().numpy(), observed=emb.cpu().numpy(),
                                costs=costs.cpu().numpy(), true_costs=true_costs.cpu().numpy())
            rows.append(dict(quad=entry['quad'], all_steps_active=valid, predicted=scores(costs),
                  true_encoded=scores(true_costs), exact_RGB_endpoints=len({array_sha(p) for p in pixels[:, -1]}),
                  cache_sha256=sha(target), actions_distinct=entry['actions_distinct']))
    model.predict = original
    for h in hooks:
        h.remove()
    assert initial_hash == state_hash(model)
    assert counts == dict(rgb=24*len(rows), encoder=len(rows), predict=5*len(rows), action_encoder=len(rows), renderer=0)
    # A score omitting state assigns the same value to rows(s0,a) and(s1,a),
    # imposing CE>=log2 for branch identity; omitting action has the dual bound.
    write(out / 'results.json', dict(passed=True, model_spec_sha256=sha(a.model_spec), acquisition_sha256=sha(a.acquisition),
          rows=rows, counts=counts, state_before=initial_hash, state_after=initial_hash,
          temperature=temperature, input_axis_omission_branch_CE_lower_bound=float(np.log(2)),
          limitation='Training-only interventional diagnostic. RGB collisions/hidden state/partial paths prevent a universal identifiability claim. No physical labels, task gain, fitting or novelty claim.'))
    print('Crossed frozen-model diagnostics complete;no model update.', flush=True)


if __name__ == '__main__':
    main()
