"""Full-model compatibility, local-query information witness and raw-image audit."""
import argparse
import gc
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, default_collate

from common import dataset, read, sha, windows, write
from as_jepa_lewm.quadrant_render import QuadrantRenderLeWM
from as_jepa_lewm.robust import RobustLeWM
from vforge.ops import OpsConfig, initialize_ops
from vforge.models.as_jepa.pixels import pack


def information_witness():
    # Two starting positions outside the right half; the same +3 displacement
    # enters that query only from the second starting position.
    starts = torch.zeros(2, 8)
    starts[0, 0] = starts[1, 1] = 1.
    ends = starts.roll(3, dims=1)
    local_anchors, targets = starts[:, 4:], ends[:, 4:]
    assert torch.equal(local_anchors[0], local_anchors[1])
    assert not torch.equal(targets[0], targets[1])
    best_constant = targets.mean(0)
    risk = (targets - best_constant).square().mean()
    assert float(risk) == .0625
    return dict(same_action=3, identical_local_anchors=True,
                different_future_queries=True, constant_state_minimum_pixel_MSE=float(risk),
                full_anchor_or_one_bit_state_minimum_MSE=0.,
                scope='Exact two-state toy at flow t=1:local anchor/query/noise cannot distinguish '
                      'the futures;full anchor can. This is not a real-task learning guarantee.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    parent = Path(args.parent)
    torch.set_num_threads(2)
    initialize_ops(OpsConfig(backend='sdpa_math', allow_fallback=False),
                   device=torch.device('cpu'), parallel_state=None)
    # Every original224px pixel belongs to exactly one of the four queries.
    image = torch.arange(224 * 224).reshape(1, 1, 224, 224)
    tiles = [QuadrantRenderLeWM.quadrants(image, torch.tensor([q])) for q in range(4)]
    reconstructed = torch.cat([torch.cat(tiles[:2], -1), torch.cat(tiles[2:], -1)], -2)
    assert torch.equal(image, reconstructed)
    rows = []
    for task in ['pusht', 'tworoom', 'cube']:
        run = parent / f'{task}-tf-seed1'
        cfg = read(run / 'config.json')
        torch.manual_seed(cfg['seed'])
        original = RobustLeWM(**cfg['model']).eval()
        original_rng = torch.get_rng_state()
        torch.manual_seed(cfg['seed'])
        query = QuadrantRenderLeWM(**cfg['model']).eval()
        assert torch.equal(torch.get_rng_state(), original_rng)
        old_state, new_state = original.state_dict(), query.state_dict()
        extra = set(new_state) - set(old_state)
        assert extra == {'renderer.query_projection.weight', 'renderer.query_projection.bias'}
        assert all(torch.equal(v, new_state[k]) for k, v in old_state.items())
        weights = torch.load(run / 'model-step0000600/weights.pt',
                             map_location='cpu', weights_only=True, mmap=True)
        original.load_state_dict(weights, strict=True)
        missing, unexpected = query.load_state_dict(weights, strict=False)
        assert set(missing) == extra and not unexpected
        ds = dataset(cfg, 'train')
        batch = default_collate([ds[0]])
        batch.pop('state', None)
        with torch.no_grad():
            px, render, anchor = original.prepare(batch)
            observed = original.encode(dict(pixels=px, action=batch['action']))
            queried_observed = query.encode(dict(pixels=px, action=batch['action']))
            assert all(torch.equal(v, queried_observed[k]) for k, v in observed.items())
            assert torch.equal(original.predict(observed['emb'][:, :-1], observed['act_emb'][:, :-1]),
                               query.predict(observed['emb'][:, :-1], observed['act_emb'][:, :-1]))
            y = pack(render[:, 1:].flatten(0, 1)); anc = pack(anchor).repeat_interleave(len(y), 0)
            t = torch.full((len(y), 1), .95)
            condition = observed['emb'][:, 1:].reshape(len(y), query.state_slots, query.state_width)
            with original.renderer.anchored(anc):
                reference = original.renderer(y, t, condition=condition)
            with query.renderer.anchored(anc):
                actual = query.renderer(y, t, condition=condition)
            assert torch.equal(reference, actual)
        before = torch.get_rng_state()
        modified, address = query.render_query_batch(batch, 600)
        assert torch.equal(before, torch.get_rng_state())
        assert modified['pixels'] is batch['pixels'] and modified['anchor224'] is batch['anchor224']
        assert torch.equal(modified['action'], batch['action'])
        try:
            with query.renderer.queried(address):
                raise ValueError('scope cleanup witness')
        except ValueError:
            pass
        assert query.renderer._query_address is None
        query.train()
        torch.manual_seed(1042)
        loss, metrics = query.loss(batch, step=600)
        assert torch.isfinite(loss) and all(torch.isfinite(v).all() for v in metrics.values())
        loss.backward()
        gradients = {k: float(v.grad.norm()) for k, v in query.named_parameters()
                     if v.grad is not None and k.startswith('renderer.query_projection')}
        assert len(gradients) == 2 and all(v > 0 for v in gradients.values())
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in query.parameters())
        assert query.renderer._query_address is None
        # Pure train-RGB measurement, no physical labels/model calls/selection.
        measured = []
        cohort = windows(ds, count=64, seed=20261101)
        for raw in DataLoader(Subset(ds, [x['index'] for x in cohort]), batch_size=2, num_workers=0):
            full = raw['pixels'].float()
            overview = raw['render'].float()
            local_energy = []
            for quadrant in range(4):
                choices = torch.full((len(full),), quadrant, dtype=torch.long)
                crop = query.quadrants(full, choices)
                rendered = F.interpolate(crop.flatten(0, 1), size=(64, 64), mode='area').round().unflatten(0, crop.shape[:2])
                local_energy.append((rendered[:, 1:] - rendered[:, :-1]).square().mean((1, 2, 3, 4)))
            values = torch.stack([(full[:, 1:] - full[:, :-1]).square().mean((1, 2, 3, 4)),
                                  (overview[:, 1:] - overview[:, :-1]).square().mean((1, 2, 3, 4)),
                                  torch.stack(local_energy).mean(0)], -1)
            measured.extend(values.tolist())
        measured = torch.tensor(measured, dtype=torch.float64)
        means = measured.mean(0)
        rows.append(dict(task=task, shared_initial_tensors_exact=len(old_state),
                         original_outer_CPU_RNG_exact=True, encoder_action_predict_and_unqueried_renderer_exact=True,
                         extra_parameters=sum(new_state[k].numel() for k in extra),
                         finite_loss=float(loss.detach()), query_gradients=gradients,
                         real_train_window=dict(episode=int(batch['episode'][0]), start=int(batch['start'][0])),
                         raw_train_measurement=dict(cohort=cohort, per_clip_motion_energy=measured.tolist(),
                             mean_full224_overview64_query64_motion_energy=means.tolist(),
                             query_over_overview_energy=float(means[2] / means[1])),
                         scope='One full-model CPU backward and64 train clips. Motion energy is not task information;'
                               'new query target differs,so FM losses cannot be compared as a likelihood improvement.'))
        del original, query, weights, old_state, new_state, loss, metrics, observed, queried_observed
        gc.collect()
    assert not torch.cuda.is_initialized()
    write(args.output, dict(passed=True, CUDA_initialized=False, all_original_pixels_covered_once=True,
          information_witness=information_witness(), full_actual_model_regressions=rows,
          source_sha256=sha(Path(__file__).resolve().parents[2] / 'as_jepa_lewm/quadrant_render.py'),
          scope='All3 full-model CPU checks;no GPU/encoder training/simulator/physical labels. '
                'Known local-query generation mechanism,not a novelty or task-success claim.'))
    print([(r['task'], r['shared_initial_tensors_exact'], r['extra_parameters'],
            r['raw_train_measurement']['query_over_overview_energy']) for r in rows])


if __name__ == '__main__':
    main()
