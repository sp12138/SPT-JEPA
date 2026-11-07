"""Frozen CPU audit: does an inverse head use the observed transition endpoint?

Targets are the same already-executed forward inputs. No desired-goal/action
policy labels, new fitting, rendering, or planner changes are involved.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.metaworld_baselines_20261107.models import PublicBaseline
from experiments.metaworld_baselines_20261107.train_native import Windows, digest


def run(args):
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    args.output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    cfg = json.loads(args.config.read_text())
    assert cfg['method'] in ['smwm', 'ac-mtm', 'lewm-idm', 'h-jepa']
    model = PublicBaseline(cfg['method'], cfg['sources'])
    model.load_state_dict(torch.load(args.checkpoint / 'weights.pt', map_location='cpu', weights_only=True), strict=True)
    model.eval().requires_grad_(False); before = digest(model)
    if cfg['method'] in ['lewm-idm', 'h-jepa']: head = model.core.get_level(1).idm_loss_fn.idm
    elif cfg['method'] == 'smwm': head = model.core.inverse_model
    else: head = model.core.inverse_predictor
    data = Windows(cfg['data'], 8, cfg['seed'], 'test')
    indices = np.linspace(0, len(data)-1, args.windows, dtype=np.int64)
    assert len(set(indices)) == args.windows
    rows = []; latent = []; controls = []; identity = []
    for offset in range(0, len(indices), 4):
        chosen = indices[offset:offset+4]
        batch = torch.utils.data.default_collate([data[int(i)] for i in chosen])
        with torch.inference_mode():
            z = model.encode(dict(pixels=model.normalize(batch['pixels'])))['emb'].float()
            x, y = z[:, :-1], z[:, 1:]
            action = batch['action'][:, :-1]; assert action.shape[-1] == 20
            variants = dict(factual=(x, y), future_copy=(x, x), future_zero=(x, torch.zeros_like(y)),
                            future_other_window=(x, y.roll(1, 0)), current_other_window=(x.roll(1, 0), y),
                            reverse=(y, x))
            baseline = action.square().reshape(len(chosen), 7, 5, 4).mean((1, 2))
            errors = {}
            for kind, (left, right) in variants.items():
                pred = head(left.flatten(0, 1), right.flatten(0, 1)).reshape_as(action)
                errors[kind] = (pred-action).square().reshape(len(chosen), 7, 5, 4).mean((1, 2)).numpy()
            for j, index in enumerate(chosen):
                episode, start = data.base.locate(int(index))
                phase_episode = int(data.base.episode_ids[episode])
                ident = dict(index=int(index), phase_episode_id=phase_episode,
                             original_episode=phase_episode//5, phase=phase_episode%5, start=int(start))
                identity.append(ident)
                rows.append(dict(**ident, normalized_mean_action_baseline_mse_by_axis=baseline[j].numpy().tolist(),
                                 mse_by_axis={k:v[j].tolist() for k,v in errors.items()}))
            latent.append(z.numpy()); controls.append(batch['action'].numpy())
        if offset % 32 == 0: print(json.dumps(dict(windows=len(rows), seconds=time.monotonic()-started)), flush=True)
    after = digest(model); assert before == after and not torch.cuda.is_initialized()
    np.savez_compressed(args.output / 'frozen-features.npz', latent=np.concatenate(latent), action=np.concatenate(controls),
                        indices=indices, original_episode_ids=np.asarray([r['original_episode'] for r in identity]),
                        phase_episode_ids=np.asarray([r['phase_episode_id'] for r in identity]))
    average = {kind:np.mean([r['mse_by_axis'][kind] for r in rows], axis=0).tolist() for kind in rows[0]['mse_by_axis']}
    baseline = np.mean([r['normalized_mean_action_baseline_mse_by_axis'] for r in rows], axis=0)
    report = dict(passed=True, method=cfg['method'], seed=cfg['seed'], checkpoint=str(args.checkpoint),
        windows=len(rows), completed_transitions=7*len(rows), unique_original_episodes=len({r['original_episode'] for r in rows}),
        mean_mse_by_axis=average, normalized_train_mean_action_baseline_mse_by_axis=baseline.tolist(),
        future_copy_over_factual_by_axis=(np.asarray(average['future_copy'])/np.maximum(average['factual'], 1e-20)).tolist(),
        model_state_before=before, model_state_after=after, model_updates=0, gpu_context_created=False,
        seconds=time.monotonic()-started, rows=rows,
        scope='Frozen fine-level native inverse-head ablation on common heldout window8s. '
              'Axes average the5 executed controls in each20D forward input. '
              'Copy/masking/swapping are input-sensitivity interventions,may be off-manifold,not a retrained current-only Bayes predictor '
              'or causal proof. Current-only action predictability and object sufficiency are different questions. '
              'No action targets from goals,policy learning,SGD,or latent-unit cross-model physical error claims.')
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed', 'method', 'windows', 'mean_mse_by_axis', 'future_copy_over_factual_by_axis', 'seconds']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['config', 'checkpoint', 'output']: p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--windows', type=int, default=128)
    args = p.parse_args()
    try: run(args)
    except BaseException:
        import traceback
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'error.json').write_text(json.dumps(dict(traceback=traceback.format_exc()), indent=2)+'\n')
        raise
