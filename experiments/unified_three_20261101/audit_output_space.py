"""Fixed global decoder span versus observed motion;zero new model forwards."""
import argparse

import numpy as np
import torch

from common import B, read, sha, write


def complement(x, basis):
    return x - (x @ basis) @ basis.T


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--output', required=True)
    args = parser.parse_args();torch.set_num_threads(2);root = B / 'runs/unified_three_20261101'
    rows = []
    for task in ['pusht', 'tworoom', 'cube']:
        for seed in [1, 2]:
            ident = f'{task}-flat-seed{seed}';previous = None
            for step in [600, 2000]:
                weights = root / ident / f'model-step{step:07d}/weights.pt'
                state = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
                matrix = state['predictor.transformer.output_proj.weight'].numpy().astype(np.float64)
                bias = state['predictor.transformer.output_proj.bias'].numpy().astype(np.float64)
                assert matrix.shape == (1024, 192) and bias.shape == (1024,)
                u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
                threshold = max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
                rank = int((singular > threshold).sum());basis = u[:, :rank]
                assert rank == 192
                ap = root / 'audits' / f'{ident}-step{step}' / 'state-audit.npz'
                with np.load(ap) as data:
                    observed = data['observed'].astype(np.float64)
                    predicted = data['clean_rollout'].astype(np.float64)
                    other_actions = data['other_episode_actions'].astype(np.float64)
                if previous is not None:assert np.array_equal(observed, previous)
                previous = observed
                delta = (observed[:, 1:] - observed[:, :-1]).reshape(-1, 1024)
                centered = delta - delta.mean(0)
                spectrum = np.linalg.svd(centered, compute_uv=False)
                oracle_floor = float(np.square(spectrum[192:]).sum() / centered.size)
                local_energy = float(np.square(delta).mean())
                floor = float(np.square(complement(delta - bias, basis)).mean())
                delta5 = observed[:, 7] - observed[:, 2]
                floor5 = float(np.square(complement(delta5 - 5 * bias, basis)).mean())
                offsets = np.arange(1, 8)[None, :, None] * bias
                residual = complement((predicted - observed[:, :1] - offsets).reshape(-1, 1024), basis)
                action_blind = complement((predicted - other_actions).reshape(-1, 1024), basis)
                max_error = float(np.max(np.abs(residual)));action_error = float(np.max(np.abs(action_blind)))
                assert max_error < 3e-5 and action_error < 3e-5
                # Actual recorded wrong-action forecasts have the same initial
                # state. Orthogonal goal error adds only a candidate-independent
                # constant, so these cost differences must agree.
                goal = observed[:, -1]
                left, right = predicted[:, -1] - goal, other_actions[:, -1] - goal
                full_difference = (left * left).mean(1) - (right * right).mean(1)
                parallel_difference = ((left @ basis) ** 2).sum(1) / 1024 - ((right @ basis) ** 2).sum(1) / 1024
                cost_error = float(np.max(np.abs(full_difference - parallel_difference)))
                assert cost_error < 3e-5
                rows.append(dict(task=task, seed=seed, step=step, decoder_rank=rank,
                    fixed_action_blind_dimensions=1024-rank,
                    local_motion_energy=local_energy, best_possible_rank192_bias_adjusted_MSE=oracle_floor,
                    best_possible_rank192_fraction=oracle_floor/local_energy,
                    actual_decoder_one_step_MSE_floor=floor, actual_decoder_one_step_fraction=floor/local_energy,
                    actual_decoder_native_history3_H5_MSE_floor=floor5,
                    H5_motion_energy=float((delta5 * delta5).mean()),
                    cached_forecast_span_max_abs_error=max_error, cached_action_effect_span_max_abs_error=action_error,
                    projected_goal_cost_difference_max_error=cost_error,
                    weights_sha256=sha(weights), cached_states_sha256=sha(ap)))
                del state
    assert not torch.cuda.is_initialized()
    write(args.output, dict(passed=True, rows=rows, source_sha256=sha(__file__),
          calls=dict(encoder=0, model=0, renderer=0, simulator=0, GPU=0),
          mechanism='For residual z_next=z+Wh+b,all action effects at every fixed horizon lie in range(W). '
                    'The goal error in its832-dimensional orthogonal complement is candidate-independent. '
                    'A low-dimensional curved observation manifold can still have a large global linear span.',
          limits='Same64 cached validation clips per task/seed,both budgets. Oracle SVD is a diagnostic,not a trained method. '
                 'Bounds are latent-space errors,not task success or physical information. Full-spatial/local output heads '
                 'can remove this fixed-span restriction but need not learn useful dynamics;the failed indirect-head controls '
                 'already show that rank freedom alone is insufficient. No novelty/common-performance claim.'))
    print([{k:x[k] for k in ['task','seed','step','actual_decoder_one_step_fraction','best_possible_rank192_fraction',
                            'actual_decoder_native_history3_H5_MSE_floor','H5_motion_energy']} for x in rows])


if __name__ == '__main__':main()
