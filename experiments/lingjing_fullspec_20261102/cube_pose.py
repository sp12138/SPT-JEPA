"""Generic frozen Cube readout using the already fixed128/64 cohorts and published internal probe arithmetic."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from common import B, read, write, sha


@torch.no_grad()
def main(checkpoint, config, seed, output):
    from as_jepa_tools.evaluate import load_model
    from as_jepa_tools.benchmark_gap import make_benchmark_dataset
    from experiments.carrier_progress_20261031 import tf_pose as old
    from experiments.trajectory_search_20261101.common import state_hash
    from vforge.ops import OpsConfig, initialize_ops
    cfg = read(config); refs = old.references(seed); assert cfg['data'] == str(refs['data'])
    original_source = read(old.SOURCE); manifest = read(original_source['manifest'])
    for name, module in [('orientation_probe.py', old.pose), ('carrier_spatial_probe.py', old.spatial)]:
        assert sha(module.__file__) == manifest['files']['experiments/carrier_progress_20261031/' + name]
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest'); torch.manual_seed(1042)
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=torch.device('cuda'), parallel_state=None)
    model = load_model(checkpoint).eval(); model.requires_grad_(False); before = state_hash(model)
    saved = {}; actions = None
    for split, cohort in refs['manifests'].items():
        dataset = make_benchmark_dataset(cfg, split=split); observations, acts = [], []; offset = 0
        for batch in DataLoader(Subset(dataset, [r['index'] for r in cohort['windows']]), batch_size=4, num_workers=0):
            for i, ep in enumerate(batch['episode'].tolist()):
                expected = cohort['windows'][offset+i]
                assert ep == expected['episode'] and batch['rows'][i].tolist() == expected['rows']
            offset += len(batch['episode']); moved = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            observations.append(model.encode(dict(pixels=model.prepare(moved)[0]))['emb'].float().cpu())
            acts.append(batch['action'])
        observed, action = torch.cat(observations), torch.cat(acts)
        dataset.file.close()
        dataset.file = None
        dataset.actions = None
        saved[split + '_observed'] = observed.numpy()
        if split == 'validation':actions = action
    observed = torch.from_numpy(saved['validation_observed']); states = {key: [] for key in old.MODES}
    for lo in range(0, len(observed), 4):
        batch = old.prediction_states(model, observed[lo:lo+4].cuda(), actions[lo:lo+4].cuda())
        for key, value in batch.items():states[key].append(value.cpu().numpy())
    states = {key: np.concatenate(value) for key, value in states.items()}
    np.testing.assert_array_equal(actions.numpy(), refs['actions'])
    dimension = observed.shape[-1]
    if model.mode != 'L0':
        rng = np.random.default_rng(1042)
        states['observed_noisy'] = states['observed_future'] + model.sigma0 * rng.standard_normal(states['observed_future'].shape)
    import h5py, hdf5plugin
    card = read(refs['data'] / 'dataset_card.json')
    with h5py.File(card['h5_path'], 'r', swmr=True) as h5:
        tq = old.pose.load_quaternions(h5, refs['manifests']['train']).reshape(-1, 4)
        vq = old.pose.load_quaternions(h5, refs['manifests']['validation'])[:, 1:].reshape(-1, 4)
        xyz = old.spatial.load_xyz(h5, refs['manifests']['train']).reshape(-1, 3)
        vxyz = old.spatial.load_xyz(h5, refs['manifests']['validation'])[:, 1:].reshape(-1, 3)
    target = np.concatenate([old.pose.invariant(tq), xyz], -1); validation = np.concatenate([old.pose.invariant(vq), vxyz], -1)
    keys = list(states); concatenated = np.concatenate([states[key].reshape(448, dimension) for key in keys])
    outputs, normalization = old.fit_and_predict(saved['train_observed'].reshape(1024, dimension).astype(np.float64), concatenated.astype(np.float64), target)
    rows = []
    for kind, (train_pred, pred, configuration) in outputs.items():
        scores = {key: old.spatial.score(pred[i*448:(i+1)*448], validation, vq, target, tq, 64) for i, key in enumerate(keys)}
        rows.append(dict(readout=kind, configuration=configuration, normalization=normalization,
                         conditions={key: old.score_scalar(value) for key, value in scores.items()}, complete_scores=scores))
    assert before == state_hash(model)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    write(out / 'pose.json', dict(passed=True, rows=rows, fixed_manifests=refs['manifests'], reference_provenance=refs['provenance'],
          checkpoint_sha256=sha(Path(checkpoint) / 'weights.pt'), state_before=before, state_after=state_hash(model), model_updates=0,
          actual_embedding_dimension=dimension, model_mode=model.mode,
          helpers_sha256={str(module.__file__): sha(module.__file__) for module in [old.pose, old.spatial]},
          scope='Same fixed1024train/448validation observations and untuned linear/RBF readouts. Physical labels fit only the diagnostic heads. Cold initial-only rollouts are not native history3/CEM-selected actions; joint readout is not P0 or a hold guarantee.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True); p.add_argument('--config', required=True); p.add_argument('--seed', type=int, required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); main(a.checkpoint, a.config, a.seed, a.output)
