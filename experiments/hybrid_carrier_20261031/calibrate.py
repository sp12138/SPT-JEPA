"""One joint F/Q calibration on the exact original128 initial training frames."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from as_jepa_lewm.hybrid_carrier import HybridCarrierLeWM
from as_jepa_tools.benchmark_data import BenchmarkWindows
from experiments.ideas_validate_20261031.carrier.calibrate import file_hash, tensor_hash
from vforge.ops import OpsConfig, initialize_ops


def statistics(values):
    if values.shape != (128, 64) or not bool(torch.isfinite(values).all()):
        raise ValueError("Exactly128 finite training-frame measurements required")
    x = values.double()
    mean = x.mean(0)
    std = x.std(0, unbiased=False)
    rms = std.square().mean().sqrt()
    if not bool(torch.isfinite(rms) and rms > 0):
        raise ValueError("Zero/nonfinite calibration RMS: no floor or replacement")
    return mean.float(), rms.float(), std


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference-calibration', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    ref_path = Path(args.reference_calibration)
    reference = json.loads(ref_path.read_text())
    assert reference['measured_frames'] == 128 and reference['selection_seed'] == 1042
    cfg = json.loads(Path(reference['config']).read_text())
    assert file_hash(reference['config']) == reference['config_sha256']
    assert cfg['dataset_kind'] == 'benchmark_hdf5' and not cfg.get('initial_encoder_checkpoint')
    kw = dict(cfg['model'])
    assert not any(k.startswith(('carrier_', 'spatial_', 'hybrid_')) for k in kw)
    kw.update(state_slots=8, state_width=16, hybrid_query='fixed')
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(cfg['seed'])
    backend = 'sdpa' if device.type == 'cuda' else 'sdpa_math'
    initialize_ops(OpsConfig(backend=backend, allow_fallback=False), device=device, parallel_state=None)
    model = HybridCarrierLeWM(**kw).to(device).eval()
    encoder_hash = tensor_hash(model.encoder.state_dict())
    assert encoder_hash == reference['encoder_initial_sha256']
    pool_hash = tensor_hash(model.pool.state_dict())
    projection_hash = tensor_hash({'carrier_projection': model.hybrid_projection})
    assert projection_hash == reference['projection_sha256']
    initial_hash = tensor_hash(model.state_dict())
    data = Path(cfg['data'])
    assert file_hash(data/'dataset_card.json') == reference['dataset_card_sha256']
    assert file_hash(data/'train_episodes.npy') == reference['train_episode_ids_sha256']
    ds = BenchmarkWindows(data, window=model.window, split='train', seed=cfg['seed'])
    indices = reference['selected_window_indices']
    assert len(indices) == 128
    loader = DataLoader(Subset(ds, indices), batch_size=8, num_workers=0)
    values = dict(fixed=[], query=[])
    episodes, starts, rows = [], [], []
    calls = 0
    with torch.no_grad():
        for batch in loader:
            pixels = model.normalize(batch['pixels'][:, :1].to(device))
            f, q = model.raw_channels(pixels)
            values['fixed'].append(f[:, 0].cpu())
            values['query'].append(q[:, 0].cpu())
            episodes.extend(int(v) for v in batch['episode'])
            starts.extend(int(v) for v in batch['start'])
            rows.extend(int(v) for v in batch['rows'][:, 0])
            calls += 1
    assert calls == 16
    assert episodes == reference['training_episode_ids'] and starts == reference['starts']
    assert rows == reference['measured_h5_rows']
    assert len(set(episodes)) == 128 and set(episodes) <= set(np.load(data/'train_episodes.npy').tolist())
    assert tensor_hash(model.state_dict()) == initial_hash
    measured, model_kwargs = {}, {}
    arrays = {}
    for name, batches in values.items():
        array = torch.cat(batches)
        mean, scale, raw_std = statistics(array)
        arrays[name] = array.numpy()
        model_kwargs['hybrid_' + name + '_mean'] = mean.tolist()
        model_kwargs['hybrid_' + name + '_scale'] = float(scale)
        z = (array-mean)/scale
        measured[name] = dict(raw_mean=mean.tolist(), raw_coordinate_std=raw_std.tolist(),
            global_rms=float(scale), raw_std_min=float(raw_std.min()), raw_std_max=float(raw_std.max()),
            normalized_rms=float(z.square().mean().sqrt()), normalized_coordinate_std_mean=float(z.std(0, unbiased=False).mean()),
            top16_centered_energy_fraction=float(raw_std.square().topk(16).values.sum()/raw_std.square().sum()),
            floor_applied=False)
    old_mean = torch.tensor(reference['model_kwargs']['carrier_mean'])
    old_scale = torch.tensor(reference['model_kwargs']['carrier_scale'])
    new_mean = torch.tensor(model_kwargs['hybrid_fixed_mean'])
    out.parent.mkdir(parents=True, exist_ok=True)
    values_path = out.with_suffix('.values.npz')
    if values_path.exists():
        raise FileExistsError(values_path)
    np.savez(values_path, **arrays)
    result = dict(source_commit=args.source_commit, helper_source_sha256=file_hash(__file__),
        model_source_sha256=file_hash(__import__('as_jepa_lewm.hybrid_carrier', fromlist=['__file__']).__file__),
        reference_calibration=str(ref_path), reference_calibration_sha256=file_hash(ref_path),
        config=reference['config'], config_sha256=reference['config_sha256'], model_seed=cfg['seed'],
        dataset_card_sha256=reference['dataset_card_sha256'], train_episode_ids_sha256=reference['train_episode_ids_sha256'],
        selected_window_indices=indices, training_episode_ids=episodes, starts=starts, measured_h5_rows=rows,
        measured_frames=128, selection_seed=1042, encoder_forward_calls=calls, batch_size=8,
        exposure='One initial frame per original selected training episode; no physical labels or future frames.',
        device=str(device), attention_backend=backend, matmul_precision=torch.get_float32_matmul_precision(),
        allow_tf32=torch.backends.cuda.matmul.allow_tf32, input_precision='float32',
        encoder_initial_sha256=encoder_hash, original64_query_pool_sha256=pool_hash,
        original_carrier_projection_sha256=projection_hash, initial_state_sha256=initial_hash,
        state_unchanged=True, branches=measured, model_kwargs=model_kwargs,
        values_file=str(values_path), values_sha256=file_hash(values_path),
        old_fixed64_comparison=dict(old_normalization='per-coordinate diagonal scale', new_normalization='one scalar RMS',
            old_scale_floor_coordinates=reference['scale_floor_coordinates'],
            old_scales_rms=float(old_scale.double().square().mean().sqrt()),
            new_mean_minus_old_max=float((new_mean-old_mean).abs().max()),
            new_scalar_over_old_coordinate_scale_min=float((model_kwargs['hybrid_fixed_scale']/old_scale).min()),
            new_scalar_over_old_coordinate_scale_max=float((model_kwargs['hybrid_fixed_scale']/old_scale).max()),
            interpretation='Same raw projection/information, changed metric; numerical means may differ by execution precision.'),
        parameter_counts=model.hybrid_parameter_counts())
    with out.open('x') as f:
        f.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out), frames=128, encoder_calls=calls,
        fixed_rms=model_kwargs['hybrid_fixed_scale'], query_rms=model_kwargs['hybrid_query_scale'])), flush=True)


if __name__ == '__main__':
    main()
