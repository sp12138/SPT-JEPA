"""CPU-only head audit of completed gates; no encoder/predictor/data access."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


PREFIX = 'dynamic_readout_module.'
NAMES = ['input.weight', 'input.bias', 'output.weight', 'output.bias']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def head_matrix(states, head, mode):
    """Literal frozen FP32 head formula, independent of all model imports."""
    slots = states.float().reshape(*states.shape[:-1], 64, 16)
    if mode == 'static':
        slots = torch.zeros_like(slots)
    axis = torch.linspace(-1., 1., 8, dtype=torch.float32)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    pos = torch.stack([x, y], -1).reshape(64, 2)
    pos = pos.expand(*slots.shape[:-2], 64, 2)
    hidden = torch.tanh(torch.nn.functional.linear(
        torch.cat([slots, pos], -1), head['input.weight'], head['input.bias']))
    return torch.nn.functional.linear(hidden, head['output.weight'], head['output.bias'])


def measure(states, head, mode):
    assert states.ndim == 3 and states.shape[-1] == 1024
    # The head is deterministic and pointwise. Batch only to cap CPU memory.
    matrices = torch.cat([head_matrix(x, head, mode) for x in states.split(8)]).double()
    fixed_zero = head_matrix(torch.zeros(1, 1, 1024), head, mode).double()
    energy = matrices.square().mean()
    zero_departure = (matrices - fixed_zero).square().mean()
    same_position_mean = matrices.mean((0, 1), keepdim=True)
    position_variance = (matrices - same_position_mean).square().mean()
    if mode == 'static':
        assert zero_departure == 0 and position_variance < 1e-24
    assert torch.isfinite(matrices).all()
    return dict(
        states=list(states.shape), matrix_entry_rms=float(energy.sqrt()),
        fixed_zero_state_entry_rms=float(fixed_zero.square().mean().sqrt()),
        departure_from_fixed_zero_state_rms=float(zero_departure.sqrt()),
        departure_from_fixed_zero_state_energy_over_total=float(zero_departure / energy) if energy else None,
        same_position_across_states_variance=float(position_variance),
        same_position_variance_over_total=float(position_variance / energy) if energy else None,
        per_time_matrix_entry_rms=matrices.square().mean((0, 2, 3)).sqrt().tolist(),
        per_time_zero_state_departure_rms=(matrices - fixed_zero).square().mean((0, 2, 3)).sqrt().tolist(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--recovery-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    plan = read(args.root / 'plan.json')
    assert len(plan['jobs']) == 40
    # The original parent died before collecting any OS exit; do not invent0.
    assert not list((args.root / 'exits').glob('*.json'))
    recovery_plan = read(args.recovery_root / 'plan.json')
    recovery_protocol = read(args.recovery_root / 'recovery-protocol.json')
    assert recovery_protocol['original_root'] == str(args.root)
    assert recovery_protocol['original_plan_sha256'] == sha(args.root / 'plan.json')
    assert len(recovery_plan['jobs']) == 32
    original_by_id = {job['id']: job for job in plan['jobs']}
    for job in recovery_plan['jobs']:
        assert job['phase'] != 'train600'
        assert read(args.recovery_root / 'exits' / (job['id'] + '.json'))['returncode'] == 0
        original = original_by_id[job['id']]
        assert job['source'] == original['source']
        expected_command = original['command'].copy()
        expected_command[expected_command.index('--output') + 1] = job['output']
        assert job['command'] == expected_command
    rows = []
    for job in plan['jobs']:
        if job['phase'] != 'train600':
            continue
        run = Path(job['output'])
        cfg = read(run / 'config.json')
        stage = read(run / 'stage-complete.json')
        assert stage['stage_complete'] and stage['stop_step'] == 600
        assert stage['config_sha256'] == job['config_sha256'] == sha(run / 'config.json')
        fullstate = read(args.recovery_root / ('fullstate-' + run.name) / 'preflight.json')
        assert fullstate['passed'] and fullstate['checkpoint']['step'] == 600
        assert all(fullstate['checkpoint']['checks'].values())
        assert fullstate['config_sha256'] == sha(run / 'config.json')
        readout_audit = read(args.recovery_root / ('fullstate-' + run.name) / 'readout-audit.json')
        assert readout_audit['passed'] and readout_audit['original_initial_tensors_exact'] == 553
        assert readout_audit['frozen_tensors_unchanged'] == 210
        assert cfg['model'].get('spatial_chart', 'none') == 'none'
        mode = cfg['model']['spatial_readout_mode']
        assert mode in {'static', 'state'}
        folder = args.recovery_root / ('transition-' + run.name)
        report = read(folder / 'state-audit.json')
        weights_path = run / 'model-step0000600/weights.pt'
        cache_path = folder / 'state-audit.npz'
        weights_hash = sha(weights_path)
        assert weights_hash == fullstate['weights_sha256']
        assert report['weights_sha256'] == weights_hash and report['step'] == 600
        assert report['config_sha256'] == fullstate['config_sha256'] and report['run'] == str(run)
        weights = torch.load(weights_path, map_location='cpu', weights_only=True, mmap=True)
        head = {key: weights[PREFIX + key].float() for key in NAMES}
        assert sum(x.numel() for x in head.values()) == 5376
        with np.load(cache_path) as cache:
            observed = torch.from_numpy(cache['observed'].copy()).float()
            clean = torch.from_numpy(cache['clean_rollout'].copy()).float()
        assert observed.shape == (64, 8, 1024) and clean.shape == (64, 7, 1024)
        prefix_states = torch.cat([observed[:, :1], clean[:, :-1]], 1)
        with torch.no_grad():
            observed_result = measure(observed, head, mode)
            causal_result = measure(prefix_states, head, mode)
        rows.append(dict(
            id=run.name, mode=mode, observed=observed_result, clean_prefix=causal_result,
            training_os_exit_code=None, training_stage_complete=True, fullstate600_verified=True,
            source=job['source'], weights_sha256=weights_hash, cache_sha256=sha(cache_path),
            state_audit_sha256=sha(folder / 'state-audit.json'), manifest_sha256=report['manifest_sha256'],
            correction_over_original_increment=None,
            missing_quantity='Cache stores only the final corrected prediction, not the original p or delta=p-u. No inversion or surrogate correction is reported.',
        ))
    assert len(rows) == 8 and not torch.cuda.is_initialized()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        rows=rows, helper_sha256=sha(__file__), plan_sha256=sha(args.root / 'plan.json'),
        recovery_plan_sha256=sha(args.recovery_root / 'plan.json'),
        completion=dict(training_stages_complete=8, training_os_exits_unknown=8, recovery_endpoints_exit0=32),
        scope='Eight completed600 stages with unknown OS exit plus32 actual recovery exits0 and8 successful600 full-state audits; existing64 windows and saved4 head tensors only. No model/encoder/predictor invocation, labels, new data, fitting, threshold selection or GPU.',
        interpretation='Nonzero matrix output measures activation; state-dependent matrix variation does not establish useful action directions or planning gain. M(0,pos) is a fixed-position head reference, not necessarily an in-distribution state. Same-position variance uses only the existing cached cohort and is descriptive.',
        correction_unavailable=True,
    )
    (args.output / 'head-mechanism.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.output), runs=len(rows), cuda_initialized=False)))


if __name__ == '__main__':
    main()
