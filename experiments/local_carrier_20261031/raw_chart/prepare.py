"""Bounded raw-interface preflight and candidate-only 600-step plans."""
import argparse
import itertools
from pathlib import Path

from experiments.local_carrier_20261031 import prepare_spatial_chart as old

B, PY, OLD = old.B, old.PY, old.OLD
read, sha, write = old.read, old.sha, old.write
PRE = B/'runs/local_raw_chart_preflight_20261031'
ROOT = B/'runs/local_raw_chart600_20261031'
SELF = 'experiments/local_carrier_20261031/raw_chart/prepare.py'
TEST = 'experiments/local_carrier_20261031/raw_chart/test_raw_chart.py'
PREFIX = 'carrier_chart_module.'


def source(path):
    s = read(path); manifest = read(s['manifest'])
    assert s['source_commit'] == manifest['source_commit']
    for name, digest in manifest['files'].items(): assert sha(Path(s['source'])/name) == digest, name
    assert sha(__file__) == manifest['files'][SELF]
    folder = Path(s['source'])/'experiments/local_carrier_20261031/raw_chart'
    proof, legacy = read(folder/'contracts.json'), read(folder/'legacy-contracts.json')
    assert proof['passed'] and proof['tests'] == 6 and proof['device'] == 'CPU'
    assert legacy['passed'] and legacy['tests'] == 4 and legacy['device'] == 'CPU'
    for name, digest in proof['source_sha256'].items():
        assert manifest['files']['as_jepa_lewm/'+name+'.py'] == digest
    assert manifest['files'][TEST] == proof['test_sha256']
    assert manifest['files']['experiments/local_carrier_20261031/test_spatial_chart.py'] == legacy['test_source_sha256']
    return s


def allocations():
    for gpu, (task, seed) in enumerate(itertools.product(('tworoom', 'cube'), (1, 2))):
        yield dict(id=f'{task}-raw-chart-bounded-seed{seed}', task=task, training_seed=seed,
                   host='dev13', gpu=gpu, family='spatial_carrier', arm='local-raw-bounded')


def config(alloc, output, step):
    cfg, reference = old.config(alloc, output, step)
    cfg['model']['spatial_state_interface'] = 'raw'
    return cfg, reference


def prepare(args):
    s = source(args.source_record)
    is_preflight = args.phase == 'preflight'
    root, step = (PRE, 2) if is_preflight else (ROOT, 600)
    assert not root.exists(), 'Never overwrite a historical root'
    if not is_preflight:
        assert read(PRE/'review.json')['passed'] and read(PRE/'source-record.json') == s
    root.mkdir(); write(root/'source-record.json', s)
    jobs, runs = [], []
    for alloc in allocations():
        name = alloc['id']; output = root/name
        cfg, reference = config(alloc, output, step); write(output/'config.json', cfg)
        common = {k: v for k, v in alloc.items() if k != 'id'}; common['source'] = s
        previous = []
        def add(phase, dest, command, **extra):
            nonlocal previous
            ident = phase+'-'+name
            jobs.append(dict(**common, id=ident, phase=phase, output=str(dest), depends_on=previous,
                             command=command, **extra)); previous = [ident]
        add('train'+str(step), output,
            [PY, '-B', '-u', '-m', 'as_jepa_tools.train_stage', '--config', str(output/'config.json'), '--stop-step', str(step)],
            config=str(output/'config.json'), config_sha256=sha(output/'config.json'))
        dest = root/('fullstate-'+name)
        add('fullstate', dest, [PY, '-B', '-u', str(Path(s['source'])/'experiments/ideas_validate_20261031/preflight.py'),
            '--run', str(output), '--output', str(dest), '--expected-step', str(step),
            '--reference-initial', str(reference/'initial-model/weights.pt')])
        if not is_preflight:
            dest = root/('health-'+name)
            add('health', dest, [PY, '-B', '-u', '-m', 'as_jepa_tools.benchmark_health', '--run', str(output), '--output', str(dest)])
            dest = root/('transition-'+name)
            add('transition', dest, [PY, '-B', '-u', str(Path(s['source'])/'experiments/ideas_validate_20261031/audit.py'),
                '--run', str(output), '--output', str(dest), '--cases', '64', '--manifest',
                str(Path(s['source'])/f'experiments/ideas_validate_20261031/audit-windows-{alloc["task"]}.json')])
            dest = root/('planning-'+name); manifest = Path(cfg['data'])/'manifests/validation-strict.json'
            add('planning', dest, [PY, '-B', '-u', '-m', 'as_jepa_tools.benchmark_evaluate', '--manifest', str(manifest),
                '--checkpoint', str(output/'model-step0000600'), '--data', cfg['data'], '--output', str(dest)],
                manifest_sha256=sha(manifest), environment=dict(MUJOCO_EGL_DEVICE_ID=str(alloc['gpu'])))
        runs.append(dict(**alloc, reference=str(reference), calibration=cfg['spatial_calibration']))
    assert len(jobs) == (8 if is_preflight else 20)
    write(root/'candidate-plan.json', dict(root=str(root), deadline=None, environment=read(OLD/'plan.json')['environment'], jobs=jobs))
    write(root/'protocol.json', dict(source=s, step=step, runs=runs, candidate_only=True,
        intervention='Fixed local carrier; bounded shared16->64->16 chart inside predictor and renderer, raw public encode/predict/goal interface. Native sample-level loss and AR noise unchanged.',
        cached_state_coordinates='RAW: do NOT inverse-transform observed/predicted cached states in collection.',
        comparator='Original local600 primary50, exact identity compatibility proved. No fresh600 cohort.',
        parameter_departure=dict(extra_total=2128, extra_trainable=2128, original_tensors=553, extra_tensors=4),
        unchanged='Original data/calibration/sampler/native Trainer/optimizer/schedule/CEM, batch128; historical holds and snapshots untouched.',
        numerical_policy='Certified-residual inverse and adjoint fail closed; never loosen tolerances after a failure. Actual GPU precision/memory/throughput reviewed before600.',
        authorization='Four production two-step native preflights explicitly approved. A600 candidate is not launch authorization; root must review all8 actual preflight endpoints. No automatic continuation beyond600.'))
    assert not (root/'plan.json').exists()
    print(dict(root=str(root), endpoints=len(jobs), activated=False))


def review(args):
    import torch
    from as_jepa_lewm.carrier import ResidualCarrierChart
    torch.set_num_threads(2)
    s = source(args.source_record); assert read(PRE/'source-record.json') == s
    plan = read(PRE/'candidate-plan.json'); assert len(plan['jobs']) == 8
    for job in plan['jobs']: assert read(PRE/'exits'/(job['id']+'.json'))['returncode'] == 0
    rows, common_chart = [], None
    for alloc in allocations():
        name = alloc['id']; run = PRE/name; cfg, reference = config(alloc, run, 2)
        assert cfg == read(run/'config.json')
        report_path = PRE/('fullstate-'+name)/'preflight.json'; report = read(report_path)
        assert report['passed'] and report['optimizer_parameter_count']['passed']
        assert all(report['checkpoint']['checks'].values())
        assert report['frozen_tensors_unchanged']['equal'] and report['frozen_tensors_unchanged']['compared_tensors'] == 210
        assert report['gradients']['carrier_chart_module']['passed'] and report['gradients']['carrier_chart_module']['gradient_norm'] > 0
        original = torch.load(reference/'initial-model/weights.pt', map_location='cpu', weights_only=True)
        initial = torch.load(run/'initial-model/weights.pt', map_location='cpu', weights_only=True)
        final = torch.load(run/'model-step0000002/weights.pt', map_location='cpu', weights_only=True)
        extra = {PREFIX+k for k in ['input.weight', 'input.bias', 'output.weight', 'output.bias']}
        assert len(original) == 553 and set(initial) == set(original)|extra and set(final) == set(initial)
        assert all(torch.equal(value, initial[key]) for key, value in original.items())
        initial_chart = {key.removeprefix(PREFIX): initial[key] for key in extra}
        assert torch.count_nonzero(initial_chart['output.weight']) == torch.count_nonzero(initial_chart['output.bias']) == 0
        if common_chart is None: common_chart = initial_chart
        else: assert all(torch.equal(value, common_chart[key]) for key, value in initial_chart.items())
        chart = ResidualCarrierChart(16, 64, 'bounded', .5)
        chart.load_state_dict({key.removeprefix(PREFIX): final[key] for key in extra}, strict=True)
        certificate = chart.norm_certificate(); assert certificate['residual_lipschitz_upper'] <= .500001
        assert any(not torch.equal(initial[key], final[key]) for key in extra)
        # Additional actual-checkpoint CPU inverse check. Actual native GPU
        # loss/backward already required every inverse/adjoint certificate.
        x = torch.randn(4, 64, 16, generator=torch.Generator().manual_seed(1042), requires_grad=True)
        transformed = chart(x); recovered = chart.differentiable_inverse(transformed)
        error = float((recovered-x).detach().abs().max())
        gradients = torch.autograd.grad(recovered.square().mean(), [x]+list(chart.parameters()))
        assert all(bool(torch.isfinite(value).all()) for value in gradients)
        bound = (chart(recovered)-transformed).detach().norm(dim=-1)/.5
        assert bool((bound <= 1e-6+1e-6*transformed.detach().norm(dim=-1)).all())
        status = read(run/'status.json'); assert status['state'] == 'finished' and status['step'] == 2
        assert 0 < status['peak_gib'] < 140 and status['seconds_per_step'] > 0
        size, oldsize = read(run/'model_size.json'), read(reference/'model_size.json')
        assert size['total']-oldsize['total'] == size['trainable']-oldsize['trainable'] == 2128
        rows.append(dict(id=name, original_initial_tensors_exact=553, extra_chart_tensors=4, chart_parameters=2128,
            frozen_tensors_unchanged=210, optimizer=report['optimizer_parameter_count'], gradients=report['gradients'],
            certificate=certificate, cpu_actual_checkpoint_inverse=dict(roundtrip_max_abs=error, measured_solution_error_bound_max=float(bound.max()), all_gradients_finite=True),
            gpu_native_inverse_scope='Actual batch128 two-step training and held-out native backward passed the unchanged fail-closed inverse/adjoint residual checks.',
            peak_gib=status['peak_gib'], seconds_per_step=status['seconds_per_step'], elapsed_seconds=status['elapsed_seconds'], learning_rates=status['learning_rates'],
            fullstate_report_sha256=sha(report_path), initial_sha256=sha(run/'initial-model/weights.pt'),
            reference_initial_sha256=sha(reference/'initial-model/weights.pt'), passed=True))
        del original, initial, final, chart
    write(PRE/'review.json', dict(passed=True, source=s, rows=rows,
        scope='Four real batch128 native two-update runs and actual checkpoint/gradient/inverse checks. No600 launch authorization or performance claim.'))
    print(dict(passed=True, runs=4, peak_gib=[row['peak_gib'] for row in rows]))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['preflight', 'review', 'training'])
    p.add_argument('--source-record', required=True); args = p.parse_args()
    review(args) if args.phase == 'review' else prepare(args)
