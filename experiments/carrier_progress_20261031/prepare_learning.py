"""Prepare bounded Carrier continuations; never launch workers or replace training.

Preparation performs read-only CPU audits before copying full native checkpoints.
The same file supplies CPU checkpoint archival jobs between the two native caps.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import shutil
import sys
import time
from pathlib import Path


B = Path('/efs/prescottsun/SPT-JEPA')
PARENT = B / 'runs/ideas_validate600_20261031'
ROOT = B / 'runs/carrier_learn2000_20261031'
PYTHON = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'
TASKS = ('tworoom', 'cube')
ARMS = ('identity', 'unconstrained', 'bounded')
OPERATIONAL = {'output', 'checkpoint_every', 'retain_steps', 'is_resuming'}
CHECKPOINT_METADATA = {'slurm.job_id', 'slurm.task_id'}
SELF = 'experiments/carrier_progress_20261031/prepare_learning.py'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def semantic(cfg):
    return {k: v for k, v in cfg.items() if k not in OPERATIONAL}


def checkpoint_semantic(hparams):
    # Lightning appends these execution annotations to saved hyperparameters.
    return semantic({k: v for k, v in hparams.items() if k not in CHECKPOINT_METADATA})


def rng_digest(value):
    def plain(v):
        if hasattr(v, 'tolist'):
            return v.tolist()
        if isinstance(v, dict):
            return {k: plain(x) for k, x in v.items()}
        if isinstance(v, (tuple, list)):
            return [plain(x) for x in v]
        return v
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def assignments():
    for slot, (task, seed, arm) in enumerate(itertools.product(TASKS, (1, 2), ARMS)):
        yield dict(task=task, training_seed=seed, arm=arm, family='carrier',
                   host='dev2' if slot < 8 else 'dev13', gpu=slot if slot < 8 else slot - 8,
                   id=f'{task}-carrier-{arm}-seed{seed}')


def continuation_config(cfg, output):
    result = copy.deepcopy(cfg)
    result.update(output=str(output), checkpoint_every=1200, retain_steps=[1200, 2000])
    assert semantic(result) == semantic(cfg)
    assert result['model_class'] == 'carrier' and result['max_steps'] == 124910
    assert result['batch_size'] == 128 and result.get('workers', 4) == 4
    return result


def activate_source(src):
    # All heavy imports below must resolve against the supplied immutable source.
    sys.path[:0] = src['pythonpath'].split(':')


def verify_sources(src, old):
    newroot, oldroot = Path(src['source']), Path(old['source'])
    newmanifest, oldmanifest = read(src['manifest']), read(old['manifest'])
    assert newmanifest['source_commit'] == src['source_commit']
    assert oldmanifest['source_commit'] == old['source_commit']
    assert src['runtime_source'] == old['runtime_source']
    assert src['runtime_manifest'] == old['runtime_manifest']
    assert src['pythonpath'] == old['pythonpath'].replace(str(oldroot), str(newroot))
    required = [name for name in oldmanifest['files']
                if name.startswith(('as_jepa_lewm/', 'as_jepa_tools/')) and name.endswith('.py')]
    required += ['experiments/ideas_validate_20261031/audit.py',
                 'experiments/ideas_validate_20261031/preflight.py',
                 'experiments/ideas_validate_20261031/audit-windows-tworoom.json',
                 'experiments/ideas_validate_20261031/audit-windows-cube.json']
    records = {}
    for name in sorted(required):
        expected = oldmanifest['files'][name]
        assert newmanifest['files'][name] == expected, f'Resume-critical source changed: {name}'
        assert digest(oldroot / name) == digest(newroot / name) == expected, name
        records[name] = expected
    assert digest(Path(__file__)) == digest(newroot / SELF) == newmanifest['files'][SELF]
    return dict(old_source=old, new_source=src, identical_files=records,
                old_manifest_sha256=digest(old['manifest']), new_manifest_sha256=digest(src['manifest']),
                shared_runtime_manifest_sha256=digest(src['runtime_manifest']))


def state_audit(run, expected_step, parent):
    """Reuse the established full-state auditor; no model forward or GPU access."""
    import torch
    from experiments.ideas_validate_20261031.preflight import checkpoint_report, compare_tensors
    from as_jepa_lewm.train import WindowSampler
    from as_jepa_tools.benchmark_data import make_training_dataset

    torch.set_num_threads(2)
    run, parent = Path(run), Path(parent)
    cfg, original = read(run / 'config.json'), read(parent / 'config.json')
    assert semantic(cfg) == semantic(original), 'Training semantics changed'
    checkpoint_path = run / 'last.ckpt'
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    portable_path = run / f'model-step{expected_step:07d}/weights.pt'
    portable = torch.load(portable_path, map_location='cpu', weights_only=True)
    report = checkpoint_report(state, portable, cfg, expected_step)
    report['checks']['complete_model_keys'] = set(state['state_dict']) == {'model.' + k for k in portable}
    report['checks']['saved_training_semantics'] = checkpoint_semantic(state['hyper_parameters']) == semantic(cfg)
    report['checks']['scheduler_step'] = all(s['last_epoch'] == expected_step for s in state['lr_schedulers'])
    report['checks']['optimizer_scheduler_lr_equal'] = (
        [g['lr'] for g in state['optimizer_states'][0]['param_groups']]
        == state['lr_schedulers'][0]['_last_lr'])
    initial_path = parent / 'initial-model/weights.pt'
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    frozen = ['encoder', 'pool', 'carrier_projection', 'carrier_mean', 'carrier_scale']
    if cfg['model']['carrier_chart'] == 'identity':
        frozen.append('carrier_chart_module')
    assert all(any(k == prefix or k.startswith(prefix + '.') for k in portable) for prefix in frozen)
    report['frozen_tensors'] = compare_tensors(initial, portable, frozen)
    report['checks']['frozen_unchanged'] = report['frozen_tensors']['equal']
    dataset = make_training_dataset(cfg)
    cursor = state['campaign_samples']
    kwargs = dict(seed=cfg['seed'], batch_size=cfg['batch_size'])
    resumed = list(itertools.islice(iter(WindowSampler(dataset, start=cursor, **kwargs)), 32))
    # Same stateless sampler contract, compared to its uninterrupted sequence.
    uninterrupted = list(itertools.islice(iter(WindowSampler(dataset, start=0, **kwargs)), cursor, cursor + 32))
    report['checks']['sampler_cursor_sequence'] = resumed == uninterrupted and len(resumed) == 32
    report.update(run=str(run), parent=str(parent), config_sha256=digest(run / 'config.json'),
        checkpoint_sha256=digest(checkpoint_path), portable_sha256=digest(portable_path),
        initial_weights_sha256=digest(initial_path), next_32_sampler_indices=resumed,
        dataset_natural_length=dataset.natural_length,
        learning_rates=[g['lr'] for g in state['optimizer_states'][0]['param_groups']],
        scheduler_last_epoch=[s['last_epoch'] for s in state['lr_schedulers']],
        saved_runtime_metadata={k: state['hyper_parameters'][k] for k in CHECKPOINT_METADATA
                                if k in state['hyper_parameters']},
        rng_component_sha256={k: rng_digest(v)
                              for k, v in state['campaign_rng'].items()},
        passed=all(report['checks'].values()),
        scope='CPU-only saved-state audit. Full checkpoint bytes preserve native optimizer, scheduler, '
              'RNG and sampler cursor. No assertion of bitwise resumed-versus-uninterrupted GPU trajectories.')
    assert report['passed'], report
    return report


def archive_state(args):
    src = read(args.source_record)
    activate_source(src)
    out = Path(args.output)
    # The native dispatcher creates the log directory before executing the job.
    assert not (out / 'fullstate.json').exists(), out
    report = state_audit(args.audit_run, args.expected_step, args.parent_run)
    stage = read(Path(args.audit_run) / 'stage-complete.json')
    assert stage['stage_complete'] and stage['stop_step'] == args.expected_step
    archive = Path(args.audit_run) / f'checkpoint-step{args.expected_step:07d}.ckpt'
    assert not archive.exists(), archive
    shutil.copy2(Path(args.audit_run) / 'last.ckpt', archive)
    assert digest(archive) == report['checkpoint_sha256']
    report.update(archived_checkpoint=str(archive), source_commit=src['source_commit'],
                  source_record_sha256=digest(args.source_record))
    report['native_stage_complete'] = stage
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'fullstate.json', report)
    print(json.dumps(dict(passed=True, step=args.expected_step, archive=str(archive))), flush=True)


def prepare(args):
    root, parent = Path(args.root), Path(args.parent)
    assert not root.exists(), f'Refusing to reuse stage output: {root}'
    src, oldsrc = read(args.source_record), read(parent / 'source-record.json')
    compatibility = verify_sources(src, oldsrc)
    activate_source(src)
    oldplan, review = read(parent / 'plan.json'), read(parent / 'review600.json')
    assert review['all_exit_zero'] and review['all_initial_original550_equal'] and review['all_full_state_health_pass']
    oldjobs = {j['id']: j for j in oldplan['jobs'] if j.get('phase') == 'train600'}
    records = []
    for allocation in assignments():
        name = allocation['id']; oldjob = oldjobs[name]; previous = Path(oldjob['output'])
        assert oldjob['family'] == 'carrier' and oldjob['arm'] == allocation['arm']
        stage = read(previous / 'stage-complete.json')
        assert stage['stage_complete'] and stage['stop_step'] == 600
        for phase in ('', 'health-', 'info-', 'transition-', 'planning-'):
            assert read(parent / 'exits' / (phase + name + '.json'))['returncode'] == 0
        cfg = read(previous / 'config.json')
        assert digest(previous / 'config.json') == oldjob['config_sha256']
        assert digest(cfg['carrier_calibration']['path']) == cfg['carrier_calibration']['sha256']
        pre = state_audit(previous, 600, previous)
        records.append(dict(allocation=allocation, parent=str(previous),
                            config=continuation_config(cfg, root / name), pre_audit=pre))
    assert len(records) == 12
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, prepared=False, fullstate_passed=12,
                              identical_source_files=len(compatibility['identical_files']),
                              allocation=[r['allocation'] for r in records]), indent=2))
        return
    root.mkdir(parents=True)
    source_record = root / 'source-record.json'; write(source_record, src)
    write(root / 'source-compatibility.json', compatibility)
    jobs, copies = [], []
    helper = str(Path(src['source']) / SELF)
    for record in records:
        common = dict(record['allocation'], source=src)
        name = common.pop('id'); previous = Path(record['parent']); out = root / name; out.mkdir()
        config = out / 'config.json'; write(config, record['config'])
        shutil.copy2(previous / 'last.ckpt', out / 'last.ckpt')
        assert digest(out / 'last.ckpt') == record['pre_audit']['checkpoint_sha256']
        pre = dict(record['pre_audit'], copied_to=str(out / 'last.ckpt'), copy_byte_identical=True,
                   source_commit=src['source_commit'])
        write(out / 'pre-resume-fullstate.json', pre)
        copies.append(dict(id=name, parent=str(previous), checkpoint_sha256=pre['checkpoint_sha256']))
        dependency = []
        for step in (1200, 2000):
            trainid = f'learn{step}-{name}'
            jobs.append(dict(**common, id=trainid, output=str(out), config=str(config),
                config_sha256=digest(config), phase=f'learn{step}', depends_on=dependency,
                command=[PYTHON, '-B', '-u', '-m', 'as_jepa_tools.train_stage', '--config', str(config),
                         '--stop-step', str(step), '--resume']))
            dest = root / f'fullstate{step}-{name}'
            jobs.append(dict(**common, id=dest.name, output=str(dest), phase=f'fullstate{step}',
                depends_on=[trainid], command=[PYTHON, '-B', '-u', helper, '--source-record', str(source_record),
                    '--audit-run', str(out), '--parent-run', str(previous), '--expected-step', str(step),
                    '--output', str(dest)]))
            dependency = [dest.name]
        for phase, module in [('health', 'as_jepa_tools.benchmark_health'),
                              ('info', 'as_jepa_tools.benchmark_information_audit')]:
            dest = root / (phase + '-' + name)
            jobs.append(dict(**common, id=dest.name, output=str(dest), depends_on=dependency, phase=phase,
                command=[PYTHON, '-B', '-u', '-m', module, '--run', str(out), '--output', str(dest)]))
            dependency = [dest.name]
        dest = root / ('transition-' + name)
        jobs.append(dict(**common, id=dest.name, output=str(dest), depends_on=dependency, phase='transition',
            command=[PYTHON, '-B', '-u', str(Path(src['source']) / 'experiments/ideas_validate_20261031/audit.py'),
                '--run', str(out), '--output', str(dest), '--cases', '64', '--manifest',
                str(Path(src['source']) / f'experiments/ideas_validate_20261031/audit-windows-{common["task"]}.json')]))
        dependency = [dest.name]; dest = root / ('planning-' + name); data = Path(record['config']['data'])
        jobs.append(dict(**common, id=dest.name, output=str(dest), depends_on=dependency, phase='planning',
            environment={'MUJOCO_EGL_DEVICE_ID': str(common['gpu'])},
            command=[PYTHON, '-B', '-u', '-m', 'as_jepa_tools.benchmark_evaluate', '--manifest',
                str(data / 'manifests/validation-strict.json'), '--checkpoint', str(out / 'model-step0002000'),
                '--data', str(data), '--output', str(dest)]))
    assert len(jobs) == 96 and len({j['id'] for j in jobs}) == 96
    write(root / 'plan.json', dict(root=str(root), deadline=None, environment=oldplan['environment'], jobs=jobs))
    write(root / 'resume-audit.json', copies)
    write(root / 'protocol.json', dict(created_at=time.time(), root=str(root), parent=str(parent),
        source_commit=src['source_commit'], old_source_commit=oldsrc['source_commit'],
        initial_step=600, intermediate_step=1200, stop_step=2000, runs=12, jobs=96,
        allocations=list(assignments()), scheduler_horizon=124910, warmup_steps=1249,
        native_fullstate_archives=[1200, 2000], semantic_config_unchanged=True,
        original_calibration_unchanged=True, validation_cases=50, transition_windows=64,
        purpose='Bounded beyond-warmup test of all reviewed Carrier arms and seeds; factual dynamics '
                'at600 motivates the test, not an established two-task improvement or chart-bound mechanism.',
        authorization='New explicit persistent-research instruction; separate named stage preserves all old holds.',
        departures=['Fresh output and newly committed source with identical model/training/data/evaluator bytes.',
                    'Common native resume boundary at1200 permits safe full-state archival without trainer edits.',
                    'Operational checkpoint cadence changes from200 to1200; retain portable1200/2000.',
                    'New host/GPU allocation. Native high-training/highest-evaluation precision remains unchanged.'],
        stop_policy='Stop at2000. A failed audit/archive blocks the next segment. No automatic step beyond2000.',
        state_scope='Stored state and stateless sampler sequence verified; bitwise identity to an uninterrupted '
                    'GPU trajectory is not asserted. Worker RNG is not serialized, but BenchmarkWindows has '
                    'deterministic samples and no random augmentation.'))
    print(json.dumps(dict(root=str(root), runs=12, jobs=len(jobs), stop_step=2000)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-record', required=True)
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--parent', default=str(PARENT))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--audit-run')
    parser.add_argument('--parent-run')
    parser.add_argument('--expected-step', type=int, choices=[1200, 2000])
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.audit_run:
        assert args.parent_run and args.expected_step and args.output and not args.dry_run
        archive_state(args)
    else:
        prepare(args)


if __name__ == '__main__':
    main()
