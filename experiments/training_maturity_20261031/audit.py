"""Actual CPU native restore; one process per original immutable model source."""
import argparse
import importlib
import shutil
import sys
from pathlib import Path
from common import B, CAP, FAMILIES, read, sha, write, semantic, source_check, stages


def equal(a, b):
    import torch
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(equal(v, w) for v, w in zip(a, b))
    return a == b


def main():
    p = argparse.ArgumentParser()
    for key in ['run', 'parent', 'initial', 'source-record', 'family', 'output']: p.add_argument('--'+key, required=True)
    p.add_argument('--step', required=True, type=int); p.add_argument('--portable')
    p.add_argument('--archive', action='store_true'); p.add_argument('--previous-step', type=int)
    args = p.parse_args(); source = read(args.source_record)
    assert args.family in FAMILIES and source['source_commit'] == FAMILIES[args.family]['commit']
    assert args.step in [FAMILIES[args.family]['step'], *stages(args.family)]
    source_proof = source_check(source)
    sys.path[:0] = source['pythonpath'].split(':')
    # Native7a intentionally contains only its model/evaluator package. Reuse
    # the previously audited093 CPU sampler primitives AFTER original sources.
    sampler_source = read(B/'runs/local_isotropic2000_20261031/helper-source-record.json')
    assert sampler_source['source_commit'] == '0934d3c445e6c754a6ab547d801890c34740b330'
    sampler_manifest = read(sampler_source['manifest'])
    for rel in ['experiments/carrier_progress_20261031/prepare_learning.py', 'experiments/carrier_progress_20261031/prepare_next.py']:
        assert sha(Path(sampler_source['source'])/rel) == sampler_manifest['files'][rel]
    sys.path.append(sampler_source['source'])
    import torch
    from as_jepa_lewm.train import TrainModule
    from as_jepa_lewm.factory import gradient_expectations
    from as_jepa_tools.benchmark_data import make_training_dataset
    from experiments.ideas_validate_20261031.preflight import checkpoint_report
    from experiments.carrier_progress_20261031.prepare_next import resume_sequence
    from experiments.carrier_progress_20261031.prepare_learning import rng_digest
    from vforge.ops import OpsConfig, initialize_ops
    torch.set_num_threads(2)
    initialize_ops(OpsConfig(backend='sdpa_math', allow_fallback=False), device=torch.device('cpu'), parallel_state=None)
    modules = ['as_jepa_lewm.train', 'as_jepa_lewm.factory', 'as_jepa_tools.benchmark_data',
               'experiments.ideas_validate_20261031.preflight']
    for module in modules: assert Path(importlib.import_module(module).__file__).is_relative_to(Path(source['source'])), module
    sampler_file = Path(importlib.import_module(resume_sequence.__module__).__file__)
    expected_sampler = Path(source['source'])/'experiments/carrier_progress_20261031/prepare_next.py'
    if not expected_sampler.exists(): expected_sampler = Path(sampler_source['source'])/'experiments/carrier_progress_20261031/prepare_next.py'
    assert sampler_file == expected_sampler
    run, parent = Path(args.run), Path(args.parent)
    cfg = read(run/'config.json'); assert semantic(cfg) == semantic(read(parent/'config.json'))
    assert cfg['batch_size'] == 128 and cfg['max_steps'] == 124910 and cfg['model']['window'] == 8
    expected = {'local': 'spatial_carrier', 'native': 'native_lewm', 'tf': None}[args.family]
    assert cfg.get('model_class') == expected
    checkpoint = torch.load(run/'last.ckpt', map_location='cpu', weights_only=False)
    pp = Path(args.portable) if args.portable else run/f'model-step{args.step:07d}/weights.pt'
    portable = torch.load(pp, map_location='cpu', weights_only=True)
    report = checkpoint_report(checkpoint, portable, cfg, args.step); checks = report['checks']
    checks['complete_model_keys'] = set(checkpoint['state_dict']) == {'model.'+k for k in portable}
    checks['saved_training_semantics'] = semantic(checkpoint['hyper_parameters']) == semantic(cfg)
    module = TrainModule(cfg); module.load_state_dict(checkpoint['state_dict'], strict=True)
    checks['actual_native_model_load'] = True
    oc = module.configure_optimizers(); opt = oc['optimizer']; scheduler = oc['lr_scheduler']['scheduler']
    named = {id(v): k for k, v in module.model.named_parameters()}
    current = [[named[id(v)] for v in g['params']] for g in opt.param_groups]
    trainable = {k for k, v in module.model.named_parameters() if v.requires_grad}
    saved_order = [k.removeprefix('model.') for k in checkpoint['state_dict'] if k.removeprefix('model.') in trainable]
    expected_order = [saved_order] if args.family == 'native' else [
        [k for k in saved_order if not k.startswith('renderer.')], [k for k in saved_order if k.startswith('renderer.')]]
    checks['optimizer_parameter_order'] = current == expected_order
    saved = checkpoint['optimizer_states'][0]
    assert len(saved['param_groups']) == len(current) == (1 if args.family == 'native' else 2)
    checks['all_trainable_optimizer_states_present'] = len(saved['state']) == sum(map(len, current))
    for old, new in zip(saved['param_groups'], opt.param_groups):
        assert len(old['params']) == len(new['params'])
        for pid, param in zip(old['params'], new['params']):
            state = saved['state'][pid]
            assert state['exp_avg'].shape == state['exp_avg_sq'].shape == param.shape
            assert int(state['step']) == args.step
    opt.load_state_dict(saved); scheduler.load_state_dict(checkpoint['lr_schedulers'][0])
    checks['actual_optimizer_roundtrip_exact'] = equal(saved, opt.state_dict())
    checks['actual_scheduler_roundtrip_exact'] = equal(checkpoint['lr_schedulers'][0], scheduler.state_dict())
    checks['scheduler_step_and_lr'] = scheduler.last_epoch == args.step and [g['lr'] for g in opt.param_groups] == scheduler.get_last_lr()
    initial = torch.load(args.initial, map_location='cpu', weights_only=True)
    assert set(initial) == set(portable)
    frozen = [k for k, v in module.model.named_parameters() if not v.requires_grad]
    if args.family == 'local':
        frozen += ['spatial_projection', 'spatial_mean', 'spatial_scale']
        assert len(frozen) == len(set(frozen)) == 210
        cal = cfg['spatial_calibration']; assert sha(cal['path']) == cal['sha256']
        assert read(cal['path'])['normalization'] == 'global_rms'
    else: assert not frozen
    checks['frozen_tensors_exact'] = all(torch.equal(initial[k], portable[k]) for k in frozen)
    ds = make_training_dataset(cfg)
    assert ds.natural_length == 1494000
    sequence = resume_sequence(ds, checkpoint['campaign_samples'], cfg['seed'], 128)
    checks['sampler_next32_vs_start0'] = sequence['passed'] and sequence['native_loader_length'] == 1493888
    report.update(run=str(run), parent=str(parent), family=args.family,
                  checkpoint_sha256=sha(run/'last.ckpt'), portable_path=str(pp), portable_sha256=sha(pp),
                  config_sha256=sha(run/'config.json'), initial_weights=str(args.initial), initial_sha256=sha(args.initial),
                  source=source, source_proof=source_proof, imported_files={n: str(importlib.import_module(n).__file__) for n in modules},
                  auditor_sha256=sha(__file__), sampler_helper=dict(path=str(sampler_file), sha256=sha(sampler_file), fallback_source=sampler_source),
                  sampler=sequence, optimizer_parameter_tensors=list(map(len, current)), learning_rates=scheduler.get_last_lr(),
                  frozen_tensor_count=len(frozen), frozen_tensor_names=frozen,
                  rng_component_sha256={k: rng_digest(v) for k, v in checkpoint['campaign_rng'].items()},
                  gradient_expectations=gradient_expectations(module.model, cfg), passed=all(checks.values()),
                  scope='Actual CPU native model/AdamW/scheduler restore. Saved RNG fields hashed, not restored here. Actual stateless sampler next32 checked against epoch-concatenated start0. No forward/update/GPU or uninterrupted trajectory equivalence claim.')
    assert report['passed'] and not torch.cuda.is_initialized(), report
    if args.archive:
        assert args.step in stages(args.family) and args.previous_step is not None
        stage = read(run/'stage-complete.json')
        assert stage['stage_complete'] and stage['initial_step'] == args.previous_step and stage['stop_step'] == args.step
        assert stage['scheduler_total_steps'] == 124910 and stage['config_sha256'] == sha(run/'config.json')
        target = run/f'checkpoint-step{args.step:07d}.ckpt'; assert not target.exists()
        shutil.copy2(run/'last.ckpt', target); assert sha(target) == report['checkpoint_sha256']
        report.update(archive=str(target), stage_complete=stage)
    write(args.output, report)
    print(__import__('json').dumps(dict(passed=True, family=args.family, step=args.step, run=str(run))), flush=True)


if __name__ == '__main__': main()
