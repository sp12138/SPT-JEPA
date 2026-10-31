"""Four reviewed numerical/task gates; native full-state continuation for one run."""
import argparse
import json
import shutil
from pathlib import Path

from experiments.local_carrier_20261031 import prepare2000 as core
from experiments.local_carrier_20261031.raw_chart.prepare import allocations, config

B, PY = core.B, core.PYTHON
read, write, sha = core.read, core.write, core.digest
ROOT = B/'runs/local_raw_chart_precision600_20261031'
GATE = B/'runs/raw_chart_precision_gate200_20261031'
FAILED = B/'runs/local_raw_chart600_20261031'
SELF = 'experiments/local_carrier_20261031/raw_chart/prepare_precision600.py'
MODEL_COMMIT = '1060dde0e36ba4fb32a54e4a141c6e1d2cbd7a98'
PREFIXES = ['encoder', 'pool', 'predictor', 'action_encoder', 'renderer',
            'spatial_projection', 'spatial_mean', 'spatial_scale', 'carrier_chart_module']


def sources(args):
    model, helper, launcher = (read(path) for path in
        [args.model_source_record, args.helper_source_record, args.launcher_source_record])
    for source in [model, helper, launcher]:
        manifest = read(source['manifest'])
        assert manifest['source_commit'] == source['source_commit']
        for name, digest in manifest['files'].items():
            assert sha(Path(source['source'])/name) == digest, name
    assert model['source_commit'] == MODEL_COMMIT
    assert launcher['source_commit'] == '22e02a1ded00c17639267d308c21a28c0247bc6c'
    assert sha(__file__) == read(helper['manifest'])['files'][SELF]
    assert read(GATE/'review.json')['passed'] and read(GATE/'source-record.json') == model
    assert (GATE/'HOLD_DISPATCH').exists() and (FAILED/'HOLD_DISPATCH').exists()
    proof = Path(model['source'])/'experiments/local_carrier_20261031/raw_chart/precision_cpu'
    assert read(proof/'contracts.json')['passed'] and read(proof/'legacy-contracts.json')['passed']
    assert read(B/'runs/raw_chart_precision_verify_20261031/replay.json')['passed']
    # CPU resume helpers are imported only from the same immutable model source.
    assert Path(core.__file__).is_relative_to(Path(model['source']))
    return model, helper, launcher


def jobs_for(alloc, out, model, reference, resume):
    name = alloc['id']; root = out.parent; cfg_path = out/'config.json'
    common = {key: value for key, value in alloc.items() if key != 'id'}
    common['source'] = model; jobs = []; previous = []

    def add(phase, dest, command, **extra):
        nonlocal previous
        ident = phase+'-'+name
        jobs.append(dict(**common, id=ident, phase=phase, output=str(dest),
            depends_on=previous, command=command, **extra))
        previous = [ident]

    command = [PY, '-B', '-u', '-m', 'as_jepa_tools.train_stage', '--config',
               str(cfg_path), '--stop-step', '600'] + (['--resume'] if resume else [])
    add('train600', out, command, config=str(cfg_path), config_sha256=sha(cfg_path))
    dest = root/('fullstate-'+name)
    command = [PY, '-B', '-u', str(Path(model['source'])/'experiments/ideas_validate_20261031/preflight.py'),
        '--run', str(out), '--output', str(dest), '--expected-step', '600',
        '--reference-initial', str(reference/'initial-model/weights.pt')]
    for prefix in PREFIXES:
        command += ['--reference-prefix', prefix]
    add('fullstate', dest, command)
    dest = root/('health-'+name)
    add('health', dest, [PY, '-B', '-u', '-m', 'as_jepa_tools.benchmark_health',
        '--run', str(out), '--output', str(dest)])
    dest = root/('transition-'+name)
    add('transition', dest, [PY, '-B', '-u', str(Path(model['source'])/'experiments/ideas_validate_20261031/audit.py'),
        '--run', str(out), '--output', str(dest), '--cases', '64', '--manifest',
        str(Path(model['source'])/f'experiments/ideas_validate_20261031/audit-windows-{alloc["task"]}.json')])
    cfg = read(cfg_path); manifest = Path(cfg['data'])/'manifests/validation-strict.json'
    dest = root/('planning-'+name)
    add('planning', dest, [PY, '-B', '-u', '-m', 'as_jepa_tools.benchmark_evaluate',
        '--manifest', str(manifest), '--checkpoint', str(out/'model-step0000600'),
        '--data', cfg['data'], '--output', str(dest)], manifest_sha256=sha(manifest),
        environment=dict(MUJOCO_EGL_DEVICE_ID=str(alloc['gpu'])))
    return jobs


def prepare(args):
    import torch
    torch.set_num_threads(2)
    model, helper, launcher = sources(args)
    assert not ROOT.exists(), 'Never overwrite an existing or held root'
    parent = GATE/'tworoom-seed1'; initial = parent/'initial-model/weights.pt'
    pre = core.audit_state(parent, 200, parent, initial, model)
    assert pre['passed'] and pre['frozen_tensors']['compared_tensors'] == 210
    assert pre['checkpoint_sha256'] == read(GATE/'fullstate-tworoom-seed1/preflight.json')['checkpoint_sha256']
    stage = read(parent/'stage-complete.json')
    assert stage['initial_step'] == 0 and stage['stop_step'] == 200 and stage['stage_complete']
    ROOT.mkdir(); jobs = []; records = []
    for alloc in allocations():
        name = alloc['id']; out = ROOT/name; cfg, identity = config(alloc, out, 600)
        reference = FAILED/name; oldcfg = read(reference/'config.json')
        assert core.semantic(cfg) == core.semantic(oldcfg)
        assert cfg['checkpoint_every'] == 200 and cfg['retain_steps'] == [200, 400, 600]
        assert cfg['model'].get('spatial_readout_mode', 'none') == 'none'
        assert cfg.get('audit_expectations', {}).get('gradients', {}) == {'action_encoder': 'active'}
        original = torch.load(reference/'initial-model/weights.pt', map_location='cpu', weights_only=True)
        identity_weights = torch.load(identity/'initial-model/weights.pt', map_location='cpu', weights_only=True)
        assert len(original) == 557 and len(identity_weights) == 553
        assert all(key in original and torch.equal(value, original[key]) for key, value in identity_weights.items())
        del original, identity_weights
        out.mkdir(); write(out/'config.json', cfg)
        resume = alloc['task'] == 'tworoom' and alloc['training_seed'] == 1
        rec = dict(**alloc, initial_step=200 if resume else 0, reference=str(reference),
            identity_reference=str(identity), initial557_sha256=sha(reference/'initial-model/weights.pt'),
            identity553_sha256=sha(identity/'initial-model/weights.pt'), calibration=cfg['spatial_calibration'])
        if resume:
            assert core.semantic(cfg) == core.semantic(read(parent/'config.json'))
            shutil.copy2(parent/'last.ckpt', out/'last.ckpt')
            shutil.copytree(parent/'initial-model', out/'initial-model')
            shutil.copytree(parent/'model-step0000200', out/'model-step0000200')
            assert sha(out/'last.ckpt') == pre['checkpoint_sha256']
            assert sha(out/'initial-model/weights.pt') == sha(initial)
            post = core.audit_state(out, 200, parent, out/'initial-model/weights.pt', model)
            assert post['checkpoint_sha256'] == pre['checkpoint_sha256']
            assert post['rng_component_sha256'] == pre['rng_component_sha256'] and post['sampler'] == pre['sampler']
            write(out/'pre-resume-fullstate.json', pre); write(out/'post-copy-fullstate.json', post)
            rec['resume'] = dict(parent=str(parent), fullstate_sha256=pre['checkpoint_sha256'],
                copied_fullstate_byte_equal=True, original200_held=True, pre_and_post_native_restore_passed=True)
        jobs.extend(jobs_for(alloc, out, model, reference, resume)); records.append(rec)
        print(json.dumps(dict(prepared=name, initial_step=rec['initial_step'])), flush=True)
    assert len(jobs) == 20 and len({job['id'] for job in jobs}) == 20
    assert all(job['source'] == model for job in jobs)
    assert sum('--resume' in job['command'] for job in jobs) == 1
    write(ROOT/'source-record.json', model); write(ROOT/'helper-source-record.json', helper)
    write(ROOT/'launcher-source-record.json', launcher)
    write(ROOT/'candidate-plan.json', dict(root=str(ROOT), deadline=None,
        environment=read(FAILED/'plan.json')['environment'], jobs=jobs))
    write(ROOT/'protocol.json', dict(model_source=model, helper_source=helper, launcher_source=launcher,
        runs=records, cap=600, endpoints=20, candidate_only=True,
        authorization='Root approved four numerical/task600 gates after actual fresh200 recovery review. One200->600 full-state continuation and three fresh600 runs; no automatic>600.',
        coordinates='Public and cached states are RAW; never apply a second inverse during collection.',
        native_restore='Existing Trainer, optimizer, scheduler124910, RNG and WindowSampler cursor. One pre-copy and one post-copy CPU native restore, exact full-file SHA, all210 frozen parameters/measurement buffers. No fresh substitution on audit failure.',
        checkpoints='Native last.ckpt every200; portable200/400/600 retained. TRseed1 original0->200 used cadence25, retained untouched in held parent; new root starts from exact200 and uses cadence200. This is an operational cadence/resume departure, not an uninterrupted-bitwise claim.',
        initial_proof='Original failed raw557 tensors contain exactly all identity553 tensors. Each final fullstate endpoint compares all557 initial tensors including four chart tensors and three measurement buffers.',
        comparison='Original local600 same strict50 planning cases,64 transition windows,data,seed,batch128,schedule,calibration and sample-level loss. Chart adds2128 trainable parameters. No new fresh600 cohort.',
        launcher='Frozen22e02 wrapper changes control-plane atomic writes only; all20 jobs and workflow/model source remain1060. No native Trainer/model/evaluator changes.',
        stop='Any inverse failure remains fail-closed with unchanged tolerances; stop600 and review all seeds. Preserve every old HOLD, failed exit and state.'))
    assert not (ROOT/'plan.json').exists()
    print(json.dumps(dict(candidate=str(ROOT/'candidate-plan.json'), endpoints=20, launched=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-source-record', required=True)
    parser.add_argument('--helper-source-record', required=True)
    parser.add_argument('--launcher-source-record', required=True)
    prepare(parser.parse_args())
