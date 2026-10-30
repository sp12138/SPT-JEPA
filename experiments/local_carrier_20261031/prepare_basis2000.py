"""Candidate-only orthogonal-basis600→1200→2000; native full-state resume."""
import argparse, copy, hashlib, json, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.local_carrier_20261031 import prepare2000 as core
from experiments.local_carrier_20261031.prepare6000 import fresh_references
from experiments.local_carrier_20261031.prepare_basis import allocations

B, PYTHON = core.B, core.PYTHON
read, write, digest, semantic = core.read, core.write, core.digest, core.semantic
PARENT = B/'runs/local_basis600_20261031'
ROOT = B/'runs/local_basis2000_20261031'
IDENTITY = B/'runs/local_isotropic2000_20261031'
FRESH = B/'runs/followup_fresh2000_20261031'
SELF = 'experiments/local_carrier_20261031/prepare_basis2000.py'
TRAINING_COMMIT = 'da26748c866137ad1ea06cf7250cabd64431e9ac'
QKEY = 'spatial_orthogonal_basis'


def source_checks(helper, training):
    assert training['source_commit'] == TRAINING_COMMIT
    required = [SELF, 'experiments/local_carrier_20261031/prepare2000.py',
        'experiments/local_carrier_20261031/prepare6000.py', 'experiments/local_carrier_20261031/prepare_basis.py',
        'experiments/carrier_progress_20261031/prepare_learning.py', 'experiments/carrier_progress_20261031/prepare_next.py']
    records = {}
    for kind, src in [('training', training), ('helper', helper)]:
        m = read(src['manifest']); assert m['source_commit'] == src['source_commit']
        for name in (m['files'] if kind == 'training' else required):
            assert digest(Path(src['source'])/name) == m['files'][name], name
        records[kind] = dict(source=src, manifest_sha256=digest(src['manifest']))
    assert digest(__file__) == digest(Path(helper['source'])/SELF)
    assert helper['runtime_source'] == training['runtime_source'] and helper['runtime_manifest'] == training['runtime_manifest']
    for name in required[1:]:
        assert digest(Path(helper['source'])/name) == digest(Path(training['source'])/name), name
    return records


def continuation_config(cfg, out):
    assert cfg['model']['spatial_basis'] == 'orthogonal' and cfg['model']['spatial_mode'] == 'local'
    assert QKEY in cfg['audit_expectations']['initial_unchanged_prefixes']
    return core.continuation_config(cfg, out)


def audit_state(run, step, parent, initial, training):
    report = core.audit_state(run, step, parent, initial, training)
    import torch
    original = torch.load(initial, map_location='cpu', weights_only=True)
    portable = torch.load(report['portable_path'], map_location='cpu', weights_only=True)
    q = original[QKEY]
    assert q.shape == (1024, 1024) and torch.equal(q, portable[QKEY])
    assert torch.equal(q.double().T @ q.double(), torch.eye(1024, dtype=torch.double))
    frozen = report['frozen_tensors']; assert frozen['compared_tensors'] == 210 and QKEY not in frozen['names']
    frozen['names'].append(QKEY); frozen['compared_tensors'] = 211
    report['checks']['orthogonal_basis_frozen_and_exact_fp64'] = True
    report['basis_sha256'] = hashlib.sha256(q.numpy().tobytes()).hexdigest()
    report['basis_scope'] = 'Original Q equals portable Q; core portable_equal also checks fullstate Q. All frozen parameters plus projection/mean/scale/Q:211 tensors.'
    report['passed'] = all(report['checks'].values()); assert report['passed']
    return report


def jobs_for(rec, root, helper, training):
    # Keep the existing native two-segment chain; omit the redundant fixed-state
    # information probe, then append the already fixed fresh50 evaluator.
    jobs = [j for j in core.jobs_for(rec, root, helper, training) if j['phase'] != 'info']
    for j in jobs:
        if j['phase'].startswith('fullstate'):
            assert j['command'][3] == str(Path(helper['source'])/core.SELF)
            j['command'][3] = str(Path(helper['source'])/SELF)
        if j['phase'] == 'transition':
            j['depends_on'] = ['health-'+rec['id']]
        if j['phase'] == 'planning':
            j['manifest_sha256'] = rec['primary_manifest']['sha256']
            assert j['command'][j['command'].index('--manifest')+1] == rec['primary_manifest']['path']
    fresh = copy.deepcopy(jobs[-1]); assert fresh['phase'] == 'planning'
    fresh.update(id='fresh-planning-'+rec['id'], phase='fresh-planning',
        output=str(Path(root)/('fresh-planning-'+rec['id'])), depends_on=[jobs[-1]['id']],
        manifest_sha256=rec['fresh_manifest']['sha256'])
    fresh['command'][fresh['command'].index('--manifest')+1] = rec['fresh_manifest']['path']
    fresh['command'][fresh['command'].index('--output')+1] = fresh['output']
    jobs.append(fresh); assert len(jobs) == 8
    return jobs


def archive(args):
    helper, training = read(args.source_record), read(args.training_source_record)
    source_checks(helper, training); assert args.expected_step in (1200, 2000)
    out = Path(args.output); assert not (out/'fullstate.json').exists()
    report = audit_state(args.audit_run, args.expected_step, args.parent_run, args.initial_weights, training)
    stage = read(Path(args.audit_run)/'stage-complete.json')
    assert stage['stage_complete'] and stage['stop_step'] == args.expected_step
    assert stage['initial_step'] == (600 if args.expected_step == 1200 else 1200)
    target = Path(args.audit_run)/f'checkpoint-step{args.expected_step:07d}.ckpt'; assert not target.exists()
    shutil.copy2(Path(args.audit_run)/'last.ckpt', target); assert digest(target) == report['checkpoint_sha256']
    report.update(archived_checkpoint=str(target), native_stage_complete=stage, helper_source_commit=helper['source_commit'])
    out.mkdir(parents=True, exist_ok=True); write(out/'fullstate.json', report)
    print(json.dumps(dict(passed=True, step=args.expected_step, archive=str(target))), flush=True)


def prepare(args):
    root, parent = Path(args.root), Path(args.parent); assert not root.exists()
    helper, training = read(args.source_record), read(parent/'source-record.json')
    compatibility = source_checks(helper, training)
    plan = read(parent/'plan.json'); assert len(plan['jobs']) == 20 and (parent/'HOLD_DISPATCH').exists()
    for j in plan['jobs']: assert read(parent/'exits'/(j['id']+'.json'))['returncode'] == 0
    fresh, evidence = fresh_references(args.fresh_root, IDENTITY)
    records, identity_refs = [], {}
    for rec in allocations():
        old = parent/rec['id']; cfg = read(old/'config.json'); stage = read(old/'stage-complete.json')
        assert stage['stage_complete'] and stage['stop_step'] == 600
        job = next(j for j in plan['jobs'] if j['phase']=='train600' and Path(j['output'])==old)
        assert digest(old/'config.json') == job['config_sha256']
        cal = cfg['spatial_calibration']; assert digest(cal['path']) == cal['sha256']
        assert read(cal['path'])['normalization'] == 'global_rms'
        refname = f'{rec["task"]}-spatial-local-seed{rec["training_seed"]}'
        refcfg = read(IDENTITY/refname/'config.json'); comparable = copy.deepcopy(cfg)
        del comparable['model']['spatial_basis']
        comparable['audit_expectations']['initial_unchanged_prefixes'].remove(QKEY)
        assert semantic(comparable) == semantic(refcfg)
        primary = Path(cfg['data'])/'manifests/validation-strict.json'
        primary_report = read(IDENTITY/('planning-'+refname)/'metrics.json')
        old_report = read(parent/('planning-'+rec['id'])/'metrics.json')
        assert primary_report['checkpoint'] == str(IDENTITY/refname/'model-step0002000')
        assert primary_report['checkpoint_audit']['strict_equal'] and len(primary_report['pairs']) == 50
        assert read(IDENTITY/'exits'/('planning-'+refname+'.json'))['returncode'] == 0
        assert primary_report['manifest_sha256'] == digest(primary)
        for key in ['manifest_sha256','pairs','solver','plan','protocol','split','normalization']:
            assert primary_report[key] == old_report[key], key
        fresh_report = read(evidence[refname]['evaluation'])
        for key in ['solver','plan','protocol','split','normalization']:
            assert primary_report[key] == fresh_report[key], key
        initial = old/'initial-model/weights.pt'
        pre = audit_state(old, 600, old, initial, training)
        previous = read(parent/('fullstate-'+rec['id'])/'preflight.json')
        assert previous['passed'] and previous['checkpoint_sha256'] == pre['checkpoint_sha256']
        records.append(dict(**rec, parent=str(old), initial_weights=str(initial),
            config=continuation_config(cfg,root/rec['id']), pre_audit=pre,
            primary_manifest=dict(path=str(primary),sha256=digest(primary)), fresh_manifest=fresh[rec['task']]))
        identity_refs[rec['id']] = dict(primary=str(IDENTITY/('planning-'+refname)/'metrics.json'), fresh=evidence[refname])
        print(json.dumps(dict(pre_resume_passed=rec['id'], frozen_tensors=211)), flush=True)
    if args.dry_run:
        print(json.dumps(dict(dry_run=True,audits_passed=4,launched=False))); return
    root.mkdir(); write(root/'source-record.json',training); write(root/'helper-source-record.json',helper)
    write(root/'source-compatibility.json',compatibility); jobs, copies = [], []
    for rec in records:
        out=root/rec['id']; out.mkdir(); write(out/'config.json',rec['config'])
        shutil.copy2(Path(rec['parent'])/'last.ckpt',out/'last.ckpt')
        assert digest(out/'last.ckpt') == rec['pre_audit']['checkpoint_sha256']
        report=dict(rec['pre_audit'],copy_byte_identical=True,copied_to=str(out/'last.ckpt'),
            copy_scope='One native CPU load per source600 state. Full-file post-copy hash preserves every saved model/optimizer/scheduler/RNG/cursor byte; no second load claimed.')
        write(out/'pre-resume-fullstate.json',report)
        copies.append(dict(id=rec['id'],checkpoint_sha256=report['checkpoint_sha256'],passed=True))
        jobs.extend(jobs_for(rec,root,helper,training))
    assert len(jobs)==32 and len({j['id'] for j in jobs})==32
    write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment=plan['environment'],jobs=jobs))
    write(root/'resume-audit.json',copies)
    write(root/'protocol.json',dict(created_at=time.time(),parent=str(parent),root=str(root),runs=4,jobs=32,
        training_source_commit=training['source_commit'],helper_source_commit=helper['source_commit'],allocations=list(allocations()),
        initial_step=600,intermediate_step=1200,stop_step=2000,native_fullstate_archives=[1200,2000],
        scheduler_horizon=124910,warmup_steps=1249,semantic_config_unchanged=True,original_calibration_unchanged=True,
        frozen_tensor_count=211,validation_cases=50,transition_windows=64,fresh_manifests=fresh,identity2000_references=identity_refs,
        candidate_only=True,reason='One reviewed maturity control.600 has faster latent fitting under rotation but no consistent planning difference and remains inside warmup.',
        source_scope='All training and GPU evaluations remain originalda26748. New helper imports that source for CPU fullstate audits, adding explicit frozenQ check.',
        departures=['New independent output directories; original600 held state/source/config remains unchanged.',
            'Native1200 boundary matches identity continuation; checkpoint cadence1200 and retained1200/2000.',
            'No optimizer/scheduler reset, added loss/head/label, calibration change, probe, or model forward in preparation.'],
        stop_policy='Failed1200 archive blocks2000. Root alone activates candidate-plan. Stop2000; no automatic continuation.'))
    assert not (root/'plan.json').exists()
    print(json.dumps(dict(candidate=str(root/'candidate-plan.json'),runs=4,endpoints=32,launched=False)),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-record',required=True);p.add_argument('--training-source-record')
    p.add_argument('--root',default=str(ROOT));p.add_argument('--parent',default=str(PARENT));p.add_argument('--fresh-root',default=str(FRESH))
    p.add_argument('--dry-run',action='store_true');p.add_argument('--audit-run');p.add_argument('--parent-run');p.add_argument('--initial-weights')
    p.add_argument('--expected-step',type=int,choices=[1200,2000]);p.add_argument('--output');a=p.parse_args()
    if a.audit_run:
        assert a.training_source_record and a.parent_run and a.initial_weights and a.expected_step and a.output and not a.dry_run;archive(a)
    else:prepare(a)
