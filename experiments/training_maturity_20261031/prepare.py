"""Prepare six held Cube controls to two loader epochs; never activates jobs."""
import argparse
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path
from common import B, PY, ROOT, PREFIX, CAP, FAMILIES, CORR, CORR_COMMIT, allocations, stages, read, sha, write, semantic, source_check


def configure(old, output, family):
    cfg = copy.deepcopy(old)
    cfg.update(output=str(output), checkpoint_every=2000, retain_steps=stages(family))
    assert semantic(cfg) == semantic(old)
    assert cfg['batch_size'] == 128 and cfg['max_steps'] == 124910 and cfg['model']['window'] == 8
    assert cfg['model']['action_dim'] == 25
    return cfg


def audit_command(helper, record, run, step, output, portable=None, archive=False, previous=None):
    cmd = [PY, '-B', '-u', str(Path(helper['source'])/PREFIX/'audit.py'), '--run', str(run),
           '--parent', record['parent'], '--initial', record['initial'], '--family', record['family'],
           '--source-record', record['model_record'], '--step', str(step), '--output', str(output)]
    if portable: cmd += ['--portable', str(portable)]
    if archive: cmd += ['--archive', '--previous-step', str(previous)]
    return cmd


def jobs_for(root, helper, r):
    run = Path(root)/r['id']; source = r['source']; src = Path(source['source']); jobs = []; deps = []
    shared = {k: r[k] for k in ['task', 'family', 'training_seed', 'host', 'gpu']}
    def add(phase, out, cmd, **kw):
        nonlocal deps
        name = phase+'-'+r['id']
        jobs.append(dict(**shared, source=source, hold=True, id=name, phase=phase, output=str(out), command=cmd, depends_on=deps, **kw))
        deps = [name]
    previous = r['initial_step']
    for step in stages(r['family']):
        add('learn'+str(step), run, [PY, '-B', '-u', '-m', 'as_jepa_tools.train_stage', '--config', str(run/'config.json'),
            '--stop-step', str(step), '--resume'], config=str(run/'config.json'), config_sha256=r['config_sha256'])
        out = Path(root)/f'archive{step}-{r["id"]}'
        add('archive'+str(step), out, audit_command(helper, r, run, step, out/'fullstate.json', archive=True, previous=previous))
        previous = step
    out = Path(root)/('fullstate-'+r['id'])
    add('fullstate', out, [PY, '-B', '-u', str(src/'experiments/ideas_validate_20261031/preflight.py'),
        '--run', str(run), '--output', str(out), '--expected-step', str(CAP), '--initial-weights', r['initial']])
    if r['family'] != 'native':
        out = Path(root)/('health-'+r['id'])
        add('health', out, [PY, '-B', '-u', '-m', 'as_jepa_tools.benchmark_health', '--run', str(run), '--output', str(out), '--cases', '16'])
    out = Path(root)/('transition-'+r['id'])
    add('transition', out, [PY, '-B', '-u', str(src/'experiments/ideas_validate_20261031/audit.py'),
        '--run', str(run), '--output', str(out), '--cases', '64', '--manifest', r['windows']])
    for case in ['primary', 'fresh']:
        out = Path(root)/(f'original-{case}-'+r['id'])
        add('original-'+case, out, [PY, '-B', '-u', '-m', 'as_jepa_tools.benchmark_evaluate',
            '--checkpoint', str(run/f'model-step{CAP:07d}'), '--manifest', r['references'][case]['manifest'],
            '--data', r['data'], '--output', str(out)], manifest_sha256=r['references'][case]['manifest_sha256'],
            environment=dict(MUJOCO_EGL_DEVICE_ID=str(r['gpu'])))
        out = Path(root)/(f'correlated-{case}-'+r['id'])
        add('correlated-'+case, out, [PY, '-B', '-u', str(Path(helper['source'])/PREFIX/'prepare.py'),
            'correlated', '--root', str(root), '--run-id', r['id'], '--case', case],
            manifest_sha256=r['references'][case]['manifest_sha256'], environment=dict(MUJOCO_EGL_DEVICE_ID=str(r['gpu'])))
    return jobs


def validate_jobs(jobs):
    ids = {j['id'] for j in jobs}
    assert len(ids) == len(jobs) == 68
    assert all(j['hold'] and set(j['depends_on']) <= ids for j in jobs)
    assert len({(j['host'], j['gpu']) for j in jobs}) == 6
    for a in allocations():
        rows = [j for j in jobs if j['training_seed'] == a['training_seed'] and j['family'] == a['family']]
        assert len(rows) == {'local': 13, 'native': 10, 'tf': 11}[a['family']]
        prior = []
        for j in rows:
            assert j['depends_on'] == prior; prior = [j['id']]
        training = [int(j['command'][j['command'].index('--stop-step')+1]) for j in rows if j['phase'].startswith('learn')]
        assert training == stages(a['family']) and max(training) == CAP
        assert all(j['source']['source_commit'] == FAMILIES[a['family']]['commit'] for j in rows)
    return dict(passed=True, endpoints=68, training_segments=14, archive_segments=14,
                final_fullstate=6, flow_health=4, native_health_in_fullstate=2, transition64=6, planning=24)


def prepare(args):
    root = Path(args.root); assert not root.exists()
    helper = read(args.helper_record); source_check(helper)
    assert sha(__file__) == read(helper['manifest'])['files'][PREFIX+'prepare.py']
    proof = read(Path(helper['source'])/PREFIX/'cpu-contracts.json'); assert proof['passed']
    for file, h in proof['helper_sha256'].items(): assert sha(Path(helper['source'])/PREFIX/file) == h
    corr = read(CORR); assert corr['source_commit'] == CORR_COMMIT; source_check(corr)
    cal = B/'runs/action_covariance_evaluation_20261031/calibration/cube/calibration.json'
    matrix = cal.with_name('matrix.npz'); calibration = read(cal)
    assert calibration['task'] == 'cube' and sha(matrix) == calibration['matrix_sha256']
    assert calibration['dataset_card_sha256'] == sha(B/'data/hardtasks-v1/cube-metadata/dataset_card.json')
    calibration_inputs = {str(p): sha(p) for p in [cal, matrix]}
    records = []; environments = []; primary = B/'data/hardtasks-v1/cube-metadata/manifests/validation-strict.json'
    fresh = B/'runs/carrier_cube_freshcases_20261031/fresh-manifest.json'
    assert len(read(primary)['pairs']) == len(read(fresh)['pairs']) == 50
    for a in allocations():
        f = FAMILIES[a['family']]; parentroot = B/'runs'/f['parent']; parent = parentroot/a['id']
        assert (parentroot/'HOLD_DISPATCH').exists()
        plan = read(parentroot/'plan.json'); source = read(parentroot/'source-record.json')
        assert source['source_commit'] == f['commit']; source_check(source)
        for j in plan['jobs']:
            if j.get('task') == 'cube' and j.get('training_seed') == a['training_seed']:
                assert read(parentroot/'exits'/(j['id']+'.json'))['returncode'] == 0
        cfg = read(parent/'config.json'); status = read(parent/'status.json')
        assert status['state'] == 'finished' and status['step'] == f['step']
        archive = parent/f'checkpoint-step{f["step"]:07d}.ckpt'
        assert sha(archive) == sha(parent/'last.ckpt')
        initial_name = a['id']+('-gate600' if a['family'] == 'tf' else '')
        initial = B/'runs'/f['initial']/initial_name/'initial-model/weights.pt'; assert initial.exists()
        refs = {}
        for case, manifest in [('primary', primary), ('fresh', fresh)]:
            phase = 'planning' if case == 'primary' else 'fresh-planning'
            metrics_path = parentroot/(phase+'-'+a['id'])/'metrics.json'; metrics = read(metrics_path)
            assert metrics['manifest_sha256'] == sha(manifest) and metrics['pairs'] == read(manifest)['pairs']
            assert metrics['normalization'] == 'train episodes' and metrics['checkpoint_audit']['strict_equal']
            assert metrics['checkpoint'] == str(parent/f'model-step{f["step"]:07d}')
            refs[case] = dict(manifest=str(manifest), manifest_sha256=sha(manifest), metrics=str(metrics_path), metrics_sha256=sha(metrics_path))
        assert read(refs['primary']['metrics'])['solver'] == read(refs['fresh']['metrics'])['solver']
        windows = Path(source['source'])/'experiments/ideas_validate_20261031/audit-windows-cube.json'
        assert len(read(windows)['windows']) == 64
        records.append(dict(**a, parent=str(parent), parent_archive=str(archive), parent_checkpoint_sha256=sha(archive), initial=str(initial),
            initial_sha256=sha(initial), source=source, data=cfg['data'], windows=str(windows), windows_sha256=sha(windows),
            model_record=str(root/'model-sources'/f'{a["family"]}.json'), config=configure(cfg, root/a['id'], a['family']), references=refs))
        environments.append(plan['environment'])
    assert len({r['windows_sha256'] for r in records}) == 1
    # Retain the original proven EGL/CPU environment; helpers never take import precedence over model code.
    environment = read(B/'runs/patch_token2000_20261031/plan.json')['environment']
    root.mkdir(); write(root/'helper-source-record.json', helper); write(root/'correlation-source-record.json', corr)
    write(root/'correlation-calibration-inputs.json', calibration_inputs)
    for a in ['HOLD', 'HOLD_DISPATCH']: (root/a).write_text('Prepared only. Root review/authorization and actual-free/startup checks required. No dispatch has been launched.\n')
    for family in FAMILIES: write(root/'model-sources'/f'{family}.json', next(r['source'] for r in records if r['family'] == family))
    # Exact pre/post restore in separate interpreters prevents cross-family module caching.
    proofs = []
    for r in records:
        run = root/r['id']; parent = Path(r['parent']); run.mkdir(); write(run/'config.json', r['config']); r['config_sha256'] = sha(run/'config.json')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=r['source']['pythonpath'], OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        prepath = run/'pre-resume-fullstate.json'; postpath = run/'post-copy-fullstate.json'
        for label, command in [('pre', audit_command(helper, r, parent, r['initial_step'], prepath)),
                               ('post', audit_command(helper, r, run, r['initial_step'], postpath,
                                        parent/f'model-step{r["initial_step"]:07d}/weights.pt'))]:
            if label == 'post':
                shutil.copy2(r['parent_archive'], run/'last.ckpt')
                assert sha(run/'last.ckpt') == r['parent_checkpoint_sha256']
            with (run/f'{label}-restore.log').open('x') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        pre, post = read(prepath), read(postpath)
        assert pre['passed'] and post['passed']
        for key in ['checkpoint_sha256', 'portable_sha256', 'rng_component_sha256', 'sampler', 'optimizer_parameter_tensors', 'learning_rates']:
            assert pre[key] == post[key], key
        proofs.append(dict(id=r['id'], passed=True, pre=prepath.name, post=postpath.name, checkpoint_sha256=post['checkpoint_sha256'],
                           sampler=post['sampler'], optimizer_parameter_tensors=post['optimizer_parameter_tensors']))
        print(json.dumps(dict(id=r['id'], native_pre_post_passed=True)), flush=True)
    jobs = [j for r in records for j in jobs_for(root, helper, r)]; dag = validate_jobs(jobs)
    write(root/'candidate-plan.json', dict(root=str(root), deadline=None, environment=environment, jobs=jobs))
    write(root/'resume-audit.json', proofs)
    write(root/'protocol.json', dict(prepared_only=True, launched=False, runs=records, helper=helper, correlation_helper=corr,
        endpoints=dag, update_cap=CAP, loader_updates_per_epoch=11671, loader_epochs=2,
        training_schedule=dict(max_steps=124910, warmup=1249, batch128=True, clip=1),
        source_scope='Each job source remains its original32fa/7a/8e model/Trainer/evaluator; only orchestration and CPU auditing use the new helper. Fixed f4 correlation adapter reused.',
        operation='local6000→11671→18000→23342; native/TF12000→18000→23342. Full checkpoint archived/audited before next segment; portable every named cap retained.',
        native_health='The native objective/gradient fullstate audit is its health check. No dummy renderer/DiT health, newBN calibration or variant selection.',
        case_scope='Original validation primary50 and existing fresh50, same ordered pairs/scalers/300x30 CEM/history3. All raw original and fixed common f4 correlated outcomes retained; no NGC/newcases.',
        rng_scope='Pre/post actual model/AdamW/scheduler restore, copied saved RNG field hashes, native stateless next32 cursor oracle. CPU does not restore RNG; no bitwise uninterrupted-GPU claim.',
        stop='Candidate only; no plan.json/dispatcher. Original roots/HOLDs unchanged. Root review required before activation; all6 stop23342, no automatic budget/seed/method extension.'))
    assert not (root/'plan.json').exists()
    write(root/'preparation-summary.json', dict(passed=True, prepared_only=True, launched=False, actual_cpu_restore_audits=12,
        copies=proofs, dag=dag, helper=helper, candidate_sha256=sha(root/'candidate-plan.json'), protocol_sha256=sha(root/'protocol.json')))
    print(json.dumps(dict(root=str(root), actual_cpu_restores=12, endpoints=68, launched=False)), flush=True)


def correlated(args):
    root = Path(args.root); protocol = read(root/'protocol.json')
    r = next(r for r in protocol['runs'] if r['id'] == args.run_id)
    helper = protocol['helper']; assert sha(__file__) == read(helper['manifest'])['files'][PREFIX+'prepare.py']
    corr = read(root/'correlation-source-record.json'); assert corr['source_commit'] == CORR_COMMIT
    source_check(corr); source_check(r['source'])
    for path, h in read(root/'correlation-calibration-inputs.json').items(): assert sha(path) == h, path
    reference = root/(f'original-{args.case}-'+r['id'])/'metrics.json'
    old = read(r['references'][args.case]['metrics']); current = read(reference)
    for key in ['pairs', 'manifest_sha256', 'solver', 'plan', 'normalization', 'split', 'protocol']:
        assert current[key] == old[key], key
    checkpoint = root/r['id']/f'model-step{CAP:07d}'
    assert current['checkpoint'] == str(checkpoint) and current['checkpoint_audit']['strict_equal']
    cal = B/'runs/action_covariance_evaluation_20261031/calibration/cube/calibration.json'
    matrix = cal.with_name('matrix.npz'); original = read(cal)
    assert sha(matrix) == original['matrix_sha256']
    out = root/(f'correlated-{args.case}-'+r['id']); spec_path = root/'specs'/(out.name+'.json')
    folder = Path(corr['source'])/'experiments/action_covariance_20261031'
    evaluator = Path(r['source']['source'])/'as_jepa_tools/benchmark_evaluate.py'
    paths = [reference, checkpoint/'weights.pt', checkpoint/'config.json', Path(r['references'][args.case]['manifest']), Path(r['data'])/'dataset_card.json', evaluator, cal, matrix]
    spec = dict(id=out.name, task='cube', family=r['family'], step=CAP, training_seed=r['training_seed'], case_set=args.case,
        mode='correlated', source=r['source'], checkpoint=str(checkpoint), manifest=r['references'][args.case]['manifest'], data=r['data'],
        output=str(out), reference_metrics=str(reference), native_evaluator=str(evaluator), calibration=str(cal),
        helper_files={n: sha(folder/n) for n in ['adapter.py', 'evaluate.py']}, immutable_inputs={str(p): sha(p) for p in paths})
    write(spec_path, spec)
    subprocess.run([PY, '-B', '-u', str(folder/'evaluate.py'), '--spec', str(spec_path)], check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('phase', choices=['prepare', 'correlated']); p.add_argument('--root', default=str(ROOT))
    p.add_argument('--helper-record'); p.add_argument('--run-id'); p.add_argument('--case', choices=['primary', 'fresh'])
    a = p.parse_args()
    if a.phase == 'prepare': assert a.helper_record; prepare(a)
    else: assert a.run_id and a.case; correlated(a)
