"""One finite recipe: preserve full states, stop confirmed empty codes, otherwise finish prescribed budgets."""
import argparse
import datetime
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import numpy as np
from common import B, PY, read, write, sha, source_check
from hard_worker import native_audit


def main(root, ident, gpu):
    root = Path(root); p = read(root / 'protocol.json'); source_check(p['source']); source_check(p['helper'])
    row = next(r for r in p['jobs'] if r['id'] == ident); run = Path(row['run']); cfg = read(run / 'template-config.json')
    assert sha(run / 'template-config.json') == row['source_config_sha256']
    if row['config_deferred']:
        dependency = Path(row['calibration_dependency']); metadata = read(dependency)
        cfg['calibrate_state_std'] = metadata['original_std']
        if (run / 'config.json').exists():assert read(run / 'config.json') == cfg
        else:write(run / 'config.json', cfg)
        binding = dict(parent_initial_scale=str(dependency), sha256=sha(dependency), target_std=metadata['original_std'],
                       scope='Same pre-affine spread; mean shift only, no outcome-based scale selection.')
        if (run / 'sham-binding.json').exists():assert read(run / 'sham-binding.json') == binding
        else:write(run / 'sham-binding.json', binding)
    else:assert read(run / 'config.json') == cfg
    row = dict(row, config_sha256=sha(run / 'config.json'))
    env = dict(os.environ, **p['environment'], PYTHONPATH=p['source']['pythonpath'], CUDA_VISIBLE_DEVICES=str(gpu),
               MUJOCO_EGL_DEVICE_ID=str(gpu), PYTHONDONTWRITEBYTECODE='1')
    env['TMPDIR'] = '/tmp/spt-lingjing-fullspec'; Path(env['TMPDIR']).mkdir(exist_ok=True, parents=True)
    results = []; stages = []; started = time.monotonic(); outcome = 'execution_error'; error = None
    helper = Path(p['helper']['source']); cp = None
    io_binding = root / 'SAVE_IO_RUNTIME.json'
    if io_binding.exists():
        binding = read(io_binding); source_check(binding['helper'])
        assert binding['scope'] == 'checkpoint_io_only'
        assert binding['model_source_commit'] == p['source']['source_commit']
        helper = Path(binding['helper']['source'])
        row['safe_checkpoint_io'] = True
    def execute(name, command, timeout=7200):
        if (root / 'STOP').exists() or (run / 'STOP').exists():raise InterruptedError('User stop marker')
        log = root / 'logs' / (ident + '-' + name + '.log')
        with log.open('x') as f:
            child = subprocess.Popen(command, cwd=p['source']['source'], env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            write(root / 'children' / (ident + '-' + name + '.json'), dict(pid=child.pid, command=command, host=row['host'], gpu=gpu))
            try:code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid, signal.SIGKILL);child.wait()
                code = 124
        results.append(dict(name=name, command=command, returncode=code, log=str(log)))
        assert code == 0, str(log)
    def tool(name, module, arguments, external=False, timeout=7200):
        command = [PY, '-B', '-u']
        command += [str(helper / module)] if external else ['-m', module]
        execute(name, command + [str(x) for x in arguments], timeout)
    def plans(step, checkpoint, folder, seeds, suffix=''):
        for seed in seeds:
            tool(f'{step}-plan{seed}{suffix}', 'as_jepa_tools.planning',
                 ['--checkpoint', checkpoint, '--output', folder / ('plan' + str(seed)),
                  '--upstream', B / 'sources/stable-worldmodel', '--data-root', B / 'data/hardtasks-v1/planning',
                  '--test-episodes', Path(cfg['evaluation_data']) / 'test_episodes.npy', '--seed', seed, '--episodes', 50, '--split', 'heldout'])
    def probes(step, folder):
        values = {}
        for noisy in [False, True] if row['model_class'] != 'base' else [False]:
            label = 'noisy' if noisy else 'mean'; out = folder / ('probes-' + label)
            args = ['--checkpoint', cp, '--data', cfg['evaluation_data'], '--output', out, '--train', 20000, '--test', 5000, '--fp32']
            if noisy:args += ['--noisy']
            tool(f'{step}-probe-{label}', 'as_jepa_tools.probes', args)
            values[label] = read(out / 'probes.json')
        return values
    def cube_evaluate(step, folder):
        tool(f'{step}-state', 'as_jepa_tools/round2_state_check.py',
             ['--checkpoint', cp, '--config', run / 'config.json', '--output', folder / 'state'], True)
        tool(f'{step}-content', 'experiments/lingjing_fullspec_20261102/hard_content.py',
             ['--checkpoint', cp, '--config', run / 'config.json', '--output', folder / 'content'], True)
        tool(f'{step}-pose', 'experiments/lingjing_fullspec_20261102/cube_pose.py',
             ['--checkpoint', cp, '--config', run / 'config.json', '--seed', row['seed'], '--output', folder / 'pose'], True)
        for case in ['primary', 'fresh']:
            manifest = p['manifests']['cube'][case]; white = folder / ('white-' + case)
            tool(f'{step}-white-{case}', 'as_jepa_tools.benchmark_evaluate',
                 ['--checkpoint', cp, '--data', cfg['data'], '--manifest', manifest, '--output', white])
            calibration = B / 'reports/action-covariance-20261031/calibration/cube/calibration.json'
            old = Path(p['source']['source']) / 'experiments/action_covariance_20261031'
            spec = dict(task='cube', source=p['source'], mode='correlated', checkpoint=str(cp), data=cfg['data'], manifest=manifest,
                        output=str(folder / ('correlated-' + case)), reference_metrics=str(white / 'metrics.json'),
                        native_evaluator=str(Path(p['source']['source']) / 'as_jepa_tools/benchmark_evaluate.py'),
                        calibration=str(calibration), helper_files={name: sha(old / name) for name in ['adapter.py', 'evaluate.py']},
                        immutable_inputs={str(path): sha(path) for path in [cp / 'weights.pt', cp / 'config.json', Path(manifest),
                                          Path(cfg['data']) / 'dataset_card.json', calibration, calibration.with_name('matrix.npz'), white / 'metrics.json']})
            sp = folder / ('correlated-' + case + '-spec.json'); write(sp, spec)
            execute(f'{step}-correlated-{case}', [PY, '-B', '-u', str(old / 'evaluate.py'), '--spec', str(sp)])
    try:
        reuse_step = row.get('reuse_completed_training_step')
        if reuse_step:
            assert reuse_step == row['stages'][0] and not row.get('resume_parent')
            parent_terminal = Path(row['repair_parent_terminal'])
            binding = read(root / 'bindings' / (ident + '.json'))
            assert sha(parent_terminal) == binding['parent_terminal_sha256']
            if row.get('recovered_stage_proof'):
                complete = read(row['recovered_stage_proof'])
                assert complete['passed'] and complete['step'] == reuse_step
                assert complete['config_sha256'] == sha(run / 'config.json')
                assert complete['full_checkpoint_sha256'] == sha(run / 'last.ckpt')
                assert complete['portable_sha256'] == sha(run / f'model-step{reuse_step:07d}/weights.pt')
            else:
                complete = read(run / 'stage-complete.json')
                assert complete['stage_complete'] and complete['stop_step'] == reuse_step
                assert complete['config_sha256'] == sha(run / 'config.json')
            results.append(dict(name='reuse-completed-training', returncode=0, step=reuse_step,
                parent_terminal=str(parent_terminal), parent_terminal_sha256=sha(parent_terminal),
                reason='Training completed exactly; repair only the missing-versus-zero gradient audit. No repeated SGD.'))
        if row.get('resume_parent'):
            parent = Path(row['resume_parent']); oldcfg = read(parent / 'config.json')
            for key in ['model', 'model_class', 'data', 'seed', 'batch_size', 'max_steps', 'calibrate_state_std', 'gap', 'benchmark_anchor_rule']:
                assert cfg.get(key) == oldcfg.get(key), key
            native_audit(root, dict(row, id=ident + '-parent', run=str(parent)), p['source'], step=2000)
            os.link(parent / 'full-archives/step0002000.ckpt', run / 'last.ckpt')
            shutil.copytree(parent / 'initial-model', run / 'initial-model', copy_function=os.link)
            shutil.copytree(parent / 'model-step0002000', run / 'model-step0002000', copy_function=os.link)
            for name in ['metrics.jsonl', 'stage-runtime-source-0002000.json', 'initial_scale.json', 'initial_encoder.json']:
                if (parent / name).exists():shutil.copy2(parent / name, run / name)
            native_audit(root, row, p['source'], step=2000)
            write(run / 'copy-binding.json', dict(parent=str(parent), parent_sha256=sha(parent / 'full-archives/step0002000.ckpt'),
                  copied_sha256=sha(run / 'last.ckpt'), native_pre_and_post_copy_CPU_restore=True))
        for index, step in enumerate(row['stages']):
            command = [PY, '-B', '-u', '-m', 'as_jepa_tools.train_stage', '--config', str(run / 'config.json'), '--stop-step', str(step)]
            if row.get('safe_checkpoint_io'):
                command = [PY, '-B', '-u', str(helper / 'experiments/lingjing_fullspec_20261102/safe_train_stage.py'), '--config', str(run / 'config.json'), '--stop-step', str(step)]
            if index or row.get('resume_parent'):command.append('--resume')
            if step != reuse_step:execute(str(step) + '-train', command, 172800)
            native_audit(root, row, p['source'], step=step)
            cp = run / f'model-step{step:07d}'; before = sha(cp / 'weights.pt'); folder = root / 'evaluations' / ident / str(step)
            if row['task'] == 'cube':
                cube_evaluate(step, folder); assert before == sha(cp / 'weights.pt'); stages.append(dict(step=step, complete=True)); continue
            probe = probes(step, folder); first = step == row.get('original_first_stage_step', row['stages'][0]); final = step == row['stages'][-1]
            evidence = dict(step=step, probes=probe, gate=None)
            if row['model_class'] == 'decoder_free':
                windows = 256 if final else 64
                tool(f'{step}-latent', 'as_jepa_tools/latent_eval.py',
                     ['--checkpoint', cp, '--data', cfg['evaluation_data'], '--output', folder / 'latent', '--windows', windows, '--fp32'], True)
                latent = read(folder / 'latent/latent-eval.json')
                info = latent['information']; snr = info['state_spread_over_sigma0']; directions = info['diagnostics']['info_dirs']
                evidence['gate'] = dict(snr=snr, noisy_agent_r2=probe['noisy']['agent_r2'], info_dirs=directions,
                     deep_collapse=snr < .3 and probe['noisy']['agent_r2'] <= .05 and directions < 2)
                if cfg['model'].get('domain', 'standard') == 'color':
                    tool(f'{step}-colour', 'as_jepa_tools/colour_invariance.py',
                         ['--checkpoint', cp, '--data', cfg['evaluation_data'], '--output', folder / 'colour'], True)
                if not first or final:plans(step, cp, folder / 'planning', [42, 43] if not final else [42, 43, 44, 1042, 1043, 1044])
            elif row['model_class'] == 'forecast_scored':
                gains = {}
                for fp32 in [True, False]:
                    name = 'fp32' if fp32 else 'bf16'; args = ['--checkpoint', cp, '--data', cfg['evaluation_data'], '--output', folder / ('gain-' + name)]
                    if fp32:args += ['--fp32']
                    tool(f'{step}-gain-{name}', 'as_jepa_tools.rollout_gain', args)
                    gains[name] = read(folder / ('gain-' + name) / 'rollout_gain.json')
                collapsed = {}
                conditions = ['posterior', 'forecast'] if cfg['model']['lam'] > 0 else ['posterior']
                for condition in conditions:
                    for fp32 in [True, False]:
                        name = condition + ('-fp32' if fp32 else '-bf16')
                        args = ['--checkpoint', cp, '--data', cfg['evaluation_data'], '--pose', p['assets']['pose'],
                                '--output', folder / ('collapse-' + name), '--condition', condition,
                                '--cases', 64 if first else 256, '--steps', 20 if first else 50]
                        if fp32:args += ['--fp32']
                        tool(f'{step}-collapse-{name}', 'as_jepa_tools/collapse_audit.py', args, True)
                        collapsed[name] = read(folder / ('collapse-' + name) / 'metrics.json')
                authority = ('forecast' if cfg['model']['lam'] > 0 else 'posterior') + '-fp32'
                evidence['gate'] = dict(snr=gains['fp32']['state_spread_over_sigma0'], noisy_agent_r2=probe['noisy']['agent_r2'],
                     trained_channel_collapsed=collapsed[authority]['collapsed'],
                     deep_collapse=gains['fp32']['state_spread_over_sigma0'] < .3 and probe['noisy']['agent_r2'] <= .05 and collapsed[authority]['collapsed'])
                if final:
                    plans(step, cp, folder / 'planning', [42, 43, 44, 1042, 1043, 1044])
                    tool(f'{step}-quality', 'as_jepa_tools/quality_complete.py', ['--checkpoint', cp, '--data', cfg['evaluation_data'], '--pose', p['assets']['pose'],
                         '--i3d', p['assets']['i3d'], '--output', folder / 'quality', '--cases', 64, '--samples', 1, '--steps', 20, '--selection-seed', 20261029], True)
                    tool(f'{step}-counterfactual', 'as_jepa_tools.counterfactual', ['--checkpoint', cp, '--data', cfg['evaluation_data'], '--pose', p['assets']['pose'],
                         '--output', folder / 'counterfactual', '--cases', 16, '--steps', 20, '--selection-seed', 20261029])
                    for fp32 in [True, False]:
                        args = ['--checkpoint', cp, '--data', cfg['evaluation_data'], '--pose', p['assets']['pose'], '--output', folder / ('decomposition-fp32' if fp32 else 'decomposition-bf16'), '--cases', 128, '--steps', 50]
                        if fp32:args += ['--fp32']
                        tool(f'{step}-decomposition-{fp32}', 'as_jepa_tools.decode_decomposition', args)
            elif row['model_class'] == 'nce':
                evidence['gate'] = dict(snr=probe['mean']['feature_std'] / cfg['model']['sigma0'],
                     deep_collapse=probe['mean']['feature_std'] / cfg['model']['sigma0'] < .3 and probe['noisy']['agent_r2'] <= .05)
                if final:plans(step, cp, folder / 'planning', [42, 43, 44, 1042, 1043, 1044])
            else:
                assert row['model_class'] == 'base'
                tool(f'{step}-colour', 'as_jepa_tools/colour_invariance.py', ['--checkpoint', cp, '--data', cfg['evaluation_data'], '--output', folder / 'colour'], True)
                if final:
                    tool(f'{step}-bn', 'experiments/lingjing_fullspec_20261102/l0_bn.py', ['--run', run, '--step', step, '--output', folder / 'bn'], True)
                    for variant, checkpoint in [('raw', cp), ('bn', folder / 'bn/model')]:
                        plans(step, checkpoint, folder / variant, [42, 43, 44, 1042, 1043, 1044], suffix='-' + variant)
            assert before == sha(cp / 'weights.pt')
            stages.append(evidence); write(folder / 'stage-evidence.json', evidence)
            if first and not final and evidence['gate'] and evidence['gate']['deep_collapse']:
                outcome = 'early_collapsed_at_first_epoch'; write(run / 'EARLY_STOP.json', dict(evidence=evidence, user_priority='Insight-first; no empty-code scaling.',
                     scope='A completed early diagnostic,not the nominal budget or a proof of irreversible collapse.')); break
        else:outcome = 'nominal_budget_complete'
    except Exception as exc:error = repr(exc)
    write(root / 'terminals' / (ident + '.json'), dict(passed=outcome != 'execution_error', id=ident, outcome=outcome,
          nominal_budget_complete=outcome == 'nominal_budget_complete', results=results, stages=stages, error=error,
          config_sha256=sha(run / 'config.json'), seconds=time.monotonic() - started, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    if outcome == 'execution_error':raise RuntimeError(error)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--id', required=True); p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args(); main(a.root, a.id, a.gpu)
