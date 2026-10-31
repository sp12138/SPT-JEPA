"""Four matched Cube continuations after a completed positive controller screen."""
import argparse
import concurrent.futures
import copy
import os
from pathlib import Path
import shutil
import subprocess
from common import B, PY, read, sha, write

INITIAL, GATE, CAP = 23342, 23942, 35013


def semantic(config):
    return {k: v for k, v in config.items() if k not in
            ('output', 'checkpoint_every', 'retain_steps', 'is_resuming', 'slurm.job_id', 'slurm.task_id')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    a = p.parse_args()
    root = Path(a.root)
    assert not root.exists()
    root.mkdir()
    prior_root = B / 'runs/cube_maturity23342_20261031'
    old = read(prior_root / 'plan.json')
    protocol = read(prior_root / 'protocol.json')
    evidence_path = B / 'reports/trajectory-search-20261101/results/results.json'
    evidence = read(evidence_path)
    assert evidence['passed'] and evidence['endpoints'] == 24
    # This is a preset eligibility check, not a best-of search over controllers.
    for cases in ('primary', 'fresh'):
        contrast = next(x for x in evidence['contrasts'] if x['family'] == 'tf' and x['cases'] == cases
                        and x['treatment'] == 'retrieval' and x['control'] == 'original' and x['seed'] == 'mean_of_fixed_two')
        assert contrast['mean'] >= 10 and contrast['ci95'][0] > 0
    helper = protocol['helper']
    auditor = Path(helper['source']) / 'experiments/training_maturity_20261031/audit.py'
    assert sha(auditor) == read(helper['manifest'])['files']['experiments/training_maturity_20261031/audit.py']
    rows, commands = [], []
    for family, slug, host in [('native', 'native-lewm', 'dev2'), ('tf', 'temporal-forecast', 'dev13')]:
        for seed in (1, 2):
            name = f'cube-{slug}-seed{seed}'
            parent, run = prior_root / name, root / name
            assert read(parent / 'status.json')['state'] == 'finished' and read(parent / 'status.json')['step'] == INITIAL
            before = sha(parent / 'last.ckpt')
            prior_proof = read(prior_root / f'archive{INITIAL}-{name}' / 'fullstate.json')
            assert prior_proof['passed'] and prior_proof['checkpoint_sha256'] == before
            prior_cfg = read(parent / 'config.json')
            config = copy.deepcopy(prior_cfg)
            config.update(output=str(run), checkpoint_every=2000, retain_steps=[INITIAL, GATE, 30000, CAP])
            assert semantic(config) == semantic(prior_cfg)
            assert config['batch_size'] == 128 and config['max_steps'] == 124910
            run.mkdir()
            write(run / 'config.json', config)
            shutil.copy2(parent / 'last.ckpt', run / 'last.ckpt')
            shutil.copytree(parent / f'model-step{INITIAL:07d}', run / f'model-step{INITIAL:07d}')
            assert before == sha(parent / 'last.ckpt') == sha(run / 'last.ckpt')
            source = next(j['source'] for j in old['jobs'] if j['id'] == 'original-primary-' + name)
            source_path = root / 'model-sources' / (name + '.json')
            write(source_path, source)
            full = next(j for j in old['jobs'] if j['id'] == 'fullstate-' + name)
            health_command = full['command'].copy()
            def argument(cmd, key, value):
                if key in cmd:
                    cmd[cmd.index(key) + 1] = str(value)
                else:
                    cmd.extend([key, str(value)])
            argument(health_command, '--run', run)
            argument(health_command, '--output', root / ('gate-health-' + name))
            argument(health_command, '--expected-step', GATE)
            argument(health_command, '--initial-weights', parent / f'model-step{INITIAL:07d}' / 'weights.pt')
            output = root / 'resume-audits' / (name + '.json')
            cmd = [PY, '-B', str(auditor), '--run', str(run), '--parent', str(parent), '--initial',
                   str(parent / f'model-step{INITIAL:07d}' / 'weights.pt'), '--source-record', str(source_path),
                   '--family', family, '--output', str(output), '--step', str(INITIAL)]
            commands.append((cmd, source, output))
            rows.append(dict(id=name, family=family, seed=seed, host=host, gpu=seed-1, run=str(run), parent=str(parent),
                  source=source, config_sha256=sha(run / 'config.json'), initial_full_checkpoint_sha256=before,
                  reused_original_CPU_restore=str(prior_root / f'archive{INITIAL}-{name}' / 'fullstate.json'),
                  actual_postcopy_CPU_restore=str(output),
                  gate_command=[PY, '-B', '-u', '-m', 'as_jepa_tools.train_stage', '--config', str(run / 'config.json'), '--stop-step', str(GATE), '--resume'],
                  gate_health_command=health_command,
                  long_command=[PY, '-B', '-u', '-m', 'as_jepa_tools.train_stage', '--config', str(run / 'config.json'), '--stop-step', str(CAP), '--resume']))
    def restore(entry):
        cmd, source, output = entry
        env = dict(os.environ, PYTHONPATH=source['pythonpath'], PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        log = output.with_suffix('.log'); log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('x') as handle:
            result = subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=600)
        assert result.returncode == 0, str(log)
        assert read(output)['passed']
        return str(output)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(restore, commands))
    write(root / 'protocol.json', dict(runs=rows, initial_step=INITIAL, health_gate_step=GATE, stop_step=CAP,
          original_scheduler_max_steps=124910, original_batch_size=128, old_environment=old['environment'],
          performance_evidence=str(evidence_path), performance_evidence_sha256=sha(evidence_path),
          actual_postcopy_CPU_restores=actual, before_copy_original_CPU_restores_reused=4,
          semantic_training_config_unchanged=True, new_parameters=0, model_sources_unchanged=True,
          root_review_required_before_gate=True, root_review_required_after_gate_before_long=True,
          gate_review='All600 new updates finite with active original module gradients; native full checkpoint/backward health; no persistent collapse. Review all four runs before continuation.',
          endpoint_evaluation='At35013 retain all models/seeds. Evaluate original,random-training-library,retrieval under the same common controllers and primary/fresh/newcases. No performance claim from longer exposure alone.',
          scope='Evidence-selected third-loader-epoch extension of unchanged TF/native learners with a useful common controller. No new learned objective/novelty claim; no fourth epoch or automatic old-grid resume;no continuous monitor.'))
    write(root / 'preparation.json', dict(passed=True, runs=4, actual_CPU_restore_after_copy=4,
          original_CPU_restore_proofs_reused=4, launched=False, protocol_sha256=sha(root / 'protocol.json')))
    (root / 'HOLD_DISPATCH').write_text('Prepared;root actual-free/health-gate review required. Preserve old HOLDs.\n')
    print('Four matched full-state continuations prepared;four actual post-copy CPU restores pass.', flush=True)


if __name__ == '__main__':
    main()
