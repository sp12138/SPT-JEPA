"""Read-only final index. Requires the finite campaign to have ended; never launches work."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric(path, seed):
    value = read(path)
    assert value['seed'] == seed
    assert len(value['metrics']['episode_successes']) == 50
    assert len(value['episodes']) == len(value['start_steps']) == 50
    assert abs(value['metrics']['success_rate'] - 2 * sum(value['metrics']['episode_successes'])) < 1e-8
    return dict(path=str(path), sha256=sha(path), value=value)


def main(root, coverage, frozen_gates, output):
    root, output = Path(root), Path(output)
    # No partial snapshot can be published as final. The existing collector owns this dependency.
    summary_path = root / 'results/summary.json'
    summary = read(summary_path)
    protocol = read(root / 'protocol.json')
    assert summary['passed'] and summary['jobs'] == len(protocol['jobs']) == 106
    assert summary['protocol_sha256'] == sha(root / 'protocol.json')
    assert (root / 'HOLD').exists()
    assert all((root / 'dispatchers' / f'{host}-finished.json').exists() for host in ['dev2', 'dev13', 'dev15'])
    assert {r['job']['id'] for r in summary['rows']} == {j['id'] for j in protocol['jobs']}
    allowed = {'nominal_budget_complete', 'inherited_nominal_budget_complete', 'early_collapsed_at_first_epoch'}
    assert all(r['terminal']['passed'] and r['terminal']['outcome'] in allowed for r in summary['rows'])
    catalog = read(coverage)
    statuses, performance, proofs, mechanisms = {}, [], [], []
    for row in summary['rows']:
        job, terminal = row['job'], row['terminal']
        ident = job['id']; base = root
        if row['correction_binding']:
            binding = row['correction_binding']; base = Path(binding['root'])
            assert sha(row['original_terminal']['path']) == binding['original_terminal_sha256']
            assert read(binding['terminal']) == terminal
        if terminal['outcome'] == 'inherited_nominal_budget_complete':
            assert sha(terminal['parent']['path']) == terminal['parent']['sha256']
            base = Path(terminal['parent']['path']).parent.parent
        nominal = terminal.get('nominal_budget_complete', False)
        step = job['stages'][-1] if nominal else terminal['stages'][-1]['step']
        audit_path = base / 'audits' / f'{ident}-step{step}.json'
        audit = read(audit_path)
        assert audit['passed'] and audit['native_cpu_model_optimizer_scheduler_restore']
        assert audit['learning_paths_active'] and all(audit['checkpoint']['checks'].values())
        assert audit['checkpoint']['step'] == step
        run = Path(job['run'])
        for file in [run / f'full-archives/step{step:07d}.ckpt', run / f'model-step{step:07d}/weights.pt']:
            assert file.is_file() and file.stat().st_size > 0, str(file)
        proofs.append(dict(id=ident, step=step, nominal_budget_complete=nominal, audit=str(audit_path), audit_sha256=sha(audit_path),
                           full_checkpoint=str(run / f'full-archives/step{step:07d}.ckpt'),
                           full_checkpoint_sha256=audit['full_checkpoint_sha256'], portable_sha256=audit['portable_sha256']))
        statuses[ident] = dict(status=terminal['outcome'], nominal_budget_complete=nominal,
                               last_step=step, evidence=str(base / 'terminals' / f'{ident}.json'))
        if not nominal or job['task'] != 'pusht':
            continue
        arms = []
        for variant in (['raw', 'bn'] if job['model_class'] == 'base' else ['planning']):
            folder = base / 'evaluations' / ident / str(step) / variant
            metrics = [metric(folder / f'plan{s}/metrics.json', s) for s in [42, 43, 44, 1042, 1043, 1044]]
            arms.append(dict(variant=variant, primary=statistics.mean(m['value']['metrics']['success_rate'] for m in metrics[:3]),
                             fresh=statistics.mean(m['value']['metrics']['success_rate'] for m in metrics[3:]), metrics=metrics))
        performance.append(dict(id=ident, track=job['track'], block=job['block'], arm=job['arm'], seed=job['seed'],
                                steps=step, model_class=job['model_class'], model=job['model'], arms=arms))
        if job['model_class'] == 'decoder_free':
            path = base / 'evaluations' / ident / str(step) / 'latent/latent-eval.json'
            latent = read(path)
            assert latent['model_updates'] == 0 and latent['state_before'] == latent['state_after']
            # Preserve scalar diagnostics without duplicating large case banks in the compact index.
            info = {k: v for k, v in latent['information'].items() if isinstance(v, (int, float, bool, str))}
            info['diagnostics'] = latent['information']['diagnostics']
            rollout = latent['rollout']
            mechanisms.append(dict(id=ident, path=str(path), sha256=sha(path), information=info,
                                   noisy_r2=latent['probes']['noisy_r2'],
                                   h7_agent_ratio=rollout['errors']['agent']['h7']['ratio_to_copy'],
                                   h7_moving_block_ratio=rollout['h7_moving_block_ratio']))
    # P1 and the TwoRoom state gates are separate campaigns, not missing main-queue jobs.
    runs = root.parent
    p1_root = runs / 'lingjing_fullspec_P1_20261102'
    p1 = read(p1_root / 'results/summary.json')
    assert p1['passed'] and p1['training_complete'] and len(p1['terminals']) == 3
    hard = read(runs / 'lingjing_fullspec_hard2000_20261102/results/summary.json')
    assert hard['passed']
    hard_rows = {r['job']['id']: r for r in hard['rows']}
    frozen = {r['job']['id']: r for r in read(frozen_gates)['frozen']}
    for row in catalog['rows']:
        ident = row['id']
        if ident in statuses:
            continue
        if row['block'] == 'P1':
            t = next(t for t in p1['terminals'] if t['id'] == f"L0-s{row['seed']}")
            assert t['passed']
            statuses[ident] = dict(status='matched_baseline_nominal_complete', nominal_budget_complete=True,
                                   last_step=152650, evidence=str(p1_root / 'terminals' / f"{t['id']}.json"))
        else:
            assert row['task'] == 'tworoom' and row['coverage_status'] == 'gate_failed_stop_no_6000'
            if ident in hard_rows:
                proof = hard_rows[ident]
                assert proof['terminal']['passed'] and not proof['gate_passed']
            else:
                assert len(row['evidence']) == 1
                ref = row['evidence'][0].split('#')[1]
                proof = frozen[ref]['result']
                assert proof.get('state_gate', proof.get('use_gate')) is False
            statuses[ident] = dict(status='gate_failed_stop_no_6000', nominal_budget_complete=False,
                                   last_step=2000, evidence=row['evidence'])
    assert len(statuses) == len(catalog['rows']) == 135
    counts = dict(Counter(r['status'] for r in statuses.values()))
    result = dict(passed=True, checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  main_summary=dict(path=str(summary_path), sha256=sha(summary_path)),
                  main_counts=summary['outcomes'], primary_counts=counts, primary_recipes=135, unfinished=0,
                  preserved_original_errors=sum(r['original_terminal'] is not None for r in summary['rows']),
                  archive_proofs=proofs, primary_dispositions=statuses,
                  scope='Final evidence collection only. Existing native restore/audit proofs indexed; no repeated model restore, SGD, GPU or simulator calls. '
                        'Early gates are dispositions, not full nominal training. P2c/new seeds/Stage2/DFPhase2 not launched. '
                        'Process exit/resource verification is recorded separately; this file alone does not prove GPUs idle.')
    output.mkdir(parents=True, exist_ok=False)
    for name, data in [('summary.json', result), ('completed-performance.json', dict(rows=performance)),
                       ('decoder-mechanisms.json', dict(rows=mechanisms))]:
        (output / name).write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(dict(passed=True, primary_counts=counts, completed_performance=len(performance),
                          decoder_mechanisms=len(mechanisms), output=str(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['root', 'coverage', 'frozen-gates', 'output']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    main(args.root, args.coverage, args.frozen_gates, args.output)
