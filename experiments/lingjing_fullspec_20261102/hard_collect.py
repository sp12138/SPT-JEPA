"""Complete-only paired mechanism report; an early gate is not nominal maturity."""
import argparse
from pathlib import Path
from common import read, write, sha, source_check


def main(root):
    root = Path(root); p = read(root / 'protocol.json'); source_check(p['source']); source_check(p['helper'])
    for host in ['dev2', 'dev13', 'dev15']:
        done = read(root / 'dispatchers' / (host + '-finished.json')); assert len(done['results']) == 12
    rows = []; groups = {}
    for job in p['jobs']:
        terminal = read(root / 'terminals' / (job['id'] + '.json'))
        row = dict(job=job, terminal=terminal, audit=None, state=None, content=None)
        if terminal['passed']:
            run = Path(job['run']); cfg = read(run / 'config.json'); assert sha(run / 'config.json') == job['config_sha256']
            audit = read(root / 'audits' / (job['id'] + '.json'))
            assert audit['passed'] and audit['checkpoint']['step'] == 2000
            assert audit['native_cpu_model_optimizer_scheduler_restore'] and audit['initial_common_non_head_tensors_exact'] > 0
            assert audit['full_checkpoint_sha256'] == sha(run / 'full-archives/step0002000.ckpt')
            state = read(root / 'evaluations' / job['id'] / 'state/state-check.json')
            assert state['state_before'] == state['state_after'] and state['model_updates'] == 0
            content = None
            if job['model_class'] == 'forecast_scored':
                content = read(root / 'evaluations' / job['id'] / 'content/content.json')
                assert content['passed'] and content['state_before'] == content['state_after']
            row.update(audit=audit, state=state, content=content, configuration=cfg,
                       nominal_budget_completed=False,
                       gate_passed=bool(state.get('use_gate', state.get('state_gate'))) and audit['learning_paths_active'])
        rows.append(row)
        groups.setdefault((job['track'], job['task'], job['arm'], bool(job['conditional_diagnostic'])), []).append(row)
    paired = []
    for key, members in groups.items():
        assert {r['job']['seed'] for r in members} == {1, 2}
        complete = all(r['terminal']['passed'] for r in members)
        both = complete and all(r['gate_passed'] for r in members)
        paired.append(dict(track=key[0], task=key[1], arm=key[2], conditional=key[3],
                           ids=[r['job']['id'] for r in members], complete=complete, both_gate_pass=both,
                           eligible_for_prescribed6000=both and key[1] == 'tworoom',
                           cube12000_status='requires paired scientific review; not silently counted as finished' if key[1] == 'cube' else None))
    report = dict(passed=all(r['terminal']['passed'] for r in rows), executed_chains=len(rows),
                  primary_recipes=32, conditional_diagnostics=4, rows=rows, paired=paired,
                  protocol_sha256=sha(root / 'protocol.json'),
                  scope='Exactly2000 pernewrun. Complete execution is distinct from passing state/use gates and from nominal6000/12000 performance. All seeds and negative evidence retained. No full task-planning scores are manufactured from these diagnostics.')
    write(root / 'results/summary.json', report)
    if not (root / 'HOLD').exists():write(root / 'HOLD', dict(reason='All finite2000 chains ended; preserve evidence and review paired gates before longer candidate training.'))
    print(dict(passed=report['passed'], chains=len(rows), eligible_tworoom=[r['ids'] for r in paired if r['eligible_for_prescribed6000']]))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    main(p.parse_args().root)
