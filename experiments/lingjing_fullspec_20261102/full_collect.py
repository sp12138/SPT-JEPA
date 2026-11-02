"""Index every outcome without selecting a winner or treating early evidence as full maturity."""
import argparse
import datetime
from pathlib import Path
import time
from common import read, write, sha


def main(root):
    root = Path(root); p = read(root / 'protocol.json'); started = time.monotonic()
    endings = [root / 'dispatchers' / (host + '-finished.json') for host in ['dev2', 'dev13', 'dev15']]
    def corrections_complete():
        return all(Path(read(path)['terminal']).exists() for path in (root / 'CORRECTIONS').glob('*.json'))
    while not all(path.exists() for path in endings) or not corrections_complete():
        if time.monotonic() - started > 259200:raise TimeoutError('Finite72h collection dependency ended; no automatic extension.')
        time.sleep(30)
    rows = []
    for job in p['jobs']:
        terminal_path = root / 'terminals' / (job['id'] + '.json'); terminal = read(terminal_path)
        original_terminal = None
        correction = root / 'CORRECTIONS' / (job['id'] + '.json')
        if correction.exists():
            binding = read(correction); assert binding['original_terminal_sha256'] == sha(terminal_path)
            original_terminal = dict(path=str(terminal_path), sha256=sha(terminal_path), terminal=terminal)
            terminal_path = Path(binding['terminal']); terminal = read(terminal_path)
        evidence = []
        for step in job['stages']:
            base = root / 'evaluations' / job['id'] / str(step)
            if not base.exists():continue
            keep = []
            for pattern in ['**/metrics.json', '**/probes.json', '**/latent-eval.json', '**/state-check.json', '**/pose.json',
                            '**/colour-invariance.json', '**/rollout_gain.json', '**/decomposition.json', '**/counterfactual.json', 'stage-evidence.json']:
                keep += list(base.glob(pattern))
            evidence.append(dict(step=step, artifacts=[dict(path=str(path), sha256=sha(path)) for path in sorted(set(keep))]))
        audits = [dict(path=str(path), sha256=sha(path)) for path in (root / 'audits').glob(job['id'] + '*.json')]
        rows.append(dict(job=job, terminal=terminal, terminal_sha256=sha(terminal_path), evidence=evidence, audits=audits,
                         original_terminal=original_terminal, correction_binding=read(correction) if correction.exists() else None))
    outcomes = {}
    for row in rows:outcomes[row['terminal']['outcome']] = outcomes.get(row['terminal']['outcome'], 0) + 1
    report = dict(passed=all(row['terminal'].get('passed') for row in rows), outcomes=outcomes, jobs=len(rows), rows=rows,
                  protocol_sha256=sha(root / 'protocol.json'), finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope='All primary-job outcomes, including early-budget collapse, numerical/implementation errors and full maturity. No best-of seeds/controllers/cases; early stops are not nominal completion. P2c/fallbacks/DFPhase2 remain separately gated scientific decisions; no fourth epoch or new research round is launched.')
    write(root / 'results/summary.json', report)
    if not (root / 'HOLD').exists():write(root / 'HOLD', dict(reason='Finite proposed-recipe queue complete; all evidence retained, no new round.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    main(p.parse_args().root)
