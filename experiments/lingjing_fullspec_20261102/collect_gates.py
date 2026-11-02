"""Complete-only collection; retain every arm/seed/precision, with no winner selection."""
import argparse
from pathlib import Path
import numpy as np
from common import read, write, sha


def main(smoke, frozen, output):
    smoke, frozen = Path(smoke), Path(frozen)
    p, q = read(smoke / 'protocol.json'), read(frozen / 'protocol.json')
    for root in [smoke, frozen]:
        for host in ['dev2', 'dev13', 'dev15']:
            done = read(root / 'dispatchers' / (host + '-finished.json'))
            assert done['passed'], done
    smokes = []
    for job in p['jobs']:
        terminal = read(smoke / 'smoke-results' / (job['id'] + '.json'))
        audit = read(smoke / 'audits' / (job['id'] + '-200.json'))
        assert terminal['passed'] and audit['passed']
        times = [row['seconds_per_step'] for row in audit['trace'] if row['state'] == 'running']
        smokes.append(dict(job=job, terminal=terminal, audit=audit,
                           seconds_per_step=float(np.mean(times)),
                           peak_gib=max(row['peak_gib'] for row in audit['trace'])))
    frozen_rows = []
    for job in q['jobs']:
        terminal = read(frozen / 'terminals' / (job['id'] + '.json'))
        assert terminal['returncode'] == 0
        proof = terminal['proof']
        assert proof['state_before'] == proof['state_after'] and proof['model_updates'] == 0
        frozen_rows.append(dict(job=job, result=proof))
    report = dict(passed=True, technical_smokes=len(smokes), frozen_diagnostics=len(frozen_rows),
                  smoke_protocol_sha256=sha(smoke / 'protocol.json'), frozen_protocol_sha256=sha(frozen / 'protocol.json'),
                  smokes=smokes, frozen=frozen_rows,
                  scope='passed refers to execution/provenance, not algorithm success. All state/use failures are retained; calibrated initial models are controls, not trained solutions.')
    write(output, report)
    for row in smokes:
        print('smoke', row['job']['id'], round(row['seconds_per_step'], 3), round(row['peak_gib'], 2))
    for row in frozen_rows:
        result = row['result']
        print('frozen', row['job']['id'], result['feature_std_over_sigma0'],
              result.get('state_swap_relative_error_increase'), result.get('clean_agent_r2'), result.get('noisy_agent_r2'),
              result.get('use_gate', result.get('state_gate')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--smoke', required=True); p.add_argument('--frozen', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); main(a.smoke, a.frozen, a.output)
