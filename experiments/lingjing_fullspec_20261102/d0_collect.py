"""Collect every prespecified D0 job, including both precisions and all aliases."""
import argparse
import math
from pathlib import Path

from common import read, write, sha, source_check


def sanitise(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: sanitise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitise(v) for v in value]
    return value


def main(root):
    root = Path(root)
    p = read(root / 'protocol.json')
    source_check(p['source'])
    terminal = {}
    for host in ['dev2', 'dev13', 'dev15']:
        done = read(root / 'dispatchers' / f'{host}-finished.json')
        assert done['passed'], done
        terminal.update({r['id']: r for r in done['results']})
    assert set(terminal) == {r['id'] for r in p['jobs']}
    rows = []
    for job in p['jobs']:
        out = root / 'jobs' / job['id']
        proof = read(out / 'proof.json')
        result = read(out / 'rollout_gain.json')
        assert terminal[job['id']]['returncode'] == 0
        assert proof['passed'] and proof['job'] == job
        assert proof['state_before'] == proof['state_after']
        assert proof['model_updates'] == proof['simulation_calls'] == 0
        assert result['fp32'] == job['fp32'] and result['cases'] == 128
        rows.append(dict(job=job, result=sanitise(result), proof=proof,
                         result_sha256=sha(out / 'rollout_gain.json'), proof_sha256=sha(out / 'proof.json')))
    report = dict(passed=True, complete_jobs=len(rows), references=len(p['references']),
                  protocol_sha256=sha(root / 'protocol.json'), rows=rows,
                  scope='All requested existing references; no training or simulation. Undefined ratios from zero truth carry are null in this summary; original JSON is preserved. No causal mediation or benchmark superiority claim.')
    write(root / 'results' / 'summary.json', report)
    for family in sorted({r['job']['family'] for r in rows}):
        selected = [r for r in rows if r['job']['family'] == family and r['job']['fp32']]
        print(family, [dict(seed=r['job']['seed'], step=r['job']['step'],
                           snr=round(r['result']['state_spread_over_sigma0'], 4),
                           noisy_slope=round(r['result']['one_step_noisy_gain'], 4),
                           h1carry=r['result']['open_loop']['carry_ratio'][0],
                           h7carry=r['result']['open_loop']['carry_ratio'][6],
                           h7error=r['result']['open_loop']['error_vs_copy'][6]) for r in selected])
    print(dict(passed=True, jobs=len(rows)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    main(p.parse_args().root)
