"""Reuse completed experiments to answer state/use questions before repeating training."""
import argparse
from pathlib import Path
from common import B, PY, read, write, sha, source_check


def main(root, source, helper):
    root = Path(root); assert not root.exists(); root.mkdir()
    source, helper = read(source), read(helper); source_check(source); source_check(helper)
    jobs = []
    parent = B / 'runs/branch_trials_20261102'
    repaired = B / 'runs/branch_repairs_20261102'
    p = read(parent / 'protocol.json')
    q = read(repaired / 'protocol.json')
    def add(row, checkpoint, kind, fp32, tag):
        ident = f'{tag}-{row["id"]}-{"fp32" if fp32 else "bf16"}'
        out = root / 'jobs' / ident
        tool = 'df_state_check' if kind == 'decoder' else 'round2_state_check'
        cmd = [PY, '-B', '-u', helper['source'] + f'/as_jepa_tools/{tool}.py', '--checkpoint', str(checkpoint),
               '--config', str(Path(row['run']) / 'config.json'), '--output', str(out)]
        if kind == 'decoder': cmd += ['--count', '256']
        elif not fp32: cmd += ['--bf16']
        jobs.append(dict(id=ident, checkpoint=str(checkpoint), weights_sha256=sha(checkpoint / 'weights.pt'),
                         config=str(Path(row['run']) / 'config.json'), config_sha256=sha(Path(row['run']) / 'config.json'),
                         output=str(out), kind=kind, fp32=fp32, command=cmd))
    for row in p['runs']:
        if row['family'] == 'round2':
            for fp32 in [True, False]: add(row, Path(row['run']) / 'model-step0002000', 'round2', fp32, 'original2000')
        if row['family'] == 'decoder' and row['task'] == 'tworoom':
            add(row, Path(row['run']) / 'model-step0002000', 'decoder', True, 'original2000')
            if row['arm'] == 'pooled': add(row, Path(row['run']) / 'initial-model', 'decoder', True, 'initial-control')
    for row in q['runs']:
        if row['family'] == 'decoder' and row['task'] == 'tworoom':
            add(row, Path(row['run']) / 'model-step0002000', 'decoder', True, 'repaired2000')
    assert len(jobs) == 26
    hosts = ['dev2', 'dev13', 'dev15']
    for i, job in enumerate(jobs):job['host'] = hosts[i % 3]
    write(root / 'protocol.json', dict(source=source, helper=helper, jobs=jobs,
          reserved_slots=dict(dev2=[5, 6, 7], dev13=[5, 6, 7], dev15=[4, 5, 6, 7]),
          model_updates=0, simulation_calls=0, continuous_monitor=False,
          scope='All eight original round2 TwoRoom2x2 cells with both precisions; all original/repaired DF pooled/window TwoRoom models plus two calibrated initial positive controls. Reuse retained checkpoints, no stopped training is resumed.'))
    print(dict(jobs=len(jobs)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--source', required=True); p.add_argument('--helper', required=True)
    a = p.parse_args(); main(a.root, a.source, a.helper)
