"""Prescribed six existing Cube11671 controls; no replacement with later checkpoints."""
import argparse
from pathlib import Path
from common import B, read, write, sha, source_check


def main(root, helper):
    root = Path(root); assert not root.exists(); root.mkdir()
    source = read(B / 'reports/lingjing_fullspec_20261102/source-record-v1.json')
    helper = read(helper); source_check(source); source_check(helper)
    refs, missing = [], []
    for family in ['original', 'lower-beta', 'temporal-forecast']:
        for seed in [1, 2]:
            cp = B / 'runs/hardtasks_cube_epoch1_20261030' / f'cube-{family}-seed{seed}-epoch1' / 'model-step0011671'
            if not (cp / 'weights.pt').exists():
                missing.append(str(cp)); continue
            refs.append(dict(family=family, seed=seed, step=11671, checkpoint=str(cp),
                             weights_sha256=sha(cp / 'weights.pt'), aliases=[]))
    jobs = []
    for ref in refs:
        for fp32 in [True, False]:
            jobs.append(dict(ref, id=f'{ref["family"]}-s{ref["seed"]}-n11671-{"fp32" if fp32 else "bf16"}',
                             fp32=fp32, host=['dev2', 'dev13', 'dev15'][len(jobs) % 3]))
    write(root / 'protocol.json', dict(source=source, helper=helper, references=refs, jobs=jobs, missing=missing,
          dataset='benchmark', data=str(B / 'data/hardtasks-v1/cube-metadata'), cases=128, window=16, context=3,
          model_updates=0, simulation_calls=0, scope='Exactly the prescribed original/lower-beta/TF11671 Cube controls; missing exports are recorded rather than retrained or replaced.'))
    print(dict(references=len(refs), jobs=len(jobs), missing=missing))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--helper', required=True)
    a = p.parse_args(); main(a.root, a.helper)
