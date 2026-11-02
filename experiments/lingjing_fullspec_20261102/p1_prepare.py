"""The three missing matched L0 baselines, with original training settings and explicit finite caps."""
import argparse
from pathlib import Path
from common import B, read, write, sha, source_check


def main(root, helper):
    root = Path(root); assert not root.exists(); root.mkdir()
    report = B / 'reports/lingjing_fullspec_20261102'
    source, helper = read(report / 'source-record-v1.json'), read(helper); source_check(source); source_check(helper)
    old = read(B / 'runs/lingjing_fullspec_20261102/protocol.json')
    jobs = []
    for seed in [0, 1, 2]:
        template = report / 'P1-templates' / f'full-L0-10epoch-seed{seed}.json'
        cfg = read(template); assert cfg['seed'] == seed and cfg['max_steps'] == 152650 and cfg['model']['mode'] == 'L0'
        ident = f'L0-s{seed}'; run = root / ident
        cfg.update(output=str(run), retain_steps=[30530, 45795, 152650], record_initial_weights=True,
                   export_initial=True, record_runtime_source=True,
                   audit_expectations=dict(gradients=dict(encoder='active', predictor='active', action_encoder='active',
                                                          projector='active', pred_proj='active', renderer='optional')))
        write(run / 'config.json', cfg)
        jobs.append(dict(id=ident, seed=seed, run=str(run), task='pusht', model_class='base', reference_initial=None,
                         config_sha256=sha(run / 'config.json'), original_template=str(template), template_sha256=sha(template),
                         host='dev2', gpu=7 if seed == 0 else seed - 1))
    write(root / 'protocol.json', dict(source=source, helper=helper, jobs=jobs, environment=old['environment'],
          stages=[30530, 45795, 152650], evaluation_stages=[45795, 152650], plan_seeds=[42, 43, 44, 1042, 1043, 1044],
          seed0_dependency='After GPU7 interface test exits; slots0–6 are disjoint hard-task training.',
          seed12_dependency='After all36 hard-task chains complete and all3 dispatchers exit.',
          scope='P1: three exact original L0 training templates; output/retention and operational provenance/gradient logging flags only. No random-encoder transplant, LR/model/loss/data change. Raw and fixed train-only precise-BN scores both retained. Baseline uses SIGReg/BatchNorm and is not a sample-level candidate. Finite152650 cap, no monitor or next epoch.'))
    print(dict(jobs=3, cap=152650))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--helper', required=True)
    a = p.parse_args(); main(a.root, a.helper)
