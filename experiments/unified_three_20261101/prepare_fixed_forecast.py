"""Twelve fixed-measurement clean-forecast technical gates;no long promotion."""
import argparse
import copy
from pathlib import Path

from common import B, read, sha, source_check, write
from fixed_forecast_contracts import model_kwargs


def main():
    parser = argparse.ArgumentParser()
    for key in ['root', 'source', 'cpu']:parser.add_argument('--'+key, required=True)
    parser.add_argument('--bind-encoder', action='store_true')
    parser.add_argument('--worker');parser.add_argument('--audit-helper')
    args = parser.parse_args();root = Path(args.root);assert not root.exists()
    source, cpu = read(args.source), read(args.cpu);source_check(source)
    assert cpu['passed'] and len(cpu['full_actual_models']) == 6 and not cpu['CUDA_initialized']
    parent = B / 'runs/unified_three_20261101';original = read(parent / 'protocol.json');runs = []
    for row in original['runs']:
        if row['arm'] != 'flat':continue
        for kind in ['flat', 'spatial']:
            ident = f'{row["task"]}-{"flat" if kind == "flat" else "field"}-seed{row["seed"]}'
            cfg = copy.deepcopy(read(Path(row['run']) / 'config.json'))
            cfg['output'] = str(root / ident);cfg['model_class'] = 'fixed_forecast';cfg['model'] = model_kwargs(cfg, kind)
            if args.bind_encoder:
                cfg['initial_encoder_checkpoint']=str(Path(row['run'])/'initial-model')
                cfg['frozen_encoder_sha256']=sha(Path(cfg['initial_encoder_checkpoint'])/'weights.pt')
            path = root / ident / 'config.json';write(path, cfg)
            runs.append(dict(row, id=ident, arm='flat' if kind == 'flat' else 'field', run=str(path.parent),
                             predictor_kind=kind, gpu=(row['seed']-1)*3 + (0 if kind == 'flat' else 1),
                             config_sha256=sha(path)))
    assert len(runs) == 12
    for task in ['pusht','tworoom','cube']:write(root / 'cohorts' / (task+'.json'), read(parent / 'cohorts' / (task+'.json')))
    write(root / 'source-record.json', source)
    env=dict(original['environment'])
    if args.bind_encoder:env['TMPDIR']='/tmp/spt-jepa-fixed-forecast-20261101'
    write(root / 'protocol.json', dict(original, source=source, runs=runs,environment=env,
          worker_sha256=sha(args.worker) if args.worker else None,
          audit_helper=args.audit_helper,audit_helper_sha256=sha(args.audit_helper) if args.audit_helper else None,
          loss='Only mean per-clip clean5-step squared forecasting error from3 real context observations',
          cpu_contract_sha256=sha(args.cpu), review_required_before600=True, automatic_long_training=False,
          note='Two matched predictors,ALL3 tasks/two seeds. Same frozen local1024 observation and native planning. '
               'No renderer/FM/rate/noise/optimal-action labels/batch coupling;known fixed-feature rollout learning control. '
               'All common nonrenderer initial tensors exact;training RNG differs from old FM,while flat/spatial '
               'share native initialization/body and all data cohorts. No new public graph or longer run yet.'))
    print('Prepared12 fixed-observation clean-forecast technical gates')


if __name__ == '__main__':main()
