"""Reuse the physical case replay and original RGB-goal CEM evaluator."""
import argparse
import json
from pathlib import Path


def run(args):
    from as_jepa_tools.supervision_contract import validate_training_supervision
    run_dir = args.checkpoint.parent
    cfg = json.loads((run_dir/'config.json').read_text())
    validate_training_supervision(cfg)
    assert cfg['model_class'] == 'executed_inverse'
    from experiments.metaworld_eval_20261103.evaluate import main
    manifest = args.manifest
    if args.technical:
        source = json.loads(manifest.read_text())
        source['rows'] = source['rows'][:1]
        manifest = args.output.parent/(args.output.name+'-case.json')
        manifest.write_text(json.dumps(source, indent=2)+'\n')
    main(str(args.checkpoint), str(args.data), str(manifest), str(args.output), budget=args.budget)
    report = json.loads((args.output/'metrics.json').read_text())
    report['rows'] = report['episodes']
    report['model_unchanged'] = report['model_state_before'] == report['model_state_after']
    report['inverse_deployed'] = False
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ['checkpoint', 'data', 'manifest', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--budget', type=int, default=500)
    parser.add_argument('--technical', action='store_true')
    run(parser.parse_args())
