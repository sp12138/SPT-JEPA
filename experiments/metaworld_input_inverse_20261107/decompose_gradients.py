"""Frozen checkpoint diagnosis of main/inverse gradient scale and agreement."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run, output):
    import torch
    from torch.utils.data import DataLoader, Subset
    from vforge.ops import OpsConfig, initialize_ops
    from as_jepa_lewm.factory import build_model
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    from experiments.metaworld_eval_20261103.evaluate import model_sha
    cfg = json.loads((run/'config.json').read_text())
    assert cfg['model_class'] == 'executed_inverse'
    audit_file = run/'audit-io-v2-5000.json'
    audit = json.loads(audit_file.read_text()); assert audit['passed']
    cache_helper = Path('/efs/prescottsun/SPT-JEPA/reports/metaworld_input_inverse_20261107/helper-io-v2/io_stage.py')
    spec = importlib.util.spec_from_file_location('gradient_cache', cache_helper)
    cache = importlib.util.module_from_spec(spec); spec.loader.exec_module(cache); cache.install_cache()
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=torch.device('cuda'), parallel_state=None)
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    weights = run/'model-step0005000/weights.pt'
    assert digest(weights) == audit['portable_sha256']
    model = build_model(cfg).cuda().eval()
    model.load_state_dict(torch.load(weights, map_location='cuda', weights_only=True), strict=True)
    before = model_sha(model)
    dataset = BenchmarkWindows(cfg['data'], window=8, seed=cfg['seed'], split='test')
    manifest = audit['heldout_windows']
    batch = next(iter(DataLoader(Subset(dataset, manifest['indices']), batch_size=2, num_workers=0)))
    assert batch['episode'].tolist() == manifest['episode'] and batch['rows'].tolist() == manifest['rows']
    batch = {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
    torch.manual_seed(20261103+cfg['seed'])
    loss, metrics = model.loss(batch, step=5000, ramp_steps=0)
    main_loss = metrics['fm'] + model.beta*metrics['rate']
    auxiliary = model.inverse_weight*metrics['inverse']
    assert torch.allclose(main_loss+auxiliary, loss, rtol=1e-6, atol=1e-6)
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    params = [p for n,p in named]
    primary = torch.autograd.grad(main_loss, params, retain_graph=True, allow_unused=True)
    inverse = torch.autograd.grad(auxiliary, params, allow_unused=True)
    sums = {}
    for (name, _), a, b in zip(named, primary, inverse):
        group = name.split('.')[0]
        row = sums.setdefault(group, dict(main2=0., inverse2=0., dot=0.))
        if a is not None: row['main2'] += float(a.double().square().sum())
        if b is not None: row['inverse2'] += float(b.double().square().sum())
        if a is not None and b is not None: row['dot'] += float((a.double()*b.double()).sum())
    groups = {}
    for name, row in sums.items():
        a, b = row['main2']**.5, row['inverse2']**.5
        groups[name] = dict(main_norm=a, inverse_norm=b, inverse_over_main=b/a if a else None,
                            cosine=row['dot']/(a*b) if a and b else None)
    after = model_sha(model); assert before == after
    report = dict(passed=True, run=str(run), step=5000, weights_sha256=audit['portable_sha256'],
        audit_sha256=digest(audit_file), actual_model_source=inspect.getfile(type(model)),
        actual_model_source_sha256=digest(inspect.getfile(type(model))), heldout_windows=manifest,
        model_state_before=before, model_state_after=after, model_unchanged=True, model_updates=0,
        main_loss=float(main_loss.detach()), weighted_inverse_loss=float(auxiliary.detach()),
        inverse_observed=float(metrics['inverse_observed'].detach()),
        inverse_predicted=float(metrics['inverse_predicted'].detach()) if 'inverse_predicted' in metrics else None,
        groups=groups, scope='Read-only decomposition on two actual heldout windows,permodel/pergroup. '
          'No optimizer step,simulator,baseline evaluation or additional training. Norm/cosine are not control performance or causal proof.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, model=run.name, model_unchanged=True, groups=groups)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        main(args.run, args.output)
    except BaseException as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_name(args.output.stem+'-error.json').write_text(json.dumps(dict(error=repr(error), traceback=traceback.format_exc()), indent=2)+'\n')
        raise
