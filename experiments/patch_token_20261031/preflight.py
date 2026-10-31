"""Original strict native preflight plus patch-specific CPU evidence.

One original FP32 backward only; no new training loop or relaxed expectations.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch

from experiments.ideas_validate_20261031.audit import file_hash
from experiments.patch_token_20261031.test_production import same


def read(p):return json.loads(Path(p).read_text())
def required_gradient_groups():
    groups=['predictor.input_proj','predictor.cond_proj','predictor.output_proj','predictor.pos_embedding']
    groups += [f'predictor.layers.{i}.{part}' for i in range(6) for part in ['attn','mlp','adaLN_modulation']]
    return groups


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--reference-initial',required=True);p.add_argument('--paired-initial',required=True)
    p.add_argument('--expected-step',type=int,default=2);args=p.parse_args()
    assert args.expected_step==2,'This helper authorizes only the two-update technical stage'
    root,out=Path(args.run),Path(args.output);cfg=read(root/'config.json')
    assert cfg['model_class']=='patch_carrier' and cfg['model']['patch_predictor_seed']==cfg['seed']
    assert cfg['batch_size']==128 and cfg['max_steps']==124910
    cal=cfg['spatial_calibration'];assert file_hash(cal['path'])==cal['sha256']
    original=Path(__file__).resolve().parents[1]/'ideas_validate_20261031/preflight.py'
    spec=importlib.util.spec_from_file_location('patch_original_preflight',original)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    captured={};native_gradient_report=native.gradient_report
    def record_gradients(model,expectations):
        rows=native_gradient_report(model,expectations)
        captured['parameter_names']=[n for n,_ in model.named_parameters()]
        captured['patch_groups']=native_gradient_report(model,{key:'active' for key in required_gradient_groups()})
        captured['predictor_per_tensor']={n:dict(shape=list(p.shape),gradient_present=p.grad is not None,
            finite=p.grad is None or bool(torch.isfinite(p.grad).all()),
            norm=None if p.grad is None else float(p.grad.detach().double().square().sum().sqrt()))
            for n,p in model.named_parameters() if n.startswith('predictor.')}
        return rows
    native.gradient_report=record_gradients
    previous=sys.argv
    sys.argv=[str(original),'--run',str(root),'--output',str(out),'--expected-step','2',
              '--reference-initial',args.reference_initial,'--gradient-cases','2']
    for prefix in ['encoder','pool','action_encoder','renderer','sigreg','spatial_projection','spatial_mean','spatial_scale']:
        sys.argv += ['--reference-prefix',prefix]
    try:native.main()
    finally:sys.argv=previous
    native_result=read(out/'preflight.json');assert native_result['passed']
    assert captured['patch_groups'] and all(r['passed'] for r in captured['patch_groups'].values())
    assert all(r['gradient_present'] and r['finite'] for r in captured['predictor_per_tensor'].values())
    initial_path=root/'initial-model/weights.pt'
    initial=torch.load(initial_path,map_location='cpu',weights_only=True,mmap=True)
    old=torch.load(args.reference_initial,map_location='cpu',weights_only=True,mmap=True)
    peer=torch.load(args.paired_initial,map_location='cpu',weights_only=True,mmap=True)
    old_common={k:v for k,v in old.items() if not k.startswith('predictor.')}
    new_common={k:v for k,v in initial.items() if not k.startswith('predictor.')}
    assert same(old_common,new_common)
    for name in captured['parameter_names']:assert torch.equal(initial[name],peer[name]),name
    for name,value in initial.items():
        if name!='predictor.coordinate_basis':assert torch.equal(value,peer[name]),name
    checkpoint=torch.load(root/'last.ckpt',map_location='cpu',weights_only=False)
    from as_jepa_lewm.train import TrainModule,WindowSampler
    from as_jepa_tools.benchmark_data import make_training_dataset
    module=TrainModule(cfg);module.load_state_dict(checkpoint['state_dict'],strict=True)
    configured=module.configure_optimizers();optimizer=configured['optimizer'];scheduler=configured['lr_scheduler']['scheduler']
    optimizer.load_state_dict(checkpoint['optimizer_states'][0]);scheduler.load_state_dict(checkpoint['lr_schedulers'][0])
    assert same(module.state_dict(),checkpoint['state_dict'])
    assert same(optimizer.state_dict(),checkpoint['optimizer_states'][0])
    assert same(scheduler.state_dict(),checkpoint['lr_schedulers'][0])
    trainable=list(p for p in module.model.parameters() if p.requires_grad)
    actual=[p for group in optimizer.param_groups for p in group['params']]
    assert len(actual)==len(trainable) and {id(p) for p in actual}=={id(p) for p in trainable}
    assert all(p in optimizer.state for p in trainable)
    assert all(p not in optimizer.state for p in module.model.encoder.parameters())
    assert all(p not in optimizer.state for p in module.model.pool.parameters())
    dataset=make_training_dataset(cfg);cursor=checkpoint['campaign_samples'];assert cursor==256
    def upcoming():
        iterator=iter(WindowSampler(dataset,start=cursor,seed=cfg['seed'],batch_size=128))
        return [next(iterator) for _ in range(16)]
    indices=upcoming();assert upcoming()==indices
    status=read(root/'status.json');assert status['state']=='finished' and status['step']==2
    binding=[root/'config.json',root/'last.ckpt',initial_path,root/'model-step0000002/weights.pt',
             Path(args.reference_initial),Path(args.paired_initial),Path(cal['path']),original,out/'preflight.json']
    report=dict(passed=True,expected_step=2,run=str(root),nonpredictor_initial_tensors_exact=len(new_common),
        original_initial_tensors=len(old),paired_parameter_initial_tensors_exact=len(captured['parameter_names']),
        paired_initial_all_common_buffers_exact=True,predictor_seed=cfg['model']['patch_predictor_seed'],
        native_report_sha256=file_hash(out/'preflight.json'),binding={str(path):file_hash(path) for path in binding},
        original_native_checks_unmodified=True,additional_backward_calls=0,gradient= captured,
        optimizer=dict(native_full_state_roundtrip=True,all_trainable_parameters_have_state=True,
            parameter_tensors=len(actual),frozen_encoder_pool_have_no_state=True,
            sampler_cursor=cursor,next16_window_indices=indices),
        parameter_counts=module.model.patch_parameter_counts(),peak_gib=status['peak_gib'],
        scope='Two-update technical gate only; no task learning claim, no600 authorization.')
    (out/'patch-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,patch_audit=str(out/'patch-audit.json'),peak_gib=status['peak_gib'])),flush=True)


if __name__=='__main__':main()
