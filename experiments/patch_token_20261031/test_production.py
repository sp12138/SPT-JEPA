"""CPU integration contracts; no campaign training or GPU calls."""
import copy
import gc
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from as_jepa_lewm.factory import build_model,model_target
from as_jepa_lewm.patch_carrier import PatchCarrierLeWM,PatchTokenPredictor
from as_jepa_lewm.robust import RobustLeWM
from as_jepa_lewm.spatial_carrier import SpatialCarrierLeWM
from as_jepa_lewm.train import TrainModule
from vforge.ops import OpsConfig,initialize_ops

B=Path('/efs/prescottsun/SPT-JEPA')
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def same(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b
def cfg_for(task,seed):return read(B/f'runs/local_isotropic600_20261031/{task}-spatial-local-seed{seed}/config.json')
def candidate(cfg,basis):
    cfg=copy.deepcopy(cfg);cfg['model_class']='patch_carrier'
    cfg['model'].update(patch_basis=basis,patch_predictor_seed=cfg['seed'])
    return cfg
def tensor_hash(values):
    h=hashlib.sha256()
    for k,v in sorted(values.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def make_batch(action_dim):
    g=torch.Generator().manual_seed(91)
    pixels=torch.randint(0,256,(1,8,3,224,224),generator=g,dtype=torch.uint8)
    render=torch.randint(0,256,(1,8,3,64,64),generator=g,dtype=torch.uint8)
    return dict(pixels=pixels,render=render,anchor=render[:,0],anchor224=pixels[:,0],
                action=torch.randn(1,8,action_dim,generator=g))


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    started=time.monotonic();torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    initialize_ops(OpsConfig(backend='sdpa_math',allow_fallback=False),device=torch.device('cpu'),parallel_state=None)
    previous=read(B/'runs/hybrid_carrier_source_20261031/manifest.json')
    unchanged=['as_jepa_lewm/model.py','as_jepa_lewm/robust.py','as_jepa_lewm/spatial_carrier.py',
               'as_jepa_lewm/carrier.py','as_jepa_lewm/train.py']
    for name in unchanged:assert sha(ROOT/name)==previous['files'][name],name
    old_factory=(Path(previous['source'])/'as_jepa_lewm/factory.py').read_text()
    actual_factory=(ROOT/'as_jepa_lewm/factory.py').read_text()
    assert actual_factory.replace('    "patch_carrier": "as_jepa_lewm.patch_carrier.PatchCarrierLeWM",\n','')==old_factory
    assert PatchCarrierLeWM.loss is RobustLeWM.loss
    assert PatchCarrierLeWM.encode is SpatialCarrierLeWM.encode
    assert PatchCarrierLeWM.predict is SpatialCarrierLeWM.predict
    assert PatchCarrierLeWM.prepare is SpatialCarrierLeWM.prepare
    initial=[];predictor_hashes={}
    for task in ('tworoom','cube'):
        for seed in (1,2):
            cfg=cfg_for(task,seed);cal=cfg['spatial_calibration'];assert sha(cal['path'])==cal['sha256']
            torch.manual_seed(seed);baseline=build_model(cfg);baseline_rng=torch.get_rng_state().clone()
            old=torch.load(Path(cfg['output'])/'initial-model/weights.pt',map_location='cpu',weights_only=True,mmap=True)
            assert same(baseline.state_dict(),old)
            common={k:v.clone() for k,v in baseline.state_dict().items() if not k.startswith('predictor.')}
            basecounts=dict(total=sum(p.numel() for p in baseline.parameters()),predictor=sum(p.numel() for p in baseline.predictor.parameters()))
            del baseline,old;gc.collect()
            torch.manual_seed(seed)
            with patch('torch.cuda.manual_seed_all',side_effect=AssertionError('Constructor must not reseed CUDA')):
                identity=build_model(candidate(cfg,'identity'))
            assert torch.equal(torch.get_rng_state(),baseline_rng)
            for k,v in common.items():assert torch.equal(v,identity.state_dict()[k]),k
            torch.manual_seed(seed);mixed=build_model(candidate(cfg,'orthogonal'))
            assert torch.equal(torch.get_rng_state(),baseline_rng)
            for k,v in identity.state_dict().items():assert torch.equal(v,mixed.state_dict()[k]),k
            assert set(mixed.state_dict())-set(identity.state_dict())=={'predictor.coordinate_basis'}
            counts=identity.patch_parameter_counts();assert counts['predictor']==523536
            assert basecounts['total']-counts['total']==11389184-523536
            assert len(common)==len(identity.state_dict())-len(identity.predictor.state_dict())
            predictor_hashes[(task,seed)]=tensor_hash(identity.predictor.state_dict())
            initial.append(dict(task=task,seed=seed,nonpredictor_tensors=len(common),
                paired_learned_tensors=len(list(identity.named_parameters())),
                original_model_tensors=553,original_total=basecounts['total'],counts=counts,
                calibration=cal,all_nonpredictor_exact=True,outer_rng_exact=True,
                predictor_sha256=predictor_hashes[(task,seed)]))
            del identity,mixed,common;gc.collect()
    assert predictor_hashes[('tworoom',1)]!=predictor_hashes[('tworoom',2)]
    assert predictor_hashes[('cube',1)]!=predictor_hashes[('cube',2)]
    assert predictor_hashes[('tworoom',1)]==predictor_hashes[('cube',1)]
    # Same declared training seed intentionally shares the predictor across tasks.
    losses=[]
    for task in ('tworoom','cube'):
        cfg=cfg_for(task,1)
        # Smaller encoder/renderer only in this synthetic CPU loss fixture.
        # Full production-sized initial tensors were all checked above.
        cfg=copy.deepcopy(cfg);cfg['model'].update(encoder_depth=1,renderer_width=48,renderer_depth=1,renderer_heads=3)
        torch.manual_seed(1);identity=build_model(candidate(cfg,'identity')).train()
        torch.manual_seed(1);mixed=build_model(candidate(cfg,'orthogonal')).train()
        batch=make_batch(cfg['model']['action_dim'])
        torch.manual_seed(73);loss,metrics=identity.loss(batch,step=0);loss.backward();rng=torch.get_rng_state().clone()
        torch.manual_seed(73);other,other_metrics=mixed.loss(batch,step=0);other.backward()
        assert torch.equal(loss,other) and same(metrics,other_metrics) and torch.equal(rng,torch.get_rng_state())
        shared=0
        for name,param in identity.named_parameters():
            if name.startswith('predictor.'):continue
            peer=dict(mixed.named_parameters())[name]
            assert (param.grad is None)==(peer.grad is None)
            if param.grad is not None:assert torch.equal(param.grad,peer.grad);shared+=1
        step_rows=[]
        for basis,model in [('identity',identity),('orthogonal',mixed)]:
            state_cfg=candidate(cfg,basis)
            native=TrainModule.configure_optimizers(SimpleNamespace(model=model,cfg=state_cfg))
            optimizer=native['optimizer'];scheduler=native['lr_scheduler']['scheduler']
            for step in range(2):
                optimizer.zero_grad(set_to_none=True);torch.manual_seed(100+step)
                value,_=model.loss(batch,step=step);assert torch.isfinite(value)
                value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();scheduler.step()
            model.eval();optimizer.zero_grad(set_to_none=True);torch.manual_seed(102)
            value,_=model.loss(batch,step=2);value.backward()
            groups={}
            for prefix in ['predictor.input_proj','predictor.cond_proj','predictor.output_proj',
                           'predictor.pos_embedding','predictor.layers','action_encoder','renderer']:
                values=[p.grad for n,p in model.named_parameters() if (n==prefix or n.startswith(prefix+'.')) and p.grad is not None]
                norm=sum(float(v.double().square().sum()) for v in values)**.5
                assert values and norm>0 and all(torch.isfinite(v).all() for v in values),prefix
                groups[prefix]=norm
            assert all(p.grad is None for p in model.encoder.parameters())
            assert all(p.grad is None for p in model.pool.parameters())
            stream=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),stream)
            stream.seek(0);saved=torch.load(stream,weights_only=False)
            torch.manual_seed(1);restored=build_model(state_cfg);restored.load_state_dict(saved['model'],strict=True)
            configured=TrainModule.configure_optimizers(SimpleNamespace(model=restored,cfg=state_cfg))
            configured['optimizer'].load_state_dict(saved['optimizer']);configured['lr_scheduler']['scheduler'].load_state_dict(saved['scheduler'])
            assert same(model.state_dict(),restored.state_dict())
            assert same(optimizer.state_dict(),configured['optimizer'].state_dict())
            assert same(scheduler.state_dict(),configured['lr_scheduler']['scheduler'].state_dict())
            step_rows.append(dict(arm=basis,steps=2,post2_actual_gradient_norms=groups,
                native_AdamW_scheduler_state_roundtrip=True,strict_model_roundtrip=True))
            del optimizer,scheduler,native,restored,configured,saved,stream;gc.collect()
        losses.append(dict(task=task,initial_loss=float(loss.detach()),initial_loss_metrics_rng_exact=True,
            initial_shared_gradients_exact=shared,fixture='B1,W8,encoder_depth1,DiTwidth48/depth1; full native TF objective',native_steps=step_rows))
        del identity,mixed;gc.collect()
    for bad in [dict(patch_predictor_seed=None),dict(patch_predictor_seed=-1),dict(patch_predictor_seed=1,spatial_chart='bounded'),dict(patch_predictor_seed=1,carry=False)]:
        try:PatchCarrierLeWM(**bad)
        except ValueError:pass
        else:raise AssertionError('Bad configuration accepted')
    # Reuse the already declared mask/native-rollout contracts against the
    # actual production class, not merely the separate design prototype.
    import contracts as mask_contracts
    class ProductionFixture(PatchTokenPredictor):
        def __init__(self,basis='identity',seed=1042):super().__init__(basis=basis,seed=seed)
    mask_output=Path(args.output).with_name(Path(args.output).stem+'-mask.json')
    previous_class=mask_contracts.PatchTokenPredictor;argv=sys.argv
    try:
        mask_contracts.PatchTokenPredictor=ProductionFixture
        sys.argv=['contracts.py','--native-source',str(B/'sources/stable-worldmodel'),'--output',str(mask_output)]
        mask_contracts.main()
    finally:
        mask_contracts.PatchTokenPredictor=previous_class;sys.argv=argv
    mask_proof=read(mask_output)
    mask_proof['actual_predictor_implementation']='as_jepa_lewm.patch_carrier.PatchTokenPredictor'
    mask_proof['actual_predictor_source_sha256']=sha(ROOT/'as_jepa_lewm/patch_carrier.py')
    mask_output.write_text(json.dumps(mask_proof,indent=2)+'\n')
    source_paths=['as_jepa_lewm/patch_carrier.py','as_jepa_lewm/factory.py',
                  'experiments/patch_token_20261031/test_production.py',
                  'experiments/patch_token_20261031/prototype.py','experiments/patch_token_20261031/contracts.py',*unchanged]
    result=dict(passed=True,CUDA_initialized=torch.cuda.is_initialized(),elapsed_seconds=time.monotonic()-started,
        source_sha256={n:sha(ROOT/n) for n in source_paths},default_model_source_exact=True,
        default_factory_only_adds_new_entry=True,native_loss_encode_prepare_predict_methods_inherited=True,
        initial=initial,loss_and_optimizer=losses,independent_training_seed_initialization=True,
        no_CUDA_reseed_called=True,actual_production_mask_contracts=dict(path=str(mask_output),sha256=sha(mask_output),passed=mask_proof['passed']),
        runtime_RNG_scope='Flat and token consume different dropout tensor sizes; no claim of identical runtime noise draws across those architectures. Distributions/loss/native training source unchanged.',
        sample_level_scope='Inherited Robust TF objective; frame-local frozen encoder, fixed scalar calibration, no cross-sample coupling or extra target.',
        production_GPU_preflight_required=True,authorized_stage_steps=2)
    assert not result['CUDA_initialized']
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,seconds=result['elapsed_seconds'],initial_pairs=len(initial),synthetic_loss_pairs=len(losses))))


if __name__=='__main__':main()
