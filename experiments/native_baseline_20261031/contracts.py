"""CPU-only native-function equivalence; no dataset fitting or GPU calls."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import hydra
import torch

B=Path('/efs/prescottsun/SPT-JEPA')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(left,right):
    if torch.is_tensor(left):return torch.is_tensor(right)and torch.equal(left,right)
    if isinstance(left,dict):return isinstance(right,dict)and left.keys()==right.keys()and all(equal(v,right[k])for k,v in left.items())
    if isinstance(left,(list,tuple)):return isinstance(right,(list,tuple))and len(left)==len(right)and all(equal(a,b)for a,b in zip(left,right))
    return left==right


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if(out/'contracts.json').exists():raise FileExistsError('Preserve earlier CPU evidence')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    from as_jepa_lewm.factory import build_model,model_target,gradient_expectations
    from as_jepa_lewm.train import TrainModule
    from stable_pretraining.optim.lr_scheduler import LinearWarmupCosineAnnealingLR
    from stable_pretraining.optim.utils import create_optimizer
    from stable_worldmodel.wm.loss import SIGReg
    from stable_worldmodel.wm.utils import load_pretrained
    native_path=B/'sources/stable-worldmodel/scripts/train/lewm.py'
    spec=importlib.util.spec_from_file_location('pinned_native_lewm_training',native_path)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    rows=[]
    for task,action_dim,expected in [('tworoom',10,18035246),('cube',25,18035396)]:
        cfg=dict(model_class='native_lewm',model=dict(window=8,action_dim=action_dim),max_steps=124910)
        torch.manual_seed(1);adapter=build_model(cfg)
        native_cfg=json.loads((B/'weights/hardtasks'/task/'config.json').read_text())
        native_cfg['predictor']['num_frames']=7
        torch.manual_seed(1);native=hydra.utils.instantiate(native_cfg)
        weights=adapter.state_dict();native_weights=native.state_dict()
        assert set(native_weights)=={k for k in weights if not k.startswith('sigreg.')}
        assert all(torch.equal(v,weights[k])for k,v in native_weights.items())
        assert sum(p.numel()for p in adapter.parameters())==expected
        assert adapter.action_encoder.smoothed_dim==10 and adapter.renderer is None
        # The CPU algebra check keeps every production parameter and all8 frames,
        # but uses a28px grid to bound CPU cost. Normalization is checked separately.
        pixels=torch.randn(2,8,3,28,28);actions=torch.randn(2,8,action_dim)
        actions[0,-1,0]=float('nan')
        wrapper=SimpleNamespace(model=native,sigreg=SIGReg(),log_dict=lambda *args,**kwargs:None)
        loss_cfg=SimpleNamespace(wm=SimpleNamespace(history_size=7,num_preds=1),
                                 loss=SimpleNamespace(sigreg=SimpleNamespace(weight=.09)))
        comparisons=[]
        for training in [True,False]:
            adapter.train(training);native.train(training)
            adapter.zero_grad(set_to_none=True);native.zero_grad(set_to_none=True)
            torch.manual_seed(92)
            reference=official.lejepa_forward(wrapper,dict(pixels=pixels.clone(),action=actions.clone()),'train',loss_cfg)
            reference['loss'].backward()
            torch.manual_seed(92)
            with patch.object(adapter,'normalize',side_effect=lambda value:value):
                loss,metrics=adapter.loss(dict(pixels=pixels.clone(),action=actions.clone()))
            loss.backward()
            for ours,theirs in [('loss','loss'),('pred','pred_loss'),('sigreg','sigreg_loss')]:
                assert torch.equal(metrics[ours],reference[theirs]),(task,training,ours)
            maximum=0.;gradients=0
            for name,param in adapter.named_parameters():
                other=dict(native.named_parameters())[name]
                assert (param.grad is None)==(other.grad is None),name
                if param.grad is not None:
                    error=float((param.grad-other.grad).abs().max());maximum=max(maximum,error)
                    assert torch.equal(param.grad,other.grad),name
                    gradients+=1
            assert all(torch.equal(value,adapter.state_dict()[name])for name,value in native.state_dict().items())
            comparisons.append(dict(training=training,loss=float(loss.detach()),gradient_tensors=gradients,max_gradient_abs_error=maximum))
        rgb=torch.randint(0,256,(2,8,3,224,224),dtype=torch.uint8)
        normalized=(rgb.float()/255.-torch.tensor([.485,.456,.406]).view(1,1,3,1,1))/torch.tensor([.229,.224,.225]).view(1,1,3,1,1)
        assert torch.equal(adapter.normalize(rgb),normalized)
        # Native factory, not a second optimizer implementation.
        module=TrainModule(cfg);module.model.load_state_dict(adapter.state_dict(),strict=True)
        options=module.configure_optimizers();optimizer=options['optimizer'];scheduler=options['lr_scheduler']['scheduler']
        assert isinstance(scheduler,LinearWarmupCosineAnnealingLR)and options['lr_scheduler']['interval']=='step'
        assert len(optimizer.param_groups)==1 and optimizer.param_groups[0]['lr']==0.
        native_optimizer=create_optimizer(module.model.parameters(),dict(type='AdamW',lr=5e-5,weight_decay=1e-3),
                                          named_params=module.model.named_parameters())
        assert equal(native_optimizer.defaults,optimizer.defaults)
        assert [len(g['params'])for g in native_optimizer.param_groups]==[len(g['params'])for g in optimizer.param_groups]
        # Use the actual native objective gradients already computed above for
        # every parameter; no additional forward. First update has lr0.
        gradient_template={name:param.grad.detach().clone()for name,param in adapter.named_parameters()}
        def assign_gradients(model):
            for name,param in model.named_parameters():param.grad=gradient_template[name].clone()
        first=next(module.model.parameters());before=first.detach().clone();assign_gradients(module.model)
        optimizer.step();scheduler.step();assert torch.equal(first,before)
        optimizer.zero_grad(set_to_none=True);assign_gradients(module.model)
        optimizer.step();scheduler.step();assert not torch.equal(first,before)
        assert len(optimizer.state)==len(gradient_template)==297
        with tempfile.TemporaryDirectory(dir=out)as temp:
            temp=Path(temp);torch.save(module.model.state_dict(),temp/'weights.pt')
            (temp/'config.json').write_text(json.dumps(dict(_target_=model_target(cfg),**cfg['model'])))
            reloaded=load_pretrained(str(temp))
            assert equal(reloaded.state_dict(),module.model.state_dict())
            fresh=reloaded.training_optimizer_config(cfg['max_steps'])
            fresh['optimizer'].load_state_dict(copy.deepcopy(optimizer.state_dict()))
            fresh['lr_scheduler']['scheduler'].load_state_dict(copy.deepcopy(scheduler.state_dict()))
            assert equal(fresh['optimizer'].state_dict(),optimizer.state_dict())
            assert equal(fresh['lr_scheduler']['scheduler'].state_dict(),scheduler.state_dict())
            # Continued next update agrees after native model/optimizer/scheduler load.
            for model,opt,sch in [(module.model,optimizer,scheduler),(reloaded,fresh['optimizer'],fresh['lr_scheduler']['scheduler'])]:
                opt.zero_grad(set_to_none=True);assign_gradients(model);opt.step();sch.step()
            assert equal(module.model.state_dict(),reloaded.state_dict())
        rows.append(dict(task=task,parameters=expected,identical_initial_native_state=True,
            loss_gradient_comparisons=comparisons,normalization_exact=True,export_load_exact=True,
            optimizer_scheduler_roundtrip_exact=True,resumed_next_update_exact=True,
            official_optimizer_factory_defaults_equal=True,
            populated_optimizer_parameter_states=len(optimizer.state),
            expectations=gradient_expectations(adapter,cfg)))
        del adapter,native,module,reloaded,optimizer,native_optimizer,scheduler,fresh
    # Actual native scheduler .step(), using a scalar parameter only.
    value=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.AdamW([value],lr=5e-5)
    sch=LinearWarmupCosineAnnealingLR(opt,warmup_steps=1249,max_steps=124910)
    rates={0:opt.param_groups[0]['lr']}
    for step in range(1,2001):
        opt.step();sch.step()
        if step in [1,2,600,1248,1249,1250,2000]:rates[step]=opt.param_groups[0]['lr']
    for step,rate in rates.items():
        expected=5e-5*step/1249 if step<1249 else 5e-5*.5*(1+math.cos(math.pi*(step-1249)/(124910-1249)))
        assert abs(rate-expected)<1e-20
    result=dict(passed=True,rows=rows,scheduler_rates_after_updates=rates,
        native_training_source=str(native_path),native_training_sha256=digest(native_path),
        source_sha256=digest(__file__),native_adapter_sha256=digest(Path(__file__).parents[2]/'as_jepa_lewm/native_lewm.py'),
        factory_sha256=digest(Path(__file__).parents[2]/'as_jepa_lewm/factory.py'),
        trainer_sha256=digest(Path(__file__).parents[2]/'as_jepa_lewm/train.py'),
        scope='CPU only. Same native full-size parameters/window8;28px spatial grid for bounded algebra checks,224px preprocessing separately. Native function, train/eval BatchNorm states, RNG, exact losses and gradients; no world-model training or GPU launch. Actual two-step production-batch GPU check remains required.')
    (out/'contracts.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
