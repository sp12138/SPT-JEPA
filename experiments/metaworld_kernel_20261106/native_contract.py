"""Actual data/model integration before native image-transport training;0updates."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset


def run(a):
    from as_jepa_lewm.factory import build_model
    from as_jepa_tools.native_rgb_pairs import NativeRGBPairs
    from experiments.metaworld_kernel_20261106.audit import equal
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    cfg=json.loads(a.config.read_text());cfg['native_rgb_cache']=str(a.cache)
    data=NativeRGBPairs(cfg,'test');chosen={0:[],1:[]}
    for index in np.linspace(0,len(data)-1,min(256,len(data)),dtype=int):
        role=int(data[int(index)]['boundary'])
        if len(chosen[role])<4:chosen[role].append(int(index))
        if all(len(v)==4 for v in chosen.values()):break
    assert all(len(v)==4 for v in chosen.values())
    batch=next(iter(DataLoader(Subset(data,chosen[0]+chosen[1]),batch_size=8)))
    batch={k:v.cuda()if torch.is_tensor(v)else v for k,v in batch.items()}
    initial=None;reports={}
    for mode in ['free','linear','signed','stencil']:
        torch.manual_seed(cfg['seed']);cfg['model']['mode']=mode;model=build_model(cfg).cuda().eval()
        state={k:v.cpu().clone()for k,v in model.state_dict().items()}
        if initial is None:initial=state
        else:assert equal(initial,state),'Unpaired initial weights'
        loss,metrics=model.loss(batch);assert torch.isfinite(loss);loss.backward();gradients={}
        for prefix,contract in model.audit_expected_gradients().items():
            params=[p for n,p in model.named_parameters()if n.startswith(prefix+'.')]
            norm=sum(float(p.grad.float().square().sum())for p in params if p.grad is not None)**.5
            assert all(torch.isfinite(p.grad).all()for p in params if p.grad is not None)
            if contract=='active':assert norm>0,(mode,prefix,norm)
            else:assert not any(p.requires_grad for p in params)
            gradients[prefix]=norm
        with torch.no_grad():
            individual=torch.stack([model.loss({k:v[j:j+1]if torch.is_tensor(v)else v for k,v in batch.items()})[0]for j in range(8)]).mean()
            independent=float((individual-loss.detach()).abs());assert independent<2e-5,independent
            image=model.encoder(batch['current_pixels'][:1]);zero=torch.zeros(1,2,128,128,device='cuda')
            identity=float((model.warp(image,zero)-image).abs().max());assert identity<1e-6
            past=image.new_tensor([[.3,-.6,.9,-.8]])
            action,_=model.choose_control(image,model.encoder(batch['goal_pixels'][:1]),image.new_ones(1),past)
            assert action.abs().max()<=1 and torch.isfinite(action).all()
            # With no image evidence,planning should preserve its own past force.
            blank=torch.ones_like(image)*.5
            preserved,_=model.choose_control(blank,blank,image.new_ones(1),past)
            force_error=float((preserved-past).abs().max());assert force_error<1e-5,force_error
        assert equal(state,{k:v.cpu()for k,v in model.state_dict().items()})
        reports[mode]=dict(loss=float(loss.detach()),gradients=gradients,sample_independence=independent,
            identity_warp=identity,unobserved_direction_own_force_error=force_error)
        del model
    manifests=[json.loads((a.cache/str(i)/'manifest.json').read_text())for i in range(12)]
    records=[r for m in manifests for r in m['records']]
    assert len(records)==2500 and len(set(r['episode']for r in records))==2500
    assert sum(m['frames']for m in manifests)==204806 and sum(m['native_pairs']for m in manifests)==202306
    assert sum(r['split']=='train'for r in records)==2250 and sum(r['split']=='test'for r in records)==250
    a.output.write_text(json.dumps(dict(passed=True,models=reports,frames=204806,native_pairs=202306,
        train_episodes=2250,test_episodes=250,paired_initial_parameters=True,no_model_updates=True,
        no_inverse_action_training_or_future_control_reference=True),indent=2)+'\n')
    print(json.dumps(dict(passed=True,modes=list(reports))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['config','cache','output']:p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
