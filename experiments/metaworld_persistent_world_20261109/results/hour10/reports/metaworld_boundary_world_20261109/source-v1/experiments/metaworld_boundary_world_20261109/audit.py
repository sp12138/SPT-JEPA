"""Actual FP32 multi-GPU gradient and executed-input alignment gates."""
import copy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,now
from experiments.metaworld_joint_flow_20261109.data import EndpointWindows
from .data import add_targets
from .model import random_terms
from .train import setup,gradients


def run():
    root=BASE/'runs/metaworld_boundary_world_20261109';p=read(root/'plan.json')
    cfg=read(next(x['config'] for x in p['jobs'] if x['mode']=='bridge'))
    device,model,_,_=setup(cfg);data=EndpointWindows(cfg['data'],'train',maximum_horizon=60)
    batch=default_collate([data[i*193] for i in range(64)]);batch={k:v.to(device) for k,v in batch.items()}
    terms=random_terms(batch);add_targets(data,batch,terms[0],device)
    for i,(index,duration) in enumerate(zip(batch['index'].cpu(),terms[0].cpu())):
        r,t=data.locate(int(index));d=int(duration);s=data.shards[r['shard']];o=r['offset']
        expected=np.asarray(s['inputs'][o+t:o+t+d])
        assert np.array_equal(batch['actions'][i,:d].cpu().numpy(),expected)
        assert np.array_equal(batch['reverse_actions'][i,:d].cpu().numpy(),expected[::-1])
        assert np.array_equal(batch['endpoint_pixels'][i].permute(1,2,0).cpu().numpy(),s['pixels'][o+t+d])
        assert np.array_equal(batch['reverse_endpoint_pixels'][i].permute(1,2,0).cpu().numpy(),s['pixels'][o+t])
    loss,metrics=model(batch,terms);loss.backward();gradients(model,positive=True)
    one=torch.cat([x.grad.flatten() for x in model.parameters() if x.requires_grad and x.grad is not None]).clone()
    model.zero_grad(set_to_none=True);parallel=torch.nn.DataParallel(model,device_ids=list(range(4)))
    loss4,_=parallel(batch,terms);loss4=loss4.mean();loss4.backward()
    four=torch.cat([x.grad.flatten() for x in model.parameters() if x.requires_grad and x.grad is not None])
    relative=float((one-four).norm()/one.norm());assert relative<1e-4
    # Verify finite sensitivity to the ordinary observed history.
    small={k:v[:2].clone() for k,v in batch.items()};ds=tuple(t[:2].clone() for t in terms)
    model.zero_grad(set_to_none=True);small['history_pixels']=small['history_pixels'].float().requires_grad_(True)
    value=model(small,ds)[0];g=torch.autograd.grad(value,small['history_pixels'])[0]
    assert torch.isfinite(g).all() and float(g.norm())>0
    save(root/'FOUR_GPU_AUDIT.json',dict(passed=True,utc=now(),four_gpu_FP32_gradient_relative_L2=relative,
        single_loss=float(loss.detach()),four_loss=float(loss4.detach()),
        actual64_forward_and_reverse_known_input_orders_and_endpoint_RGB_byte_checks=True,
        reverse_controls_negated=False,action_targets=False,desired_goal_in_training=False,
        native_TRAIN_starts=len(data),train_episodes=len(data.episode_ids),optimizer_updates=0,
        scope='AlignmentplusactualFP32backward;not physical-control performance or exact resume trajectory equivalence.'))


if __name__=='__main__':run()
