"""Actual-window lazy-endpoint parity and inherited factual-operator check."""
import argparse
import copy
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import default_collate
from experiments.metaworld_transition_only_20261109.common import read,save,sha,tensor_sha,now
from experiments.metaworld_transition_only_20261109.data import FactualWindows
from experiments.metaworld_belief_state_20261109.model import BeliefWorld
from .data import EndpointWindows,add_endpoint
from .model import random_terms
from .train import setup


def run(config,output):
    assert not torch.cuda.is_available();cfg=read(config)
    old=FactualWindows(cfg['data'],'train',maximum_horizon=60);new=EndpointWindows(cfg['data'],'train',maximum_horizon=60)
    assert old.episode_ids==new.episode_ids and len(old)==len(new)==181608
    started=time.monotonic();reference=[old[i*2719%len(old)] for i in range(64)];old_seconds=time.monotonic()-started
    started=time.monotonic()
    for n,a in enumerate(reference):
        index=n*2719%len(old);b=new[index]
        for key in ['history_pixels','history_actions','history_valid','actions','future_valid']:
            assert torch.equal(a[key],b[key]),key
        assert a['episode']==b['episode'] and a['start']==b['start']
        duration=1+n%int(a['future_valid'].sum());assert torch.equal(a['future_pixels'][duration-1],new.endpoint(index,duration))
    lazy_seconds=time.monotonic()-started
    device,model,_,_=setup(cfg,False);model.train();before=tensor_sha(model)
    origin=read(Path(read(cfg['head_config'])['output']).parent.parent.parent/'metaworld_belief_state_20261109'/'training'/f'joint_anchor-seed{cfg["seed"]}'/'ORIGIN.json')
    assert tensor_sha(model,True)==origin['initial_parameters_sha256']
    batch=default_collate([old[0],old[31]]);lazy=default_collate([new[0],new[31]])
    terms=random_terms(lazy);add_endpoint(new,lazy,terms[0],device)
    baseline=BeliefWorld(copy.deepcopy(model.parent),copy.deepcopy(model.renderer),'joint_anchor')
    baseline.load_state_dict(model.state_dict(),strict=True);baseline.posterior_scale.requires_grad_(False);baseline.train()
    a,_=baseline.objective(batch,terms[:5]);a.backward()
    b,_=BeliefWorld.objective(model,lazy,terms[:5]);b.backward()
    assert torch.equal(a,b),(a,b)
    named=dict(baseline.named_parameters());maximum=0
    for name,p in model.named_parameters():
        q=named[name]
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:
            assert torch.equal(p.grad,q.grad),name
            maximum=max(maximum,float((p.grad-q.grad).abs().max()))
    assert tensor_sha(model)==before
    save(output,dict(passed=True,utc=now(),actual_train_windows=64,train_starts=len(old),train_episodes=len(old.episode_ids),
        all_history_action_mask_fields_and_selected_endpoint_bytes_exact=True,full_NLL_reference_loss_and_gradients_bitexact=True,
        same_initial_parameter_values_as_inherited_NLL_control=True,maximum_gradient_absolute_difference=maximum,
        old_dense_endpoint_read_seconds=old_seconds,new_lazy_endpoint_read_seconds=lazy_seconds,
        scope='TimingincludesfirstcoldreadsandisnotacontrolledIObenchmark;no speedupclaim. No SGD/GPU/physics.',
        optimizer_updates=0,gpu_contexts=0,physics_steps=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.config,a.output)
