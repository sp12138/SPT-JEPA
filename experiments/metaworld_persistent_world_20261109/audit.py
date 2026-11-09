"""Two-GPU gradient gate and actual FP32 memory/inverse diagnostics, zero SGD."""
import copy
import torch
from torch.utils.data import default_collate
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,now
from .data import PooledWindows,add_endpoint
from .model import random_terms
from .train import setup,gradients


def run():
    root=BASE/'runs/metaworld_persistent_world_20261109_v2';p=read(root/'plan.json')
    cfg=read(next(j['config'] for j in p['jobs'] if j['mode']=='reversible' and j['pool']=='interventions'))
    device,world,_,_=setup(cfg);data=PooledWindows(cfg['datasets'],'train',maximum_horizon=60)
    batch={k:v.to(device) for k,v in default_collate([data[i*193] for i in range(64)]).items()}
    terms=random_terms(batch);add_endpoint(data,batch,terms[0],device)
    loss,_=world(batch,terms);loss.backward();gradients(world,positive=True)
    one=torch.cat([x.grad.flatten() for x in world.parameters() if x.requires_grad and x.grad is not None]).clone()
    world.zero_grad(set_to_none=True);loss2,_=torch.nn.DataParallel(world,device_ids=[0,1])(batch,terms);loss2=loss2.mean();loss2.backward()
    two=torch.cat([x.grad.flatten() for x in world.parameters() if x.requires_grad and x.grad is not None])
    relative=float((one-two).norm()/one.norm());assert relative<1e-4
    from experiments.metaworld_boundary_world_20261109.train import setup as old_setup
    _,old,_,_=old_setup(read(cfg['parent_config']))
    old.load_state_dict(torch.load(cfg['parent_weights'],map_location=device,weights_only=True),strict=True)
    with torch.no_grad():
        image=world.parent.pixels(batch['history_pixels'][:8,-1]);initial=world.parent.observe(world.parent.initial(8),image)
    generator=torch.Generator(device=device).manual_seed(1041001)
    actions=torch.randn(8,60,4,device=device,generator=generator).clamp(-1,1);direction=torch.zeros(8,device=device,dtype=torch.long)
    projection=torch.randn(initial.shape,device=device,generator=generator)
    reports={}
    for kind in ['pretrained_gru','gru','residual','reversible']:
        if kind=='pretrained_gru':model=old
        else:
            c=read(next(j['config'] for j in p['jobs'] if j['mode']==kind and j['pool']=='interventions' and j['seed']==1))
            _,model,_,_=setup(c)
        model.eval().requires_grad_(False);hidden=initial.detach().clone().requires_grad_(True);state=hidden
        for t in range(60):state=model.advance(state,actions[:,t],direction)
        g=torch.autograd.grad((state*projection).sum(),hidden)[0]
        item=dict(initial_h0_to_h60_VJP_norm_ratio=float(g.norm()/projection.norm()),endpoint_hidden_rms=float(state.square().mean().sqrt()))
        if kind=='reversible':
            with torch.no_grad():
                reconstructed=state.detach()
                for t in reversed(range(60)):
                    context=model.parent.control(actions[:,t])+model.direction(direction)
                    reconstructed=model.parent.transition.invert_state(reconstructed,context[:,:,None,None].expand(-1,-1,16,16))
                error=float((reconstructed-initial).norm()/initial.norm());assert error<1e-4
                item.update(sixty_step_FP32_roundtrip_relative_error=error,algebraic_inverse_uses_SAME_known_controls=True)
        reports[kind]=item
    save(root/'TWO_GPU_AUDIT.json',dict(passed=True,utc=now(),two_gpu_FP32_gradient_relative_L2=relative,
        actual_TRAIN_batch64_forward_backward=True,initial_memory_diagnostics=reports,optimizer_updates=0,
        scope='SyntheticOWNcontrolsandTRAINobservations/0physics. InitialVJP/roundtripNOTcontrolperformance or proofthatalltrainedsingularvaluesare1.'))


if __name__=='__main__':run()
