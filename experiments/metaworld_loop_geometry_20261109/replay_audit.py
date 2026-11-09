"""Read-only diagnosis of the one failed Drawer branch RGB restoration gate."""
from pathlib import Path
import numpy as np
import torch
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,sha,tensor_sha,now
from experiments.metaworld_horizon_match_20261109.model import build
from experiments.metaworld_horizon_match_20261109.bank import capture,restore,mjstate
from experiments.metaworld_transition_only_20261109.evaluate import rgb_tensor
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    cfg=read(BASE/'runs/metaworld_horizon_match_20261109/training/h60-seed2/config.json')
    weight=Path(cfg['output'])/'model-step0012000/weights.pt'
    assert sha(weight)==read(weight.parent/'SEALED.json')['weights_sha256']
    model=build(cfg).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(weight,map_location='cuda',weights_only=True),strict=True);before=tensor_sha(model)
    case=next(x for x in read(BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json')['rows'] if x['task']=='drawer-open-v3')
    assert sha(case['path'])==case['sha256'];z=np.load(case['path'])
    env,state,_=make_environment(case['task'],case['seed'],z['rand_vec'])
    try:
        assert np.array_equal(state,z['initial_state'])
        assert all(np.array_equal(v,z['initial_physics_'+k]) for k,v in physics_state(env).items())
        initial=pixels(env);initial_gate=rgb_replay_check(z['initial'],initial);assert initial_gate['passed']
        snapshot=capture(env);raw=pixels(env);restore(env,snapshot);canonical=mjstate(env).copy();post=pixels(env)
        rows=[]
        for i in range(10):
            restore(env,snapshot);assert np.array_equal(mjstate(env),canonical)
            first=pixels(env);model.vision(rgb_tensor(first,model));second=pixels(env)
            rows.append(dict(index=i,precanonical=rgb_replay_check(raw,first),canonical=rgb_replay_check(post,first),
                CUDA_vision_before_after=rgb_replay_check(first,second),second_precanonical=rgb_replay_check(raw,second)))
        assert tensor_sha(model)==before
        save(BASE/'runs/metaworld_loop_geometry_20261109/replay-drawer0-v1.json',dict(utc=now(),initial_gate=initial_gate,
            precanonical_to_restored=rgb_replay_check(raw,post),rows=rows,weights_unchanged=True,optimizer_updates=0,native_steps=0,
            all_integration_restores_exact=True,protected_audit_only=True))
    finally:close_environment(env)


if __name__=='__main__':main()
