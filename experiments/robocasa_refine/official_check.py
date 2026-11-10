"""Check the released client history and CEM path using a real transition."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from experiments.robocasa_compare import evaluate as base
from experiments.robocasa_compare.acquire import save
from experiments.robocasa_refine import validation


def difference(a,b):
    assert a.shape==b.shape,(a.shape,b.shape)
    return float((a.float().cpu()-b.float().cpu()).abs().max()) if a.numel() else 0.


def main(args):
    base.install_public(args.root)
    base.initialize_singleton()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    validation.ROOT=args.validation_root
    env=validation.make_env("push",2,7401)
    try:
        base.reset(env,"push",7401,"goal")
        first=base.reset(env,"push",7401,"init")
        first["_reference_native_pose"]=base.native_training_pose(env)
        issued=np.array([0.,0.,-.3,0.,0.,0.,0.],np.float32)
        second,_,_,_=env.step(issued)
        second["_reference_native_pose"]=base.native_training_pose(env)
        observations=[first,second]
        factual=second["_reference_native_pose"]-first["_reference_native_pose"]
        command14=base.control14(torch.from_numpy(issued))[...].numpy()
        adapter=base.RoboAdapter(args.root,args.reference,3,"released")
        from app.robo_jepa.world_model import WorldModelState,poses_to_diffs
        from app.robo_jepa.mpc.mpc_utils import cem
        state=torch.load(args.root/"reference"/args.reference/"world_model.pth.tar",map_location="cpu",weights_only=False)
        supplied=state.get("predictor",state.get("model"))
        supplied={k.replace("module.",""):v for k,v in supplied.items()}
        parameters=dict(adapter.model.transition_model.named_parameters())
        missing=[k for k in parameters if k not in supplied]
        assert not missing,missing
        for key,param in parameters.items():
            assert torch.equal(param.detach().cpu(),supplied[key]),key
        del state,supplied
        features=adapter.encode(observations)
        direct=[]
        for obs in observations:
            views=[]
            for key in ("img_0","img_1"):
                pixels=np.asarray(Image.fromarray(obs[key]).resize((320,240)))
                views.append(adapter.model.encode(pixels[None],size=(240,320),dtype=torch.float16))
            direct.append(torch.cat(views,0))
        direct=torch.stack(direct,0)[None].float()
        encoder_difference=difference(features,direct)
        assert encoder_difference==0.,encoder_difference
        original_encode=adapter.encode
        adapter.encode=lambda _:features
        records=[]
        for mode,pose_source in ((3,"released"),(3,"native"),(2,"native")):
            adapter.mode=mode;adapter.pose_source=pose_source
            if mode==3:
                key="proprio_robocasa" if pose_source=="released" else "_reference_native_pose"
                poses=torch.as_tensor(np.stack([o[key] for o in observations]),device="cuda").float()[None]
                actions=poses[:,1:]-poses[:,:-1]
            else:
                poses=torch.zeros(1,2,7,device="cuda")
                observed=torch.as_tensor(np.stack([o["pose"] for o in observations])).float()
                actions=poses_to_diffs(observed)[None].cuda()

            def direct_history():
                history=WorldModelState.from_features(features=direct[:,:1],states=poses[:,:1],
                    view_map=torch.tensor([[0,2]],device="cuda"),
                    embodiment_ids=torch.tensor([[mode]],device="cuda"))
                # Literal client replay loop, with the Table-14 BF16 precision.
                with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                    for k in range(actions.shape[1]):
                        _,history=adapter.model.step(actions[:,k:k+1],history,no_grad=True,
                            device="cuda",use_cache=True,batch=100)
                        history.features=direct[:,k+1:k+2]
                        history.states=poses[:,k+1:k+2]
                return history

            implemented=adapter.history(observations,[issued])
            released=direct_history()
            errors=dict(features=difference(implemented.features,released.features),
                        states=difference(implemented.states,released.states),
                        actions=difference(implemented.actions,released.actions),
                        keys=max(difference(a,b) for a,b in zip(implemented.cache.keys,released.cache.keys)),
                        values=max(difference(a,b) for a,b in zip(implemented.cache.values,released.cache.values)))
            assert implemented.feature_positions==released.feature_positions
            assert max(errors.values())==0.,errors
            proposal=torch.tensor([[[.03,-.02,.01,0.,0.,0.,0.]]],device="cuda")
            first_out,_=adapter.step(proposal,implemented,use_cache=True,no_grad=True,device="cuda")
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                second_out,_=adapter.model.step(proposal,released,use_cache=True,no_grad=True,device="cuda")
            errors["next_prediction"]=difference(first_out,second_out)
            assert errors["next_prediction"]==0.,errors
            # Two independent cache histories; same sampler state and public CEM.
            kwargs=dict(rollout=2,cem_steps=2,samples=16,num_batches=2,cutoff=10,
                translation_only=True,use_gripper=False,maxnorm=.3,robot_init_std=.3,
                view_weights=[.5,.5],use_cache=True,total_steps=18)
            torch.manual_seed(8821);np.random.seed(8821)
            a=cem(adapter,adapter.history(observations,[issued]),features[:,-1:],step_func=adapter.step,**kwargs)
            def source_step(*a,**kw):
                with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                    return adapter.model.step(*a,**kw)
            torch.manual_seed(8821);np.random.seed(8821)
            b=cem(adapter.model,direct_history(),features[:,-1:],step_func=source_step,**kwargs)
            for k in range(3):
                errors[f"cem_output_{k}"]=max(difference(x,y) for x,y in zip(a[k],b[k]))
            assert max(errors.values())==0.,errors
            records.append(dict(mode=mode,pose_source=pose_source,maximum_errors=errors,
                literal_client_inputs=mode==3 and pose_source=="released",
                intentional_input_variant="zero absolute pose for RGB-only reference" if mode==2 else
                    "native training-aligned sensors" if pose_source=="native" else None))
            del implemented,released,first_out,second_out,a,b
            gc.collect();torch.cuda.empty_cache()
        adapter.encode=original_encode
        assert adapter.unchanged()
        result=dict(passed=True,reference=args.reference,actual_frozen_checkpoint_parameters_checked=len(parameters),
            no_random_predictor_parameters=True,encoder_max_error=encoder_difference,records=records,
            issued_command=issued.tolist(),actual_native_delta=factual.tolist(),
            issued_actual_l2=float(np.linalg.norm(factual-command14)),
            precision="per-camera encoder FP16; transition BF16 as paper Table14 (literal client specifies FP16)",
            physical_transition=True,model_unchanged=True,new_sgd=0,performance_claim=False)
        save(args.output,result);print(json.dumps(result),flush=True)
    finally:
        env.close()
        torch.distributed.destroy_process_group()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--validation-root",type=Path,required=True)
    p.add_argument("--reference",default="50m");p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();main(a)
