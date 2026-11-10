"""Reference-first, one-component planning contrasts on the public predictor.

Full public RGB preprocessing -> matched square preprocessing -> SPT readout.
The last contrast changes only the goal-distance readout, retaining the exact
public encoder/predictor/history/CEM. Optional distributed CEM keeps the GLOBAL
candidate count fixed. Replicated physics is a technical implementation detail,
never counted as independent performance episodes. No SGD or action targets.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.robocasa_compare import evaluate as base
from experiments.robocasa_compare import train as common
from experiments.robocasa_compare.acquire import save,sha


class SPTReadout(nn.Module):
    def __init__(self,run,step,code):
        super().__init__();common.install_runtime(code)
        from vforge.models.as_jepa.pooling import QueryPool
        cfg=json.loads((run/"config.json").read_text())
        assert cfg["encoder"]=="vjepa" and cfg["objective"]=="spt"
        ready=json.loads((run/f"step-{step:06d}/ready.json").read_text())
        path=run/f"step-{step:06d}/weights.pt"
        assert sha(path)==ready["weights_sha256"]
        state=torch.load(path,map_location="cpu",weights_only=True)
        self.feature_adapter=nn.Sequential(nn.Linear(1664,192),nn.LayerNorm(192))
        self.pool=QueryPool(192,3,4,16)
        for prefix,module in (("feature_adapter.",self.feature_adapter),("pool.",self.pool)):
            module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
        self.eval().requires_grad_(False)
        self.parent=dict(run=str(run),step=step,full_parent_NN_hash=ready["model_sha256"],
            extracted_parameters=sum(p.numel() for p in self.parameters()))
        self.before=common.state_hash(self.state_dict())

    @torch.no_grad()
    def forward(self,value):
        # CEM passes[B,T,V,H*W*C]; the token count may vary by input recipe.
        assert value.shape[-1]%1664==0
        shape=value.shape[:-1]
        tokens=value.reshape(-1,value.shape[-1]//1664,1664).float()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            code=self.pool(self.feature_adapter(tokens)).flatten(1)
        return code.reshape(*shape,64).float()


class ReferenceAdapter(base.RoboAdapter):
    def __init__(self,root,size,recipe,readout_run=None,readout_step=10000,
                 reference_mode=2,pose_source="native"):
        super().__init__(root,size,reference_mode,pose_source)
        assert recipe in ("public","square")
        self.recipe=recipe
        self.readout=SPTReadout(readout_run,readout_step,root/"code").cuda() if readout_run else None
        inputs="RGB-only/action-key" if reference_mode==2 else f"RGB plus {pose_source} proprio/state-delta"
        self.extra_information+=(f"; {inputs}; camera recipe={recipe}; real executed-motion history")
        if self.readout:self.extra_information+="; only goal-distance readout replaced by existing SPT pool"

    def synchronize_observations(self,observations):
        import torch.distributed as dist
        if dist.get_world_size()==1:return observations
        fields=("img_0","img_1","pose","_reference_native_pose","proprio_robocasa")
        payload=[([{k:o[k] for k in fields if k in o} for o in observations]) if dist.get_rank()==0 else None]
        dist.broadcast_object_list(payload,src=0)
        assert len(payload[0])==len(observations)
        # history() consumes pose only after encode(); sharing the original
        # objects also synchronizes the executed-motion feedback across ranks.
        for observation,shared in zip(observations,payload[0]):observation.update(shared)
        return observations

    @torch.no_grad()
    def encode(self,observations):
        observations=self.synchronize_observations(observations)
        if self.recipe=="public":return super().encode(observations)
        import cv2
        cv2.setNumThreads(1)
        output=[]
        for observation in observations:
            views=[]
            for key in ("img_0","img_1"):
                image=observation[key];height,width=image.shape[:2]
                assert (height,width) in ((240,320),(480,640)) and image.dtype==np.uint8 and image.std()>1.
                left=(width-height)//2
                square=cv2.resize(image[:,left:left+height],(224,224),interpolation=cv2.INTER_AREA)
                views.append(self.model.encode(square[None],size=(224,224),dtype=torch.float16))
            output.append(torch.cat(views,0))
        return torch.stack(output,0)[None].float()

    def unchanged(self):
        return super().unchanged() and (self.readout is None or
            common.state_hash(self.readout.state_dict())==self.readout.before)


def install_cem(adapter,total_candidates,samples,technical=False):
    """Use native public distributed CEM; no extra sampled candidates."""
    import torch.distributed as dist
    from app.robo_jepa.mpc import mpc_utils
    world=dist.get_world_size();rank=dist.get_rank()
    assert total_candidates%(samples*world)==0 and total_candidates>=samples*world
    original=mpc_utils.cem
    partition_seeded=False
    def cem(world_model,state,goal,**kwargs):
        nonlocal partition_seeded
        if not partition_seeded:
            # Public translation candidates use NumPy, not Torch. Seed only at
            # the first planning call, after physical reset: changing the global
            # reset seed would invalidate the replicated physical case.
            if rank:
                current=np.random.get_state()
                seed=np.random.SeedSequence([*current[1].tolist(),int(current[2]),rank])
                np.random.seed(seed.generate_state(1)[0])
            fingerprint=hashlib.sha256(np.random.get_state()[1].tobytes()).hexdigest()
            fingerprints=[None]*world;dist.all_gather_object(fingerprints,fingerprint)
            assert len(set(fingerprints))==world
            adapter.candidate_partition_fingerprints=fingerprints
            partition_seeded=True
        kwargs.update(samples=samples,num_batches=total_candidates//(samples*world),world_size=world,rank=rank)
        if adapter.readout is not None:
            assert not isinstance(goal,dict)
            target=adapter.readout(goal.flatten(3,5))
            def objective(_target,prediction):
                return -(adapter.readout(prediction)-target).abs().mean(-1)
            kwargs["objective"]=objective
        return original(world_model,state,goal,**kwargs)
    mpc_utils.cem=cem


def initialize_distributed(gpus=None):
    import torch.distributed as dist
    local=int(os.environ.get("LOCAL_RANK",0));torch.cuda.set_device(local)
    if "RANK" in os.environ:
        dist.init_process_group("nccl",device_id=torch.device("cuda",local))
    else:base.initialize_singleton()
    # EGL enumerates physical devices independently of CUDA_VISIBLE_DEVICES.
    physical=os.environ.get("CUDA_VISIBLE_DEVICES","0").split(",")[local]
    os.environ["MUJOCO_EGL_DEVICE_ID"]=physical
    torch.set_num_threads(2);torch.set_float32_matmul_precision("high")
    return dist.get_rank(),dist.get_world_size()


def main(args):
    rank,world=initialize_distributed();base.install_public(args.root)
    common.install_runtime(args.root/"code")
    from experiments.robocasa_refine import validation
    validation.ROOT=args.root;base.make_env=validation.make_env;base.case_name=validation.name
    adapter=ReferenceAdapter(args.root,args.reference,args.recipe,args.readout_run,args.readout_step,
                             args.reference_mode,args.pose_source)
    install_cem(adapter,args.candidates,args.samples,args.technical)
    # Each rank draws an independent partition; the GLOBAL count stays fixed.
    # All ranks still execute the identical globally selected action sequence.
    seed=torch.manual_seed
    torch.manual_seed=lambda value:seed(int(value)+104729*rank)
    primary=args.output
    if rank:args.output=primary.parent.parent/"physics-replicas"/f"rank{rank}"/primary.parent.name/primary.name
    args.run=None;args.step=0;args.reference_pose_source=args.pose_source
    args.zero=False;args.history_controls="achieved";args.online_correction=False
    args.goal_metric="uniform";args.action_repeat=1
    base.evaluate(args,adapter=adapter)
    # evaluate() closes its local process group. Files remain per rank, with
    # only rank0 published as the performance episode by the external collector.
    path=args.output/"summary.json";value=json.loads(path.read_text())
    value.update(reference_camera_recipe=args.recipe,SPT_readout=adapter.readout.parent if adapter.readout else None,
        only_readout_changed=adapter.readout is not None,distributed_CEM_world_size=world,
        CEM_global_candidates=args.candidates,CEM_per_rank_batch=args.samples,
        CEM_per_rank_batches=args.candidates//(args.samples*world),rank=rank,
        independent_NumPy_candidate_streams=adapter.candidate_partition_fingerprints,
        primary_performance_episode=rank==0,physics_replica_not_independent_case=rank!=0,
        extra_new_SGD=0,no_future_action_target=True,
        public_predictor_and_encoder_retained=True,
        reference_history_is_released_actual_motion_not_budget_candidate_issued_history=True,
        not_an_equal_pretraining_or_data_SOTA_claim=True)
    save(path,value)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--reference",choices=["50m","2b450k","8b300k"],required=True)
    p.add_argument("--recipe",choices=["public","square"],required=True)
    p.add_argument("--reference-mode",type=int,choices=[2,3],default=2)
    p.add_argument("--pose-source",choices=["native","released"],default="native")
    p.add_argument("--readout-run",type=Path);p.add_argument("--readout-step",type=int,default=10000)
    p.add_argument("--task",choices=["reach","push"],required=True);p.add_argument("--kitchen",type=int,required=True)
    p.add_argument("--seed",type=int,default=7401);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--samples",type=int,default=32);p.add_argument("--candidates",type=int,default=1024)
    p.add_argument("--technical",action="store_true")
    main(p.parse_args())
