"""Same-size global versus spatially addressed predictive bottlenecks.

The global and grid variants have identical tensors and 16x16 code coordinates.
Each grid query reads one fixed region of the image token lattice. This gives
initial-goal differences an explicit spatial address, without object labels or
cross-example statistics. Frozen V-JEPA tokens still contain global context, so
local pooling is not a claim of local pixel receptive fields.
"""
import argparse,copy,json,math,time,types
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from experiments.robocasa_compare.acquire import save,sha
from experiments.robocasa_compare import train as common
from experiments.robocasa_refine import encoder_ablation as base

HOST_ARM={"dev13":"global16","dev15":"grid16","dev6":"grid16-noise"}


def grid_mask(queries,patches,device):
    k=math.isqrt(queries);p=math.isqrt(patches)
    assert k*k==queries and p*p==patches and p>=k
    pixel=torch.arange(patches,device=device)
    # Integer cells cover every patch once;14 need not be divisible by4.
    row=torch.div(pixel,p,rounding_mode="floor")*k//p;column=(pixel%p)*k//p
    owner=row*k+column
    return owner[None]!=torch.arange(queries,device=device)[:,None]


def _grid_forward(self,tokens):
    queries=self.queries.expand(tokens.shape[0],-1,-1)
    mask=grid_mask(self.queries.shape[0],tokens.shape[1],tokens.device)
    pooled,_=self.attn(queries,tokens,tokens,attn_mask=mask,need_weights=False)
    return self.head(self.norm(pooled))


def restrict_pool(pool):
    pool.forward=types.MethodType(_grid_forward,pool)


def lift(name,old,shape):
    """Repeat the old carrier; average its repeated input coordinates.

    This preserves the global deterministic function on the repeated-code
    subspace. Resized tensors get fresh optimizer moments; scalar rate/noise and
    dropout are not claimed to preserve the old optimization trajectory.
    """
    factors=tuple(n//o for n,o in zip(shape,old.shape))
    assert len(shape)==old.ndim and all(n==o*f and f in (1,4) for n,o,f in zip(shape,old.shape,factors)),(name,old.shape,shape)
    if name=="context_bridge.query.weight":
        width=old.shape[1]//2
        return torch.cat([old[:,:width].repeat(1,4),old[:,width:].repeat(1,4)],1)/4
    value=old.repeat(factors)
    if name.endswith(".weight") and old.ndim==2 and factors[1]>1:value=value/factors[1]
    return value


def initialize(root,parent):
    torch.set_num_threads(2);root.mkdir(parents=True,exist_ok=False)
    for name in ("sources","environment","features","observations","cases","data.json","validation-plan.json"):
        (root/name).symlink_to(parent/name,target_is_directory=(parent/name).is_dir())
    (root/"code").symlink_to(Path(__file__).resolve().parents[2],target_is_directory=True)
    plan=json.loads((parent/"plan.json").read_text())
    plan.update(parent=str(parent),arms=HOST_ARM,models=6,state_slots=16,state_width=16,
        max_steps=40000,stages=[30002,40000],phase_start=30000,prior_updates=30000,
        extra_updates_per_model=10000,deadline=time.time()+6*3600,world_size=4,
        new_control_episodes=48,physical_cases=4,metrics=["uniform","support"],
        no_automatic_long_training=True,all_variants_same_code_size_parameters_and_initial_tensors=True,
        transfer="Same-shaped tensors/moments retained; resize carrier with repeated outputs/averaged inputs, fresh moments on resized tensors",
        no_bitwise_original_trajectory_continuation_claim=True,LR_restart="Common10000-step cosine/100warmup",
        no_new_data=True,no_future_or_optimal_action_target=True,
        not_equal_capacity_or_FLOPs_to_old64_codes=True)
    save(root/"plan.json",plan);rows=[];hashes={}
    for arm in HOST_ARM.values():
      for seed in (1,2):
        source=parent/"training"/f"global-seed{seed}"
        old=torch.load(source/"last.ckpt",map_location="cpu",weights_only=False)
        assert old["step"]==30000 and old["world_size"]==4 and old["samples"]==960000
        assert old["config_sha256"]==sha(source/"config.json")
        cfg=json.loads((source/"config.json").read_text())
        cfg.update(state_slots=16,pool_layout="global" if arm=="global16" else "grid",
            flow_logit_shift=4. if arm=="grid16-noise" else 0.,parent=str(source),
            max_steps=40000,phase_start=30000,stages=[30002,40000],forecast_only=False,
            new_training_origin="Matched256D global/grid carrier; same compatible30000 tensors/moments and ordinary logs")
        run=root/"training"/f"{arm}-seed{seed}";run.mkdir(parents=True);save(run/"config.json",cfg)
        model=base.build(root,cfg);state=model.state_dict();retained=[];fresh=[]
        for key in state:
            if key in old["model"] and state[key].shape==old["model"][key].shape:
                state[key]=old["model"][key];retained.append(key)
            else:
                assert key in old["model"],key
                state[key]=lift(key,old["model"][key],state[key].shape);fresh.append(key)
        model.load_state_dict(state,strict=True)
        assert all(torch.equal(model.state_dict()[k],old["model"][k]) for k in retained)
        digest=common.state_hash(model.state_dict())
        if seed in hashes:assert hashes[seed]==digest
        else:hashes[seed]=digest
        opt,schedule=base.optimizer_for(model,cfg);names=common.associations(model,opt)
        prior={n:old["optimizer"]["state"][pid] for g,ns in zip(old["optimizer"]["param_groups"],old["parameter_associations"])
            for pid,n in zip(g["params"],ns) if pid in old["optimizer"]["state"]}
        optim=opt.state_dict();moved=[]
        for group,ns in zip(optim["param_groups"],names):
          for pid,name in zip(group["params"],ns):
            if name in prior and name in retained:optim["state"][pid]=copy.deepcopy(prior[name]);moved.append(name)
        opt.load_state_dict(optim)
        for group,ns in zip(opt.state_dict()["param_groups"],names):
          for pid,name in zip(group["params"],ns):
            if name in moved:assert common.checkpoint_equal(opt.state_dict()["state"][pid],prior[name])
        rates=[r*f(30000) for r,f in zip(schedule.base_lrs,schedule.lr_lambdas)]
        sched=copy.deepcopy(old["scheduler"]);sched["_last_lr"]=rates;schedule.load_state_dict(sched)
        for group,rate in zip(opt.param_groups,rates):group["lr"]=rate
        full=dict(old,model=common.cpu(model.state_dict()),optimizer=common.cpu(opt.state_dict()),scheduler=schedule.state_dict(),
            parameter_associations=names,nn_sha256=digest,config_sha256=sha(run/"config.json"))
        assert common.checkpoint_equal(full["rank_rng"],old["rank_rng"]);common.save_full_state(run,full)
        row=dict(passed=True,run=run.name,parent=str(source),retained_tensors=retained,lifted_tensors_with_fresh_moments=fresh,
            compatible_moments_retained=moved,four_rank_RNG_and_sample_cursor_retained=True,
            changed_carrier_not_a_bitwise_continuation=True,initial_NN_sha256=digest,new_SGD=0)
        save(run/"ancestry.json",row);rows.append(row);print(json.dumps(dict(initialized=run.name,fresh=len(fresh),retained_moments=len(moved))),flush=True)
        del old,model,opt,schedule,state,optim,prior,full
    save(root/"INITIALIZED.json",dict(passed=True,rows=rows,all_variant_tensors_initially_equal=True))


def worker(root,host):
    arm=HOST_ARM[host]
    assert json.loads((root/"METHOD_PROOF_APPROVED.json").read_text())["passed"]
    try:
      def one(seed):
        run=root/"training"/f"{arm}-seed{seed}";logs=run/"execution-v1";logs.mkdir(exist_ok=False)
        gpus=",".join(str(i) for i in range((seed-1)*4,seed*4))
        for stop in (30002,40000):
            base.call(root,[base.TRAIN_PYTHON,"-m","torch.distributed.run","--standalone","--nnodes=1","--nproc-per-node=4",
                "-m","--",base.MODULE,"--root",str(root),"--mode","train","--run",str(run),"--stop",str(stop),"--resume"],logs/f"train-{stop}.log",gpus)
            if stop==30002:
                base.call(root,[str(root/"environment/bin/python"),"-u","-m",base.MODULE,"--root",str(root),"--mode","evaluate",
                    "--run",str(run),"--step",str(stop),"--technical","--case","2"],logs/"entry.log",gpus.split(",")[0],True)
                save(run/"ENTRY_APPROVED.json",dict(passed=True,step=stop,actual_Push_H5=True))
            else:
                base.call(root,[base.TRAIN_PYTHON,"-u","-m",base.MODULE,"--root",str(root),"--mode","diagnostic","--run",str(run),"--step",str(stop)],logs/"diagnostic.log",gpus.split(",")[0])
                for metric in ("uniform","support"):
                    def evaluate(case):
                        base.call(root,[str(root/"environment/bin/python"),"-u","-m",base.MODULE,"--root",str(root),"--mode","evaluate",
                            "--run",str(run),"--step",str(stop),"--case",str(case),"--goal-metric",metric],logs/f"{metric}-{case}.log",gpus.split(",")[case],True)
                    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(evaluate,range(4)))
        save(run/"DONE.json",dict(passed=True,step=40000,new_updates=10000))
      with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,(1,2)))
      save(root/f"DONE-{host}.json",dict(passed=True))
    except BaseException as e:save(root/f"ERROR-{host}.json",dict(error=repr(e),method_negative=False));raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--mode",choices=("initialize","worker"),required=True)
    p.add_argument("--parent",type=Path);p.add_argument("--host",choices=HOST_ARM);a=p.parse_args()
    if a.mode=="initialize":initialize(a.root,a.parent)
    else:worker(a.root,a.host)
