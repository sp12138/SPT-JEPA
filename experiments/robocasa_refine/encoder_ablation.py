"""Finite frozen-encoder attribution: encoder x compressed-SPT/dense prediction.

Only ordinary random-command logs train the fresh heads. External weights supply
the V-JEPA 2.1 image encoder only; no released predictor, action target, proprio,
reward, expert path or protected diagnostic enters the learner.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_compare import train as common

MODULE = "experiments.robocasa_refine.encoder_ablation"
HOSTS = ("dev13", "dev15", "dev6")
TRAIN_PYTHON = "/efs/prescottsun/project_jepa/envs/lewm/bin/python"


def initialize(root, source, original):
    root.mkdir(parents=True, exist_ok=False)
    card = json.loads((source / "interventions/dataset.json").read_text())
    assert card["passed"] and len(card["records"]) == 1024
    records = sorted(card["records"], key=lambda r: r["id"])
    assert {r["split"] for r in records} == {"train", "validation"}
    assert not ({tuple(r["kitchen"]) for r in records if r["split"] == "train"} &
                {tuple(r["kitchen"]) for r in records if r["split"] == "validation"})
    save(root / "data.json", dict(records=records, mean=card["action_mean"], std=card["action_std"],
        source_sha256=sha(source / "interventions/dataset.json"),
        ordinary_commands_only=True, old_human_training_not_repeated=True))
    plan = dict(original=str(original), source=str(source), max_steps=1000, global_batch=32,
        window=8, deadline=time.time()+6*3600, encoder_shapes={"own": [257,192], "vjepa": [196,1664]},
        external_encoder=str(original / "reference/encoder.pth.tar"),
        external_encoder_sha256=sha(original / "reference/encoder.pth.tar"),
        own_parents={str(s): str(source / "training" / f"fork-flat-seed{s}" / "step-007000/weights.pt")
                     for s in (1,2)},
        source_cases=json.loads((source / "VALIDATION_CASES_READY.json").read_text())["selected"],
        encoder_preprocessing="Same square-FOV224 uint8 images; ImageNet normalization; one image per encoder call",
        not_literal_public_full_FOV_resolution=True, no_automatic_long_training=True,
        external_predictor_or_decoder_used=False, no_future_or_optimal_action_target=True,
        fresh_heads=True, own_encoder_ancestry_updates=7000,
        external_pretraining_budget_not_matched=True, same_new_data_and_samples=True,
        dense_changes_both_carrier_and_objective=True)
    save(root / "plan.json", plan)
    for encoder in ("own", "vjepa"):
        for objective in ("spt", "dense"):
            for seed in (1,2):
                run=root / "training" / f"{encoder}-{objective}-seed{seed}"
                run.mkdir(parents=True)
                save(run / "config.json", dict(encoder=encoder, objective=objective, seed=seed,
                    feature_kind=f"own{seed}" if encoder == "own" else "vjepa", global_batch=32,
                    window=8, max_steps=1000, new_training_origin="fresh heads; frozen existing encoder"))


def original(root):
    return Path(json.loads((root / "plan.json").read_text())["original"])


def initialize_budget(root,parent):
    """Keep the completed1000 states; restart a common9000-update LR schedule."""
    root.mkdir(parents=True,exist_ok=False)
    for name in ("sources","environment","features","observations","cases","data.json","validation-plan.json",
                 "CACHE-DONE-dev13.json","CACHE-DONE-dev15.json","CACHE-DONE-dev6.json",
                 "PROOF-own1.json","PROOF-own2.json","PROOF-vjepa.json"):
        (root/name).symlink_to(parent/name,target_is_directory=(parent/name).is_dir())
    code=Path(__file__).resolve().parents[2]
    (root/"code").symlink_to(code,target_is_directory=True)
    plan=json.loads((parent/"plan.json").read_text())
    plan.update(parent=str(parent),max_steps=10000,stages=[1002,5000,10000],deadline=time.time()+6*3600,
        fresh_heads=False,prior_head_updates=1000,extra_head_updates=9000,
        training_data_and_model_unchanged=True,LR_restart="Common cosine horizon9000;1% warmup; retain moments",
        same_session_check_seconds=3600,not_a_from_scratch_fixed_LR_horizon_scaling_law=True,
        no_automatic_long_training=True)
    save(root/"plan.json",plan)
    rows=[]
    for source in sorted((parent/"training").iterdir()):
        old=torch.load(source/"last.ckpt",map_location="cpu",weights_only=False)
        assert old["step"]==1000 and old["samples"]==32000
        assert old["config_sha256"]==sha(source/"config.json")
        cfg=json.loads((source/"config.json").read_text())
        cfg.update(max_steps=10000,phase_start=1000,stages=[1002,5000,10000],parent=str(source),
            new_training_origin="Exact1000-step full state; new common LR restart only")
        run=root/"training"/source.name;run.mkdir(parents=True)
        save(run/"config.json",cfg)
        model=build(root,cfg);model.load_state_dict(old["model"],strict=True)
        assert common.state_hash(model.state_dict())==old["nn_sha256"]
        opt,scheduler=optimizer_for(model,cfg)
        assert common.associations(model,opt)==old["parameter_associations"]
        opt.load_state_dict(old["optimizer"])
        assert common.checkpoint_equal(old["optimizer"]["state"],opt.state_dict()["state"])
        rates=[base*factor(1000) for base,factor in zip(scheduler.base_lrs,scheduler.lr_lambdas)]
        schedule=copy.deepcopy(old["scheduler"]);schedule["_last_lr"]=rates
        scheduler.load_state_dict(schedule)
        for group,lr in zip(opt.param_groups,rates):group["lr"]=lr
        state=dict(old,model=common.cpu(model.state_dict()),optimizer=common.cpu(opt.state_dict()),
            scheduler=scheduler.state_dict(),config_sha256=sha(run/"config.json"))
        assert common.checkpoint_equal(old["rank_rng"],state["rank_rng"])
        common.save_full_state(run,state)
        (run/"step-001000").symlink_to(source/"step-001000",target_is_directory=True)
        (run/"original-initial.json").symlink_to(source/"initial.json")
        evaluation=root/"evaluation"/source.name;evaluation.mkdir(parents=True)
        (evaluation/"step-001000").symlink_to(parent/"evaluation"/source.name/"step-001000",target_is_directory=True)
        row=dict(passed=True,parent=str(source),parent_step=1000,model_hash=old["nn_sha256"],
            optimizer_moments_RNG_cursor_exact=True,new_parameters=0,SGD_replayed=0,
            new_LR_horizon=9000,new_LR_warmup=90,restored_world_size=old["world_size"])
        save(run/"ancestry.json",row);rows.append(dict(run=source.name,**row))
        print(json.dumps(dict(initialized=source.name,parent_step=1000,new_updates=9000)),flush=True)
        del old,model,opt,scheduler,state
    save(root/"INITIALIZED.json",dict(passed=True,rows=rows,inherited_control_episodes=32,
        new_control_episodes=64,physical_cases=4,new_SGD=0,SGD_replayed=0))


def load_encoder(root, kind):
    from experiments.robocasa_compare.evaluate import install_public
    plan=json.loads((root / "plan.json").read_text())
    common.install_runtime(root / "code")
    install_public(root)
    if kind == "vjepa":
        from app.robo_jepa.utils import build_encoder
        encoder=build_encoder(version="v21", filename=plan["external_encoder"], device="cuda")
    else:
        from transformers import ViTConfig, ViTModel
        encoder=ViTModel(ViTConfig(hidden_size=192,num_hidden_layers=12,num_attention_heads=3,
            intermediate_size=768,image_size=224,patch_size=14),add_pooling_layer=False,use_mask_token=False)
        weights=torch.load(plan["own_parents"][kind[-1]],map_location="cpu",weights_only=True)
        state={k[len("encoder."):]:v for k,v in weights.items() if k.startswith("encoder.")}
        encoder.load_state_dict(state,strict=True)
        encoder=encoder.cuda()
        del weights
    return encoder.eval().requires_grad_(False)


@torch.no_grad()
def encode(encoder, images, kind):
    mean=images.new_tensor([.485,.456,.406],dtype=torch.float32)[None,:,None,None]
    std=images.new_tensor([.229,.224,.225],dtype=torch.float32)[None,:,None,None]
    value=(images.float()/255.-mean)/std
    with torch.autocast("cuda",dtype=torch.float16 if kind == "vjepa" else torch.bfloat16):
        if kind == "vjepa":
            result=encoder(value[:,:,None])
        else:
            result=encoder(value,interpolate_pos_encoding=True).last_hidden_state
    if kind == "vjepa": result=F.layer_norm(result,(result.shape[-1],))
    return result.to(torch.float16)


def cache(root, kind, shard, shards):
    torch.set_num_threads(2)
    records=json.loads((root / "data.json").read_text())["records"]
    encoder=load_encoder(root,kind)
    versions=[p._version for p in encoder.parameters()]
    directory=root / "features" / kind
    directory.mkdir(parents=True,exist_ok=True)
    count=0
    for index,r in enumerate(records):
        if index % shards != shard: continue
        assert time.time() < json.loads((root/"plan.json").read_text())["deadline"]
        assert not (root / "STOP").exists()
        out=directory / (r["id"]+".npy")
        assert not out.exists(), out
        assert sha(Path(r["path"])) == r["sha256"]
        with np.load(r["path"]) as log:
            pixels=log["pixels"].copy()
            commands=log["commands"].copy()
        assert pixels.shape == (17,2,3,224,224) and commands.shape == (16,7)
        images=torch.from_numpy(pixels).flatten(0,1).cuda()
        features=torch.cat([encode(encoder,x,kind) for x in images.split(8)],0)
        assert torch.isfinite(features).all()
        if count == 0:
            single=encode(encoder,images[:1],kind)
            error=float((single.float()-features[:1].float()).abs().max())
            assert error < .035, error  # FP16/BF16 size-changing reductions.
            exact=float((encode(encoder,images[:8],kind)-features[:8]).abs().max())
            assert exact == 0.,exact
            save(directory / f"ENTRY-{shard}.json",dict(passed=True,kind=kind,
                batch_vs_single_max=error,same_batch_exact=exact,
                encoder_frozen=True,encoder_only=True,feature_shape=list(features.shape)))
        data=features.reshape(17,2,*features.shape[1:]).cpu().numpy()
        with out.with_suffix(".partial").open("xb") as stream: np.save(stream,data)
        out.with_suffix(".partial").replace(out)
        # Only V-JEPA workers write the shared decoder images/current controls.
        if kind == "vjepa":
            common_dir=root / "observations";common_dir.mkdir(exist_ok=True)
            small=F.interpolate(images.float(),size=(64,64),mode="area").round().byte().reshape(17,2,3,64,64).cpu().numpy()
            np.savez(common_dir/(r["id"]+".npz"),render=small,commands=commands)
        save(out.with_suffix(".json"),dict(passed=True,id=r["id"],source_sha256=r["sha256"],
            feature_sha256=sha(out),shape=list(data.shape),encoder_kind=kind))
        count+=1
        if count % 8 == 0: print(json.dumps(dict(kind=kind,shard=shard,completed=count)),flush=True)
    assert versions == [p._version for p in encoder.parameters()]
    assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())
    save(directory/f"DONE-{shard}.json",dict(passed=True,kind=kind,shard=shard,shards=shards,
        records=count,encoder_unchanged=True))


class FeatureWindows:
    def __init__(self,root,cfg,split="train"):
        self.root=Path(root);self.cfg=cfg
        card=json.loads((self.root/"data.json").read_text())
        self.records=[r for r in card["records"] if r["split"] == split]
        self.mean=np.asarray(card["mean"],np.float32);self.std=np.asarray(card["std"],np.float32)
        self.arrays={};self.observations={}
    def __getitem__(self,index):
        rng=np.random.default_rng(np.random.SeedSequence([781021,self.cfg["seed"],int(index)]))
        r=self.records[int(rng.integers(len(self.records)))]
        start=int(rng.integers(18-self.cfg["window"]))
        rid=r["id"]
        if rid not in self.arrays:
            self.arrays[rid]=np.load(self.root/"features"/self.cfg["feature_kind"]/(rid+".npy"),mmap_mode="r")
            with np.load(self.root/"observations"/(rid+".npz")) as a:
                self.observations[rid]=(a["render"].copy(),a["commands"].copy())
        render,commands=self.observations[rid];w=self.cfg["window"]
        raw=np.zeros((w,14),np.float32)
        raw[:-1,7:10]=commands[start:start+w-1,:3]
        raw[:-1,13]=-.04*commands[start:start+w-1,6]
        action=np.zeros_like(raw);action[:-1]=(raw[:-1]-self.mean)/self.std
        return dict(features=torch.from_numpy(self.arrays[rid][start:start+w].transpose(1,0,2,3).copy()),
            render=torch.from_numpy(render[start:start+w].transpose(1,0,2,3,4).copy()),
            action=torch.from_numpy(action),identity=torch.tensor([self.records.index(r),start]))


def prepare(batch,device):
    features=batch["features"].to(device,non_blocking=True)
    b,v,w=features.shape[:3]
    return dict(features=features.flatten(0,1),render=batch["render"].to(device,non_blocking=True).flatten(0,1),
        action=batch["action"].to(device,non_blocking=True)[:,None].expand(b,v,w,14).reshape(b*v,w,14))


def build(root,cfg):
    common.install_runtime(root / "code")
    dim=1664 if cfg["encoder"] == "vjepa" else 192
    torch.manual_seed(61001+cfg["seed"])
    if cfg["objective"] == "spt":
        from experiments.robocasa_refine.analytic_model import AnalyticResidualSPT
        class CachedSPT(AnalyticResidualSPT):
            def __init__(self):
                super().__init__(mode="L2",beta=.25,sigma0=.1,window=8,anchor_gap=0,
                    renderer_width=384,renderer_depth=12,renderer_heads=6,action_dim=14,
                    rho=.8,forecast_fraction=.5,forecast_seed=61001+cfg["seed"],rate_prior="forecast",
                    flow_logit_shift=cfg.get("flow_logit_shift",0.),state_slots=cfg.get("state_slots",4))
                self.code_chart=cfg.get("code_chart","global")
                self.encoder=nn.Identity()
                # Adapter initialized after all common SPT heads, preserving their RNG.
                self.feature_adapter=nn.Sequential(nn.Linear(dim,192),nn.LayerNorm(192))
            def prepare(self,batch):
                self.cached=batch["features"]
                render=batch["render"].float()/127.5-1.
                return render,render,render[:,0]
            def encode(self,info):
                b,w=self.cached.shape[:2]
                features=self.cached.float()
                if self.code_chart=="delta":features=features-features[:,:1]
                tokens=self.feature_adapter(features.flatten(0,1))
                info["emb"]=self.pool(tokens).reshape(b,w,self.state_dim)
                if self.code_chart=="delta":info["emb"]=info["emb"]-info["emb"][:,:1]
                if "action" in info:info["act_emb"]=self.action_encoder(info["action"])
                return info
            def loss(self,batch,step=0,ramp_steps=0):
                try:return super().loss(batch,step,ramp_steps)
                finally:self.cached=None
        model=CachedSPT()
        if cfg.get("view_message"):
            from experiments.robocasa_refine.cross_view import attach
            attach(model,cfg["view_message"],cfg["seed"])
        if cfg.get("target_refinement"):
            from experiments.robocasa_refine.target_refinement import configure
            configure(model,cfg["target_refinement"],cfg["seed"])
        if cfg.get("context_memory"):
            from experiments.robocasa_refine.context_prior import attach
            attach(model,cfg["context_memory"],cfg["seed"])
        if cfg.get("forecast_only"):
            from experiments.robocasa_refine.conditional_codes import forecast_only
            forecast_only(model)
        if cfg.get("pool_layout")=="grid":
            from experiments.robocasa_refine.spatial_codes import restrict_pool
            restrict_pool(model.pool)
        return model
    return DenseWorld(dim,8)


class AxialBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial=nn.MultiheadAttention(192,3,dropout=0.,batch_first=True)
        self.temporal=nn.MultiheadAttention(192,3,dropout=0.,batch_first=True)
        self.norms=nn.ModuleList([nn.LayerNorm(192) for _ in range(3)])
        self.ff=nn.Sequential(nn.Linear(192,2048),nn.GELU(),nn.Linear(2048,192))
    def forward(self,x):
        b,t,p,d=x.shape
        q=self.norms[0](x).reshape(b*t,p,d)
        x=x+self.spatial(q,q,q,need_weights=False)[0].reshape(b,t,p,d)
        q=self.norms[1](x).permute(0,2,1,3).reshape(b*p,t,d)
        mask=torch.ones(t,t,device=x.device,dtype=torch.bool).triu(1)
        x=x+self.temporal(q,q,q,attn_mask=mask,need_weights=False)[0].reshape(b,p,t,d).permute(0,2,1,3)
        return x+self.ff(self.norms[2](x))


class DenseWorld(nn.Module):
    def __init__(self,dim,window):
        super().__init__();self.dim=dim;self.window=window
        # Input width must not change the RNG initialization of shared blocks.
        with torch.random.fork_rng(devices=[]):self.input=nn.Linear(dim,192)
        self.action=nn.Sequential(nn.Linear(14,192),nn.GELU(),nn.Linear(192,192))
        self.time=nn.Parameter(torch.randn(1,window,1,192)*.02)
        self.blocks=nn.ModuleList([AxialBlock() for _ in range(10)])
        self.output=nn.Linear(192,dim)
        nn.init.normal_(self.output.weight,std=.001);nn.init.zeros_(self.output.bias)
    def predict(self,features,actions):
        if features.shape[2] == 257: features=features[:,:,1:]
        t,p=features.shape[1:3]
        position=torch.arange(p,device=features.device)[:,None]
        freq=torch.exp(torch.arange(0,192,2,device=features.device)*(-math.log(10000.)/192))
        pe=torch.stack([(position*freq).sin(),(position*freq).cos()],-1).flatten(-2)
        x=self.input(features.float())+self.action(actions.float())[:,:,None]+self.time[:,:t]+pe[None,None]
        for block in self.blocks:x=block(x)
        return F.layer_norm(features.float()+self.output(x).float(),(self.dim,))
    def loss(self,batch,step=0):
        z=batch["features"].float()
        if z.shape[2] == 257:z=z[:,:,1:]
        a=batch["action"]
        tf=self.predict(z[:,:-1],a[:,:-1]);tf_loss=F.l1_loss(tf,z[:,1:])
        history=[z[:,:1]];predictions=[]
        for t in range(z.shape[1]-1):
            predicted=self.predict(torch.cat(history,1),a[:,:t+1])[:,-1:]
            predictions.append(predicted);history.append(predicted.detach())
        ar_loss=F.l1_loss(torch.cat(predictions,1),z[:,1:])
        loss=.5*(tf_loss+ar_loss)
        return loss,dict(loss=loss,tf=tf_loss,ar=ar_loss)


def optimizer_for(model,cfg):
    phase=cfg.get("phase_start",0);total=cfg["max_steps"]
    if cfg["objective"] == "spt":return common.make_optimizer(model,total,phase)
    horizon=total-phase
    assert horizon>0
    warmup=max(1,int(horizon*.01))
    def factor(step):
        n=max(0,step-phase)
        return min(1.,max(1,n)/warmup)*.5*(1+math.cos(math.pi*min(n,horizon)/horizon))
    opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.001)
    return opt,torch.optim.lr_scheduler.LambdaLR(opt,factor)


def gradient_prefixes(cfg):
    prefixes=(("pool.","predictor.","action_encoder.","renderer.","feature_adapter.")
        if cfg["objective"]=="spt" else ("input.","action.","blocks.","output."))
    if cfg.get("target_refinement")=="fixed":
        prefixes=tuple(p for p in prefixes if p not in ("pool.","feature_adapter."))
    if cfg.get("forecast_only"):
        prefixes=tuple(p for p in prefixes if p not in ("pool.","feature_adapter.","renderer."))
    return prefixes+(("view_bridge.",) if cfg.get("view_message") else ())+(("context_bridge.",) if cfg.get("context_memory") else ())


def training(root,run,resume=False,stop=1000):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    rank=int(os.environ["RANK"]);world=int(os.environ["WORLD_SIZE"]);local=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local);device=torch.device("cuda",local)
    dist.init_process_group("nccl",device_id=device)
    torch.set_num_threads(2);torch.set_float32_matmul_precision("high")
    cfg=json.loads((run/"config.json").read_text());model=build(root,cfg).to(device)
    from vforge.ops import OpsConfig,initialize_ops
    initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=device,parallel_state=None)
    opt,scheduler=optimizer_for(model,cfg)
    start=0;restored=None
    if resume:
        state=torch.load(run/"last.ckpt",map_location="cpu",weights_only=False)
        assert state["config_sha256"] == sha(run/"config.json") and state["world_size"] == world
        assert state["parameter_associations"] == common.associations(model,opt)
        model.load_state_dict(state["model"],strict=True);opt.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        assert common.state_hash(model.state_dict()) == state["nn_sha256"]
        start=state["step"];restored=state["rank_rng"][rank]
        assert state["samples"] == start*cfg["global_batch"]
        if rank==0:save(run/f"restore-{start:06d}.json",dict(passed=True,step=start,
            neural_hash=state["nn_sha256"],optimizer_parameter_associations_exact=True,
            rank_RNG_and_sample_cursor_restored=True,world_size=world,SGD_replayed=0))
    elif rank == 0:
        excluded=("feature_adapter.",) if cfg["objective"] == "spt" else ("input.","output.")
        common_parameters={k:v for k,v in model.state_dict().items() if not k.startswith(excluded)}
        save(run/"initial.json",dict(passed=True,parameters=sum(p.numel() for p in model.parameters()),
            common_heads_sha256=common.state_hash(common_parameters),fresh_heads=True,
            encoder_not_in_optimizer_or_checkpoint=True,encoder_kind=cfg["encoder"]))
    assert cfg.get("phase_start",0)<=start<stop<=cfg["max_steps"]
    wrapped=DDP(common.LossModel(model),device_ids=[local],broadcast_buffers=False,find_unused_parameters=True)
    data=FeatureWindows(root,cfg)
    loader=DataLoader(data,batch_sampler=common.CounterBatches(start,stop,cfg["global_batch"],rank,world),
        num_workers=2,pin_memory=True,prefetch_factor=2,generator=torch.Generator().manual_seed(91000+cfg["seed"]+rank))
    iterator=iter(loader)
    if restored: common.restore_rng(restored)
    else:
        torch.manual_seed(98000+cfg["seed"]+rank);random.seed(98000+cfg["seed"]+rank);np.random.seed(98000+cfg["seed"]+rank)
    began=time.time();last_grad={}
    model.train()
    for step,raw in enumerate(iterator,start+1):
        assert not (root/"STOP").exists() and time.time()<json.loads((root/"plan.json").read_text())["deadline"]
        opt.zero_grad(set_to_none=True)
        batch=prepare(raw,device)
        with torch.autocast("cuda",dtype=torch.bfloat16):loss,metrics=wrapped(batch,step)
        assert torch.isfinite(loss),metrics
        loss.backward()
        if step <= 4 or step<=start+2 or step % 100 == 0 or step == stop:
            last_grad={n:float(p.grad.detach().float().norm()) for n,p in model.named_parameters() if p.grad is not None}
            assert last_grad and all(math.isfinite(v) for v in last_grad.values())
            if cfg["objective"] == "spt":
                for prefix in gradient_prefixes(cfg):
                    norm=sum(v for n,v in last_grad.items() if n.startswith(prefix))
                    if prefix == "action_encoder." and step <= 2:
                        # Output and adaLN gates are both zero initially. Strict
                        # positivity starts after their two activation updates.
                        if step == 1:assert torch.count_nonzero(model.predictor.transformer.output_proj.weight) == 0
                    else:assert norm>0,prefix
            else:
                for prefix in ("input.","action.","blocks.","output."):
                    assert sum(v for n,v in last_grad.items() if n.startswith(prefix))>0,prefix
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step();scheduler.step()
        if rank == 0 and (step<=2 or step<=start+2 or step%20==0 or step==stop):
            save(run/"progress.json",dict(step=step,loss=float(loss),metrics={k:float(v.detach()) for k,v in metrics.items()},
                elapsed_seconds=time.time()-began,completed=step==cfg["max_steps"],gradients_checked=bool(last_grad),
                phase_start=cfg.get("phase_start",0),segment_start=start,
                gradient_norms={prefix:sum(v*v for n,v in last_grad.items() if n.startswith(prefix))**.5
                    for prefix in gradient_prefixes(cfg)}))
        if step == stop or step % 250 == 0:
            common.checkpoint(run,model,opt,scheduler,cfg,step,world,rank,
                publish=step in cfg.get("stages",(4,500,1000)))
    dist.destroy_process_group()


class Adapter:
    def __init__(self,root,run,step):
        self.root=root;self.cfg=json.loads((run/"config.json").read_text())
        self.model=build(root,self.cfg).cuda().eval()
        weights=torch.load(run/f"step-{step:06d}/weights.pt",map_location="cpu",weights_only=True)
        self.model.load_state_dict(weights,strict=True)
        self.before=common.state_hash(self.model.state_dict());self.encoder=load_encoder(root,self.cfg["feature_kind"])
        self.versions=[p._version for p in self.encoder.parameters()]
        card=json.loads((root/"data.json").read_text());self.mean=torch.tensor(card["mean"],device="cuda");self.std=torch.tensor(card["std"],device="cuda")
        self.extra_information=f"Frozen {self.cfg['encoder']} RGB encoder; fresh {self.cfg['objective']} predictor; no proprio; same ordinary-command training"
        self.online_correction=False;self.residual=None
    @torch.no_grad()
    def dense(self,observations):
        import cv2
        cv2.setNumThreads(1)
        def pixels(observation):
            result=[]
            for key in ("img_0","img_1"):
                image=observation[key];h,w=image.shape[:2]
                assert (h,w) in ((240,320),(480,640)) and image.dtype == np.uint8 and image.std()>1.
                left=(w-h)//2
                result.append(cv2.resize(image[:,left:left+h],(224,224),interpolation=cv2.INTER_AREA).transpose(2,0,1))
            return np.stack(result)
        images=torch.from_numpy(np.stack([pixels(o) for o in observations])).flatten(0,1).cuda()
        return encode(self.encoder,images,self.cfg["feature_kind"])
    @torch.no_grad()
    def encode(self,observations):
        features=self.dense(observations)
        if self.cfg["objective"] == "spt":
            if self.cfg.get("context_memory"):self._latest_dense_initial=features[:2].detach()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                pooled=self.model.pool(self.model.feature_adapter(features.float()))
            return pooled.reshape(1,len(observations),2,self.model.state_slots,1,self.model.state_width).float()
        if features.shape[1] == 257:features=features[:,1:]
        p,d=features.shape[1:];side=math.isqrt(p);assert side*side == p
        return features.reshape(1,len(observations),2,side,side,d).float()
    def history(self,observations,commands,realized_controls=None):
        from experiments.robocasa_compare.evaluate import RollState,control14
        a=torch.tensor(np.asarray(commands,np.float32).reshape(1,-1,7),device="cuda")
        encoded=self.encode(observations)
        if self.cfg.get("context_memory"):
            # Only real history sets dynamics context; encoding a goal never does.
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                self.model.context_tokens=self.model.feature_adapter(self._latest_dense_initial.float())
        return RollState(encoded,(control14(a)-self.mean)/self.std)
    @torch.no_grad()
    def step(self,action,state,**kwargs):
        from experiments.robocasa_compare.evaluate import RollState,control14
        known=(control14(action.cuda())-self.mean)/self.std
        controls=torch.cat([state.actions,known],1)
        b,t,v,h,w,d=state.features.shape
        controls=controls[:,None].expand(b,v,t,14).reshape(b*v,t,14)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            if self.cfg["objective"] == "spt":
                z=state.features.permute(0,2,1,3,4,5).reshape(b*v,t,h*w*d)
                future=self.model.predict(z,self.model.action_encoder(controls))[:,-1].reshape(b,v,h,w,d)[:,None].float()
            else:
                z=state.features.permute(0,2,1,3,4,5).reshape(b*v,t,h*w,d)
                future=self.model.predict(z,controls)[:,-1].reshape(b,v,h,w,d)[:,None].float()
        features=torch.cat([state.features,future],1)[:,-7:]
        actions=controls.reshape(b,v,t,14)[:,0][:,-(features.shape[1]-1):]
        return future,RollState(features,actions)
    def unchanged(self):
        return (common.state_hash(self.model.state_dict()) == self.before and
                self.versions == [p._version for p in self.encoder.parameters()])


def evaluate(root,run,step,technical=False,case=None,proposal="released",goal_metric="uniform"):
    from experiments.robocasa_compare import evaluate as base
    from experiments.robocasa_refine import validation
    base.install_public(root);validation.ROOT=root
    if proposal!="released":
        from app.robo_jepa.mpc import mpc_utils
        from experiments.robocasa_refine.endpoint_cem import make_cem
        mpc_utils.cem=make_cem(mpc_utils.cem,proposal)
    base.make_env=validation.make_env;base.case_name=validation.name
    base.initialize_singleton()
    common.install_runtime(root/"code")
    from vforge.ops import OpsConfig,initialize_ops
    initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda"),parallel_state=None)
    cfg=json.loads((run/"config.json").read_text())
    if cfg.get("code_chart","global")=="delta":
        from experiments.robocasa_refine.conditional_codes import ConditionalAdapter
        adapter=ConditionalAdapter(root,run,step)
    else:adapter=Adapter(root,run,step)
    cases=json.loads((root/"plan.json").read_text())["source_cases"]
    if case is not None:cases=[cases[case]]
    elif technical:cases=cases[:1]
    for c in cases:
        tier="evaluation" if goal_metric=="uniform" else "evaluation-"+goal_metric
        output=root/("entry" if technical else tier)/run.name/f"step-{step:06d}"/validation.name(**c)
        args=argparse.Namespace(root=root,run=run,step=step,reference=None,reference_mode=2,
            reference_pose_source="native",zero=False,history_controls="issued",online_correction=False,
            goal_metric=goal_metric,technical=technical,output=output,action_repeat=1,**c)
        base.evaluate(args,adapter=adapter)
        if proposal!="released":
            value=json.loads((output/"summary.json").read_text())
            value.update(CEM_proposal=proposal,
                planner_forced_gc="Omitted for acyclic tensor state; sampling/covariance intentionally replaced")
            save(output/"summary.json",value)


def diagnostic(root,run,step):
    """Fixed held-out windows; no SGD; within-representation copy normalization."""
    from torch.utils.data import DataLoader
    cfg=json.loads((run/"config.json").read_text());model=build(root,cfg).cuda().eval()
    model.load_state_dict(torch.load(run/f"step-{step:06d}/weights.pt",map_location="cpu",weights_only=True),strict=True)
    before=common.state_hash(model.state_dict())
    from vforge.ops import OpsConfig,initialize_ops
    initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda"),parallel_state=None)
    data=FeatureWindows(root,cfg,"validation");rows=[];identities=[]
    with torch.no_grad():
        for start in range(0,128,4):
            raw=torch.utils.data.default_collate([data[i] for i in range(start,start+4)])
            identities.extend(raw["identity"].tolist());batch=prepare(raw,"cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16):
                if cfg["objective"] == "spt":
                    model.cached=batch["features"]
                    if cfg.get("context_memory"):
                        model.context_tokens=model.feature_adapter(batch["features"][:,0].float())
                    z=model.encode(dict(pixels=batch["render"],action=batch["action"]))["emb"].float()
                    model.cached=None
                    pred=model.mean_trajectory(z[:,0],model.action_encoder(batch["action"])[:,:-1])
                    if cfg.get("context_memory"):model.context_tokens=None
                else:
                    z=batch["features"].float()
                    if z.shape[2] == 257:z=z[:,:,1:]
                    history=[z[:,:1]]
                    for t in range(z.shape[1]-1):history.append(model.predict(torch.cat(history,1),batch["action"][:,:t+1])[:,-1:])
                    pred=torch.cat(history,1)
            for h in (1,3,7):
                error=(pred[:,h]-z[:,h]).abs().flatten(1).mean(1)
                copy_error=(z[:,0]-z[:,h]).abs().flatten(1).mean(1)
                rows.append(dict(horizon=h,prediction=float(error.mean()),copy=float(copy_error.mean())))
    assert common.state_hash(model.state_dict()) == before
    save(run/f"diagnostic-{step:06d}.json",dict(passed=True,rows=rows,window_identities=identities,
        no_SGD=True,model_unchanged=True,raw_losses_not_comparable_between_representations=True))


def proof(root,kind):
    """Real TRAIN images, frozen encoder, sample independence and action support."""
    torch.set_num_threads(2);torch.set_float32_matmul_precision("high")
    common.install_runtime(root/"code")
    from vforge.ops import OpsConfig,initialize_ops
    initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda"),parallel_state=None)
    card=json.loads((root/"data.json").read_text());r=next(r for r in card["records"] if r["split"] == "train")
    with np.load(r["path"]) as log:images=torch.from_numpy(log["pixels"][:8].copy()).permute(1,0,2,3,4).cuda()
    cached=root/"features"/kind/(r["id"]+".npy")
    encoder=None
    if cached.exists():
        features=torch.from_numpy(np.load(cached)[:8].transpose(1,0,2,3).copy()).cuda()
        shards=16 if kind == "vjepa" else 4
        assert all(json.loads((root/"features"/kind/f"DONE-{s}.json").read_text())["encoder_unchanged"] for s in range(shards))
    else:
        encoder=load_encoder(root,kind);versions=[p._version for p in encoder.parameters()]
        features=torch.cat([encode(encoder,x,kind) for x in images.flatten(0,1).split(8)],0).reshape(2,8,-1,1664 if kind == "vjepa" else 192)
    preprocessing_error=None
    if kind == "vjepa":
        from app.robo_jepa.world_model import WorldModel
        raw=images.flatten(0,1)[:2]
        official=WorldModel.preprocess_frames(None,raw.permute(0,2,3,1).cpu().numpy(),(224,224)).permute(1,0,2,3).cuda()
        ours=(raw.float()/255.-raw.new_tensor([.485,.456,.406],dtype=torch.float32)[None,:,None,None])/raw.new_tensor([.229,.224,.225],dtype=torch.float32)[None,:,None,None]
        preprocessing_error=float((official-ours).abs().max())
        assert preprocessing_error < 1e-6,preprocessing_error
    with np.load(r["path"]) as log:
        controls=np.zeros((8,14),np.float32);controls[:-1,7:10]=log["commands"][:7,:3]
        controls[:-1,13]=-.04*log["commands"][:7,6]
    controls[:-1]=(controls[:-1]-np.asarray(card["mean"]))/np.asarray(card["std"])
    render=F.interpolate(images.flatten(0,1).float(),size=(64,64),mode="area").round().reshape(2,8,3,64,64)
    batch=dict(features=features,render=render,action=torch.from_numpy(controls).cuda()[None].expand(2,-1,-1).clone())
    rows=[]
    for objective in ("spt","dense"):
        cfg=dict(encoder="vjepa" if kind == "vjepa" else "own",objective=objective,seed=int(kind[-1]) if kind != "vjepa" else 1)
        model=build(root,cfg).cuda().eval()
        before=common.state_hash(model.state_dict())
        initial=common.cpu(model.state_dict())
        poisoned=dict(batch,action=batch["action"].clone());poisoned["action"][:,-1]=1000.
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            torch.manual_seed(89217);a,_=model.loss(batch,step=1)
            torch.manual_seed(89217);b,_=model.loss(poisoned,step=1)
        assert float((a-b).abs()) == 0.,(objective,float(a),float(b))
        model.train();model.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):loss,_=model.loss(batch,step=1)
        loss.backward()
        norms={n:float(p.grad.float().norm()) for n,p in model.named_parameters() if p.grad is not None}
        assert torch.isfinite(loss) and all(math.isfinite(x) for x in norms.values())
        prefixes=("pool.","predictor.","action_encoder.","renderer.","feature_adapter.") if objective == "spt" else ("input.","action.","blocks.","output.")
        temporary_updates=0
        if objective == "spt":
            assert torch.count_nonzero(model.predictor.transformer.output_proj.weight) == 0
            scratch_optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5)
            while sum(v for n,v in norms.items() if n.startswith("action_encoder.")) == 0 and temporary_updates < 4:
                scratch_optimizer.step();scratch_optimizer.zero_grad(set_to_none=True)
                temporary_updates+=1
                with torch.autocast("cuda",dtype=torch.bfloat16):loss,_=model.loss(batch,step=temporary_updates+1)
                loss.backward()
                norms={n:float(p.grad.float().norm()) for n,p in model.named_parameters() if p.grad is not None}
            model.load_state_dict(initial,strict=True)
        grouped_norms={prefix:sum(v*v for n,v in norms.items() if n.startswith(prefix))**.5 for prefix in prefixes}
        assert all(v>0 and math.isfinite(v) for v in grouped_norms.values()),(objective,grouped_norms)
        assert common.state_hash(model.state_dict()) == before
        rows.append(dict(objective=objective,loss=float(loss),terminal_poison_error=float((a-b).abs()),
            gradient_norms=grouped_norms,
            trainable_parameters=sum(p.numel() for p in model.parameters()),retained_SGD_updates=0,
            temporary_gradient_activation_updates=temporary_updates,all_weights_restored=True))
        del model
    if encoder is not None:assert versions == [p._version for p in encoder.parameters()]
    save(root/f"PROOF-{kind}.json",dict(passed=True,rows=rows,frozen_encoder=True,encoder_unchanged=True,
        real_train_source_sha256=r["sha256"],retained_SGD_updates=0,no_future_action_target=True,
        published_encoder_preprocessing_max_error=preprocessing_error))


def env(root,gpus,simulator=False):
    from experiments.robocasa_compare.campaign import environment
    return environment(root,gpus,simulator)


def call(root,argv,log,gpus,simulator=False):
    plan=json.loads((root/"plan.json").read_text());remaining=plan["deadline"]-time.time()
    assert remaining>0 and not (root/"STOP").exists()
    with Path(log).open("x") as stream:
        subprocess.run(argv,cwd=root/"code",env=env(root,gpus,simulator),stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=remaining)


def worker(root,host,adopt_cache=False,attempt="v5"):
    def caching(gpu):
        if host == "dev13":kind=f"own{1+gpu//4}";shard=gpu%4;shards=4
        else:kind="vjepa";shard=gpu+(8 if host == "dev6" else 0);shards=16
        call(root,[TRAIN_PYTHON,"-u","-m",MODULE,"--root",str(root),"--mode","cache",
            "--kind",kind,"--shard",str(shard),"--shards",str(shards)],
            root/f"cache-{kind}-{shard}.log",str(gpu))
    try:
        if adopt_cache:
            for kind,shards in (("own1",4),("own2",4),("vjepa",16)):
                rows=[json.loads((root/"features"/kind/f"DONE-{s}.json").read_text()) for s in range(shards)]
                assert all(r["passed"] and r["encoder_unchanged"] for r in rows)
                assert sum(r["records"] for r in rows) == 1024
        else:
            with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(caching,range(8)))
            save(root/f"CACHE-DONE-{host}.json",dict(passed=True,host=host))
        deadline=json.loads((root/"plan.json").read_text())["deadline"]
        while not all((root/f"CACHE-DONE-{h}.json").exists() for h in HOSTS):
            assert time.time()<deadline
            time.sleep(10)
        for kind in ("own1","own2","vjepa"):
            assert json.loads((root/f"PROOF-{kind}.json").read_text())["passed"]
        jobs=[]
        if host in ("dev13","dev15"):
            encoder="own" if host == "dev13" else "vjepa"
            jobs=[(encoder,"spt",s,",".join(str(g) for g in range((s-1)*4,s*4))) for s in (1,2)]
        else:
            jobs=[(e,"dense",s,f"{2*i},{2*i+1}") for i,(e,s) in enumerate((('own',1),('own',2),('vjepa',1),('vjepa',2)))]
        def one(job):
            encoder,objective,seed,gpus=job;run=root/"training"/f"{encoder}-{objective}-seed{seed}"
            logs=run/f"execution-{attempt}";logs.mkdir(exist_ok=True)
            world=len(gpus.split(","))
            cfg=json.loads((run/"config.json").read_text())
            stages=cfg.get("stages",(4,1000))
            for stop in stages:
                argv=[TRAIN_PYTHON,"-m","torch.distributed.run","--standalone","--nnodes=1",f"--nproc-per-node={world}",
                    "-m","--",MODULE,"--root",str(root),"--mode","train","--run",str(run),"--stop",str(stop)]
                if (run/"last.ckpt").exists():argv.append("--resume")
                if not (run/f"step-{stop:06d}/ready.json").exists():
                    call(root,argv,logs/f"train-{stop}.log",gpus)
                if stop == stages[0]:
                    if not list((root/"entry"/run.name/f"step-{stop:06d}").glob("*/summary.json")):
                        call(root,[str(root/"environment/bin/python"),"-u","-m",MODULE,"--root",str(root),
                            "--mode","evaluate","--run",str(run),"--step",str(stop),"--technical"],logs/"entry.log",gpus.split(",")[0],True)
                else:
                    if not (run/f"diagnostic-{stop:06d}.json").exists():
                        call(root,[TRAIN_PYTHON,"-u","-m",MODULE,"--root",str(root),"--mode","diagnostic",
                            "--run",str(run),"--step",str(stop)],logs/f"diagnostic-{stop}.log",gpus.split(",")[0])
                    def control(i):
                        from experiments.robocasa_refine.validation import name
                        case=json.loads((root/"plan.json").read_text())["source_cases"][i]
                        if (root/"evaluation"/run.name/f"step-{stop:06d}"/name(**case)/"summary.json").exists():return
                        call(root,[str(root/"environment/bin/python"),"-u","-m",MODULE,"--root",str(root),
                            "--mode","evaluate","--run",str(run),"--step",str(stop),"--case",str(i)],
                            logs/f"evaluate-{stop}-{i}.log",gpus.split(",")[i%world],True)
                    # Dense jobs use two GPUs, two independent cases per wave.
                    for begin in range(0,4,world):
                        with ThreadPoolExecutor(max_workers=world) as pool:list(pool.map(control,range(begin,min(4,begin+world))))
                    save(run/f"STAGE-{stop:06d}.json",dict(passed=True,step=stop,control_episodes=4))
            save(run/"DONE.json",dict(passed=True,new_updates=cfg["max_steps"]-cfg.get("phase_start",0),
                max_steps=cfg["max_steps"],no_automatic_long_training=True))
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:list(pool.map(one,jobs))
        save(root/f"DONE-{attempt}-{host}.json",dict(passed=True,host=host,cache_adopted=adopt_cache))
    except BaseException as error:
        save(root/f"ERROR-{'training-' if adopt_cache else ''}{attempt}-{host}.json",dict(error=repr(error),method_negative=False))
        raise


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",choices=["initialize","continue","cache","worker","train","evaluate","diagnostic","proof"],required=True)
    for name in ("source","original","run"):p.add_argument("--"+name,type=Path)
    p.add_argument("--kind",choices=["own1","own2","vjepa"]);p.add_argument("--shard",type=int);p.add_argument("--shards",type=int)
    p.add_argument("--host",choices=HOSTS);p.add_argument("--resume",action="store_true");p.add_argument("--technical",action="store_true")
    p.add_argument("--adopt-cache",action="store_true")
    p.add_argument("--attempt",default="v5")
    p.add_argument("--stop",type=int,default=1000);p.add_argument("--step",type=int,default=1000);p.add_argument("--case",type=int)
    p.add_argument("--proposal",choices=["released","gaussian","endpoint"],default="released")
    p.add_argument("--goal-metric",choices=["uniform","support","direction"],default="uniform")
    a=p.parse_args()
    if a.mode == "initialize":initialize(a.root,a.source,a.original)
    elif a.mode == "continue":initialize_budget(a.root,a.source)
    elif a.mode == "cache":cache(a.root,a.kind,a.shard,a.shards)
    elif a.mode == "worker":worker(a.root,a.host,a.adopt_cache,a.attempt)
    elif a.mode == "train":training(a.root,a.run,a.resume,a.stop)
    elif a.mode == "evaluate":evaluate(a.root,a.run,a.step,a.technical,a.case,a.proposal,a.goal_metric)
    elif a.mode == "proof":proof(a.root,a.kind)
    else:diagnostic(a.root,a.run,a.step)
