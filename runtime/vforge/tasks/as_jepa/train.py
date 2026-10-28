"""Native lifecycle integration; no project-owned trainer or checkpoint manager."""
import hashlib,json
from pathlib import Path
import torch
from vforge.data.loader import build_resumable_loader
from vforge.distributed.parallel_state import get_parallel_state
from vforge.models.factory import ModelSpec
from vforge.models.as_jepa.model import ASJepaSPT
from vforge.training.recipe import TrainComponents,StepOutput
from vforge.utils.training import build_optimizer_factory,build_scheduler_factory
from as_jepa_tools.data import PushTWindows,FullPushTWindows,full_phase_sampler
from vforge.data.spec import DataSpec


def build_train_components(request):
    cfg=dict(request.extras);ramp_steps=cfg.pop("ramp_steps",0);all_phase_data=cfg.pop("all_phase_data",False);warm_start=cfg.pop("warm_start_checkpoint",None);initial_state_std=cfg.pop("initial_state_std",None);export_every=cfg.pop("export_every",1000);data=Path(cfg.pop("data_root"));model_cfg=dict(cfg.pop("as_jepa"));kind=model_cfg.pop("kind","S2");legacy_output=cfg.pop("campaign_output",None);output=Path(request.output_dir);output.mkdir(parents=True,exist_ok=True)
    dataset=(FullPushTWindows if all_phase_data else PushTWindows)(data,window=model_cfg.get("window",8),gap=model_cfg.get("anchor_gap",8) if kind=="S2" else 0,seed=request.sampler_seed)
    loader=build_resumable_loader(DataSpec(dataset,sampler_factory=full_phase_sampler) if all_phase_data else dataset,parallel_state=get_parallel_state(),batch_size=request.batch_size,num_workers=request.num_workers,drop_last=request.drop_last,seed=request.sampler_seed,shuffle=request.shuffle,pin_memory=request.pin_memory,persistent_workers=request.persistent_workers,prefetch_factor=request.prefetch_factor)
    def build():
        with torch.device(request.init_device):
            if kind=="S2":return ASJepaSPT(**model_cfg)
            if kind=="S1Carry":
                from vforge.models.as_jepa.semantic_carry import SemanticCarry
                return SemanticCarry(**model_cfg)
            from vforge.models.as_jepa.spt_baseline import ActionRepWorld
            return ActionRepWorld(**model_cfg)
    def step_fn(model,batch,step,context):
        frames=batch["render"].to(context.device,dtype=torch.float32)/127.5-1.;anchor=batch["anchor"].to(context.device,dtype=torch.float32)/127.5-1.;actions=batch["action"].to(context.device)
        result=model(frames=frames,actions=actions,anchor=anchor,return_metrics=True,**(dict(beta_scale=min(1.,step/max(1,ramp_steps))) if kind=="S2" and ramp_steps else {}))
        return StepOutput(loss=result["loss"],metrics=result["metrics"],workload={"local_batch":len(frames),"local_tokens":len(frames)*(frames.shape[1]-1)*256},metadata={"sample_ids":[f"{int(ep)}:{int(start)}" for ep,start in zip(batch["episode"],batch["start"])]})
    def export(model,step,context,output_record):
        completed=step+1
        diagnostics={}
        if completed%100==0 and kind=="S2":
            for label,prefixes in [("encoder",("encoder.","pool.")),("prior",("prior.",)),("renderer",("renderer.",))]:
                values=[p.grad.detach() for name,p in model.named_parameters() if name.startswith(prefixes) and p.grad is not None]
                local=[g.to_local() if hasattr(g,"to_local") else g for g in values]
                total=sum((g.float().square().sum() for g in local),torch.zeros((),device=context.device))
                diagnostics["grad_norm_"+label+"_post_clip"]=float(total.sqrt())
        if completed%2000==0 and hasattr(model,"training_variable_rate"):
            with (output/"training_dimension_rates.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(step=completed,rate=model.training_variable_rate.cpu().tolist(),scope="Training last microbatch,variable per-window rate; formal activity is measured separately on heldout windows."))+"\n")
        if completed%export_every:return diagnostics
        folder=output/f"model-step{completed:07d}";folder.mkdir(exist_ok=True)
        weights={k:(v.full_tensor() if hasattr(v,"full_tensor") else v).detach().cpu() for k,v in model.state_dict().items()}
        torch.save(weights,folder/"weights.pt");(folder/"config.json").write_text(json.dumps(dict(_target_="vforge.models.as_jepa.model.ASJepaSPT" if kind=="S2" else "vforge.models.as_jepa.semantic_carry.SemanticCarry" if kind=="S1Carry" else "vforge.models.as_jepa.spt_baseline.ActionRepWorld",**model_cfg),indent=2))
        return diagnostics
    def initialize(model):
        model.initialize_weights()
        if warm_start:
            weights=torch.load(Path(warm_start)/"weights.pt",map_location=next(model.parameters()).device,weights_only=True)
            incompatible=model.load_state_dict(weights,strict=False)
            if incompatible.unexpected_keys or any(not k.startswith(("transition.","spatial.")) for k in incompatible.missing_keys):raise RuntimeError(str(incompatible))
            (output/"warm_start.json").write_text(json.dumps(dict(source=warm_start,scope="ExistingS1 initialization; new carry transition only remains fresh; training time counted separately."),indent=2))
        if initial_state_std is None or kind!="S2":return
        from torch.utils.data import DataLoader,Subset
        was_training=model.training;model.eval();values=[];device=next(model.parameters()).device
        ids=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(91))[:128].tolist()
        with torch.no_grad():
            for batch in DataLoader(Subset(dataset,ids),batch_size=16,num_workers=0):
                frames=batch["render"].to(device,dtype=torch.float32)/127.5-1
                values.append(model.encoded(frames).flatten(0,1).float())
            values=torch.cat(values);mean=values.mean(0).mean(0);std=values.flatten(1).std(0).mean().clamp_min(1e-8)
            scale=float(initial_state_std)/float(std);model.pool.head.weight.mul_(scale);model.pool.head.bias.sub_(mean).mul_(scale)
        model.train(was_training)
        (output/"initial_scale.json").write_text(json.dumps(dict(target_std=initial_state_std,original_std=float(std),scale=scale,scope="One-time Native initializer before training or checkpoint restore; any full restore replaces these initial weights."),indent=2))
    contract=dict(dataset="as_jepa_pusht_lance_allstarts_v1" if all_phase_data else "as_jepa_pusht_phase0_v1",data_card_sha256=hashlib.sha256((data/"dataset_card.json").read_bytes()).hexdigest(),window=dataset.window,gap=dataset.gap)
    return TrainComponents(model=ModelSpec(build),dataloader=loader,optimizer_factory=build_optimizer_factory(learning_rate=request.learning_rate,weight_decay=request.weight_decay,betas=request.betas,eps=request.eps),step_fn=step_fn,scheduler_factory=build_scheduler_factory(name=request.scheduler_name,factor=request.scheduler_factor,warmup_steps=request.warmup_steps,start_factor=request.scheduler_start_factor,end_factor=request.scheduler_end_factor,total_iters=request.scheduler_total_iters),model_initializer=initialize,dataloader_resume_contract=contract,post_optimizer_step=export)
