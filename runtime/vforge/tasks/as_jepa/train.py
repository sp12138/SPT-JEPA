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
from as_jepa_tools.data import PushTWindows


def build_train_components(request):
    cfg=dict(request.extras);data=Path(cfg.pop("data_root"));model_cfg=dict(cfg.pop("as_jepa"));kind=model_cfg.pop("kind","S2");legacy_output=cfg.pop("campaign_output",None);output=Path(request.output_dir);output.mkdir(parents=True,exist_ok=True)
    dataset=PushTWindows(data,window=model_cfg.get("window",8),gap=model_cfg.get("anchor_gap",8) if kind=="S2" else 0,seed=request.sampler_seed)
    loader=build_resumable_loader(dataset,parallel_state=get_parallel_state(),batch_size=request.batch_size,num_workers=request.num_workers,drop_last=request.drop_last,seed=request.sampler_seed,shuffle=request.shuffle,pin_memory=request.pin_memory,persistent_workers=request.persistent_workers,prefetch_factor=request.prefetch_factor)
    def build():
        with torch.device(request.init_device):
            if kind=="S2":return ASJepaSPT(**model_cfg)
            from vforge.models.as_jepa.spt_baseline import ActionRepWorld
            return ActionRepWorld(**model_cfg)
    def step_fn(model,batch,step,context):
        frames=batch["render"].to(context.device,dtype=torch.float32)/127.5-1.;anchor=batch["anchor"].to(context.device,dtype=torch.float32)/127.5-1.;actions=batch["action"].to(context.device)
        result=model(frames=frames,actions=actions,anchor=anchor,return_metrics=True)
        return StepOutput(loss=result["loss"],metrics=result["metrics"],workload={"local_batch":len(frames),"local_tokens":len(frames)*(frames.shape[1]-1)*256},metadata={"sample_ids":[f"{int(ep)}:{int(start)}" for ep,start in zip(batch["episode"],batch["start"])]})
    def export(model,step,context,output_record):
        completed=step+1
        if completed%2000:return {}
        folder=output/f"model-step{completed:07d}";folder.mkdir(exist_ok=True)
        weights={k:(v.full_tensor() if hasattr(v,"full_tensor") else v).detach().cpu() for k,v in model.state_dict().items()}
        torch.save(weights,folder/"weights.pt");(folder/"config.json").write_text(json.dumps(dict(_target_="vforge.models.as_jepa.model.ASJepaSPT" if kind=="S2" else "vforge.models.as_jepa.spt_baseline.ActionRepWorld",**model_cfg),indent=2))
        return {}
    contract=dict(dataset="as_jepa_pusht_phase0_v1",data_card_sha256=hashlib.sha256((data/"dataset_card.json").read_bytes()).hexdigest(),window=dataset.window,gap=dataset.gap)
    return TrainComponents(model=ModelSpec(build),dataloader=loader,optimizer_factory=build_optimizer_factory(learning_rate=request.learning_rate,weight_decay=request.weight_decay,betas=request.betas,eps=request.eps),step_fn=step_fn,scheduler_factory=build_scheduler_factory(name=request.scheduler_name,factor=request.scheduler_factor,warmup_steps=request.warmup_steps,start_factor=request.scheduler_start_factor,end_factor=request.scheduler_end_factor,total_iters=request.scheduler_total_iters),model_initializer=lambda model:model.initialize_weights(),dataloader_resume_contract=contract,post_optimizer_step=export)
