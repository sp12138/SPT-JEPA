"""Insight campaign entry point. Lightning owns training and checkpointing."""
import argparse, datetime, json, math, os, random, signal, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Sampler
import lightning as pl
from lightning.pytorch.callbacks import Callback
from as_jepa_tools.data import PushTWindows,FullPushTWindows,seed_worker
from as_jepa_lewm.model import ASJepaLeWM
from vforge.ops import OpsConfig,initialize_ops


class WindowSampler(Sampler):
    """Stateless permutations, with a precise, batch-aligned resume offset."""
    def __init__(self,dataset,start=0,seed=0,batch_size=128):
        self.dataset=dataset;self.start=start;self.seed=seed;self.length=(dataset.natural_length//batch_size)*batch_size
    def __len__(self):return self.length
    def __iter__(self):
        n=self.dataset.natural_length;epoch=-1;start=self.start
        for k in range(start,start+self.length):
            current=k//n
            if current!=epoch:
                epoch=current;generator=np.random.default_rng(self.seed+epoch*1000003)
                a=int(generator.integers(1,n));b=int(generator.integers(0,n))
                while math.gcd(a,n)!=1:a=(a+1)%n or 1
            yield (a*(k%n)+b)%n


class TrainModule(pl.LightningModule):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg;self.model=(__import__("as_jepa_lewm.research",fromlist=["ResearchLeWM"]).ResearchLeWM if cfg.get("research_variant",False) else ASJepaLeWM)(**cfg["model"]);self.last_metrics={};self.initial_step=0;self.samples_consumed=0
        self.save_hyperparameters(cfg)
    def training_step(self,batch,batch_idx):
        if self.cfg.get("fork_data"):
            if not hasattr(self,"_fork_bank"):
                root=Path(self.cfg["fork_data"])
                self._fork_bank={k:np.load(root/(name+".npy"),mmap_mode="r") for k,name in [("context","contexts"),("futures","futures"),("actions","actions"),("indices","train_indices")]}
            bank=self._fork_bank;rng=np.random.default_rng(self.cfg["seed"]+int(self.global_step)*1009);ids=rng.choice(bank["indices"],size=16,replace=False)
            batch["fork_context"]=torch.from_numpy(np.array(bank["context"][ids])).permute(0,3,1,2).to(self.device)
            batch["fork_futures"]=torch.from_numpy(np.array(bank["futures"][ids])).permute(0,1,4,2,3).to(self.device)
            batch["fork_actions"]=torch.from_numpy(np.array(bank["actions"][ids])).to(self.device)
        loss,metrics=self.model.loss(batch,self.global_step,self.cfg.get("ramp_steps",0))
        if not torch.isfinite(loss):raise FloatingPointError(f"nonfinite loss at {self.global_step}")
        self.last_metrics={k:float(v.detach()) for k,v in metrics.items()}
        if self.cfg.get("research_variant") and self.cfg["model"].get("method")=="gradient_constraint":
            named=[(name,p) for name,p in self.model.named_parameters() if name.startswith(("encoder.","pool.")) and p.requires_grad]
            fm,rate=self.model._loss_terms
            gf=torch.autograd.grad(fm,[p for _,p in named],retain_graph=True,allow_unused=True)
            gr=torch.autograd.grad(rate,[p for _,p in named],retain_graph=True,allow_unused=True)
            dot=sum((a.float()*b.float()).sum() for a,b in zip(gf,gr) if a is not None and b is not None)
            norm=sum(a.float().square().sum() for a in gf if a is not None)
            coefficient=torch.minimum(torch.as_tensor(0.,device=loss.device),dot)/torch.as_tensor(norm,device=loss.device).clamp_min(1e-20)
            no_signal=float(torch.as_tensor(norm).detach())<1e-20
            self._constrained=[(p,None if a is None else (a if no_signal else a+(b if b is not None else 0)-coefficient*a).detach()) for (_,p),a,b in zip(named,gf,gr)]
            self.last_metrics["constraint_no_signal"]=float(no_signal)
            self.last_metrics["constraint_projection"]=float(coefficient.detach())
        if hasattr(self.model,"rate_dimensions") and self.global_step%2000==0:
            with (Path(self.cfg["output"])/"dimension_rates.jsonl").open("a") as f:f.write(json.dumps(dict(step=int(self.global_step),rate=self.model.rate_dimensions))+"\n")
        self.last_metrics["encoder_frozen"]=float(self.global_step<self.cfg.get("freeze_encoder_steps",0))
        if self.cfg.get("trace_batches",False):
            with (Path(self.cfg["output"])/"batch_trace.jsonl").open("a") as trace:
                trace.write(json.dumps(dict(step=int(self.global_step),episodes=batch["episode"].detach().cpu().tolist(),starts=batch["start"].detach().cpu().tolist()))+"\n")
        return loss
    def on_before_optimizer_step(self,optimizer):
        if hasattr(self,"_constrained"):
            for p,g in self._constrained:p.grad=g
            del self._constrained
        if self.global_step%100==0:
            for prefix in ["encoder","predictor","renderer"]:
                total=sum(p.grad.float().square().sum() for name,p in self.model.named_parameters() if name.startswith(prefix+".") and p.grad is not None)
                self.last_metrics["grad_norm_"+prefix]=float(torch.as_tensor(total).sqrt())
        if self.global_step>=self.cfg.get("freeze_encoder_steps",0):return
        # Diagnostic curriculum only: preserve a known informative initial code
        # while the existing prior/renderer optimizers learn to use it.
        for name,param in self.model.named_parameters():
            if name.startswith(("encoder.","pool.")):param.grad=None
    def configure_optimizers(self):
        renderer=[];other=[]
        for name,param in self.model.named_parameters():
            if param.requires_grad:(renderer if name.startswith("renderer.") else other).append(param)
        groups=[dict(params=other,lr=5e-5,weight_decay=1e-3,betas=(.9,.999))]
        if renderer:groups.append(dict(params=renderer,lr=1e-4,weight_decay=0.,betas=(.9,.95)))
        optimizer=torch.optim.AdamW(groups,foreach=True)
        total=self.cfg["max_steps"];warmup=max(1,int(.01*total))
        def schedule(step):
            if step<warmup:return max(1,step)/warmup
            return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,total-warmup))))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
        return dict(optimizer=optimizer,lr_scheduler=dict(scheduler=scheduler,interval="step"))
    def on_fit_start(self):
        if self.cfg.get("calibrate_state_std") is None or self.cfg.get("is_resuming",False):return
        # One-time affine head initialization. No per-batch normalization is added to AS-JEPA.
        dataset=(FullPushTWindows if self.cfg.get("all_phase_data",False) else PushTWindows)(self.cfg["data"],window=self.cfg["model"]["window"],gap=self.cfg.get("gap",8),seed=self.cfg["seed"])
        generator=torch.Generator().manual_seed(91);indices=torch.randperm(len(dataset),generator=generator)[:128].tolist()
        from torch.utils.data import Subset
        loader=DataLoader(Subset(dataset,indices),batch_size=16,num_workers=0)
        was_training=self.model.training;self.model.eval();values=[]
        with torch.no_grad():
            for batch in loader:
                batch={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in batch.items()}
                pixels,_,_=self.model.prepare(batch);values.append(self.model.encode(dict(pixels=pixels))["emb"].float().flatten(0,1))
            values=torch.cat(values);mean=values.mean(0).unflatten(-1,(self.model.state_slots,self.model.state_width)).mean(0);current=values.std(0).mean().clamp_min(1e-8);scale=float(self.cfg["calibrate_state_std"])/float(current)
            head=self.model.pool.head;head.weight.mul_(scale);head.bias.sub_(mean).mul_(scale)
        self.model.train(was_training)
        if self.cfg.get("export_initial",False):
            initial=Path(self.cfg["output"])/"initial-model";initial.mkdir(exist_ok=True)
            torch.save(self.model.state_dict(),initial/"weights.pt")
            (initial/"config.json").write_text(json.dumps(dict(_target_="as_jepa_lewm.research.ResearchLeWM" if self.cfg.get("research_variant") else "as_jepa_lewm.model.ASJepaLeWM",**self.cfg["model"]),indent=2))
        (Path(self.cfg["output"])/"initial_scale.json").write_text(json.dumps(dict(target_std=self.cfg["calibrate_state_std"],original_std=float(current),scale=scale,mean_before=values.mean(0).cpu().tolist(),note="One fixed affine head initialization from 128 train windows; no target labels, batch normalization, stop-gradient or objective change."),indent=2))
    def on_save_checkpoint(self,checkpoint):
        checkpoint["campaign_rng"]=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
        checkpoint["campaign_samples"]=self.samples_consumed
    def on_load_checkpoint(self,checkpoint):
        rng=checkpoint.get("campaign_rng")
        if rng:
            random.setstate(rng["python"]);np.random.set_state(rng["numpy"]);torch.set_rng_state(rng["torch"]);torch.cuda.set_rng_state_all(rng["cuda"])
        self.initial_step=int(checkpoint.get("global_step",0));self.samples_consumed=int(checkpoint.get("campaign_samples",0))


class PersistentProgress(Callback):
    def __init__(self,root,cfg):self.root=Path(root);self.cfg=cfg;self.started=time.time();self.last_time=self.started;self.last_step=0
    def on_train_start(self,trainer,module):self.last_time=time.time();self.last_step=trainer.global_step
    def status(self,trainer,module,state):
        now=time.time();step=int(trainer.global_step)
        row=dict(state=state,step=step,elapsed_seconds=now-self.started,updated_at=now,pid=os.getpid(),host=os.uname().nodename,**module.last_metrics)
        row["seconds_per_step"]=(now-self.last_time)/max(1,step-self.last_step)
        row["peak_gib"]=torch.cuda.max_memory_allocated()/2**30 if self.cfg.get("device")!="cpu" else 0.
        row["learning_rates"]=[group["lr"] for group in (trainer.optimizers[0].param_groups if trainer.optimizers else [])]
        path=self.root/"status.json";tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(row,indent=2));tmp.replace(path)
        with (self.root/"metrics.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
        self.last_time=now;self.last_step=step
    def save(self,trainer,module):
        path=self.root/"last.ckpt";tmp=self.root/"last.ckpt.tmp"
        trainer.save_checkpoint(tmp);tmp.replace(path)
        checkpoint=self.root/f"model-step{trainer.global_step:07d}";checkpoint.mkdir(exist_ok=True)
        torch.save(module.model.state_dict(),checkpoint/"weights.pt")
        (checkpoint/"config.json").write_text(json.dumps(dict(_target_="as_jepa_lewm.research.ResearchLeWM" if self.cfg.get("research_variant") else "as_jepa_lewm.model.ASJepaLeWM",**self.cfg["model"]),indent=2))
        # Keep just the last two portable model exports, plus the resumable checkpoint.
        exports=sorted(self.root.glob("model-step*"))
        import shutil
        for older in exports[:-2]:
            if int(older.name.split("step")[1]) not in self.cfg.get("retain_steps",[]):shutil.rmtree(older)
    def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
        step=int(trainer.global_step)
        module.samples_consumed+=self.cfg["batch_size"]
        if step%100==0:self.status(trainer,module,"running")
        if step and step%self.cfg.get("checkpoint_every",2000)==0:self.save(trainer,module)
        if (self.root/"STOP").exists():trainer.should_stop=True
        if self.cfg.get("stop_after")==step and module.initial_step==0:trainer.should_stop=True
    def on_train_epoch_end(self,trainer,module):
        trainer.train_dataloader.sampler.start=module.samples_consumed
    def on_train_end(self,trainer,module):self.save(trainer,module);self.status(trainer,module,"finished")
    def on_exception(self,trainer,module,exception):
        (self.root/"error.txt").write_text(repr(exception));self.status(trainer,module,"failed")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("config");parser.add_argument("--resume",action="store_true");args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());cfg["is_resuming"]=args.resume;root=Path(cfg["output"]);root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(cfg.get("cpu_threads",2));pl.seed_everything(cfg["seed"],workers=True)
    torch.backends.cuda.matmul.allow_tf32=True;torch.set_float32_matmul_precision(cfg.get("float32_matmul_precision","high"))
    use_cpu=cfg.get("device")=="cpu"
    initialize_ops(OpsConfig(backend="sdpa_math" if use_cpu else "sdpa",allow_fallback=False),device=torch.device("cpu") if use_cpu else torch.device("cuda",0),parallel_state=None)
    module=TrainModule(cfg)
    if cfg.get("initial_encoder_checkpoint") and not args.resume:
        import hashlib
        source=Path(cfg["initial_encoder_checkpoint"])
        weights=torch.load(source/"weights.pt",map_location="cpu",weights_only=True)
        encoder={k.removeprefix("encoder."):v for k,v in weights.items() if k.startswith("encoder.")}
        module.model.encoder.load_state_dict(encoder,strict=True)
        digest=hashlib.sha256()
        with (source/"weights.pt").open("rb") as handle:
            for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
        (root/"initial_encoder.json").write_text(json.dumps(dict(source=str(source),sha256=digest.hexdigest(),scope="Encoder-only warm initialization from our L0; pool, prior, renderer and optimizer are fresh"),indent=2))
        del weights,encoder
    (root/"model_size.json").write_text(json.dumps(dict(total=sum(p.numel() for p in module.model.parameters()),encoder=sum(p.numel() for p in module.model.encoder.parameters()),prior=sum(p.numel() for p in module.model.predictor.parameters()),renderer=sum(p.numel() for p in module.model.renderer.parameters()) if module.model.renderer else 0),indent=2))
    checkpoint=root/"last.ckpt";resume=str(checkpoint) if args.resume and checkpoint.exists() else None
    start=int(torch.load(checkpoint,map_location="cpu",weights_only=False).get("campaign_samples",0)) if resume else 0
    dataset=(FullPushTWindows if cfg.get("all_phase_data",False) else PushTWindows)(cfg["data"],window=cfg["model"]["window"],gap=cfg.get("gap",8),seed=cfg["seed"])
    sampler=WindowSampler(dataset,start=start,seed=cfg["seed"],batch_size=cfg["batch_size"])
    loader=DataLoader(dataset,batch_size=cfg["batch_size"],sampler=sampler,generator=torch.Generator().manual_seed(cfg["seed"]+999),num_workers=cfg.get("workers",4),persistent_workers=cfg.get("workers",4)>0,pin_memory=True,drop_last=True,worker_init_fn=seed_worker)
    progress=PersistentProgress(root,cfg)
    trainer=pl.Trainer(accelerator="cpu" if use_cpu else "gpu",devices=1,precision="32-true" if use_cpu else cfg.get("precision","bf16-mixed"),max_steps=cfg["max_steps"],max_epochs=-1,
                       max_time=datetime.timedelta(seconds=cfg.get("max_seconds",36000)),gradient_clip_val=1.,logger=False,enable_checkpointing=False,
                       enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,callbacks=[progress],log_every_n_steps=100,default_root_dir=str(root))
    trainer.fit(module,train_dataloaders=loader,ckpt_path=resume,weights_only=False)

if __name__=="__main__":main()
