"""Independent supervised evaluator; labels never enter a world-model loss."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import lightning as pl
from torchvision.models import resnet18
from lightning.pytorch.callbacks import ModelCheckpoint

class PoseFrames(Dataset):
    def __init__(self,root,split):
        self.root=Path(root);lengths=np.load(self.root/"lengths.npy");offsets=np.load(self.root/"offsets.npy");test=set(np.load(self.root/"test_episodes.npy").tolist())
        self.rows=np.concatenate([np.arange(o,o+n) for ep,(o,n) in enumerate(zip(offsets,lengths)) if ((ep in test)==(split=="test"))]);self.pixels=None
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        if self.pixels is None:self.pixels=np.load(self.root/"pixels64.npy",mmap_mode="r");self.state=np.load(self.root/"states.npy",mmap_mode="r")
        row=int(self.rows[i]);state=np.array(self.state[row]);target=np.r_[state[:4]/512,np.sin(state[4]),np.cos(state[4])].astype(np.float32)
        return torch.from_numpy(np.array(self.pixels[row])).permute(2,0,1),torch.from_numpy(target)

class PoseModule(pl.LightningModule):
    def __init__(self,output,max_steps=10000,dot_aug=False,learning_rate=1e-3):
        super().__init__();self.output=Path(output);self.max_steps=max_steps;self.dot_aug=dot_aug;self.learning_rate=learning_rate;self.net=resnet18(weights=None,num_classes=6);self.net.conv1=nn.Conv2d(3,64,3,1,1,bias=False);self.net.maxpool=nn.Identity()
    def forward(self,x):return self.net(x)
    def training_step(self,batch,index):
        image,target=batch;x=image.float()/127.5-1
        # Evaluator-only nuisance augmentation; never used to train AS-JEPA.
        factors=.35+.65*torch.rand(len(x),3,1,1,device=x.device);bias=.25*torch.rand(len(x),3,1,1,device=x.device)
        if self.global_step%2==0:x=((x+1)*factors+bias-1).clamp(-1,1)
        if self.dot_aug and self.global_step%4 in (1,2):
            # Match the random-dot nuisance distribution without using pose labels.
            values=torch.rand(len(x),5,device=x.device)
            side=x.shape[-1];height=x.shape[-2]
            yy=torch.arange(height,device=x.device)[None,:,None]
            xx=torch.arange(side,device=x.device)[None,None,:]
            mask=((xx-values[:,0,None,None]*side)**2+(yy-values[:,1,None,None]*height)**2)<(.04*side)**2
            x=torch.where(mask[:,None],(values[:,2:,None,None]*2-1),x)
        x=x+.025*torch.randn_like(x)
        prediction=self(x);loss=(prediction[:,:4]-target[:,:4]).square().mean()+.2*(prediction[:,4:]-target[:,4:]).square().mean()
        if self.global_step%100==0:
            (self.output/"status.json").write_text(json.dumps(dict(step=self.global_step,loss=float(loss.detach()),position_error=float((prediction[:,:4]-target[:,:4]).abs().mean().detach()*512),updated_at=time.time())))
        return loss
    def validation_step(self,batch,index):
        image,target=batch;prediction=self(image.float()/127.5-1);self.log("val_pose",(prediction-target).square().mean(),on_epoch=True)
    def configure_optimizers(self):
        opt=torch.optim.AdamW(self.parameters(),lr=self.learning_rate,weight_decay=1e-4)
        return dict(optimizer=opt,lr_scheduler=dict(scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,self.max_steps),interval="step"))


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--output",required=True);p.add_argument("--steps",type=int,default=10000);p.add_argument("--batch",type=int,default=512);p.add_argument("--dot-aug",action="store_true");p.add_argument("--initial-weights");p.add_argument("--learning-rate",type=float,default=1e-3);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);pl.seed_everything(123,workers=True);torch.set_float32_matmul_precision("high")
    (out/"config.json").write_text(json.dumps(vars(a),indent=2))
    model=PoseModule(out,a.steps,dot_aug=a.dot_aug,learning_rate=a.learning_rate)
    if a.initial_weights:model.net.load_state_dict(torch.load(a.initial_weights,map_location="cpu",weights_only=True),strict=True)
    checkpoint=ModelCheckpoint(dirpath=out,every_n_train_steps=2000,save_last=True,save_top_k=1,monitor="step",mode="max",filename="pose-{step:07d}")
    trainer=pl.Trainer(accelerator="gpu",devices=1,precision="bf16-mixed",max_steps=a.steps,max_epochs=-1,logger=False,enable_progress_bar=False,enable_model_summary=False,callbacks=[checkpoint],default_root_dir=out,limit_val_batches=20,num_sanity_val_steps=0,gradient_clip_val=1.)
    train=DataLoader(PoseFrames(a.data,"train"),batch_size=a.batch,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)
    val=DataLoader(PoseFrames(a.data,"test"),batch_size=a.batch,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
    trainer.fit(model,train_dataloaders=train,val_dataloaders=val)
    torch.save(model.net.state_dict(),out/"weights.pt")
    errors=[]
    model=model.cuda().eval()
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        for image,target in val:
            prediction=model(image.cuda().float()/127.5-1).float().cpu();errors.append(prediction-target)
    diff=torch.cat(errors);position=(diff[:,:4].reshape(-1,2,2)*512).norm(dim=-1)
    (out/"floor.json").write_text(json.dumps(dict(agent_position_error=float(position[:,0].mean()),block_position_error=float(position[:,1].mean()),frames=len(diff),scope="held-out standard PushT real-frame evaluator floor"),indent=2))

if __name__=="__main__":main()
