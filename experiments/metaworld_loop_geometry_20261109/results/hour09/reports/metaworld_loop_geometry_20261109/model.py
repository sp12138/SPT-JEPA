"""Observed-transition worlds, with optional reverse-time physical inference.

Reverse examples receive the SAME known, executed controls, in reversed time
order, as INPUTS. They predict past RGB, never controls or an expert path.
No desired goal enters learning. Boundary matching is inference only.
"""
import torch
from torch import nn
from torch.nn import functional as F


class BoundaryWorld(nn.Module):
    def __init__(self,parent,mode):
        super().__init__();assert mode in ['residual','warp','bridge'];self.parent=parent;self.mode=mode
        parent.inverse.requires_grad_(False);parent.rgb.requires_grad_(False)
        self.direction=nn.Embedding(2,96);nn.init.zeros_(self.direction.weight)
        self.readout=nn.Conv2d(32,5,1);nn.init.normal_(self.readout.weight,std=.001);nn.init.zeros_(self.readout.bias)
        c=(torch.arange(128).float()+.5)*(2/128)-1;y,x=torch.meshgrid(c,c,indexing='ij')
        self.register_buffer('grid',torch.stack([x,y],-1)[None],persistent=False)

    def advance(self,hidden,action,direction):
        context=self.parent.control(action.float())+self.direction(direction.long())
        return self.parent.transition(hidden,context[:,:,None,None].expand(-1,-1,16,16))

    def prepare(self,batch,direction):
        images=self.parent.pixels(batch['history_pixels'].flatten(0,1)).unflatten(0,batch['history_pixels'].shape[:2])
        hidden=self.parent.initial(len(images),images.device)
        for t in range(images.shape[1]):
            if t:
                prior=self.advance(hidden,batch['history_actions'][:,t-1],direction)
                hidden=torch.where(batch['history_valid'][:,t-1,None,None,None],prior,hidden)
            posterior=self.parent.observe(hidden,images[:,t])
            hidden=torch.where(batch['history_valid'][:,t,None,None,None],posterior,hidden)
        return hidden,images[:,-1]

    def render(self,hidden,anchor):
        fields=F.interpolate(self.readout(self.parent.decoder(hidden)).float(),(128,128),mode='bilinear',align_corners=False)
        if self.mode=='residual':return (anchor+fields[:,:3]).clamp(0,1)
        grid=self.grid+fields[:,3:].permute(0,2,3,1)*(2/128)
        return F.grid_sample(anchor,grid,mode='bilinear',padding_mode='border',align_corners=False)

    def objective(self,batch,terms):
        assert not {'goal_pixels','boundary','desired_goal','reward','state','task_id','expert_actions'}&set(batch)
        durations,directions=terms
        direction=directions.long() if self.mode=='bridge' else torch.zeros_like(directions,dtype=torch.long)
        reverse=direction.bool()
        def select(name):
            value=batch[name];other=batch['reverse_'+name]
            return torch.where(reverse.reshape(-1,*([1]*(value.ndim-1))),other,value)
        context={k:select(k) for k in ['history_pixels','history_actions','history_valid']}
        controls=select('actions');target=select('endpoint_pixels');hidden,anchor=self.prepare(context,direction)
        last=hidden
        for t in range(int(durations.max())):
            hidden=self.advance(hidden,controls[:,t],direction)
            last=torch.where((durations==t+1)[:,None,None,None],hidden,last)
        prediction=self.render(last,anchor);target=self.parent.pixels(target)
        per=(prediction-target).square().mean((1,2,3));loss=per.mean()
        return loss,dict(loss=loss,RGB_MSE=loss,duration=durations.float().mean(),reverse_fraction=reverse.float().mean())

    def forward(self,batch,terms):return self.objective(batch,terms)


def random_terms(batch):
    valid=batch['future_valid'].sum(1)
    durations=(torch.rand(len(valid),device=valid.device)*valid).long()+1
    directions=(torch.rand(len(valid),device=valid.device)<.5).long()
    return durations,directions
