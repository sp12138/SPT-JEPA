"""Forward-only signed actuation fields;future observation targets,no inverse head."""
import itertools

import torch
from torch import nn

from as_jepa_lewm.predictive_density import PredictiveDensity,quadratic_box


@torch.no_grad()
def signed_projection(drift,effect,target,variance,center=None):
    """Exact weighted projection for b+L*u+R*abs(u),u in[-1,1]^4.

    Each of16 sign chambers is a convex four-dimensional box problem.
    All chambers use the same objective;ties retain the agent's own past control.
    """
    basis=effect.flatten(2).transpose(1,2).double()
    weight=variance.flatten(1).double().reciprocal();rhs=(target-drift).flatten(1).double()
    gram=torch.einsum('npi,np,npj->nij',basis,weight,basis)
    linear=torch.einsum('npi,np,np->ni',basis,weight,rhs)
    signs=gram.new_tensor(list(itertools.product([-1.,1.],repeat=4)))
    transform=torch.cat([torch.diag_embed(signs),torch.eye(4,device=gram.device,dtype=gram.dtype)[None].expand(16,-1,-1)],1)
    H=torch.einsum('sai,nab,sbj->nsij',transform,gram,transform)
    g=torch.einsum('sai,na->nsi',transform,linear)
    previous=gram.new_zeros(len(gram),4)if center is None else center.double()
    w_center=2*previous[:,None]*signs[None]-1
    w=quadratic_box((.25*H).flatten(0,1),(.5*g-.25*H.sum(-1)).flatten(0,1),w_center.flatten(0,1)).double().reshape(len(gram),16,4)
    commands=signs[None]*(w+1)*.5;lifted=torch.cat([commands,commands.abs()],-1)
    cost=torch.einsum('nsi,nij,nsj->ns',lifted,gram,lifted)-2*torch.einsum('nsi,ni->ns',lifted,linear)
    minimum=cost.min(-1).values;tolerance=torch.finfo(torch.float64).eps*128*(1+minimum.abs())
    distance=(commands-previous[:,None]).square().sum(-1).masked_fill(cost>minimum[:,None]+tolerance[:,None],float('inf'))
    return commands[torch.arange(len(gram),device=gram.device),distance.argmin(-1)].float()


class SignedDensity(PredictiveDensity):
    def __init__(self,**kwargs):
        kwargs.update(control_steps=5,passive='stencil',variance='point',actuation_history=True)
        super().__init__(**kwargs)
        width,channels=self.mean_head.in_channels,self.mean_head.out_channels
        self.absolute_head=nn.Conv2d(width,channels*4*5,1)
        self.absolute_transient_head=nn.Conv2d(width,channels*4*5,1)
        for head in[self.absolute_head,self.absolute_transient_head]:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def fields(self,previous,current,goal,control,role,history_valid,previous_control=None,sequence=False):
        past=torch.zeros_like(control)if previous_control is None else previous_control
        boundary=(role==1).float()[:,None,None,None]
        x=self.predictor(torch.cat([previous,current,(goal-current)*boundary],1))
        context=torch.cat([past,role[:,None],history_valid[:,None]],-1)
        gamma,beta=self.context(context.float()).chunk(2,-1)
        x=x*(1+.1*gamma[:,:,None,None])+beta[:,:,None,None]
        channels=current.shape[1]
        def field(head):return head(x).float().unflatten(1,(5,4,channels))
        J,K,R,T=[field(head)for head in[self.effect_head,self.transient_head,self.absolute_head,self.absolute_transient_head]]
        drift=-torch.einsum('na,nachw->nchw',past,K[:,0])-torch.einsum('na,nachw->nchw',past.abs(),T[:,0])
        mean=torch.where((role==0)[:,None,None,None],drift,self.mean_head(x).float())
        def telescope(value):return value-torch.cat([value[:,1:],torch.zeros_like(value[:,:1])],1)
        effect=torch.cat([J+telescope(K),R+telescope(T)],2)
        if not sequence:effect=effect.sum(1)
        return mean,effect,torch.ones_like(mean)

    def loss(self,batch,step=0,ramp_steps=0):
        transformed=dict(batch);known=batch['executed_sequence']
        transformed['executed_sequence']=torch.cat([known,known.abs()],-1)
        return super().loss(transformed,step,ramp_steps)

    @torch.no_grad()
    def box_projection(self,drift,effect,target,variance,center=None):
        return signed_projection(drift,effect,target,variance,center)

    @torch.no_grad()
    def command(self,previous,current,goal,valid,objective,previous_control=None):
        assert objective in['projection','marginal']
        zero=current.new_zeros(len(current),4);past=zero if previous_control is None else previous_control
        flag=valid.new_zeros(len(valid))
        drift,effect,var=self.fields(previous,current,torch.zeros_like(goal),zero,flag,valid,past)
        desired,_,goalvar=self.fields(previous,current,goal,zero,flag-1 if objective=='marginal'else flag+1,valid,past)
        action=self.box_projection(drift,effect,desired,var+goalvar,past)
        future=current+drift+torch.einsum('na,nachw->nchw',torch.cat([action,action.abs()],-1),effect)
        return action,future

    def audit_expected_gradients(self):
        out=super().audit_expected_gradients();out.update(absolute_head='active',absolute_transient_head='active');return out
