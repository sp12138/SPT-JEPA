"""Frozen-model planning costs; original GoalMSE remains available as the control."""
import os,time
import numpy as np
import torch
from torch import nn

class GeometryGoal(nn.Module):
    def __init__(self,covariance,kind="covariance",alpha=.1,rank=8,epsilon=.1):
        super().__init__();self.register_buffer("covariance",torch.as_tensor(covariance,dtype=torch.float32))
        self.kind=kind;self.alpha=alpha;self.rank=rank;self.epsilon=epsilon;self.signature=None;self.precision=None;self.extra_rollouts=0;self.prepare_seconds=0.
    @torch.no_grad()
    def prepare(self,model,info,actions):
        pixels=info["pixels"][:,0]
        signature=[pixels.float().sum(tuple(range(1,pixels.ndim))),pixels.float().square().sum(tuple(range(1,pixels.ndim)))]
        if "action_history" in info:signature.append(info["action_history"][:,0].float().flatten(1).sum(1))
        signature=torch.stack(signature).cpu()
        if self.signature is not None and torch.equal(signature,self.signature):return
        self.signature=signature;started=time.perf_counter();b=actions.shape[0];dim=actions.shape[-2]*actions.shape[-1];latent=self.covariance.shape[0]
        covariance=self.covariance.to(actions.device)
        gramian=self.alpha*covariance[None].expand(b,-1,-1)
        if self.kind=="gramian":
            r=min(self.rank,dim);generator=torch.Generator(device=actions.device).manual_seed(20261028)
            directions=torch.linalg.qr(torch.randn(dim,r,device=actions.device,generator=generator)).Q
            center=actions.float().mean(1)
            perturb=self.epsilon*directions.T.reshape(r,*actions.shape[-2:])
            candidates=torch.cat([center[:,None]+perturb[None],center[:,None]-perturb[None]],1)
            work={}
            for key,value in info.items():
                if torch.is_tensor(value) and value.ndim>=2 and value.shape[0]==b:
                    work[key]=value[:,:1].expand(-1,2*r,*value.shape[2:]).clone()
            work.pop("predicted_emb",None);work.pop("emb",None)
            forecast=model.rollout(work,candidates)["predicted_emb"][:,:,-1].float()
            differential=(forecast[:,:r]-forecast[:,r:])/(2*self.epsilon)
            gramian=gramian+torch.einsum("brd,bre->bde",differential,differential)*(dim/r)
            self.extra_rollouts+=b*2*r
        eigen,vectors=torch.linalg.eigh(gramian.double())
        # Positive training covariance is required; a numerical eigenvalue floor is recorded.
        self.precision=(vectors*eigen.clamp_min(1e-10).reciprocal()[:,None])@vectors.transpose(-1,-2)
        self.prepare_seconds+=time.perf_counter()-started
    def forward(self,info):
        difference=info["predicted_emb"][:,:,-1].float()-info["goal_emb"][:,-1,None].float()
        return torch.einsum("bsd,bde,bse->bs",difference.double(),self.precision,difference.double()).float()

def install():
    import stable_worldmodel as swm
    from stable_worldmodel.planning.evaluator import ShootingCostEvaluator
    covariance=np.load(os.environ["ASJEPA_GEOMETRY_COVARIANCE"])
    kind=os.environ.get("ASJEPA_GEOMETRY_COST","covariance")
    if kind not in ["covariance","gramian"]:raise ValueError(kind)
    original_init=ShootingCostEvaluator.__init__;original_get=ShootingCostEvaluator.get_cost
    def init(self,model,objective,*args,**kwargs):
        original_init(self,model,GeometryGoal(covariance,kind,float(os.environ.get("ASJEPA_GEOMETRY_ALPHA",".1"))),*args,**kwargs)
    def get(self,info,actions):
        self.objective.prepare(self.model,info,actions)
        return original_get(self,info,actions)
    ShootingCostEvaluator.__init__=init;ShootingCostEvaluator.get_cost=get
