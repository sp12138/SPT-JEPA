"""Frozen-model planning costs; original GoalMSE remains available as the control."""
import os,time
import numpy as np
import torch
from torch import nn

class GeometryGoal(nn.Module):
    def __init__(self,covariance,kind="covariance",alpha=.1,rank=8,epsilon=.1):
        super().__init__();self.register_buffer("covariance",torch.as_tensor(covariance,dtype=torch.float32))
        self.kind=kind;self.alpha=alpha;self.rank=rank;self.epsilon=epsilon;self.signature=None;self.precision=None;self.extra_rollouts=0;self.prepare_seconds=0.;self.prepare_calls=0
    @torch.no_grad()
    def prepare(self,model,info,actions):
        pixels=info["pixels"][:,0]
        # Image moments are invariant to object translation and cannot identify context.
        signature=[pixels.detach()]
        if "action_history" in info:signature.append(info["action_history"][:,0].detach())
        if self.signature is not None and len(signature)==len(self.signature) and all(torch.equal(a,b) for a,b in zip(signature,self.signature)):return
        self.signature=[value.clone() for value in signature];self.prepare_calls+=1
        started=time.perf_counter();b=actions.shape[0];dim=actions.shape[-2]*actions.shape[-1];latent=self.covariance.shape[0]
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
    def statistics(self):
        return dict(kind=self.kind,alpha=self.alpha,rank=self.rank,epsilon=self.epsilon,
                    prepare_calls=self.prepare_calls,extra_rollouts=self.extra_rollouts,
                    prepare_seconds=self.prepare_seconds,cache="exact pixels and action history equality",
                    numerical_eigenvalue_floor=1e-10)
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
    import atexit,json
    from pathlib import Path
    objectives=[]
    def save_statistics():
        destination=os.environ.get("ASJEPA_PLANNER_OUTPUT")
        if destination:
            (Path(destination)/"geometry-statistics.json").write_text(json.dumps([x.statistics() for x in objectives],indent=2))
    atexit.register(save_statistics)
    def init(self,model,objective,*args,**kwargs):
        objective=GeometryGoal(covariance,kind,float(os.environ.get("ASJEPA_GEOMETRY_ALPHA",".1")))
        objectives.append(objective)
        original_init(self,model,objective,*args,**kwargs)
    def get(self,info,actions):
        self.objective.prepare(self.model,info,actions)
        return original_get(self,info,actions)
    ShootingCostEvaluator.__init__=init;ShootingCostEvaluator.get_cost=get
