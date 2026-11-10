"""Per-episode Bayesian correction from naturally observed transitions."""
import torch


class ResidualPosterior:
    """Affine residual, unit Gaussian coefficient prior, known channel noise.

    Each factor targets the observed next RGB code minus the frozen model's
    forecast for the issued current command. No action or goal is a target.
    """
    def __init__(self, input_width, output_width, sigma, device):
        self.precision=torch.eye(input_width+1,dtype=torch.float64)*(sigma**2)
        self.cross=torch.zeros(input_width+1,output_width,dtype=torch.float64)
        self.mean=torch.zeros(input_width+1,output_width,device=device)
        self.updates=0

    def predict(self, action):
        design=torch.cat([torch.ones_like(action[...,:1]),action],-1)
        return design@self.mean

    def update(self, action, residual):
        design=torch.cat([torch.ones(1,dtype=torch.float64),action.detach().flatten().double().cpu()])
        target=residual.detach().flatten().double().cpu()
        assert target.numel()==self.cross.shape[1]
        self.precision.add_(design[:,None]*design[None,:])
        self.cross.add_(design[:,None]*target[None,:])
        self.mean.copy_(torch.linalg.solve(self.precision,self.cross).to(self.mean))
        assert torch.isfinite(self.mean).all()
        self.updates+=1


def goal_support_cost(initial,goal,slots,width):
    """Fixed initial-goal change energy weights; no semantic/task mask."""
    shape=goal.shape
    views=shape[2]
    weights=(goal-initial).float().reshape(1,1,views,slots,width).square().mean(-1)
    weights=torch.where(weights.sum(-1,keepdim=True)>0,weights,torch.ones_like(weights))
    weights=weights/weights.sum(-1,keepdim=True)

    def objective(target,prediction):
        error=(prediction-target).reshape(-1,1,views,slots,width).abs().mean(-1)
        return -(error*weights).sum(-1)
    return objective
