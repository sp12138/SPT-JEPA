"""Original CEM actions/budget; change only its terminal spatial objective."""
import argparse
from pathlib import Path
import torch
from experiments.metaworld_transition_only_20261109 import evaluate as original
from .metric import cost

SPATIAL=False


@torch.no_grad()
def plan(model,hidden,image,goal,generator,horizon,warm=None,particles=64,iterations=4):
    assert horizon in [30,60] and horizon%5==0
    blocks=horizon//5
    mean=image.new_zeros(blocks,4) if warm is None else warm.reshape(blocks,5,4).mean(1)
    std=torch.ones_like(mean);best=None;best_score=float('inf');best_pixel=None;first_bank=None
    for iteration in range(iterations):
        raw=torch.randn(particles,12,4,device=image.device,generator=generator)
        candidates=(mean[None]+std[None]*raw[:,:blocks]).clamp(-1,1);candidates[0]=mean
        if best is not None:candidates[1]=best
        if iteration==0:first_bank=__import__('hashlib').sha256(candidates[:,:6].cpu().numpy().tobytes()).hexdigest()
        state=hidden.expand(particles,-1,-1,-1).clone();future=image.expand(particles,-1,-1,-1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for t in range(horizon):state,future=model.predict(state,future,candidates[:,t//5])
        pixel=(future-goal).square().mean((1,2,3));costs=cost(future,goal) if SPATIAL else pixel
        assert torch.isfinite(costs).all();values,indices=costs.topk(min(10,particles),largest=False);elite=candidates[indices]
        if float(values[0])<best_score:
            best_score=float(values[0]);best=elite[0].clone();best_pixel=float(pixel[indices[0]])
        mean=elite.mean(0);std=elite.std(0,unbiased=False).clamp_min(1e-6)
    return best[:,None].expand(-1,5,-1).reshape(horizon,4),dict(predicted_terminal_goal_mse=best_pixel,
        planner_objective=best_score,planner_objective_kind='fixed_DCT_Hminus1' if SPATIAL else 'pixel_MSE',first_bank_prefix_sha256=first_bank)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['config','manifest','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--kind',choices=['pixel','spatial'],required=True);p.add_argument('--budget',type=int,default=500);a=p.parse_args()
    SPATIAL=a.kind=='spatial';original.plan=plan;torch.cuda.set_per_process_memory_fraction(.30)
    original.run(a.config,10000,a.manifest,a.output,30,15,a.budget)
