"""Use the exact previous planner with the horizon-only model constructor."""
import argparse
from pathlib import Path
import torch
from experiments.metaworld_transition_only_20261109 import evaluate as original
from .model import build


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['config','manifest','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--horizon',type=int,required=True)
    p.add_argument('--execute',type=int,required=True);p.add_argument('--budget',type=int,default=500);a=p.parse_args()
    torch.cuda.set_per_process_memory_fraction(.30)
    original.build=build
    original.run(a.config,a.step,a.manifest,a.output,a.horizon,a.execute,a.budget)
