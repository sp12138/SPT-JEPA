"""Actual frozen CUDA steering and Gaussian-bin parity, zero SGD/physics."""
import runpy
import torch
from pathlib import Path
from experiments.metaworld_transition_only_20261109.common import BASE,save,now,tensor_sha,sha


@torch.no_grad()
def main():
    d=runpy.run_path(str(BASE/'reports/metaworld_random_roadmap_20261109/edges-v1.py'))
    world,inverse,ancestry=d['NODE']['load_models'](1);bank=d['load_bank'](1);before=(tensor_sha(world),tensor_sha(inverse))
    source=torch.tensor([0,7,31],device='cuda');target=d['neighbors'](bank,bank['image'][source],[bank['hashes'][i] for i in [0,7,31]])
    repeated=source[:,None].expand(-1,32).flatten();target=target.flatten()
    a,b,control=d['steer'](world,inverse,bank['hidden'][repeated],bank['image'][repeated],bank['feature'][repeated],bank['image'][target],bank['feature'][target])
    assert torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(control).all() and control.abs().max()<=1
    mean=bank['image'][source,:,:4,:4];outcome=bank['image'][source,:,:4,:4]+.001;variance=torch.ones_like(mean)*.0025
    gpu=d['quantized_nll'](mean,outcome,variance);cpu=d['quantized_nll'](mean.cpu(),outcome.cpu(),variance.cpu())
    error=float((gpu.cpu()-cpu).abs().max());assert error<1e-5
    assert before==(tensor_sha(world),tensor_sha(inverse))
    save(d['ROOT']/'EDGE_RUNTIME_PREFLIGHT.json',dict(passed=True,utc=now(),actual_CUDA_steering_candidates=96,own_controls_bounded=True,
        CUDA_CPU_quantized_NLL_max_abs_error=error,models_unchanged=True,ancestry=ancestry,physics_steps=0,optimizer_updates=0,
        helper_sha256=sha(__file__),scope='NN/interfacegate,NOTactualphysicaledgefeasibility or controllerperformance.'))


if __name__=='__main__':main()
