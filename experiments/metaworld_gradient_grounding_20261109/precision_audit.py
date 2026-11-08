"""Same frozen computation, test cuDNN TF32 versus IEEE FP32 derivatives."""
from pathlib import Path
import numpy as np
import torch
from .probe import forecast
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,sha,tensor_sha,now
from experiments.metaworld_horizon_match_20261109.model import build


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.30)
    cfg=read(BASE/'runs/metaworld_horizon_match_20261109/training/h60-seed1/config.json')
    model=build(cfg).cuda().eval().requires_grad_(False)
    weights=Path(cfg['output'])/'model-step0012000/weights.pt'
    model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True));before=tensor_sha(model)
    path=BASE/'runs/metaworld_realized_bank_20261109/push-v3-seed6042.npz'
    assert sha(path)==read(path.with_suffix('.json'))['bank_sha256']
    data=np.load(path)
    with torch.no_grad():
        past=torch.from_numpy(data['warm_history'].copy()).permute(0,3,1,2)[None].cuda()
        hidden,image=model.prepare(dict(history_pixels=past,history_actions=torch.zeros(1,7,4,device='cuda'),history_valid=torch.ones(1,8,device='cuda',dtype=torch.bool)))
        goal=model.pixels(torch.from_numpy(data['goal'].copy()).permute(2,0,1)[None].cuda())
        generator=torch.Generator(device='cuda').manual_seed(771009+6042)
        candidates=torch.randn(64,12,4,device='cuda',generator=generator).clamp(-1,1);candidates[0]=0
        costs=(forecast(model,hidden,image,candidates)-goal).square().mean((1,2,3))
        point=(candidates[costs.topk(4,largest=False).indices]*.9).detach()
    direction=np.random.default_rng(20261109).standard_normal(point.numel());direction/=np.linalg.norm(direction)
    direction=torch.tensor(direction.reshape(point.shape),device='cuda',dtype=torch.float32)
    rows=[];initial=dict(cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    for label,allow in [('original',initial['cudnn_allow_tf32']),('ieee',False)]:
        torch.backends.cudnn.allow_tf32=allow
        commands=point.detach().clone().requires_grad_(True)
        loss=(forecast(model,hidden,image,commands)-goal).square().mean()
        gradient=torch.autograd.grad(loss,commands)[0]
        derivative=float((gradient*direction).sum());scale=float(loss.detach());differences=[]
        with torch.no_grad():
            repeated=[float((forecast(model,hidden,image,point)-goal).square().mean()) for _ in range(2)]
            for epsilon in [.02,.002,.0002]:
                a=float((forecast(model,hidden,image,point+epsilon*direction)-goal).square().mean())
                b=float((forecast(model,hidden,image,point-epsilon*direction)-goal).square().mean())
                finite=(a-b)/(2*epsilon)
                differences.append(dict(epsilon=epsilon,analytic_normalized=derivative/scale,
                    finite_normalized=finite/scale,relative_error=abs(finite-derivative)/max(abs(finite),abs(derivative),1e-10)))
        rows.append(dict(mode=label,cudnn_allow_tf32=allow,repeated_identical=repeated[0]==repeated[1],differences=differences))
        print(rows[-1],flush=True)
    assert tensor_sha(model)==before
    save(BASE/'runs/metaworld_gradient_grounding_20261109/PRECISION_AUDIT.json',dict(passed=True,utc=now(),initial_flags=initial,rows=rows,
        weights_sha256=sha(weights),model_before=before,model_after=before,model_updates=0,physics_steps=0,
        candidate_performance=False,training_data_eligible=False,
        scope='Protected Push6042 warm frames used only for this numerical audit;no calibration/SGD/proposal reuse. Same fixed state and controls across backend modes.'))
