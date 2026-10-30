"""Frozen-model audit of scalar posterior coupling versus dynamics transport.

No model update. This measures a concrete mismatch; it does not validate a new
posterior objective or assert that removing a regularizer improves control.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from as_jepa_lewm.robust import RobustLeWM
from as_jepa_tools.benchmark_data import BenchmarkWindows
from as_jepa_tools.benchmark_health import finite
from vforge.ops import OpsConfig, initialize_ops


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--step', required=True, type=int)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root/'config.json').read_text())
    torch.set_num_threads(2)
    initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=torch.device('cuda'),parallel_state=None)
    model = RobustLeWM(**cfg['model']).cuda().eval()
    model.load_state_dict(torch.load(root/f'model-step{args.step:07d}/weights.pt',map_location='cpu',weights_only=True))
    ds = BenchmarkWindows(cfg['data'], window=model.window, split='validation')
    indices = np.random.default_rng(1042).choice(len(ds), 32, replace=False).tolist()
    rows = []
    draws = 16
    for batch in DataLoader(Subset(ds,indices),batch_size=4,num_workers=0):
        batch = {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        pixels, _, _ = model.prepare(batch)
        info = model.encode(dict(pixels=pixels,action=batch['action']))
        mu, embedded = info['emb'], info['act_emb']
        b,w,d = mu.shape
        clean = model.predict(mu[:,:-1],embedded[:,:-1])
        delta = mu[:,1:]-clean
        fixed, component, cross, exact_errors, final_variance, prefix_variance = [],[],[],[],[],[]
        for draw in range(draws):
            torch.manual_seed(1042+draw+100*len(rows))
            eps = torch.randn_like(mu)
            for t in range(1,w):
                eps[:,t]=model.rho*eps[:,t-1]+math.sqrt(1-model.rho**2)*eps[:,t]
            noise = model.sigma0*eps
            noisy = model.predict((mu+noise)[:,:-1],embedded[:,:-1])
            mismatch = model.rho*noise[:,:-1]-(noisy-clean)
            residual = delta+mismatch
            fixed.append(residual.square().mean((1,2)))
            component.append(mismatch.square().mean((1,2)))
            cross.append((2*delta*mismatch).mean((1,2)))
            # Same total innovation budget sigma0² over this observed window.
            # Prior and posterior share F, with observation-anchored correction.
            history=[mu[:,0]+noise[:,0]]
            variances=[]
            for t in range(w-1):
                prior=model.predict(torch.stack(history,1),embedded[:,:t+1])[:,-1]
                conditional=mu[:,t+1]+prior-clean[:,t]
                exact_errors.append(float(((conditional-prior)-delta[:,t]).abs().max()))
                history.append(conditional+model.sigma0/math.sqrt(w-1)*torch.randn_like(prior))
                variances.append((history[-1]-mu[:,t+1]).square().mean(-1))
            prefix_variance.append(torch.stack(variances,1))
            final_variance.append(variances[-1])
        values=dict(clean_mse=delta.square().mean((1,2)),
            scalar_coupled_mean_mse=torch.stack(fixed).mean(0),
            noise_mismatch_energy=torch.stack(component).mean(0),
            clean_noise_cross_term=torch.stack(cross).mean(0),
            transported_final_noise_mse=torch.stack(final_variance).mean(0))
        closure=values['scalar_coupled_mean_mse']-values['clean_mse']-values['noise_mismatch_energy']-values['clean_noise_cross_term']
        assert float(closure.abs().max())<2e-6
        assert max(exact_errors)<2e-6
        for i in range(b):
            rows.append(dict(episode=int(batch['episode'][i]),start=int(batch['start'][i]),
                **{k:float(v[i]) for k,v in values.items()},
                transported_noise_by_step=torch.stack(prefix_variance).mean(0)[i].cpu().tolist()))
    summary={k:float(np.mean([x[k] for x in rows])) for k in rows[0] if k not in {'episode','start','transported_noise_by_step'}}
    summary['transported_final_to_initial_noise_ratio']=summary['transported_final_noise_mse']/model.sigma0**2
    report=dict(run=str(root),step=args.step,rho=model.rho,sigma0=model.sigma0,draws=draws,cases=rows,summary=summary,
        scope='Read-only eval-mode FP32. Decomposes existing rate mean-error into clean error, scalar noise/dynamics mismatch and cross term. Transport check uses a changed sequential posterior and fixed total innovation budget, not a trained candidate or identical marginal noise. No task gain asserted.')
    assert finite(report)
    (out/'noise-transport-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
