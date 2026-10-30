"""CPU-only reanalysis of saved carrier endpoint states and chart weights.

No encoder, predictor rollout, optimizer, simulator, or accelerator is used.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from as_jepa_lewm.carrier import ResidualCarrierChart


def mse(a, b):
    return float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))


def distance_ratios(u, z):
    def ratios(a, b):
        da = np.sqrt(np.sum(a*a, axis=-1))
        db = np.sqrt(np.sum(b*b, axis=-1))
        keep = da > 1e-10
        ratio = db[keep]/da[keep]
        return dict(count=int(keep.sum()), minimum=float(ratio.min()),
                    median=float(np.median(ratio)), maximum=float(ratio.max()))
    a, b = u[:,0].astype(np.float64), z[:,0].astype(np.float64)
    i, j = np.triu_indices(len(a), k=1)
    return dict(initial_frame_pairs=ratios(a[i]-a[j],b[i]-b[j]),
                adjacent_frame_pairs=ratios(np.diff(u.astype(np.float64),axis=1),
                                           np.diff(z.astype(np.float64),axis=1)))


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    root=Path('/efs/prescottsun/SPT-JEPA/runs/ideas_validate600_20261031')
    rows=[]
    with torch.no_grad():
        for task in ['tworoom','cube']:
            for seed in [1,2]:
                reference=np.load(root/f'transition-{task}-carrier-identity-seed{seed}'/'state-audit.npz')
                u=reference['observed'];target=u[:,1:]
                initial=mse(target,u[:,:1]);local=mse(target,u[:,:-1])
                for mode in ['identity','unconstrained','bounded']:
                    name=f'{task}-carrier-{mode}-seed{seed}'
                    folder=root/('transition-'+name)
                    report=json.loads((folder/'state-audit.json').read_text())
                    stored=np.load(folder/'state-audit.npz')
                    checkpoint=root/name/'model-step0000600/weights.pt'
                    weights=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
                    parameters={k.removeprefix('carrier_chart_module.'):v for k,v in weights.items()
                                if k.startswith('carrier_chart_module.')}
                    chart=ResidualCarrierChart(mode=mode)
                    chart.load_state_dict(parameters,strict=True)
                    del parameters,weights
                    certificate=chart.norm_certificate()
                    assert certificate['residual_lipschitz_upper'] < 1
                    first,second=chart.effective_weights()
                    def residual(value):
                        return F.linear(torch.tanh(F.linear(value,first,chart.input.bias)),
                                        chart.bound*second,chart.output.bias)
                    def inverse(value):
                        z=torch.from_numpy(value.copy()).float();x=z.clone()
                        for _ in range(32):x=z-residual(x)
                        defect=x+residual(x)-z
                        return x.numpy(),float(defect.abs().max())
                    observed_inverse,observed_defect=inverse(stored['observed'])
                    alignment=float(np.max(np.abs(observed_inverse-u)))
                    assert alignment < 2e-4, (name,alignment)
                    conditions={}
                    for key in ['clean_rollout','teacher_forced_prefix','other_initial',
                                'zero_initial','wrong_action_order','other_episode_actions']:
                        aligned,defect=inverse(stored[key])
                        error=mse(aligned,target)
                        conditions[key]=dict(mse=error,over_initial_carry=error/initial,
                                             over_shifted_carry=error/local,inverse_max_defect=defect)
                    for key in ['other_initial','zero_initial','wrong_action_order','other_episode_actions']:
                        conditions[key]['mse_change_vs_factual']=conditions[key]['mse']-conditions['clean_rollout']['mse']
                    row=dict(name=name,task=task,seed=seed,mode=mode,
                        recorded_weights_sha256=report['weights_sha256'],manifest_sha256=report['manifest_sha256'],
                        residual_lipschitz_upper=certificate['residual_lipschitz_upper'],
                        chart_displacement_rms=float(np.sqrt(mse(stored['observed'],u))),
                        chart_displacement_max=float(np.max(np.abs(stored['observed']-u))),
                        chart_distance_ratios=distance_ratios(u,stored['observed']),
                        inverse_observation_alignment_max=alignment,inverse_observation_defect=observed_defect,
                        common_initial_carry_mse=initial,common_shifted_carry_mse=local,
                        common_carrier_prediction=conditions)
                    rows.append(row)
                    print(json.dumps(dict(name=name,residual_bound=row['residual_lipschitz_upper'],
                        displacement_rms=row['chart_displacement_rms'],distance_ratios=row['chart_distance_ratios'],
                        common_rollout=conditions['clean_rollout']['over_initial_carry'],
                        common_teacher=conditions['teacher_forced_prefix']['over_shifted_carry'],
                        inverse_alignment=alignment)),flush=True)
    result=dict(scope='CPU-only inversion/reanalysis of saved outputs; no new image encoding, dynamics rollout, model update or GPU',
                root=str(root),source_commit=json.loads((root/'source-record.json').read_text())['source_commit'],
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                inverse_iterations=32,rows=rows)
    output=Path('/tmp/carrier-endpoint-common-coordinates-20261031.json')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output))),flush=True)


if __name__ == '__main__':
    main()
