"""CPU-only requested/executed action comparison from the saved CUDA bank."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

B=Path('/efs/prescottsun/SPT-JEPA')
ROOT=B/'runs/basis_counterfactual_pilot_20261031'
OUT=B/'reports/basis-counterfactual-20261031/action-contract'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    result=json.loads((ROOT/'results.json').read_text())
    for path,digest in result['provenance']['native_files'].items():assert sha(path)==digest
    cache=np.load(ROOT/'bank-endpoints.npz');bank=cache['bank'];assert bank.shape==(64,5,10) and bank.dtype==np.float32
    scaler=StandardScaler();scaler.mean_=np.asarray(result['provenance']['action_mean'])
    scaler.scale_=np.asarray(result['provenance']['action_scale']);scaler.var_=scaler.scale_**2;scaler.n_features_in_=2
    requested=scaler.inverse_transform(bank.reshape(-1,2)).reshape(64,25,2)
    executed=np.clip(requested,-1,1)
    renormalized=scaler.transform(executed.reshape(-1,2)).reshape(64,5,10)
    outside=np.abs(requested)>1;fractions=outside.mean((1,2))
    np.testing.assert_array_equal(fractions,[b['clipped_coordinate_fraction'] for b in result['branches']])
    raw_error=requested-executed;z_error=bank-renormalized
    rows=[]
    for i in range(64):
        rows.append(dict(index=i,clipped_coordinate_fraction=float(fractions[i]),
            affected_microaction_fraction=float(outside[i].any(-1).mean()),
            raw_max_overshoot=float(np.abs(raw_error[i]).max()),raw_clip_rmse=float(np.sqrt(np.mean(raw_error[i]**2))),
            normalized_difference_rmse=float(np.sqrt(np.mean(z_error[i]**2))),
            normalized_difference_max=float(np.abs(z_error[i]).max()),
            raw_min=float(requested[i].min()),raw_max=float(requested[i].max())))
    selected=dict(mean_candidate=0,identity=result['models']['identity']['ranking']['selected_index'],
        basis=result['models']['basis']['ranking']['selected_index'],oracle=result['models']['identity']['ranking']['oracle_index'])
    assert selected['oracle']==result['models']['basis']['ranking']['oracle_index']
    report=dict(source_result=dict(path=str(ROOT/'results.json'),sha256=sha(ROOT/'results.json')),
        bank_file=dict(path=str(ROOT/'bank-endpoints.npz'),sha256=sha(ROOT/'bank-endpoints.npz')),
        cpu_only=True,new_model_forwards=0,action_mean=scaler.mean_.tolist(),action_std=scaler.scale_.tolist(),
        normalized_environment_lower=((-1-scaler.mean_)/scaler.scale_).tolist(),
        normalized_environment_upper=((1-scaler.mean_)/scaler.scale_).tolist(),
        candidates_with_any_clipping=int(np.count_nonzero(fractions)),total_candidates=64,
        clipped_coordinate_fraction_mean=float(fractions.mean()),
        clipped_coordinate_fraction_quantiles=dict(zip(['min','q25','median','q75','max'],np.quantile(fractions,[0,.25,.5,.75,1]).tolist())),
        normalized_difference_rmse_all=float(np.sqrt(np.mean(z_error**2))),
        selected={k:rows[i] for k,i in selected.items()},per_candidate=rows,
        formula='z_requested -> a_requested=mean+std*z -> a_executed=clip(a_requested,-1,1) -> z_executed=(a_executed-mean)/std',
        scope='One fixed first-round CUDA bank/case/seed. Comparison reproduces actual sklearn float32 inverse/forward transforms. '
              'Numerical requested/executed discrepancy is established; its causal role in ranking or the basis gap is not. '
              'A world model of requested commands may in principle learn actuator saturation. Any explicit bounds control belongs '
              'to a common planner protocol for all baselines, not a new representation algorithm.')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Requested versus executed actions: existing bank audit','',
        f"The native model receives unbounded normalized candidates. WorldModelPolicy then inverses the train action scaler; TwoRoom clips raw actions to[-1,1]. This saved bank has clipping in{report['candidates_with_any_clipping']}/64 candidates; mean clipped-coordinate fraction={100*fractions.mean():.2f}% (median{100*np.median(fractions):.1f}%, range{100*fractions.min():.0f}–{100*fractions.max():.0f}%). No CEM action-bound mapping is present.",'',
        '| Selection | Index | Clipped coordinates | Raw max overshoot | Normalized action RMSE | Normalized max difference |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in report['selected'].items():
        lines.append(f"| {name} | {r['index']} | {100*r['clipped_coordinate_fraction']:.0f}% | {r['raw_max_overshoot']:.3f} | {r['normalized_difference_rmse']:.3f} | {r['normalized_difference_max']:.3f} |")
    lines+=['','`CEMSolver.configure` uses Box only for shape/type (`cem.py:63–71`); sampling goes straight to cost (`cem.py:196–215`). `ShootingCostEvaluator` forwards candidates unchanged (`evaluator.py:156`); LeWM flattens them into action_encoder (`lewm.py:101–103`). Policy unblocks the25 micro-actions and inverse-normalizes (`policy.py:513–537`); TwoRoom applies the clipping (`env.py:262–267`). `scripts/plan/config/solver/cem.yaml:1–9` and `tworoom.yaml:14–17` contain no action_bounds setting. All source hashes match the completed pilot.','',
        'This confirms a numerical command/execution difference, not its causal contribution. Clipping fraction is not ordered with regret: identity44%, basis26%, oracle34%. One case cannot establish that clipping caused bad selection or the basis gap. Models could learn saturation as part of requested-command dynamics. A proposed explicit bounds experiment must change the common planner consistently for every baseline and preserve original-protocol results; it is not a new learning algorithm. No model forward, GPU or new environment rollout was performed.','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines));print('\n'.join(lines))


if __name__=='__main__':main()
