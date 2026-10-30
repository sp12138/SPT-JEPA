"""CPU-only review of native600 and the one posthoc train-only BN control."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

B=Path('/efs/prescottsun/SPT-JEPA')
ROOT=B/'runs/native_lewm600_20261031'
REPORT=B/'reports/native-lewm600-20261031'


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def compare(left,right):
    for key in ['manifest_sha256','pairs','solver','plan','protocol','normalization','split']:assert left[key]==right[key],key
    assert len(left['metrics']['episode_successes'])==len(right['metrics']['episode_successes'])==50


def main():
    from experiments.ideas_validate_20261031.collect import paired_outcomes
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(REPORT));a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    plan=read(ROOT/'plan.json');assert len(plan['jobs'])==16
    for job in plan['jobs']:assert read(ROOT/'exits'/(job['id']+'.json'))['returncode']==0
    rows=[];reference_rows=[]
    for task in ['tworoom','cube']:
        for seed in [1,2]:
            name=f'{task}-native-lewm-seed{seed}';run=ROOT/name
            planning=read(ROOT/('planning-'+name)/'metrics.json');state=read(ROOT/('transition-'+name)/'state-audit.json')
            full=read(ROOT/('fullstate-'+name)/'preflight.json');assert full['passed']
            bn=read(REPORT/('bn-'+name)/'bn-audit.json');assert bn['buffers_restored']
            assert bn['weights_sha256']==state['weights_sha256']
            assert read(REPORT/('bn-'+name)/'exit.json')['returncode']==0
            cal=read(REPORT/('calibrated-'+name)/'calibration.json');assert cal['all_other_state_bitexact']and cal['training_only']
            assert cal['original_weights_sha256']==state['weights_sha256']
            assert set(cal['changed_buffers'])=={prefix+'.'+kind for prefix in ['projector.net.1','pred_proj.net.1']for kind in ['running_mean','running_var']}
            assert [s['vectors']for s in cal['statistics']]==[1024,896]
            assert all(s['num_batches_tracked']==600 for s in cal['statistics'])
            assert cal['manifest_sha256']==digest(cal['manifest'])
            assert read(REPORT/('calibration-exit-'+name+'.json'))['returncode']==0
            cp=read(REPORT/('planning-calibrated-'+name)/'metrics.json');compare(planning,cp)
            assert read(REPORT/('planning-exit-'+name+'.json'))['returncode']==0
            clean=state['conditions']['clean_rollout'];teacher=state['conditions']['teacher_forced_prefix']
            row=dict(task=task,seed=seed,id=name,parameters=read(run/'model_size.json'),training=read(run/'status.json'),
                original_success=planning['metrics']['success_rate'],calibrated_success=cp['metrics']['success_rate'],
                original_outcomes=planning['metrics']['episode_successes'],calibrated_outcomes=cp['metrics']['episode_successes'],
                calibrated_minus_original=paired_outcomes(cp['metrics']['episode_successes'],planning['metrics']['episode_successes']),
                fullstate_passed=full['passed'],eval_two_case_metrics=full['metrics'],
                state=dict(observed_std=state['observed_state_std_mean'],observed_rms=state['observed_state_rms'],
                    clean_mse=clean['mse'],clean_over_initial_copy=clean['normalized_by_initial_variation'],
                    teacher_mse=teacher['mse'],teacher_over_initial_copy=teacher['normalized_by_initial_variation'],
                    initial_copy_mse=state['conditions']['initial_carry']['mse'],shifted_observed_copy_mse=state['conditions']['shifted_observed_carry']['mse'],
                    interventions={key:{k:v for k,v in state['conditions'][key].items()if k in ['mse','mse_change_vs_clean_rollout','intervention_effect_over_initial_variation']}for key in ['other_initial','wrong_action_order','other_episode_actions']}),
                bn_attribution=bn['rows'],calibration={k:v for k,v in cal.items()if k!='statistics'},
                calibration_statistics=[{k:v for k,v in s.items()if k not in ['mean','unbiased_variance']}for s in cal['statistics']],
                planning_manifest_sha256=planning['manifest_sha256'],state_manifest_sha256=state['manifest_sha256'],
                artifacts=dict(native_planning=str(ROOT/('planning-'+name)/'metrics.json'),calibrated_planning=str(REPORT/('planning-calibrated-'+name)/'metrics.json'),
                    transition=str(ROOT/('transition-'+name)/'state-audit.json'),bn=str(REPORT/('bn-'+name)/'bn-audit.json'),calibration=str(REPORT/('calibrated-'+name)/'calibration.json')))
            for arm in ['identity','unconstrained','bounded']:
                fr=B/'runs/ideas_validate600_20261031'/f'planning-{task}-carrier-{arm}-seed{seed}'/'metrics.json'
                f=read(fr);compare(planning,f)
                reference_rows.append(dict(task=task,seed=seed,arm='F-'+arm,success=f['metrics']['success_rate'],outcomes=f['metrics']['episode_successes'],
                    path=str(fr),sha256=digest(fr),exact_cases_planner_protocol_normalization=True,
                    native_minus_F=paired_outcomes(row['original_outcomes'],f['metrics']['episode_successes']),
                    calibrated_minus_F=paired_outcomes(row['calibrated_outcomes'],f['metrics']['episode_successes'])))
            rows.append(row)
    joint=[];rng=np.random.default_rng(1042)
    for task in ['tworoom','cube']:
        selected=[r for r in rows if r['task']==task]
        assert len({r['calibration']['manifest_sha256']for r in selected})==1
        delta=np.stack([np.asarray(r['calibrated_outcomes'],float)-np.asarray(r['original_outcomes'],float)for r in selected])
        draws=delta[:,rng.integers(50,size=(10000,50))].mean((0,2))*100
        joint.append(dict(task=task,difference_pp=float(delta.mean()*100),case_bootstrap_ci95_pp=np.quantile(draws,[.025,.975]).tolist(),
            scope='Same50 cases resampled jointly across the two fixed training seeds; not uncertainty over training-seed population.'))
    result=dict(rows=rows,carrier_references=reference_rows,calibration_case_bootstrap=joint,
        tf600_scalar_context={'tworoom':[6,8],'cube':[14,10],'scope':'Previously reported native-evaluator scalar reference; original per-case metrics not located during this bounded review, so no exact-case certificate or paired TF interval claimed.'},
        completed_main_endpoints=16,fullstate_audits=4,source_commit=read(ROOT/'protocol.json')['source']['source_commit'],
        planning_launch_recovery=read(REPORT/'bncal-planning-first-attempt-failure.json'),
        bn_buffer_history=read(REPORT/'bn-buffer-history.json'),source_sha256=digest(__file__),
        scope='Native learning objective and modules were retained; history7/split/exposure/budget adaptation remains explicit. PosthocBN control uses train-only128clips and changes4 buffers, not weights. Models differ in capacity/compute from TF/F.600 remains inside1249warmup; no mature superiority or automatic2000 continuation.')
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Native LeWM600 and one train-only BN calibration control','',
        '| Task | Seed | Native | Calibrated BN | Fidentity | Ffree | Fbounded | TF600 context |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        refs=[next(x['success']for x in reference_rows if x['task']==row['task']and x['seed']==row['seed']and x['arm']=='F-'+arm)for arm in ['identity','unconstrained','bounded']]
        tf=result['tf600_scalar_context'][row['task']][row['seed']-1]
        lines.append(f"| {row['task']} | {row['seed']} | {row['original_success']:.0f}% | {row['calibrated_success']:.0f}% | {refs[0]:.0f}% | {refs[1]:.0f}% | {refs[2]:.0f}% | {tf}% |")
    lines+=['','Train-only BN calibration removes the large one-step prediction bias but shows no consistent planning improvement across tasks or seeds. The raw baseline cannot support a healthy-baseline advantage claim; the calibrated result also does not establish mature LeWM inferiority. All four checkpoints are still inside warmup.','',
        'Native/calibrated/F comparisons have identical50 cases, manifest hash, solver, horizon, budget, strict semantics and action normalization. TF is scalar context only. Capacity is18.04M for native LeWM versus approximately51.1M for TF/F; these are not equal-compute/capacity comparisons.','',
        '| Task | Seed | Clean rollout / initial copy | Teacher / initial copy | Observed RMS | Observed std |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        s=row['state'];lines.append(f"| {row['task']} | {row['seed']} | {s['clean_over_initial_copy']:.3f} | {s['teacher_over_initial_copy']:.3f} | {s['observed_rms']:.3f} | {s['observed_std']:.3f} |")
    lines+=['','Teacher-forced and shifted-copy controls receive additional real observations. Their errors must not be called clean planning performance. On Cube, wrong states/actions increase error, especially seed1; this establishes input use, not sufficiently accurate control. TwoRoom original rollouts nearly predict zero relative to the huge shared target offset.','',
        'The fixed-weight experiment reproduces the discrepancy on the exact last training batch, using the actual bf16-mixed/high training precision. Thus validation marginal shift and FP32-only evaluation are not its main cause. Replacing only observation-projector BN running statistics by current batch statistics largely removes the error; replacing prediction BN alone does not. Original BN counters equal200/400/600 and default momentum=.1: no missing-stat-update defect was found. The observation BN variance contracts sharply, especially on TwoRoom.','',
        '| Task | Seed | Last-train FP32 running MSE | Last-train BF16 running MSE | FP32 projector-batch MSE | Mean-bias share of FP32 error |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        pick=lambda policy,mode:next(x for x in row['bn_attribution']if x['split']=='train'and x['precision_policy']==policy and x['bn_mode']==mode)
        f=pick('eval_fp32','running');b=pick('train_bf16','running');p=pick('eval_fp32','projector_batch')
        lines.append(f"| {row['task']} | {row['seed']} | {f['native_pred_loss']:.4f} | {b['native_pred_loss']:.4f} | {p['native_pred_loss']:.4f} | {100*f['shared_bias_mse']/f['fp64_residual_mse']:.2f}% |")
    lines+=['',
        'Mechanism: training BN is invariant to a shared translation of its pre-BN features. A loss applied after BN therefore need not anchor that feature origin. Evaluation instead uses running means: the current-to-running mean offset is divided by sqrt(running variance + epsilon). Current TwoRoom projector inputs differ by roughly20–24 running standard deviations, and almost all prediction error is a shared offset. This supports inconsistent running statistics under changing features, amplified by shrinking variance; it does not separately establish how much comes from EMA lag versus the final parameter updates. Native-function loss/gradient equivalence was already verified; the diagnosed failure is real deployment behavior, not evidence that the native objective was replaced.','',
        'The calibrated variant uses one predeclared128-clip training manifest per task, shared across seeds. Fixed FP32 weights first estimate observation-projector moments over1024 vectors, then prediction-projector moments over896 vectors after applying the first correction. Float64 actual-count pooled moments include between-chunk variance; both use unbiased global variance. Only four running_mean/running_var buffers change. No validation labels/images fit these statistics. Fixed single-clip replay and finite-sample batch-BN corrections are verified before unchanged primary50 planning. This is a posthoc baseline adaptation, not a new objective or the literal native result.','',
        'The extra calibration exposure is 128 training clips / 1024 RGB frames per model, with no gradient updates. Native optimizer state and original exports remain unchanged. The calibration rule is identical for both tasks and seeds; its settings were not selected on planning outcomes.','',
        '| Task | Seed | Fixed train MSE | Batch-BN train MSE | Fixed/batch prediction discrepancy | Single-clip/batch discrepancy |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        c=row['calibration']['replay']
        lines.append(f"| {row['task']} | {row['seed']} | {c['fixed_one_step_mse']:.6f} | {c['batch_bn_one_step_mse']:.6f} | {c['fixed_vs_batch_prediction_mse']:.2e} | {c['fixed_single_clip_vs_batch_prediction_mse']:.2e} |")
    lines+=['',
        '| Task | Mean calibrated − native | Conditional same-case95% interval |',
        '|---|---:|---|']
    for row in joint:lines.append(f"| {row['task']} | {row['difference_pp']:+.1f} pp | {row['case_bootstrap_ci95_pp']} pp |")
    lines+=['',result['scope'],'',
        'Native adaptation: official modules, BN, target gradients, prediction MSE + .09 SIGReg, Cube smoothed_dim10 and per-update warmup scheduler are retained. History7/window8, episode-level train split, training-only action normalization and project sampler/seed match the current experiment exposure. The literal official history3/window4/random-clip split recipe is distinct. Planner history3 is unchanged. Native BN/SIGReg are batch coupled; candidate objectives remain sample-level.','',
        'The 16 main endpoints, four attribution jobs, four train-only calibrations and four calibrated evaluations have exit0. The first planning supervisor attempt failed before opening logs or starting any GPU evaluator because nested shell quoting removed Python string quotes; its source and launch records are preserved. The corrected copied Python supervisor uses argument lists and records actual evaluator PIDs. This infrastructure failure changed no model/data or evaluation setting.','',
        'All four full-state gradient audits passed, but this does not establish a healthy original deployment baseline. No2000 run is launched by this report. Preserve native and calibrated results separately; judge a next budget using deployed planning and frozen-state evidence together.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(out),successes=[dict(task=r['task'],seed=r['seed'],native=r['original_success'],calibrated=r['calibrated_success'])for r in rows],paired=joint)),flush=True)


if __name__=='__main__':main()
