"""Presentation only: summarize the six completed fixed-probe results, no fits."""
import hashlib
import json
from pathlib import Path

B=Path('/efs/prescottsun/SPT-JEPA')
R=B/'runs/native_cube_readability_20261031'
O=B/'reports/native-cube-readability-20261031'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def small(v):
    return dict(xyz_cm=v['xyz_euclidean']['mean_cm'],xyz3cm=v['xyz_euclidean']['fraction_within3cm'],
        xyz_r2=v['xyz_features']['variance_weighted_r2'],pose_deg=v['orientation_angles']['mean_deg'],
        pose15=v['orientation_angles']['fraction_within15deg'],pose_r2=v['orientation_features']['variance_weighted_r2'],
        joint=v['joint_readout_within3cm_and15deg_fraction'])


def main():
    d=read(O/'results.json');assert d['passed'] and read(R/'closed-owned-processes.json')['closed']
    assert [r['id']for r in d['rows']]==[f'{family}{step}-seed{seed}'for family,step in [('native',6000),('native',12000),('tf',12000)]for seed in [1,2]]
    rows=[]
    for r in d['rows']:
        e=r['extraction'];assert e['complete_parameters_and_buffers_unchanged'] and e['state_hash_before']==e['state_hash_after']
        row={k:r[k]for k in ['id','family','step','seed','dimension']}
        row['readouts']={kind:dict(configuration=fit['configuration'],training=small(fit['training']),validation={mode:small(v)for mode,v in fit['validation'].items()},decomposition=fit['decomposition'])for kind,fit in r['readouts'].items()}
        row['provenance']=dict(extraction=str(R/r['id']/'extraction.json'),extraction_sha256=sha(R/r['id']/'extraction.json'),
            weights_sha256=e['spec']['weights_sha256'],state_hash=e['state_hash_after'],model_source=e['model_source']['source_commit'],
            actual_imported_modules=e['actual_imported_modules'],actual_calls=e['actual_calls'],seconds=e['seconds'],peak_gib=e['peak_gib'])
        rows.append(row)
    protocol=read(B/'runs/native_cube12000_20261031/planning-cube-native-lewm-seed1/metrics.json')['protocol']
    assert protocol['cube_orientation_threshold_deg']==15 and protocol['cube_position_threshold_m']==.03 and protocol['cube_sustained_steps']==3 and protocol['cube_symmetry_aware']
    tasks=read(R/'probe-task-source-check.json');runner=B/'sources/clear-lewm-20261030/clear_lewm/runner.py'
    assert tasks['sha256']==read(B/'reports/carrier-orientation600-20261031/results.json')['pinned_task_source_sha256']
    criterion=dict(task_source=tasks,runner=dict(path=str(runner),sha256=sha(runner),lines='250–285'),
        identical_angle_function='cube_symmetry_angle_deg: min SO(3) geodesic angle over24 proper cube symmetries; normalized wxyz quaternion matrices',
        identical_threshold='angle<=15degrees; position<=.03m',task_hold_physical_steps=3,
        different_operands='Probe compares predicted/read-out future pose to actual future pose; task compares current actual pose to desired goal.',
        limitation='Constant readout accuracy is not a theoretical task upper bound. Probe failure is not missing information or unlearnability.')
    durations=[read(R/'exits'/(r['id']+'.json'))for r in rows]
    timing=dict(encoder_and_forecast_region_seconds=sum(r['provenance']['seconds']for r in rows),
        summed_job_wall_seconds=sum(x['finished_at']-x['started_at']for x in durations),campaign_wall_seconds=max(x['finished_at']for x in durations)-min(x['started_at']for x in durations))
    summary=dict(passed=True,rows=rows,references={k:small(v)for k,v in d['references'].items()},criterion=criterion,timing=timing,
        full_results=dict(path=str(O/'results.json'),sha256=sha(O/'results.json')),helper_source='8fabed3f13691bd35edc9bebb3a1f04b121bbb80',
        rgb_frames=d['total_rgb_frames'],predict_calls=d['total_predict_calls'],renderer_calls=0,scope=d['scope'],report_script_sha256=sha(__file__))
    (O/'mechanism-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Native versus TF Cube: fixed readability and forecast diagnostics','',
        'All six frozen jobs passed and are held; exact owned dispatcher179907/monitor179908 were closed after GPUs0–5 became idle. Complete parameters and buffers stayed bit-exact. Exactly7,168 new RGB frames,224 predictor calls and zero renderer/training/planning calls were used. Native6k/12k validation states were reused; only TF12k required new validation states.','',
        'Native12k has more linearly readable xyz and a more consistent linearly decoded forecast than native6k; it also beats TF12k on these linear xyz measures in both seeds. This does not establish uniformly better dynamics: with the same predeclared RBF, TF12k has lower mean xyz forecast error in both seeds. Pose remains weak under both fixed probe classes.','',
        'All position values below are mean Euclidean error in cm, on the same448 future frames. Teacher receives actual past observations; shifted carry copies the last actual observation. Neither is a causal rollout.','',
        '| Model | Probe | Observed | Causal forecast | Teacher one-step | Initial copy | Shifted observed copy |',
        '|---|---|---:|---:|---:|---:|---:|']
    modes=['observed_future','clean_rollout','teacher_forced_prefix','initial_carry','shifted_observed_carry']
    for r in rows:
        for kind,f in r['readouts'].items():lines.append(f"| {r['id']} | {kind} | "+' | '.join(f"{f['validation'][m]['xyz_cm']:.2f}"for m in modes)+' |')
    lines+=['','Mean cube-symmetry pose angle in degrees, with identical conditions:','',
        '| Model | Probe | Observed | Causal forecast | Teacher one-step | Initial copy | Shifted observed copy |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        for kind,f in r['readouts'].items():lines.append(f"| {r['id']} | {kind} | "+' | '.join(f"{f['validation'][m]['pose_deg']:.2f}"for m in modes)+' |')
    lines+=['','Threshold readout accuracy; entries are observed → causal forecast. These are not planning-success or hold rates.','',
        '| Model | Probe | xyz≤3cm | pose≤15° | Joint |','|---|---|---:|---:|---:|']
    for r in rows:
        for kind,f in r['readouts'].items():
            a,b=[f['validation'][m]for m in modes[:2]]
            lines.append(f"| {r['id']} | {kind} | "+' | '.join(f"{100*a[k]:.1f}% → {100*b[k]:.1f}%"for k in ['xyz3cm','pose15','joint'])+' |')
    lines+=['','Exact xyz MSE decomposition, in cm² per coordinate. The cross term is retained, not interpreted as dynamics error.','',
        '| Model | Probe | Observation MSE | Forecast–observation discrepancy | Cross term | Forecast MSE |','|---|---|---:|---:|---:|---:|']
    for r in rows:
        for kind,f in r['readouts'].items():
            v=f['decomposition']['clean_rollout']['xyz']
            lines.append(f"| {r['id']} | {kind} | "+' | '.join(f"{10000*v[k]:.2f}"for k in ['observation_mse','decoded_forecast_discrepancy_mse','cross_term','forecast_mse'])+' |')
    lines+=['',
        'Native linear decoded forecast–observation discrepancy falls from26.58/40.28 to4.46/3.44cm² per coordinate between6k and12k. Its observed xyz error improves5.11/5.07→3.90/3.65cm; forecast error improves9.83/11.56→4.93/4.17cm. RBF also improves within native, but TF12k RBF forecast4.98/4.77cm beats native12k5.997/5.118cm. The linear comparison alone cannot establish an intrinsic dynamics ordering.',
        'TF has large negative xyz cross terms: linear−21.10/−8.97cm² per coordinate and RBF−3.43/−2.73. Its decoded forecast can therefore be closer to the physical label than the decoded actual observation. That does not make the model more accurate than reality: observed-state probe error is correlated with the forecast displacement, and a fixed imperfect readout need not commute with prediction. Observation readout error is not a lower bound on forecast readout error.',
        'All held-out observed tensor81 variance-weighted R² values are negative. Native linear training R² is.375–.409 but validation is−.194 to−.328; TF linear training is.048/.080 and validation−.016/−.003. RBF validation remains negative for all six. This documents weak generalization of these fixed pose probes, not absent orientation information or an impossibility of learning it.',
        'Constant training-mean readout: xyz18.62cm, pose23.13°,32.8125% within15°, joint0%. True physical target decoded through the same training-only quaternion codebook: mean angle.552°,100% within15°. Thus codebook quantization alone is not the main restriction in these probe scores. The32.8% constant rate is a distribution-specific diagnostic reference, never a theoretical task-success ceiling.',
        'The pose metric and≤15° inequality exactly match the pinned strict task: tasks.py:93–103 and runner.py:274–285 use the same24-symmetry geodesic function; the actual protocol also requires≤3cm and three consecutive physical steps. Probe and task have different operands (readout versus actual future state, compared with actual state versus desired goal), and probe frames are five steps apart. No hold claim is possible.',
        'Controls/limits: exactly the old128 distinct training clips×8 frames and64 validation clips×7 future frames; all probe fits use observed training encodings, train-only standardization, fixed ridge.001 and fixed training-median RBF bandwidth. Tensor81/codebook/xyz targets are diagnostic privileged labels only. Native192-state/18.0M parameters differs from TF64-state/51.1M; probe dimension and model capacity are not matched. Native6k→12k has extra exposure. Native12k versus TF12k matches update/window exposure but not architecture/compute. No raw latent-MSE comparison or causal mediation claim is made.128 clips are128 episodes despite1,024 correlated frames.',
        'Complete per-mode training/validation metrics, feature R², pose/xyz decompositions, hashes and original source paths remain in results.json; mechanism-summary.json is the small publication artifact. No extra fits, cohorts, counterfactual rollouts, dimension controls or new planning runs were added.']
    (O/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summary=str(O/'mechanism-summary.json'),report=str(O/'RESULTS.md'),models=len(rows),timing=timing,source_angle_definition_matches=True)))


if __name__=='__main__':main()
