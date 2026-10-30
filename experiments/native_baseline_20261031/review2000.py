"""Read-only CPU review: native2000, fixed preciseBN, exact600 and F/TF controls."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.ideas_validate_20261031.collect import paired_outcomes,digest
from experiments.native_baseline_20261031.review600 import compare
from experiments.native_baseline_20261031.prepare2000 import semantic

B=Path('/efs/prescottsun/SPT-JEPA')
ROOT=B/'runs/native_lewm2000_20261031'
OLD=B/'runs/native_lewm600_20261031'
OLD_REPORT=B/'reports/native-lewm600-20261031'
REF=B/'reports/carrier-learn2000-20261031/results.json'
BUFFERS={p+'.'+k for p in ['projector.net.1','pred_proj.net.1']for k in ['running_mean','running_var']}
def read(p):return json.loads(Path(p).read_text())
def artifact(p):return dict(path=str(p),sha256=digest(p))
def compact(d):return {k:v for k,v in d.items()if not isinstance(v,(dict,list))}
def summarize_state(d):
    return dict(observed_std=d['observed_state_std_mean'],observed_rms=d['observed_state_rms'],
        manifest_sha256=d['manifest_sha256'],actual_temporal_variation=d['actual_temporal_variation'],
        conditions={k:{q:v for q,v in r.items()if q not in ['per_window_mse']}for k,r in d['conditions'].items()},
        scope=d['scope'])
def joint_compare(pairs):
    delta=np.array([np.asarray(a,float)-np.asarray(b,float)for a,b in pairs]);assert delta.shape==(2,50)
    rng=np.random.default_rng(1042);draws=delta[:,rng.integers(50,size=(10000,50))].mean((0,2))*100
    return dict(difference_pp=float(delta.mean()*100),case_bootstrap_ci95_pp=np.quantile(draws,[.025,.975]).tolist(),
        scope='Same50 cases jointly resampled across two fixed training seeds; not population uncertainty across seeds.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default=str(B/'reports/native-lewm2000-20261031'));a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    plan=read(ROOT/'plan.json');protocol=read(ROOT/'protocol.json');source=read(ROOT/'source-record.json')
    assert len(plan['jobs'])==28 and source==protocol['source']
    jobs={j['id']:j for j in plan['jobs']};helper_sources={};exits=[]
    for j in plan['jobs']:
        e=read(ROOT/'exits'/(j['id']+'.json'));assert e['returncode']==0;exits.append(e)
        assert j['source']==source
        if j['phase']in ['archive','calibration']:
            p=Path(j['command'][3]);base=p.parents[2];m=base.with_suffix('.json');man=read(m)
            assert digest(p)==man['files'][str(p.relative_to(base))]
            helper_sources[str(p)]=dict(**artifact(p),source_commit=man['source_commit'],manifest=artifact(m))
    man=read(source['manifest']);critical={}
    for name in ['as_jepa_lewm/native_lewm.py','as_jepa_lewm/factory.py','as_jepa_lewm/train.py','as_jepa_tools/train_stage.py','as_jepa_tools/benchmark_data.py','as_jepa_tools/benchmark_evaluate.py']:
        p=Path(source['source'])/name;assert digest(p)==man['files'][name];critical[name]=digest(p)
    oldsummary=read(OLD_REPORT/'results.json');refs=read(REF);rows=[];reference_rows=[];comparisons=[]
    for task in ['tworoom','cube']:
        for seed in [1,2]:
            n=f'{task}-native-lewm-seed{seed}';run=ROOT/n;cfg=read(run/'config.json');parent=OLD/n
            assert semantic(cfg)==semantic(read(parent/'config.json'))
            assert digest(run/'config.json')==jobs['learn2000-'+n]['config_sha256']
            arch=read(ROOT/f'archive-{n}/fullstate.json');full=read(ROOT/f'fullstate-{n}/preflight.json');pre=read(run/'pre-resume-fullstate.json')
            assert arch['passed']and full['passed']and pre['passed']
            assert arch['step']==arch['scheduler_last_epoch']==2000 and arch['campaign_samples']==256000
            assert arch['optimizer_group_count']==1 and arch['optimizer_state_tensors']==297 and arch['optimizer_scheduler_roundtrip_exact']and arch['sampler_cursor_exact']
            oldfull=read(OLD/f'fullstate-{n}/preflight.json');assert pre['checkpoint_sha256']==oldfull['checkpoint_sha256']
            assert digest(run/'last.ckpt')==digest(arch['archived_checkpoint'])==arch['checkpoint_sha256']==full['checkpoint_sha256']
            state=read(ROOT/f'transition-{n}/state-audit.json');cal=read(ROOT/f'calibrated-{n}/calibration.json')
            weights=run/'model-step0002000/weights.pt';cw=ROOT/f'calibrated-{n}/model/weights.pt'
            assert digest(weights)==arch['portable_sha256']==full['weights_sha256']==state['weights_sha256']==cal['original_weights_sha256']
            assert digest(cw)==cal['export_weights_sha256']
            assert state['step']==cal['step']==2000 and state['cases']==64
            assert digest(run/'model-step0002000/config.json')==cal['source_config_sha256']==digest(ROOT/f'calibrated-{n}/model/config.json')
            assert cal['source_sha256']==digest(jobs['calibration-'+n]['command'][3])
            raw=torch.load(weights,map_location='cpu',weights_only=True);calibrated=torch.load(cw,map_location='cpu',weights_only=True)
            assert raw.keys()==calibrated.keys();changed={k for k,v in raw.items()if not torch.equal(v,calibrated[k])}
            assert changed==BUFFERS==set(cal['changed_buffers'])and cal['all_other_state_bitexact']and cal['training_only']
            assert all(int(raw[p+'.num_batches_tracked'])==int(calibrated[p+'.num_batches_tracked'])==2000 for p in ['projector.net.1','pred_proj.net.1'])
            del raw,calibrated
            oldrow=next(x for x in oldsummary['rows']if x['task']==task and x['seed']==seed)
            assert digest(cal['manifest'])==cal['manifest_sha256']==oldrow['calibration']['manifest_sha256']
            assert [x['vectors']for x in cal['statistics']]==[1024,896]
            assert all(x['num_batches_tracked']==2000 for x in cal['statistics'])
            native=read(ROOT/f'planning-{n}/metrics.json');bn=read(ROOT/f'planning-calibrated-{n}/metrics.json')
            oldnative=read(OLD/f'planning-{n}/metrics.json');oldbn=read(OLD_REPORT/f'planning-calibrated-{n}/metrics.json')
            for x in [bn,oldnative,oldbn]:compare(native,x)
            assert native['checkpoint_audit']['strict_equal']and bn['checkpoint_audit']['strict_equal']
            assert native['checkpoint']==str(weights.parent)and bn['checkpoint']==str(cw.parent)
            assert native['checkpoint_audit']['parameters']==bn['checkpoint_audit']['parameters']==read(run/'model_size.json')['total']
            oldstate=read(OLD/f'transition-{n}/state-audit.json');assert state['manifest_sha256']==oldstate['manifest_sha256']
            status=read(run/'status.json');stage=read(run/'stage-complete.json');assert status['step']==2000 and status['state']=='finished'
            assert stage['initial_step']==600 and stage['stop_step']==2000 and stage['stage_complete']and stage['scheduler_total_steps']==124910
            row=dict(task=task,seed=seed,id=n,native_success=native['metrics']['success_rate'],calibrated_success=bn['metrics']['success_rate'],
                native_outcomes=native['metrics']['episode_successes'],calibrated_outcomes=bn['metrics']['episode_successes'],
                old_native_success=oldrow['original_success'],old_calibrated_success=oldrow['calibrated_success'],old_native_outcomes=oldrow['original_outcomes'],old_calibrated_outcomes=oldrow['calibrated_outcomes'],
                training=status,training600=oldrow['training'],fullstate_metrics=full['metrics'],fullstate_metrics600=oldrow['eval_two_case_metrics'],
                state=summarize_state(state),state600=summarize_state(oldstate),calibration_replay=cal['replay'],calibration_replay600=oldrow['calibration']['replay'],
                calibration_statistics=[compact(x)for x in cal['statistics']],calibration_manifest=artifact(cal['manifest']),
                archive={k:v for k,v in arch.items()if k!='optimizer_group_names'},optimizer_order_sha256=__import__('hashlib').sha256(json.dumps(arch['optimizer_group_names']).encode()).hexdigest(),
                direct_buffer_only_check=True,changed_buffers=sorted(changed),fullstate_passed=True,gradients=full['gradients'],model_size=read(run/'model_size.json'),
                planning_signature={k:native[k]for k in ['manifest_sha256','pairs','solver','plan','protocol','normalization','split']},
                artifacts={key:artifact(path)for key,path in dict(config=run/'config.json',archive=ROOT/f'archive-{n}/fullstate.json',fullstate=ROOT/f'fullstate-{n}/preflight.json',state=ROOT/f'transition-{n}/state-audit.json',calibration=ROOT/f'calibrated-{n}/calibration.json',native_planning=ROOT/f'planning-{n}/metrics.json',calibrated_planning=ROOT/f'planning-calibrated-{n}/metrics.json').items()})
            for label,left,right in [('BN2000-minus-raw2000',row['calibrated_outcomes'],row['native_outcomes']),('raw2000-minus-raw600',row['native_outcomes'],row['old_native_outcomes']),('BN2000-minus-BN600',row['calibrated_outcomes'],row['old_calibrated_outcomes'])]:
                comparisons.append(dict(task=task,seed=seed,label=label,**paired_outcomes(left,right)))
            controls=[x for x in refs['rows']if x['task']==task and x['seed']==seed and x['step']==2000]
            controls +=[x for x in refs['baselines']if x['task']==task and x['seed']==seed and x['method']=='temporal-forecast']
            assert len(controls)==4
            for c in controls:
                label='TF'if c.get('method')=='temporal-forecast'else 'F-'+c['arm']
                arts=c.get('source_artifacts',c.get('artifacts'));pa=arts['planning'];assert digest(pa['path'])==pa['sha256'];m=read(pa['path']);compare(native,m)
                assert c['episode_successes']==m['metrics']['episode_successes']
                assert abs(c['success_rate']-m['metrics']['success_rate'])<1e-10  # e.g.14 vs14.000000000000002
                for key in ['data','dataset_kind','seed','batch_size','max_steps']:assert cfg[key]==c['config'][key]
                for key in ['window','action_dim']:assert cfg['model'][key]==c['config']['model'][key]
                reference_rows.append(dict(task=task,seed=seed,label=label,success=c['success_rate'],outcomes=c['episode_successes'],model_size=c['model_size'],planning=pa))
                for variant,values in [('raw',row['native_outcomes']),('BN',row['calibrated_outcomes'])]:comparisons.append(dict(task=task,seed=seed,label=variant+'2000-minus-'+label,**paired_outcomes(values,c['episode_successes'])))
            rows.append(row)
    joint=[]
    for task in ['tworoom','cube']:
        rr=[r for r in rows if r['task']==task]
        for label,lk,rk in [('BN2000-minus-raw2000','calibrated_outcomes','native_outcomes'),('raw2000-minus-raw600','native_outcomes','old_native_outcomes'),('BN2000-minus-BN600','calibrated_outcomes','old_calibrated_outcomes')]:
            joint.append(dict(task=task,label=label,**joint_compare([(r[lk],r[rk])for r in rr])))
        for label in ['TF','F-identity','F-unconstrained','F-bounded']:
            pairs=[(r['calibrated_outcomes'],next(x['outcomes']for x in reference_rows if x['task']==task and x['seed']==r['seed']and x['label']==label))for r in rr]
            joint.append(dict(task=task,label='BN2000-minus-'+label,**joint_compare(pairs)))
    result=dict(rows=rows,references=reference_rows,per_seed_comparisons=comparisons,joint_case_comparisons=joint,
        completed_endpoints=28,all_exit_zero=True,source=source,verified_critical_source=critical,helper_sources=helper_sources,source_manifest=artifact(source['manifest']),
        inputs=dict(plan=artifact(ROOT/'plan.json'),protocol=artifact(ROOT/'protocol.json'),parent=artifact(OLD_REPORT/'results.json'),references=artifact(REF)),collector=artifact(__file__),
        scope='All four seeds retained. Native2000 resumes original fullstate600. Train-only preciseBN is a separate posthoc deployment variant, not the official raw recipe or a sample-level new objective. Same gradient-exposure updates/RGB/actions and exact50case planner; capacity, architecture, objective and compute differ. Raw64window state audits are not calibrated state audits. No extra assay or GPU run.')
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');report(result,out)
    print(json.dumps(dict(output=str(out),success=[{k:r[k]for k in ['task','seed','native_success','calibrated_success']}for r in rows],joint=joint)),flush=True)

def report(d,out):
    lines=['# Native LeWM2000: learning progress and fixed train-only BN control','',
        'The 2000-step model has learned substantially more useful control than at600. Fixed train-only BN calibration reveals TwoRoom82/76% and Cube14/16%; raw deployment remains8/0% and12/6%. The normalization mismatch hides the TwoRoom learning signal. It is not evidence of a broken native loss, and the posthoc calibrated variant must remain separate from literal native inference.','',
        '| Task | Seed | Raw600 | BN600 | Raw2000 | BN2000 | Fidentity2000 | Ffree2000 | Fbounded2000 | TF2000 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in d['rows']:
        vals=[next(x['success']for x in d['references']if x['task']==r['task']and x['seed']==r['seed']and x['label']==q)for q in ['F-identity','F-unconstrained','F-bounded','TF']]
        numbers=[r[k]for k in ['old_native_success','old_calibrated_success','native_success','calibrated_success']]+vals
        lines.append('| '+r['task']+' | '+str(r['seed'])+' | '+' | '.join(f'{x:g}%'for x in numbers)+' |')
    lines+=['','All native600/2000/raw/BN and F/TF2000 comparisons use exactly the same ordered50cases, case hash, solver, planner horizon/budget, strict success semantics and training-only action normalization. Equal2000 updates mean256,000 clips with8RGB frames/7conditioned action blocks each. Native~18.04M parameters and192-state coordinates differ from TF/F~51.1M and64-state coordinates; F also freezes its encoder. This is matched data/update/evaluator exposure, not equal capacity or compute. PreciseBN adds fixed128training clips (1024 RGB frames) without gradients.','',
        '| Task | Contrast | Mean difference | Conditional same-case95% interval |','|---|---|---:|---|']
    for r in d['joint_case_comparisons']:
        lo,hi=r['case_bootstrap_ci95_pp'];lines.append(f"| {r['task']} | {r['label']} | {r['difference_pp']:+.1f} pp | [{lo:.1f}, {hi:.1f}] pp |")
    lines+=['','Intervals resample the50cases jointly across the two fixed training seeds. They do not establish stability over a population of training seeds. The calibrated TwoRoom result is competitive with F2000 and far above this TF2000 checkpoint; the small Cube differences do not establish superiority over F or TF. The fixed-weight raw/BN contrast isolates the deployment-statistics intervention. LeWM-versus-TF/F changes architecture, objective and capacity jointly, so it cannot attribute the gap specifically to SIGReg or a single learned mechanism.','',
        '| Task | Seed | Raw clean/copy600→2000 | Raw teacher/copy2000 | Raw clean/shifted-observed-copy2000 | Train preciseBN one-step MSE600→2000 |','|---|---:|---|---:|---:|---|']
    for r in d['rows']:
        s=r['state']['conditions'];old=r['state600']['conditions'];cal=r['calibration_replay'];oc=r['calibration_replay600']
        lines.append(f"| {r['task']} | {r['seed']} | {old['clean_rollout']['normalized_by_initial_variation']:.3f}→{s['clean_rollout']['normalized_by_initial_variation']:.3f} | {s['teacher_forced_prefix']['normalized_by_initial_variation']:.3f} | {s['clean_rollout']['mse']/s['shifted_observed_carry']['mse']:.3f} | {oc['fixed_one_step_mse']:.4f}→{cal['fixed_one_step_mse']:.4f} |")
    lines+=['','Cube raw clean forecasts now beat retaining the initial state for both seeds. Wrong initial states and wrong actions substantially increase error, supporting learned action-conditioned dynamics rather than a copy shortcut. TwoRoom raw forecast errors remain dominated by deployment inconsistency; its wrong-action error effects are mixed, so raw sensitivity alone cannot claim useful prediction. Teacher forcing and shifted-observed copy receive extra real observations and are diagnostic controls. Only raw2000 has the64window audit; calibrated planning success and training replay must not be relabeled as calibrated heldout rollout MSE. MSE is in each learned coordinate system, not a common physical error unit.','',
        'The calibration rule was fixed before2000 outcomes: identical training128clip manifests as600 and shared across seeds, observation BN1024vectors first, then prediction BN896vectors, global actual-count mean/unbiased variance, FP32 evaluation with dropout off. Direct CPU tensor comparison verifies exactly four running_mean/running_var buffers changed, all other state bitexact, and num_batches_tracked remains2000. Fixed vs batch-BN prediction discrepancies are below5e-7; single-clip vs full-batch replay errors are below3e-12. No validation examples or physical/optimal-action labels fit calibration.','',
        'Integrity: all28 endpoints exit0; four2000 checkpoint archives match last.ckpt hashes, portable tensors match full-state and state audits, native AdamW has one group/297 parameter tensors with exact optimizer/scheduler roundtrip, scheduler step2000 and sample cursor256000. Config semantics match600; original fullstate600 was resumed, never the calibrated export. Native training source remains7a1ab4f; helper source0934d3c is independently hash-checked. Native MSE+.09SIGReg, target gradients, BN, Cube smoothed_dim10, history7/window8 and per-update scheduler are preserved. All four gradient audits pass.','',
        'Recommendation: a bounded continuation of all four original native2000 full states to6000 is justified as a fair baseline budget, not a new method promotion. Actual Cube state/action learning and TwoRoom deployment performance improve strongly beyond the1249-step warmup. Keep the native objective/trainer/scheduler unchanged, preserve raw and the same predeclared train-only BN variant at the endpoint, and stop for review at6000. No launch follows from this report. Do not claim final baseline maturity, broad two-task superiority, or an unmodified official-recipe reproduction.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
