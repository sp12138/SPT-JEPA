"""Complete-only CPU review of native6000 and its separately authorized raw fresh cases."""
import argparse
import json
from pathlib import Path
import torch

from experiments.ideas_validate_20261031.collect import digest,paired_outcomes
from experiments.native_baseline_20261031.prepare2000 import semantic
from experiments.native_baseline_20261031.review600 import compare
from experiments.native_baseline_20261031.review2000 import BUFFERS,artifact,compact,joint_compare,summarize_state

B=Path('/efs/prescottsun/SPT-JEPA')
ROOT=B/'runs/native_lewm6000_20261031'
FRESH=B/'runs/native_rawfresh6000_20261031'
PARENT=B/'runs/native_lewm2000_20261031'
OLD_REPORT=B/'reports/native-lewm2000-20261031/results.json'
REF=B/'reports/carrier6000-20261031/results.json'
def read(p):return json.loads(Path(p).read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default=str(B/'reports/native-lewm6000-20261031'));a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);plan=read(ROOT/'plan.json');freshplan=read(FRESH/'plan.json');source=read(ROOT/'source-record.json');helper=read(ROOT/'helper-source-record.json')
    assert len(plan['jobs'])==32 and len(freshplan['jobs'])==4
    for root,p in [(ROOT,plan),(FRESH,freshplan)]:
        for job in p['jobs']:
            assert read(root/'exits'/(job['id']+'.json'))['returncode']==0
            assert job['source']==source
    jobs={j['id']:j for j in plan['jobs']};fj={j['id']:j for j in freshplan['jobs']}
    for src,names in [(source,['as_jepa_lewm/native_lewm.py','as_jepa_lewm/train.py','as_jepa_tools/train_stage.py','as_jepa_tools/benchmark_evaluate.py']),
                      (helper,['experiments/native_baseline_20261031/prepare6000.py','experiments/native_baseline_20261031/bn_calibrate.py','experiments/carrier_progress_20261031/prepare_next.py'])]:
        man=read(src['manifest'])
        for name in names:assert digest(Path(src['source'])/name)==man['files'][name]
    parent=read(OLD_REPORT);refs=read(REF);rows=[];reference_rows=[];paired=[]
    for task in ['tworoom','cube']:
        for seed in [1,2]:
            n=f'{task}-native-lewm-seed{seed}';run=ROOT/n;cfg=read(run/'config.json');pr=PARENT/n
            assert semantic(cfg)==semantic(read(pr/'config.json'))
            assert digest(run/'config.json')==jobs['learn6000-'+n]['config_sha256']
            assert cfg['retain_steps']==[4000,6000]and (run/'model-step0004000/weights.pt').exists()
            arch=read(ROOT/f'archive-{n}/fullstate.json');full=read(ROOT/f'fullstate-{n}/preflight.json');pre=read(run/'pre-resume-fullstate.json');old=next(x for x in parent['rows']if x['id']==n)
            assert arch['passed']and full['passed']and pre['passed']and arch['step']==arch['scheduler_last_epoch']==6000
            assert arch['campaign_samples']==768000 and arch['sampler']['passed']and arch['sampler_at6000']['passed']
            assert arch['optimizer_parameter_order_exact']and arch['optimizer_scheduler_roundtrip_exact']and arch['optimizer_group_count']==1 and arch['optimizer_state_tensors']==297
            assert pre['checkpoint_sha256']==old['archive']['checkpoint_sha256']
            assert digest(run/'last.ckpt')==digest(arch['archived_checkpoint'])==arch['checkpoint_sha256']==full['checkpoint_sha256']
            assert digest(arch['initial_weights'])==arch['initial_weights_sha256']and '/runs/native_lewm600_20261031/'in arch['initial_weights']
            status=read(run/'status.json');stage=read(run/'stage-complete.json');assert status['step']==6000 and status['state']=='finished'
            assert stage['initial_step']==2000 and stage['stop_step']==6000 and stage['stage_complete']and stage['scheduler_total_steps']==124910
            state=read(ROOT/f'transition-{n}/state-audit.json');cal=read(ROOT/f'calibrated-{n}/calibration.json');w=run/'model-step0006000/weights.pt';cw=ROOT/f'calibrated-{n}/model/weights.pt'
            sha=digest(w);assert sha==arch['portable_sha256']==full['weights_sha256']==state['weights_sha256']==cal['original_weights_sha256']==fj['fresh-planning-raw-'+n]['checkpoint_weights_sha256']
            assert digest(cw)==cal['export_weights_sha256']
            assert state['step']==cal['step']==6000 and state['cases']==64 and state['manifest_sha256']==old['state']['manifest_sha256']
            assert digest(cal['manifest'])==cal['manifest_sha256']==old['calibration_manifest']['sha256']
            assert digest(jobs['calibration-'+n]['command'][3])==cal['source_sha256']
            assert digest(w.parent/'config.json')==digest(cw.parent/'config.json')==cal['source_config_sha256']
            raw=torch.load(w,map_location='cpu',weights_only=True);bn=torch.load(cw,map_location='cpu',weights_only=True)
            assert raw.keys()==bn.keys();changed={k for k,v in raw.items()if not torch.equal(v,bn[k])};assert changed==BUFFERS==set(cal['changed_buffers'])
            assert all(int(raw[p+'.num_batches_tracked'])==int(bn[p+'.num_batches_tracked'])==6000 for p in ['projector.net.1','pred_proj.net.1']);del raw,bn
            assert cal['all_other_state_bitexact']and cal['training_only']and [x['vectors']for x in cal['statistics']]==[1024,896]
            metrics={('primary','raw'):read(ROOT/f'planning-{n}/metrics.json'),('primary','cal'):read(ROOT/f'planning-calibrated-{n}/metrics.json'),('fresh','cal'):read(ROOT/f'fresh-planning-calibrated-{n}/metrics.json'),('fresh','raw'):read(FRESH/f'fresh-planning-raw-{n}/metrics.json')}
            for case in ['primary','fresh']:
                compare(metrics[(case,'raw')],metrics[(case,'cal')])
                for variant in ['raw','cal']:
                    m=metrics[(case,variant)];assert m['checkpoint_audit']['strict_equal']and m['checkpoint']==str(w.parent if variant=='raw'else cw.parent)
            for variant,phase in [('raw','planning'),('cal','planning-calibrated')]:compare(metrics[('primary',variant)],read(PARENT/f'{phase}-{n}/metrics.json'))
            row=dict(id=n,task=task,seed=seed,training=status,training2000=old['training'],state=summarize_state(state),state2000=old['state'],fullstate_passed=True,
                archive={k:v for k,v in arch.items()if k!='optimizer_group_names'},gradients=full['gradients'],fullstate_metrics=full['metrics'],
                direct_buffer_only_check=True,calibration_replay=cal['replay'],calibration_replay2000=old['calibration_replay'],calibration_statistics=[compact(x)for x in cal['statistics']],
                model_size=read(run/'model_size.json'),planning={},primary2000={'raw':dict(success=old['native_success'],outcomes=old['native_outcomes']),'cal':dict(success=old['calibrated_success'],outcomes=old['calibrated_outcomes'])},
                artifacts={k:artifact(p)for k,p in dict(config=run/'config.json',archive=ROOT/f'archive-{n}/fullstate.json',state=ROOT/f'transition-{n}/state-audit.json',fullstate=ROOT/f'fullstate-{n}/preflight.json',calibration=ROOT/f'calibrated-{n}/calibration.json').items()})
            for case in ['primary','fresh']:
                row['planning'][case]={v:dict(success=metrics[(case,v)]['metrics']['success_rate'],outcomes=metrics[(case,v)]['metrics']['episode_successes'])for v in ['raw','cal']}
                row['planning'][case]['signature']={k:metrics[(case,'raw')][k]for k in ['manifest_sha256','pairs','solver','plan','protocol','normalization','split']}
                paired.append(dict(task=task,seed=seed,case_set=case,label='cal-minus-raw',**paired_outcomes(row['planning'][case]['cal']['outcomes'],row['planning'][case]['raw']['outcomes'])))
            for v in ['raw','cal']:paired.append(dict(task=task,seed=seed,case_set='primary',label=v+'6000-minus-'+v+'2000',**paired_outcomes(row['planning']['primary'][v]['outcomes'],row['primary2000'][v]['outcomes'])))
            for ref in [x for x in refs['rows']if x['task']==task and x['seed']==seed]:
                label='TF'if ref['family']=='temporal-forecast'else 'F-'+ref['arm'];record=dict(task=task,seed=seed,label=label,model_size=ref['model_size'],planning={})
                for k in ['data','dataset_kind','seed','batch_size','max_steps']:assert cfg[k]==ref['config'][k]
                for k in ['window','action_dim']:assert cfg['model'][k]==ref['config']['model'][k]
                for case in ['primary','fresh']:
                    a=ref['planning'][case]['source'];assert digest(a['path'])==a['sha256'];m=read(a['path']);compare(metrics[(case,'raw')],m)
                    assert ref['planning'][case]['episode_successes']==m['metrics']['episode_successes']
                    record['planning'][case]=dict(success=m['metrics']['success_rate'],outcomes=m['metrics']['episode_successes'],artifact=a)
                    for v in ['raw','cal']:paired.append(dict(task=task,seed=seed,case_set=case,label=v+'-minus-'+label,**paired_outcomes(row['planning'][case][v]['outcomes'],m['metrics']['episode_successes'])))
                reference_rows.append(record)
            rows.append(row)
    joint=[]
    for task in ['tworoom','cube']:
        rr=[r for r in rows if r['task']==task]
        for case in ['primary','fresh']:
            joint.append(dict(task=task,case_set=case,label='cal-minus-raw',**joint_compare([(r['planning'][case]['cal']['outcomes'],r['planning'][case]['raw']['outcomes'])for r in rr])))
            for label in sorted({x['label']for x in reference_rows if x['task']==task}):
                for v in ['raw','cal']:
                    pairs=[(r['planning'][case][v]['outcomes'],next(x['planning'][case]['outcomes']for x in reference_rows if x['task']==task and x['seed']==r['seed']and x['label']==label))for r in rr]
                    joint.append(dict(task=task,case_set=case,label=v+'-minus-'+label,**joint_compare(pairs)))
        for v in ['raw','cal']:joint.append(dict(task=task,case_set='primary',label=v+'6000-minus-'+v+'2000',**joint_compare([(r['planning']['primary'][v]['outcomes'],r['primary2000'][v]['outcomes'])for r in rr])))
    result=dict(rows=rows,references=reference_rows,per_seed_comparisons=paired,joint_case_comparisons=joint,completed_endpoints={'native6000':32,'rawfresh':4},source=source,helper_source=helper,
        inputs={k:artifact(p)for k,p in dict(plan=ROOT/'plan.json',rawfresh=FRESH/'plan.json',parent2000=OLD_REPORT,reference6000=REF).items()},
        dispatcher_closures=dict(native6000=read(ROOT/'closed-dev2-58484.json'),rawfresh=read(FRESH/'closed-dispatcher.json')),collector=artifact(__file__),
        scope='Both uniform variants retained across both tasks: raw native and fixed train128 preciseBN. No task-specific best-of variant. Same gradient exposure and exact within-set cases/planner; architecture/capacity/compute differ. Primary and fresh sets are not pooled or paired across sets. Raw state audits only; calibration replay is train-only. No new training or assays.')
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');report(result,out)
    print(json.dumps(dict(output=str(out),native=[dict(task=r['task'],seed=r['seed'],planning={case:{v:r['planning'][case][v]['success']for v in ['raw','cal']}for case in ['primary','fresh']})for r in rows]),indent=2))

def report(d,out):
    lines=['# Native LeWM6000: retain raw and calibrated variants on both tasks','',
        'The raw native world model improves substantially by6000. Its Cube advantage over calibration on primary cases does not replicate on fresh cases: raw24/22% versus calibrated16/18% becomes raw12/14% versus calibrated20/16%. Train-only BN recalibration remains a separate uniform variant. Choosing calibrated TwoRoom and raw Cube after seeing primary outcomes would create a task-specific selected system, not a single evaluated method.','',
        '| Task | Seed | Raw2000 primary | Cal2000 primary | Raw6000 primary | Cal6000 primary | Raw6000 fresh | Cal6000 fresh |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in d['rows']:
        vals=[r['primary2000'][v]['success']for v in ['raw','cal']]+[r['planning'][c][v]['success']for c in ['primary','fresh']for v in ['raw','cal']]
        lines.append('| '+r['task']+' | '+str(r['seed'])+' | '+' | '.join(f'{v:g}%'for v in vals)+' |')
    lines+=['','| Task | Reference6000 | Primary seed1/2 | Fresh seed1/2 |','|---|---|---|---|']
    for t in ['tworoom','cube']:
        for label in sorted({x['label']for x in d['references']if x['task']==t}):
            rr=sorted([x for x in d['references']if x['task']==t and x['label']==label],key=lambda x:x['seed'])
            vals=[' / '.join(f"{r['planning'][c]['success']:g}%"for r in rr)for c in ['primary','fresh']];lines.append(f'| {t} | {label} | {vals[0]} | {vals[1]} |')
    lines+=['','All comparisons verify identical ordered50cases, case hashes, planner, budget, strict success and normalization within each case set. Primary and fresh remain separate; fresh means new pairs from the existing validation episode pool, not an independent test set. Cube Ffree/Fbounded were not continued to6000, so no missing result is filled with its2000 score. Native~18.04M parameters versus old F/TF~51.1M; architectures, latent dimensions and frozen/trainable capacity differ. This is matched6000 gradient-update exposure, not equal capacity or compute. BN calibration adds the same fixed128 training clips without gradients.','',
        '| Task | Case set | Cal − raw mean | Same-case conditional95% interval |','|---|---|---:|---|']
    for r in d['joint_case_comparisons']:
        if r['label']!='cal-minus-raw':continue
        lo,hi=r['case_bootstrap_ci95_pp'];lines.append(f"| {r['task']} | {r['case_set']} | {r['difference_pp']:+.1f}pp | [{lo:.1f}, {hi:.1f}]pp |")
    lines+=['','Intervals resample cases jointly across the two fixed training seeds; they are not uncertainty over the training-seed population. Full per-seed and F/TF paired results are in results.json.','',
        '| Task | Seed | Raw clean / initial copy2000→6000 | Raw clean / shifted-observed copy6000 | Wrong-order MSE / clean MSE6000 |','|---|---:|---|---:|---:|']
    for r in d['rows']:
        s=r['state']['conditions'];o=r['state2000']['conditions'];lines.append(f"| {r['task']} | {r['seed']} | {o['clean_rollout']['normalized_by_initial_variation']:.3f}→{s['clean_rollout']['normalized_by_initial_variation']:.3f} | {s['clean_rollout']['mse']/s['shifted_observed_carry']['mse']:.3f} | {s['wrong_action_order']['mse']/s['clean_rollout']['mse']:.3f} |")
    lines+=['','All four raw models now predict better than retaining the initial state and better than the shifted-observed copy control; wrong initial states or wrong action order substantially increase error. This is useful action-conditioned latent learning. Teacher/shifted-observed controls receive extra real observations. The same-case planning gains supply separate task evidence; latent MSE is not physical accuracy or a common scale across architectures.','',
        'The large TwoRoom raw deployment deficit at2000 largely resolves during continued native training. Cube calibration effects change with the evaluation case set; primary-only raw superiority is therefore not a stable conclusion. Precise training moments are not a universally beneficial controller correction. This experiment does not isolate whether finite calibration support, training-versus-planning context distributions or another mechanism explains the reversal. No tuning or new assay is added to select among them.','',
        'Integrity:32 original endpoints and4 independent raw-fresh endpoints completed0. Full6000 archives match last.ckpt/portable hashes; all four full-state/gradient checks pass, native optimizer order and scheduler roundtrip pass, cursor768000 and cross-epoch sampler checks pass. Original2000 full states were resumed;4000/6000 exports and source7a remain preserved. Direct CPU comparison confirms calibration changes exactly four BN mean/variance buffers, not parameters or tracked counts; fixed training128 manifests and1024/896 vector counts are unchanged. Calibrated replay is training-only; there is no calibrated heldout state audit.','',
        'Both idle owned dispatchers are closed and their roots held. No further training is launched. Retain raw native as its original deployment result and preciseBN as a fully reported separate variant across both tasks; do not publish a posthoc task-wise maximum.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
