"""Complete-only CPU review of two native Cube maturity references; no GPU work."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import torch

from experiments.ideas_validate_20261031.collect import digest,paired_outcomes
from experiments.native_baseline_20261031.prepare2000 import semantic
from experiments.native_baseline_20261031.review600 import compare
from experiments.native_baseline_20261031.review2000 import BUFFERS,joint_compare,summarize_state

B=Path('/efs/prescottsun/SPT-JEPA')
ROOT=B/'runs/native_cube12000_20261031'
PARENT=B/'runs/native_lewm6000_20261031'
FRESH=B/'runs/native_rawfresh6000_20261031'
TF=B/'runs/cube_tf12000_20261031'
OUT=B/'reports/native-cube12000-20261031'


def read(p):return json.loads(Path(p).read_text())
@lru_cache(None)
def sha(p):return digest(p)
def artifact(p):return dict(path=str(p),sha256=sha(str(p)))
def verify_source(s):
    for key in ['manifest','runtime_manifest']:
        m=read(s[key])
        for rel,h in m['files'].items():assert sha(str(Path(m['source'])/rel))==h
    return dict(source=s,manifest=artifact(s['manifest']),runtime_manifest=artifact(s['runtime_manifest']))


def main():
    torch.set_num_threads(2)
    plan,protocol,source,helper=[read(ROOT/n) for n in ['plan.json','protocol.json','source-record.json','helper-source-record.json']]
    assert len(plan['jobs'])==26 and (ROOT/'HOLD_DISPATCH').exists()
    closure=read(ROOT/'closed-owned-processes.json');assert closure['closed']
    activation=read(ROOT/'activation.json')
    assert sha(str(ROOT/'plan.json'))==activation['plan_sha256'] and sha(str(ROOT/'protocol.json'))==activation['protocol_sha256']
    assert source['source_commit']=='7a1ab4feb1a44cb3a6a37f5f7f2ad6a0e110487f'
    assert helper['source_commit']=='a34257e12e31ecfe0c1c095bf0410fd9f1caa44f'
    oldhelper=read(PARENT/'helper-source-record.json');assert oldhelper['source_commit']=='a8b77922f26a7ee9397d4bbad78e65649c361eef'
    sources=[verify_source(s) for s in [source,helper,oldhelper]]
    script=Path(helper['source'])/'experiments/native_baseline_20261031/prepare12000.py'
    jobs={j['id']:j for j in plan['jobs']}
    for j in jobs.values():
        assert read(ROOT/'exits'/(j['id']+'.json'))['returncode']==0 and j['source']==source
        if j['phase'].startswith('archive'):assert j['command'][3]==str(script)
    oldpath=B/'reports/native-lewm6000-20261031/results.json';old_report=read(oldpath)
    tfpath=B/'reports/cube-tf12000-20261031/results.json';tf=read(tfpath)
    assert tf['all_checks_passed'] and tf['completed_endpoint_count']==10
    assert read(tfpath.with_name('summary.json'))['full_results']['sha256']==sha(str(tfpath))
    sources.extend(verify_source(s) for s in [tf['source'],tf['helper']])
    rows=[];comparisons=[]
    for seed in [1,2]:
        name=f'cube-native-lewm-seed{seed}';run,parent=ROOT/name,PARENT/name
        cfg=read(run/'config.json');assert semantic(cfg)==semantic(read(parent/'config.json'))
        assert cfg['retain_steps']==[8000,10000,12000] and cfg['max_steps']==124910 and cfg['batch_size']==128
        config_sha=sha(str(run/'config.json'));assert all(jobs[f'learn{s}-{name}']['config_sha256']==config_sha for s in [8000,10000,12000])
        pre=read(run/'pre-resume-fullstate.json');assert pre['passed'] and pre['copy_byte_identical'] and pre['step']==6000
        assert pre['checkpoint_sha256']==sha(str(parent/'last.ckpt'))==sha(str(parent/'checkpoint-step0006000.ckpt'))
        archives=[];shapes=None
        for step,initial in [(8000,6000),(10000,8000),(12000,10000)]:
            ap=ROOT/f'archive{step}-{name}/fullstate.json';a=read(ap)
            assert a['passed'] and all(a['checks'].values()) and a['step']==a['scheduler_last_epoch']==step and a['campaign_samples']==step*128
            assert a['optimizer_group_count']==1 and a['optimizer_state_tensors']==297 and a['optimizer_parameter_order_exact'] and a['optimizer_scheduler_roundtrip_exact']
            assert all(v['passed'] for v in a['sampler_at_stage_boundaries'].values())
            stage=a['stage_complete'];assert stage['stage_complete'] and stage['initial_step']==initial and stage['stop_step']==step and stage['scheduler_total_steps']==124910 and stage['config_sha256']==config_sha
            assert a['model_source']==source and a['reused_helpers']==oldhelper and a['archive_helper_sha256']==sha(str(script))
            assert a['native_dispatch']['id']==f'learn{step}-{name}' and a['native_dispatch']['source']==source
            archive=run/f'checkpoint-step{step:07d}.ckpt';assert str(archive)==a['archived_checkpoint'] and sha(str(archive))==a['checkpoint_sha256']
            weights=run/f'model-step{step:07d}/weights.pt';assert sha(str(weights))==a['portable_sha256']
            state=torch.load(weights,map_location='cpu',weights_only=True);assert all(torch.isfinite(v).all() for v in state.values())
            current={k:list(v.shape) for k,v in state.items()}
            if shapes is not None:assert current==shapes
            shapes=current
            assert int(state['projector.net.1.num_batches_tracked'])==int(state['pred_proj.net.1.num_batches_tracked'])==step
            archives.append(dict(**{k:v for k,v in a.items() if k!='optimizer_group_names'},audit_artifact=artifact(ap),portable_finite=True,parameter_shapes_unchanged=True))
            del state
        assert sha(str(run/'last.ckpt'))==archives[-1]['checkpoint_sha256']
        full=read(ROOT/f'fullstate-{name}/preflight.json');assert full['passed'] and full['checkpoint_sha256']==archives[-1]['checkpoint_sha256']
        old=next(r for r in old_report['rows'] if r['id']==name)
        state=read(ROOT/f'transition-{name}/state-audit.json');assert state['step']==12000 and state['cases']==64 and state['manifest_sha256']==old['state']['manifest_sha256']
        rawpath=run/'model-step0012000/weights.pt';calroot=ROOT/f'calibrated-{name}';cal=read(calroot/'calibration.json')
        assert sha(str(rawpath))==archives[-1]['portable_sha256']==full['weights_sha256']==state['weights_sha256']==cal['original_weights_sha256']
        assert cal['step']==12000 and cal['training_only'] and cal['all_other_state_bitexact']
        assert sha(cal['manifest'])==cal['manifest_sha256']==protocol['fixed_calibration']['sha256']
        assert cal['manifest_sha256']==read(PARENT/f'calibrated-{name}/calibration.json')['manifest_sha256']
        assert [s['vectors'] for s in cal['statistics']]==[1024,896]
        assert cal['source_sha256']==sha(jobs['calibration-'+name]['command'][3])
        assert sha(str(rawpath.parent/'config.json'))==sha(str(calroot/'model/config.json'))==cal['source_config_sha256']
        raw=torch.load(rawpath,map_location='cpu',weights_only=True);bn=torch.load(calroot/'model/weights.pt',map_location='cpu',weights_only=True)
        assert raw.keys()==bn.keys() and {k for k,v in raw.items() if not torch.equal(v,bn[k])}==BUFFERS==set(cal['changed_buffers'])
        assert sha(str(calroot/'model/weights.pt'))==cal['export_weights_sha256']
        assert all(int(raw[p+'.num_batches_tracked'])==int(bn[p+'.num_batches_tracked'])==12000 for p in ['projector.net.1','pred_proj.net.1'])
        del raw,bn
        tfrow=next(r for r in tf['rows'] if r['seed']==seed);tfname=f'cube-temporal-forecast-seed{seed}';tfcfg=read(TF/tfname/'config.json')
        assert sha(str(TF/tfname/'config.json'))==tfrow['artifacts']['config']['sha256']
        assert sha(str(TF/tfname/'model-step0012000/weights.pt'))==tfrow['fullstate']['portable_sha256']
        assert sha(tfrow['fullstate']['archived_checkpoint'])==tfrow['fullstate']['checkpoint_sha256']
        for k in ['data','dataset_kind','seed','batch_size','max_steps']:assert cfg[k]==tfcfg[k]
        for k in ['window','action_dim']:assert cfg['model'][k]==tfcfg['model'][k]
        assert tfrow['fullstate']['campaign_samples']==1536000 and tfrow['fullstate']['step']==12000
        training=read(run/'status.json');assert training['state']=='finished' and training['step']==12000
        size=read(run/'model_size.json');assert size==old['model_size'] and size['trainable']==18035396
        row=dict(id=name,seed=seed,training=training,training6000=old['training'],archives=archives,fullstate_passed=True,gradients=full['gradients'],
            state=summarize_state(state),state6000=old['state'],model_size=size,calibration_replay=cal['replay'],calibration_replay6000=old['calibration_replay'],
            calibration_manifest=artifact(cal['manifest']),changed_buffers=sorted(BUFFERS),buffer_only_direct_check=True,planning={},planning6000=old['planning'],tf12000={},
            artifacts=dict(config=artifact(run/'config.json'),fullstate=artifact(ROOT/f'fullstate-{name}/preflight.json'),state=artifact(ROOT/f'transition-{name}/state-audit.json'),calibration=artifact(calroot/'calibration.json')))
        for case,phase in [('primary','planning'),('fresh','fresh-planning')]:
            tfmetrics=read(tfrow['planning'][case]['new_source']['path']);assert sha(tfrow['planning'][case]['new_source']['path'])==tfrow['planning'][case]['new_source']['sha256']
            assert tfmetrics['checkpoint']==str(TF/tfname/'model-step0012000') and tfmetrics['checkpoint_audit']['strict_equal']
            row['tf12000'][case]=dict(success=tfmetrics['metrics']['success_rate'],outcomes=tfmetrics['metrics']['episode_successes'],artifact=tfrow['planning'][case]['new_source'])
            row['planning'][case]={}
            for variant,suffix in [('raw',''),('cal','-calibrated')]:
                path=ROOT/f'{phase}{suffix}-{name}/metrics.json';m=read(path);compare(m,tfmetrics)
                expected=run/'model-step0012000' if variant=='raw' else calroot/'model'
                assert m['checkpoint']==str(expected) and m['checkpoint_audit']['strict_equal']
                assert m['checkpoint_audit']['parameters']==size['total']
                command=jobs[f'{phase}{suffix}-{name}']['command'];manifest=command[command.index('--manifest')+1]
                assert sha(manifest)==jobs[f'{phase}{suffix}-{name}']['manifest_sha256']==m['manifest_sha256']
                previous=(FRESH/f'fresh-planning-raw-{name}/metrics.json') if case=='fresh' and variant=='raw' else PARENT/f'{phase}{suffix}-{name}/metrics.json'
                oldm=read(previous);compare(m,oldm);assert oldm['metrics']['episode_successes']==old['planning'][case][variant]['outcomes']
                current=m['metrics']['episode_successes'];assert len(current)==50 and np.isin(current,[0,1]).all()
                np.testing.assert_allclose(m['metrics']['success_rate'],np.mean(current)*100)
                row['planning'][case][variant]=dict(success=m['metrics']['success_rate'],outcomes=current,artifact=artifact(path),parent_artifact=artifact(previous))
                comparisons.append(dict(seed=seed,case_set=case,label=variant+'12000-minus-'+variant+'6000',**paired_outcomes(current,oldm['metrics']['episode_successes'])))
                comparisons.append(dict(seed=seed,case_set=case,label=variant+'12000-minus-TF12000',**paired_outcomes(current,tfmetrics['metrics']['episode_successes'])))
            row['planning'][case]['signature']={k:tfmetrics[k] for k in ['manifest_sha256','pairs','solver','plan','protocol','normalization','split']}
            comparisons.append(dict(seed=seed,case_set=case,label='cal-minus-raw',**paired_outcomes(row['planning'][case]['cal']['outcomes'],row['planning'][case]['raw']['outcomes'])))
        rows.append(row)
    joint=[]
    for case in ['primary','fresh']:
        for label in ['raw12000-minus-raw6000','cal12000-minus-cal6000','raw12000-minus-TF12000','cal12000-minus-TF12000','cal-minus-raw']:
            pairs=[]
            for row in rows:
                if label=='cal-minus-raw':a,b=row['planning'][case]['cal']['outcomes'],row['planning'][case]['raw']['outcomes']
                else:
                    variant=label.split('12000')[0];a=row['planning'][case][variant]['outcomes']
                    b=row['tf12000'][case]['outcomes'] if label.endswith('TF12000') else row['planning6000'][case][variant]['outcomes']
                pairs.append((a,b))
            joint.append(dict(case_set=case,label=label,**joint_compare(pairs)))
    result=dict(completed_endpoints=26,rows=rows,per_seed_comparisons=comparisons,joint_case_comparisons=joint,sources=sources,protocol=protocol,closure=closure,
        artifacts=dict(collector=artifact(__file__),plan=artifact(ROOT/'plan.json'),parent6000=artifact(oldpath),tf12000=artifact(tfpath)),
        scope='Two native Cube seeds only. Original raw fullstate6000 resumed; staged8000/10000/12000 full archives gate continuation. Raw and fixed train128 preciseBN are separate variants. Original-command50-case protocols are unchanged; no projection treatment. Equal12000 updates/window exposure is not equal architecture, capacity or compute. Bootstrap conditions on two fixed seeds and realized environment outcomes, excluding seed-population and environment repeat variance. No automatic continuation or posthoc variant maximum.')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Native Cube maturity reference at12000','',
        'All26 bounded endpoints exit0; both raw and train-only preciseBN variants are retained. All8000/10000/12000 full archives, finite portable states, optimizer/scheduler/RNG/cursor audits and source/config/case checks pass.','',
        '| Cases | Seed | Raw6000 | Cal6000 | Raw12000 | Cal12000 | TF12000 |','|---|---:|---:|---:|---:|---:|---:|']
    for case in ['primary','fresh']:
        for row in rows:
            values=[row['planning6000'][case][v]['success'] for v in ['raw','cal']]+[row['planning'][case][v]['success'] for v in ['raw','cal']]+[row['tf12000'][case]['success']]
            lines.append(f"| {case} | {row['seed']} | "+' | '.join(f'{v:g}%' for v in values)+' |')
    lines+=['','| Cases | Contrast | Mean change pp | Conditional95% interval |','|---|---|---:|---|']
    for row in joint:
        lo,hi=row['case_bootstrap_ci95_pp'];lines.append(f"| {row['case_set']} | {row['label']} | {row['difference_pp']:+.1f} | [{lo:.1f}, {hi:.1f}] |")
    lines+=['',result['scope'],'','Native has18,035,396 trainable parameters versus approximately51.1M total for TF. Native192-state and TF64-state errors are not comparable physical units. BN calibration changes exactly four running mean/variance buffers and no parameters, using the same training128 clips and1024/896 actual vectors. Historical and current outcomes are both preserved; neither variant is chosen after viewing task results.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n');print((OUT/'RESULTS.md').read_text())


if __name__=='__main__':main()
