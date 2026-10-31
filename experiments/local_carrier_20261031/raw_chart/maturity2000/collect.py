"""Completed-only CPU audit of four raw-public charts and existing2000 controls."""
import copy,json,time
from pathlib import Path
import numpy as np
from experiments.local_carrier_20261031.results.basis600.collect import finite
from experiments.ideas_validate_20261031.collect import read,digest,canonical,paired_outcomes
from experiments.carrier_progress_20261031.collect_learning import artifact,paired_summary,planning_signature,successes
from experiments.carrier_progress_20261031.prepare_learning import semantic

B=Path('/efs/prescottsun/SPT-JEPA');ROOT=B/'runs/local_raw_chart2000_20261031'
OUT=B/'reports/local-raw-chart2000-20261031'


def collect():
    plan,proto,source,helper=[read(ROOT/n) for n in ['plan.json','protocol.json','source-record.json','helper-source-record.json']]
    assert len(plan['jobs'])==36 and (ROOT/'HOLD_DISPATCH').exists() and read(ROOT/'closed-dev13.json')['all36_exit0']
    assert source['source_commit']=='6e1693345786f2c190d0191685e4b485d24b4ab8'
    assert helper['source_commit']=='341644178ca978df09468e7f36237cbef8a4a8ea'
    sources=[]
    for s in [source,helper]:
        m=read(s['manifest']);assert m['source']==s['source'] and m['source_commit']==s['source_commit']
        for p,h in m['files'].items():assert digest(Path(s['source'])/p)==h
        sources.append(dict(source=s,manifest=artifact(s['manifest'])))
    exits=[]
    for j in plan['jobs']:
        path=ROOT/'exits'/(j['id']+'.json');e=read(path);assert e['returncode']==0 and j['source']==source
        if 'config' in j:assert digest(j['config'])==j['config_sha256']
        exits.append(dict(id=j['id'],artifact=artifact(path),returncode=0))
    # Reuse audited complete controls, not partial/live outputs or newly fitted probes.
    refpath=B/'reports/local-spatial-chart2000-20261031/results.json'
    refsummary=read(refpath.parent/'summary.json');assert digest(refpath)==refsummary['full_results']['sha256']
    ref=read(refpath);assert ref['completed_endpoint_count']==32
    previous=read(B/'reports/local-raw-chart-repaired600-20261031/results.json')
    assert previous['new_endpoints']==17 and previous['prior_successful_endpoints']==3
    rows=[];checks=[];pairs=[]
    for rec in proto['allocations']:
        name,task,seed=rec['id'],rec['task'],rec['training_seed'];run=ROOT/name
        cfg=read(run/'config.json');assert semantic(cfg)==semantic(read(Path(rec['parent'])/'config.json'))
        assert cfg['model']['beta']==.25 and cfg['model']['spatial_state_interface']=='raw' and cfg['model']['spatial_chart']=='bounded'
        pre,post=read(run/'pre-resume-fullstate.json'),read(run/'post-copy-fullstate.json')
        assert pre['passed'] and post['passed'] and pre['step']==post['step']==600
        assert pre['checkpoint_sha256']==post['checkpoint_sha256'] and pre['sampler']==post['sampler']
        assert pre['rng_component_sha256']==post['rng_component_sha256']
        oldcheck=next(c for c in previous['checks'] if c['task']==task and c['seed']==seed)
        assert pre['checkpoint_sha256']==oldcheck['fullstate_sha256'] and oldcheck['initial557_exact'] and oldcheck['identity553_exact']
        assert digest(run/'initial-model/weights.pt')==digest(rec['initial'])
        archives=[]
        for step in [1200,2000]:
            path=ROOT/f'archive{step}-{name}'/'fullstate.json';a=read(path)
            assert a['passed'] and all(a['checks'].values()) and a['step']==step and a['campaign_samples']==step*128
            assert a['native_stage_complete']['initial_step']==(600 if step==1200 else 1200) and a['native_stage_complete']['stop_step']==step
            assert a['source_commit']==source['source_commit'] and a['source_provenance']['helper']==helper
            assert a['frozen_tensors']['equal'] and a['frozen_tensors']['compared_tensors']==210
            assert len(a['sigreg_buffers'])==3 and a['chart']['parameters']==2128
            assert digest(a['archived_checkpoint'])==a['checkpoint_sha256'] and digest(a['portable_path'])==a['portable_sha256']
            archives.append(dict(step=step,report=artifact(path),checkpoint_sha256=a['checkpoint_sha256'],chart=a['chart']))
        path=ROOT/f'fullstate2000-{name}'/'preflight.json';f=read(path)
        assert f['passed'] and all(f['checkpoint']['checks'].values()) and f['checkpoint']['step']==2000
        assert f['paired_initial_equal']['equal'] and f['paired_initial_equal']['compared_tensors']==557
        assert f['frozen_tensors_unchanged']['equal'] and f['frozen_tensors_unchanged']['compared_tensors']==210
        assert f['optimizer_parameter_count']['passed'] and all(v['passed'] for v in f['gradients'].values()) and f['all_parameter_gradients_finite']
        assert digest(run/'last.ckpt')==f['checkpoint_sha256']==archives[-1]['checkpoint_sha256']
        checks.append(dict(task=task,seed=seed,archives=archives,fullstate=artifact(path),initial557=True,identity553=True,frozen210=True,
            sigreg3_unchanged=True,optimizer=f['optimizer_parameter_count'],gradients=f['gradients']))
        paths=dict(config=run/'config.json',stage=run/'stage-complete.json',status=run/'status.json',size=run/'model_size.json',
            health=ROOT/('health-'+name)/'health.json',state=ROOT/('transition-'+name)/'state-audit.json',windows=ROOT/('transition-'+name)/'windows.json',
            cache=ROOT/('transition-'+name)/'state-audit.npz',primary=ROOT/('planning-'+name)/'metrics.json',fresh=ROOT/('fresh-planning-'+name)/'metrics.json',trace=run/'metrics.jsonl')
        stage,status,size,h,s,w,p,fr=[read(paths[k]) for k in ['stage','status','size','health','state','windows','primary','fresh']]
        assert stage['stage_complete'] and stage['initial_step']==1200 and stage['stop_step']==status['step']==h['audit']['step']==s['step']==2000
        assert status['state']=='finished' and all(v for v in h['audit'].values() if isinstance(v,bool))
        assert s['cases']==w['count']==64 and s['manifest_sha256']==digest(paths['windows']) and s['config_sha256']==digest(paths['config'])
        assert size['total']==h['parameters']==p['checkpoint_audit']['parameters']==fr['checkpoint_audit']['parameters']
        assert Path(p['checkpoint'])==Path(fr['checkpoint'])==run/'model-step0002000'
        values={};signatures={}
        for case,x in [('primary',p),('fresh',fr)]:
            oo,rate=successes(x);values[case]=dict(success_rate=rate,episode_successes=oo);signatures[case]=planning_signature(x)
            assert x['manifest_sha256']==rec[case+'_manifest']['sha256']==digest(rec[case+'_manifest']['path'])
        trace=[json.loads(line) for line in paths['trace'].read_text().splitlines() if line.strip()]
        assert finite(trace) and finite(h) and finite(s) and finite(status)
        for key in ['predictor','renderer','action_encoder','carrier_chart_module']:assert status['grad_norm_'+key]>0
        assert status['grad_norm_encoder']==status['grad_norm_pool']==0
        c=s['conditions'];clean=c['clean_rollout']['mse'];cert=archives[-1]['chart']['certificate']
        for k in ['input_operator_norm','output_operator_norm','residual_lipschitz_upper']:assert abs(cert[k]-s['carrier_chart_certificate'][k])<1e-6
        current=dict(task=task,seed=seed,arm='raw-chart',values=values,common_rollout=clean/c['initial_carry']['mse'],
            common_one_step=c['teacher_forced_prefix']['mse']/c['shifted_observed_carry']['mse'],wrongstate_ratio=c['other_initial']['mse']/clean,
            otheractions_ratio=c['other_episode_actions']['mse']/clean,wrongorder_ratio=c['wrong_action_order']['mse']/clean,
            renderer={k:h['summary']['fm_t0.95_'+k]/h['summary']['fm_t0.95_forecast'] for k in ['wrong_state','wrong_actions','carry','zero']},
            certificate=cert,status=status,training_tail=trace[-6:],model_size=size,planning_seconds=dict(primary=p['seconds'],fresh=fr['seconds']),
            planning_signatures=signatures,windows=w,artifacts={k:artifact(v) for k,v in paths.items()})
        controls=[copy.deepcopy(x) for x in ref['rows'] if x['task']==task and x['seed']==seed];assert len(controls)==2
        for control in controls:
            assert canonical(control['planning_signatures'])==canonical(signatures) and control['windows']==w
            for case in ['primary','fresh']:
                a=control['artifacts'][case];assert digest(a['path'])==a['sha256']
                pairs.append(dict(task=task,seed=seed,case_set=case,reference=control['arm'],**paired_outcomes(values[case]['episode_successes'],control['values'][case]['episode_successes'],10000)))
            assert current['model_size']['total']-control['model_size']['total']==(2128 if control['arm']=='identity' else 0)
        identity=next(x for x in controls if x['arm']=='identity')
        with np.load(identity['artifacts']['cached_states']['path']) as ac,np.load(paths['cache']) as bc:
            assert np.array_equal(ac['observed'],bc['observed']) and np.array_equal(ac['actions'],bc['actions'])
            mse=float(np.mean((bc['clean_rollout'].astype(np.float64)-bc['observed'][:,1:].astype(np.float64))**2))
            assert abs(mse-clean)<1e-6
        current['raw_observed_exact_identity']=True
        rows.extend(controls+[current])
    comparisons=[dict(task=t,case_set=c,reference=a,**paired_summary([p for p in pairs if (p['task'],p['case_set'],p['reference'])==(t,c,a)],10000))
        for t in ['tworoom','cube'] for c in ['primary','fresh'] for a in ['identity','shared-chart']]
    return dict(created_at=time.time(),completed_endpoint_count=36,exits=exits,sources=sources,checks=checks,rows=rows,comparisons=pairs,comparison_summaries=comparisons,
        reference_report=artifact(refpath),collector=artifact(__file__),scope='CPU completed files/archive hashes and cached raw arrays only. No encoder/dynamics/renderer inference, fit, new cohort or GPU. Raw caches never inverted; old chart common-coordinate quantities reused from its completed audited report.')


def report(s):
    lines=['四组raw-chart2000成熟判别完成并HOLD：36/36真实exit0，8份1200/2000完整归档、557/553原初值、210冻结张量、3个sigreg buffer、optimizer/RNG/cursor及实际最终梯度检查通过。沿用β=.25、原数据/校准/两seed、64窗口和primary/fresh各50病例；新raw及旧chart均比identity多2128参数。','',
        '|任务/病例|identity seed1/2|旧chart seed1/2|raw-chart seed1/2|raw−identity Δpp [95%CI]|raw−旧chart Δpp [95%CI]|','|---|---:|---:|---:|---:|---:|']
    for task in ['tworoom','cube']:
        for case in ['primary','fresh']:
            vals=['/'.join(f"{x['values'][case]['success_rate']:g}" for x in sorted(s['rows'],key=lambda x:x['seed']) if x['task']==task and x['arm']==arm) for arm in ['identity','shared-chart','raw-chart']]
            ci=[]
            for arm in ['identity','shared-chart']:
                q=next(x for x in s['comparison_summaries'] if (x['task'],x['case_set'],x['reference'])==(task,case,arm));lo,hi=q['shared_case_bootstrap_ci95_pp'];ci.append(f"{q['mean_difference_pp']:+g} [{lo:+g},{hi:+g}]")
            lines.append('|'+ '|'.join([task+'/'+case,*vals,*ci])+'|')
    lines+=['','|任务/seed|raw rollout/复制 (identity/旧chart)|raw单步/复制 (identity/旧chart)|raw错初态/动作|DiT错初态/动作|chart L界 / raw rate|raw规划秒 primary/fresh|','|---|---:|---:|---:|---:|---:|---:|']
    for r in [x for x in s['rows'] if x['arm']=='raw-chart']:
        controls=[next(x for x in s['rows'] if (x['task'],x['seed'],x['arm'])==(r['task'],r['seed'],arm)) for arm in ['identity','shared-chart']]
        roll=f"{r['common_rollout']:.4f} ({controls[0]['common_rollout']:.4f}/{controls[1]['common_rollout']:.4f})"
        one=f"{r['common_one_step']:.4f} ({controls[0]['common_one_step']:.4f}/{controls[1]['common_one_step']:.4f})"
        lines.append(f"|{r['task']}/{r['seed']}|{roll}|{one}|{r['wrongstate_ratio']:.3f}/{r['otheractions_ratio']:.3f}|{r['renderer']['wrong_state']:.3f}/{r['renderer']['wrong_actions']:.3f}|{r['certificate']['residual_lipschitz_upper']:.3f}/{r['status']['rate']:.3f}|{r['planning_seconds']['primary']:.1f}/{r['planning_seconds']['fresh']:.1f}|")
    lines+=['','raw预测/目标已经是原始坐标，观测缓存与identity逐元素相同，没有二次inverse；旧chart的同坐标误差复用既有审计逆映射结果。DiT比值取t=.95。配对区间共同重采样50病例10000次，条件于两seed，不包含训练seed总体不确定性；primary/fresh分别保留。规划时间来自已有端到端记录，含仿真/路径差异，不是独立FLOP或延迟测定。2000已过1249 warmup，但不代表收敛；本批没有>2000续训。']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    s=collect();OUT.mkdir(exist_ok=False);(OUT/'results.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    keep=['task','seed','arm','values','common_rollout','common_one_step','wrongstate_ratio','otheractions_ratio','wrongorder_ratio','renderer','certificate','status','training_tail','model_size','planning_seconds','raw_observed_exact_identity']
    small={k:v for k,v in s.items() if k not in ['rows','comparisons','exits']};small['rows']=[{k:r[k] for k in keep if k in r} for r in s['rows']];small['full_results']=artifact(OUT/'results.json')
    (OUT/'summary.json').write_text(json.dumps(small,indent=2,allow_nan=False)+'\n');(OUT/'RESULTS.md').write_text(report(s));print(json.dumps(s['comparison_summaries']),flush=True)
