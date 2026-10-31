"""Complete-only CPU collection of chart2000; no model forward or new assay."""
import copy
import json
from pathlib import Path
import time
import torch
from as_jepa_lewm.carrier import ResidualCarrierChart
from experiments.local_carrier_20261031.results.chart600.collect import cached_geometry

from experiments.local_carrier_20261031.results.basis600.collect import finite
from experiments.carrier_progress_20261031.prepare_learning import semantic
from experiments.ideas_validate_20261031.collect import read, digest, canonical, paired_outcomes
from experiments.carrier_progress_20261031.collect_learning import (
    artifact, paired_summary, planning_signature, successes, verify_manifest,
)

B=Path('/efs/prescottsun/SPT-JEPA')
ROOT=B/'runs/local_spatial_chart2000_20261031'
OLD=B/'runs/local_isotropic2000_20261031'
OUT=B/'reports/local-spatial-chart2000-20261031'
PREFIX='carrier_chart_module.'


def collect():
    torch.set_num_threads(2)
    plan,proto=read(ROOT/'plan.json'),read(ROOT/'protocol.json')
    src,helper,oldsrc=[read(p/'source-record.json') for p in [ROOT,B/'runs/local_spatial_chart2000_helpers_20261031',OLD]]
    assert len(plan['jobs'])==32 and proto['runs']==4 and (ROOT/'HOLD_DISPATCH').exists()
    assert read(ROOT/'closed-dev13.json')['all32_exit0']
    assert src['source_commit']==proto['training_source_commit']=='9c3b9fdadf81116242ef9e9c5bf48603aa3c3a1c'
    assert helper['source_commit']==proto['helper_source_commit']=='1006be1d6183515e674fd4fc1578fced1d7a880e'
    sources=[dict(source=s,manifest=verify_manifest(s)) for s in [src,helper,oldsrc]]
    jobs={j['id']:j for j in plan['jobs']};exits=[]
    for j in plan['jobs']:
        e=read(ROOT/'exits'/(j['id']+'.json'));assert e['returncode']==0 and e['id']==j['id']
        assert j['source']==(helper if j['phase'].startswith('fullstate') else src)
        exits.append(e)
    oldplan=read(OLD/'plan.json')
    priorpath=B/'reports/local-spatial-chart600-20261031/results.json';prior=read(priorpath)
    assert prior['completed_endpoint_count']==20
    rows,checks,pairs=[],[],[]
    for rec in proto['allocations']:
        name,task,seed=rec['id'],rec['task'],rec['training_seed'];run=ROOT/name
        ref=OLD/f'{task}-spatial-local-seed{seed}';parent=Path(proto['parent'])/name
        cfg,oldcfg=read(run/'config.json'),read(ref/'config.json')
        expected=copy.deepcopy(oldcfg);expected['output']=str(run)
        expected['model']['spatial_chart']='bounded'
        assert cfg==expected and semantic(cfg)==semantic(read(parent/'config.json'))
        pre=read(run/'pre-resume-fullstate.json')
        assert pre['passed'] and pre['copy_byte_identical'] and pre['step']==600 and all(pre['checks'].values())
        pc=next(x for x in prior['checks'] if x['task']==task and x['seed']==seed)
        assert digest(pc['initial']['path'])==pc['initial']['sha256']
        assert pc['original_initial_tensors_exact']==553 and pc['frozen_tensors600']==210
        assert pre['portable_sha256']==read(pc['fullstate']['path'])['weights_sha256']
        for step in [1200,2000]:
            fp=ROOT/f'fullstate{step}-{name}'/'fullstate.json';f=read(fp)
            assert f['passed'] and all(f['checks'].values()) and f['step']==step
            assert f['source_commit']==src['source_commit'] and f['helper_source_commit']==helper['source_commit']
            assert f['frozen_tensors']['equal'] and f['frozen_tensors']['compared_tensors']==210
            assert f['chart']['parameters']==2128 and f['chart']['certificate']['residual_lipschitz_upper']<=.500001
            assert f['native_stage_complete']['initial_step']==(600 if step==1200 else 1200)
            assert f['native_stage_complete']['stop_step']==step and f['campaign_samples']==step*128
            assert f['config_sha256']==digest(run/'config.json')
            assert digest(f['archived_checkpoint'])==f['checkpoint_sha256']
            assert digest(f['portable_path'])==f['portable_sha256']
            checks.append(dict(task=task,seed=seed,step=step,report=artifact(fp),
                archive=dict(path=f['archived_checkpoint'],sha256=f['checkpoint_sha256']),
                portable=dict(path=f['portable_path'],sha256=f['portable_sha256']),
                frozen_tensors=210,chart=f['chart'],passed=True))
        initial=torch.load(pre['initial_weights'],map_location='cpu',weights_only=True)
        final=torch.load(run/'model-step0002000/weights.pt',map_location='cpu',weights_only=True)
        frozen=f['frozen_tensors']['names'];assert len(frozen)==210 and all(torch.equal(initial[k],final[k]) for k in frozen)
        chart=ResidualCarrierChart(16,64,'bounded',.5)
        chart.load_state_dict({k.removeprefix(PREFIX):v for k,v in final.items() if k.startswith(PREFIX)},strict=True)
        certificate=chart.norm_certificate();assert certificate['residual_lipschitz_upper']<=.500001
        changed_chart=[k for k in final if k.startswith(PREFIX) and not torch.equal(final[k],initial[k])]
        assert changed_chart
        del initial,final
        phases=['learn1200','fullstate1200','learn2000','fullstate2000','health','transition','planning','fresh-planning']
        previous=[]
        for phase in phases:
            j=jobs[phase+'-'+name];assert j['depends_on']==previous;previous=[j['id']]
        pairrows=[]
        for arm,root,rr in [('identity',OLD,ref),('shared-chart',ROOT,run)]:
            rn=rr.name
            relevant=[j for j in (oldplan if arm=='identity' else plan)['jobs'] if j['id'].endswith('-'+rn)]
            assert len(relevant)==8
            for j in relevant:assert read(root/'exits'/(j['id']+'.json'))['returncode']==0
            fresh_path=(Path(proto['identity2000_references'][name]['fresh']['evaluation']) if arm=='identity'
                        else root/('fresh-planning-'+rn)/'metrics.json')
            if arm=='identity':
                assert digest(fresh_path)==proto['identity2000_references'][name]['fresh']['sha256']
                oldfull=read(root/('fullstate2000-'+rn)/'fullstate.json')
                assert oldfull['passed'] and all(oldfull['checks'].values()) and oldfull['frozen_tensors']['compared_tensors']==210
            paths=dict(config=rr/'config.json',stage=rr/'stage-complete.json',status=rr/'status.json',size=rr/'model_size.json',
                health=root/('health-'+rn)/'health.json',transition=root/('transition-'+rn)/'state-audit.json',
                windows=root/('transition-'+rn)/'windows.json',primary=root/('planning-'+rn)/'metrics.json',fresh=fresh_path,
                training=rr/'metrics.jsonl',cached_states=root/('transition-'+rn)/'state-audit.npz')
            stage,status,size,h,s,w,p,f=[read(paths[k]) for k in ['stage','status','size','health','transition','windows','primary','fresh']]
            assert stage['stage_complete'] and stage['initial_step']==1200
            assert stage['stop_step']==status['step']==h['audit']['step']==s['step']==2000
            assert status['state']=='finished' and all(v for v in h['audit'].values() if isinstance(v,bool))
            assert s['cases']==w['count']==64 and s['manifest_sha256']==digest(paths['windows'])
            assert s['config_sha256']==digest(paths['config'])
            assert size['total']==h['parameters']==p['checkpoint_audit']['parameters']==f['checkpoint_audit']['parameters']
            assert Path(p['checkpoint'])==Path(f['checkpoint'])==rr/'model-step0002000'
            assert f['manifest_sha256']==proto['fresh_manifests'][task]['sha256']
            outcomes={};signatures={}
            for case,x in [('primary',p),('fresh',f)]:
                oo,rate=successes(x);outcomes[case]=dict(success_rate=rate,episode_successes=oo)
                signatures[case]=planning_signature(x)
            trace=[json.loads(line) for line in paths['training'].read_text().splitlines() if line.strip()]
            assert finite(trace) and finite(h) and finite(s) and finite(status)
            for prefix in ('predictor','renderer','action_encoder'):
                assert status['grad_norm_'+prefix]>0
            assert status['grad_norm_encoder']==status['grad_norm_pool']==0
            if arm=='shared-chart':assert status['grad_norm_carrier_chart_module']>0
            c=s['conditions'];clean=c['clean_rollout']['mse']
            row=dict(task=task,seed=seed,arm=arm,values=outcomes,
                rollout=clean/c['initial_carry']['mse'],one_step=c['teacher_forced_prefix']['mse']/c['shifted_observed_carry']['mse'],
                wrongstate_ratio=c['other_initial']['mse']/clean,otheractions_ratio=c['other_episode_actions']['mse']/clean,
                wrongorder_ratio=c['wrong_action_order']['mse']/clean,
                renderer={k:h['summary']['fm_t0.95_'+k]/h['summary']['fm_t0.95_forecast'] for k in ['wrong_state','wrong_actions','carry']},
                rms=s['observed_state_rms'],coordinate_std_mean=s['observed_state_std_mean'],model_size=size,
                planning_seconds=dict(primary=p['seconds'],fresh=f['seconds']),state=s,health=h,training=trace,
                status=status,training_tail=trace[-5:],planning_signatures=signatures,windows=w,artifacts={k:artifact(v) for k,v in paths.items()})
            rows.append(row);pairrows.append(row)
        a,b=pairrows
        assert canonical(a['planning_signatures'])==canonical(b['planning_signatures'])
        assert a['windows']==b['windows']
        assert b['model_size']['total']-a['model_size']['total']==b['model_size']['trainable']-a['model_size']['trainable']==2128
        assert [{k:v[k] for k in ['episode','start']} for v in a['health']['cases']]==[{k:v[k] for k in ['episode','start']} for v in b['health']['cases']]
        for key in ['input_operator_norm','output_operator_norm','residual_lipschitz_upper']:
            assert abs(b['state']['carrier_chart_certificate'][key]-certificate[key])<1e-6
        b['chart_analysis']=cached_geometry(chart,a['artifacts']['cached_states']['path'],b['artifacts']['cached_states']['path'])
        b['certificate']=certificate;b['changed_chart_tensors']=changed_chart
        a['common_rollout'],a['common_one_step']=a['rollout'],a['one_step']
        common=b['chart_analysis']['common_coordinates']
        b['common_rollout']=common['clean_rollout']['over_initial_copy'];b['common_one_step']=common['teacher_forced_prefix']['over_shifted_copy']
        for case in ['primary','fresh']:
            pairs.append(dict(task=task,seed=seed,case_set=case,
                **paired_outcomes(b['values'][case]['episode_successes'],a['values'][case]['episode_successes'],10000)))
        print(json.dumps(dict(checked=name,archives=2,primary=b['values']['primary']['success_rate'],fresh=b['values']['fresh']['success_rate'])),flush=True)
    summaries=[dict(task=t,case_set=c,**paired_summary([x for x in pairs if x['task']==t and x['case_set']==c],10000))
               for t in ['tworoom','cube'] for c in ['primary','fresh']]
    return dict(created_at=time.time(),completed_endpoint_count=32,exits=exits,sources=sources,checks=checks,rows=rows,
        comparisons=pairs,comparison_summaries=summaries,collector=artifact(__file__),chart600_report=artifact(priorpath),
        scope='Completed CPU file/hash checks of all8 archived native full-state audits, including210 frozen tensors and chart certificates; cached chart inverse gives common coordinates. No encoder/dynamics/renderer forward, probe, GPU, training, or new simulation; only cached-code2128-parameter chart evaluation on CPU.')


def report(s):
    lines=['共享local chart的2000步有限成熟度门已完成：32/32端点exit0，8份1200/2000完整状态归档及portable哈希核对通过，210个冻结张量与chart证书通过；沿用553原初值一致证据。原数据、恢复边界、校准、64诊断窗口及primary/fresh各50有序病例/CEM预算匹配。新增2128可训练参数，全部四组保留，调度已关闭并HOLD。','',
        '| 任务/病例集 | 原local seed1/2 | chart seed1/2 | 均值Δpp [配对95%区间] |','|---|---:|---:|---:|']
    for q in s['comparison_summaries']:
        vals=[]
        for arm in ['identity','shared-chart']:
            rr=sorted([r for r in s['rows'] if r['task']==q['task'] and r['arm']==arm],key=lambda r:r['seed'])
            vals.append('/'.join(f"{r['values'][q['case_set']]['success_rate']:g}" for r in rr))
        lo,hi=q['shared_case_bootstrap_ci95_pp']
        lines.append(f"| {q['task']}/{q['case_set']} | {vals[0]} | {vals[1]} | {q['mean_difference_pp']:+g} [{lo:+g},{hi:+g}] |")
    lines+=['','| 任务/seed | 表征 | 同坐标rollout/复制 | 同坐标单步/复制 | 错初态/别段动作 | DiT错初态/错动作 | 偏移RMS% / 残差L界 |','|---|---|---:|---:|---:|---:|---:|']
    for r in s['rows']:
        extra='0 / 0' if r['arm']=='identity' else f"{100*r['chart_analysis']['identity_deviation']['relative_rms']:.2f} / {r['certificate']['residual_lipschitz_upper']:.3f}"
        lines.append(f"| {r['task']}/{r['seed']} | {r['arm']} | {r['common_rollout']:.3f} | {r['common_one_step']:.3f} | {r['wrongstate_ratio']:.3f}/{r['otheractions_ratio']:.3f} | {r['renderer']['wrong_state']:.3f}/{r['renderer']['wrong_actions']:.3f} | {extra} |")
    lines+=['','同坐标预测误差由缓存预测经最终chart收缩逆映射回原local坐标计算并检查重建，没有新encoder/world-model前向；干预与DiT比值仍使用各自训练坐标，DiT取t=.95。chart偏移和可逆证书不证明控制几何有效。2000步已过1249步warmup，仍不是收敛性能证明。区间为各病例集50病例共同bootstrap10000次，条件于两训练seed，不含seed总体不确定性或多重比较修正。没有继续训练。']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    result=collect();OUT.mkdir(exist_ok=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    keep=['task','seed','arm','values','rollout','one_step','wrongstate_ratio','otheractions_ratio','wrongorder_ratio','renderer',
          'rms','coordinate_std_mean','model_size','planning_seconds','common_rollout','common_one_step','chart_analysis','certificate','changed_chart_tensors','status','training_tail']
    summary=dict(full_results=artifact(OUT/'results.json'),completed_endpoint_count=32,
        rows=[{k:r[k] for k in keep if k in r} for r in result['rows']],checks=result['checks'],
        comparison_summaries=result['comparison_summaries'],sources=result['sources'],collector=result['collector'],
        chart600_report=result['chart600_report'],scope=result['scope'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (OUT/'RESULTS.md').write_text(report(summary))
    print(json.dumps(summary['comparison_summaries'],ensure_ascii=False),flush=True)
