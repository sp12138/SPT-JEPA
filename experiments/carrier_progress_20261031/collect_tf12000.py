"""CPU-only completed TF12000 provenance, finite-portable and paired-case review."""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

REPO=next(p for p in Path(__file__).resolve().parents if (p/'experiments/ideas_validate_20261031/collect.py').is_file())
sys.path.insert(0,str(REPO))
from experiments.ideas_validate_20261031.collect import canonical,read,paired_outcomes
from experiments.carrier_progress_20261031.collect_learning import artifact,paired_summary,planning_signature,successes,verify_manifest
from experiments.carrier_progress_20261031.prepare_learning import digest,semantic

B=Path('/efs/prescottsun/SPT-JEPA')
ROOT=B/'runs/cube_tf12000_20261031'
PARENT=B/'runs/carrier_learn6000_20261031'
OUT=B/'reports/cube-tf12000-20261031'


def finite(value):
    if torch.is_tensor(value):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(tuple,list)):return all(finite(v) for v in value)
    return not isinstance(value,float) or math.isfinite(value)


def training_windows(run,windows):
    trace=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines() if x.strip()]
    assert finite(trace)
    unique={r['step']:r for r in trace if 'loss' in r}
    output={}
    for name,lo,hi in windows:
        rows=[r for step,r in unique.items() if lo<=step<=hi]
        assert len(rows)>=5
        output[name]=dict(step_range=[lo,hi],records=len(rows),**{
            k:float(np.mean([r[k] for r in rows])) for k in ['loss','fm','rate','feature_std']})
    return output


def collect():
    plan,protocol,source,helper=[read(ROOT/f) for f in ['plan.json','protocol.json','source-record.json','helper-source-record.json']]
    assert protocol['initial_step']==6000 and protocol['stop_step']==12000
    assert protocol['retained_steps']==[8000,10000,12000] and protocol['runs']==2
    assert len(plan['jobs'])==protocol['jobs']==10
    assert source==read(PARENT/'source-record.json')
    assert source['source_commit']==protocol['original_training_source_commit']=='8e0f821bba84e7ed8c0f16500d7c880fde56d4cf'
    assert helper['source_commit']=='0934d3c445e6c754a6ab547d801890c34740b330'
    provenance=dict(original=verify_manifest(source),helper=verify_manifest(helper))
    assert source['runtime_manifest']==helper['runtime_manifest']
    runtime=read(source['runtime_manifest'])
    for name,sha in runtime['files'].items():assert digest(Path(source['runtime_source'])/name)==sha
    provenance['runtime']=artifact(source['runtime_manifest'])
    helper_path=Path(helper['source'])/'experiments/carrier_progress_20261031/prepare_cube_tf12000.py'
    assert digest(helper_path)==protocol['helper_sha256']
    activation=read(ROOT/'activation-review.json')
    assert activation['passed'] and activation['no_auto_extension']
    jobs={j['id']:j for j in plan['jobs']};exits=[]
    expected={f'{phase}-cube-temporal-forecast-seed{s}' for s in [1,2]
              for phase in ['learn12000','fullstate12000','health','planning','fresh-planning']}
    assert set(jobs)==expected
    for job in jobs.values():
        x=read(ROOT/'exits'/(job['id']+'.json'))
        assert x['id']==job['id'] and x['returncode']==0
        assert job['source']==source
        exits.append(x)
        if job['phase']=='fullstate12000':assert job['command'][3]==str(helper_path)
    old_report_path=B/'reports/carrier6000-20261031/results.json'
    old_report=read(old_report_path);oldrows={r['id']:r for r in old_report['rows']}
    rows=[];comparisons=[]
    for seed in [1,2]:
        name=f'cube-temporal-forecast-seed{seed}';run,parent=ROOT/name,PARENT/name
        cfg,oldcfg=read(run/'config.json'),read(parent/'config.json')
        assert semantic(cfg)==semantic(oldcfg)
        assert cfg['max_steps']==124910 and cfg['retain_steps']==[8000,10000,12000]
        assert cfg['checkpoint_every']==2000 and cfg['batch_size']==128 and cfg['seed']==seed
        cfg_sha=digest(run/'config.json');assert cfg_sha==jobs['learn12000-'+name]['config_sha256']
        pre=read(run/'pre-resume-fullstate.json');fullpath=ROOT/('fullstate12000-'+name)/'fullstate.json';full=read(fullpath)
        assert pre['passed'] and pre['copy_byte_identical'] and pre['step']==6000 and all(pre['checks'].values())
        assert pre['parent_source']==source and pre['config_sha256']==digest(parent/'config.json')
        assert pre['checkpoint_sha256']==digest(parent/'last.ckpt')==digest(parent/'checkpoint-step0006000.ckpt')
        assert pre['checkpoint_sha256']==oldrows[name]['fullstate']['checkpoint_sha256']
        stage,status=read(run/'stage-complete.json'),read(run/'status.json')
        assert stage['initial_step']==6000 and stage['stop_step']==12000 and stage['stage_complete']
        assert status['state']=='finished' and status['step']==12000
        assert full['passed'] and all(full['checks'].values()) and full['step']==12000
        assert full['native_stage_complete']==stage and full['config_sha256']==cfg_sha
        assert full['campaign_samples']==12000*128 and full['source_commit']==source['source_commit']
        assert full['source_record_sha256']==digest(ROOT/'source-record.json')
        archive=Path(full['archived_checkpoint']);assert archive==run/'checkpoint-step0012000.ckpt'
        assert full['checkpoint_sha256']==digest(archive)==digest(run/'last.ckpt')
        portable=[];shapes=None
        for step in [6000,8000,10000,12000]:
            path=(parent if step==6000 else run)/f'model-step{step:07d}'
            weights=torch.load(path/'weights.pt',map_location='cpu',weights_only=True)
            assert weights and finite(weights)
            current={k:list(v.shape) for k,v in weights.items()}
            if shapes is not None:assert shapes==current
            else:shapes=current
            config=read(path/'config.json');assert all(config[k]==v for k,v in cfg['model'].items())
            record=dict(step=step,finite=True,tensors=len(weights),shapes_equal=True,**artifact(path/'weights.pt'))
            if step==6000:assert record['sha256']==pre['portable_sha256']
            if step==12000:assert record['sha256']==full['portable_sha256']
            portable.append(record);del weights
        healthpath=ROOT/('health-'+name)/'health.json';oldhealthpath=PARENT/('health-'+name)/'health.json'
        health,oldhealth=read(healthpath),read(oldhealthpath)
        assert health['audit']['step']==12000 and finite(health)
        assert all(v for v in health['audit'].values() if isinstance(v,bool))
        assert all(health['audit']['gradient_max'][k]>0 for k,v in health['audit']['gradient_expectations'].items() if v=='active')
        assert [{k:r[k] for k in ['episode','start']} for r in health['cases']]==[
            {k:r[k] for k in ['episode','start']} for r in oldhealth['cases']]
        for k in ['parameters','inference_shift','training_shift']:assert health[k]==oldhealth[k]
        assert digest(oldhealthpath)==oldrows[name]['artifacts']['health']['sha256']
        planning={}
        for case,phase in [('primary','planning'),('fresh','fresh-planning')]:
            path=ROOT/(phase+'-'+name)/'metrics.json';oldpath=PARENT/(phase+'-'+name)/'metrics.json'
            new,old=read(path),read(oldpath);job=jobs[phase+'-'+name]
            manifest=Path(job['command'][job['command'].index('--manifest')+1])
            assert digest(manifest)==job['manifest_sha256']==new['manifest_sha256']
            assert read(manifest)['pairs']==new['pairs'] and new['split']=='validation'
            assert canonical(planning_signature(new))==canonical(planning_signature(old))
            assert new['checkpoint']==str(run/'model-step0012000') and old['checkpoint']==str(parent/'model-step0006000')
            assert new['checkpoint_audit']['parameters']==old['checkpoint_audit']['parameters']==health['parameters']
            a,ar=successes(new);b,br=successes(old)
            assert digest(oldpath)==oldrows[name]['planning'][case]['source']['sha256']
            comparisons.append(dict(seed=seed,case_set=case,**paired_outcomes(a,b,10000)))
            planning[case]=dict(old_success=br,new_success=ar,old_outcomes=b,new_outcomes=a,
                signature=planning_signature(new),old_source=artifact(oldpath),new_source=artifact(path))
        hchange={k:100*(health['summary'][k]/oldhealth['summary'][k]-1) for k in health['summary']}
        train=training_windows(parent,[('old_late',5500,6000)])
        train.update(training_windows(run,[('recent',10500,11000),('late',11500,12000)]))
        rows.append(dict(seed=seed,id=name,planning=planning,pre_resume=pre,fullstate=full,portable=portable,
            health_old=oldhealth,health_new=health,health_change_percent=hchange,training=train,
            artifacts=dict(config=artifact(run/'config.json'),fullstate=artifact(fullpath),
                health=artifact(healthpath),parent_health=artifact(oldhealthpath))))
    summaries=[dict(case_set=c,**paired_summary([x for x in comparisons if x['case_set']==c],10000)) for c in ['primary','fresh']]
    return dict(root=str(ROOT),completed_endpoint_count=10,source=source,helper=helper,provenance=provenance,
        protocol=protocol,exits=exits,rows=rows,comparisons=comparisons,comparison_summaries=summaries,
        parent_report=artifact(old_report_path),collector=artifact(__file__),all_checks_passed=True,
        scope='CPU read-only review. No model forward, training, GPU, or added evaluation. Case bootstrap10000 seed1042 jointly '
              'resamples50 cases across the same2 trained seeds; conditional on checkpoints, not training-seed-population uncertainty. '
              'Primary/fresh remain distinct validation sets. Health is matched-noise renderer error, not clean latent outcome error.')


def compact(r,path):
    return dict(full_results=artifact(path),completed_endpoint_count=10,all_checks_passed=True,
        source_commit=r['source']['source_commit'],helper_commit=r['helper']['source_commit'],
        provenance=r['provenance'],parent_report=r['parent_report'],comparison_summaries=r['comparison_summaries'],
        rows=[dict(seed=x['seed'],planning={c:{k:v for k,v in a.items() if k in ['old_success','new_success']} for c,a in x['planning'].items()},
            portable=x['portable'],health_old={k:v for k,v in x['health_old'].items() if k!='cases'},
            health_new={k:v for k,v in x['health_new'].items() if k!='cases'},
            health_change_percent=x['health_change_percent'],training=x['training']) for x in r['rows']],scope=r['scope'])


def report(s):
    c={x['case_set']:x for x in s['comparison_summaries']}
    ci=lambda x:f"{x['mean_difference_pp']:+g}pp [{x['shared_case_bootstrap_ci95_pp'][0]:+g},{x['shared_case_bootstrap_ci95_pp'][1]:+g}]"
    lines=['# Cube TF：6000→12000','',
        f"10/10端点exit0；primary平均变化{ci(c['primary'])}，fresh变化{ci(c['fresh'])}，未见规划收益。区间为50病例配对bootstrap95%区间（10000次，同一索引同时重采样两个固定训练seed），不包含训练seed总体不确定性，未作多重校正。原primary/fresh有序病例、CEM、strict判据及train-episode动作归一化均与6000精确一致。原8e0f821训练源、0934d3c收集/归档helper及runtime全部manifest哈希通过；6000完整状态恢复、12000完整状态归档/last哈希、8000/10000/12000保留权重有限性与相同键/形状均通过，优化器/scheduler/RNG/sampler审计通过。",'',
        '| 病例 | seed | 6000成功率 | 12000成功率 | Δpp |','|---|---:|---:|---:|---:|']
    for case in ['primary','fresh']:
        for r in s['rows']:
            v=r['planning'][case];lines.append(f"| {case} | {r['seed']} | {v['old_success']:g}% | {v['new_success']:g}% | {v['new_success']-v['old_success']:+g} |")
    old='/'.join(f"{r['health_old']['summary']['fm_t0.95_forecast']:.5f}" for r in s['rows'])
    new='/'.join(f"{r['health_new']['summary']['fm_t0.95_forecast']:.5f}" for r in s['rows'])
    changes='/'.join(f"{r['health_change_percent']['fm_t0.95_forecast']:.1f}%" for r in s['rows'])
    stdold='/'.join(f"{r['health_old']['feature_std']:.3f}" for r in s['rows'])
    stdnew='/'.join(f"{r['health_new']['feature_std']:.3f}" for r in s['rows'])
    lines+=['',f"同16 validation窗口的健康检查保持有限且encoder/predictor/renderer均有非零训练梯度；t=.95 forecast条件去噪MSE由{old}降至{new}（{changes}），feature std从{stdold}变为{stdnew}。这是带噪初态和renderer的匹配噪声诊断，不是独立clean-latent预测误差；改善没有转化为这两组规划收益。原max_steps=124910/warmup=1249保持不变，训练在12000停止，保留全部指定checkpoint；不据此自动续跑。逐case比较和全部健康指标保留EFS，summary.json含完整结果SHA。",'']
    return '\n'.join(lines)


if __name__=='__main__':
    torch.set_num_threads(2)
    r=collect();OUT.mkdir(parents=True,exist_ok=True);path=OUT/'results.json'
    path.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');s=compact(r,path)
    (OUT/'summary.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    (OUT/'RESULTS.md').write_text(report(s));print(report(s))
