"""Complete-only CPU collection of basis2000; no model forward or new assay."""
import copy
import json
from pathlib import Path
import time

from experiments.local_carrier_20261031.results.basis600.collect import finite
from experiments.carrier_progress_20261031.prepare_learning import semantic
from experiments.ideas_validate_20261031.collect import read, digest, canonical, paired_outcomes
from experiments.carrier_progress_20261031.collect_learning import (
    artifact, paired_summary, planning_signature, successes, verify_manifest,
)

B=Path('/efs/prescottsun/SPT-JEPA')
ROOT=B/'runs/local_basis2000_20261031'
OLD=B/'runs/local_isotropic2000_20261031'
OUT=B/'reports/local-basis2000-20261031'
QHASH='f5b62cef8c9c8893c8dad083b4e24b3c1ec6f40bddc926aff146a1de1796f51a'


def collect():
    plan,proto=read(ROOT/'plan.json'),read(ROOT/'protocol.json')
    src,helper,oldsrc=[read(p/'source-record.json') for p in [ROOT,B/'runs/local_basis2000_helpers_20261031',OLD]]
    assert len(plan['jobs'])==32 and proto['runs']==4 and (ROOT/'HOLD_DISPATCH').exists()
    assert src['source_commit']==proto['training_source_commit']=='da26748c866137ad1ea06cf7250cabd64431e9ac'
    assert helper['source_commit']==proto['helper_source_commit']=='cf6a4101c13e3ddef5fa10c60c18166048fd1c8c'
    sources=[dict(source=s,manifest=verify_manifest(s)) for s in [src,helper,oldsrc]]
    jobs={j['id']:j for j in plan['jobs']};exits=[]
    for j in plan['jobs']:
        e=read(ROOT/'exits'/(j['id']+'.json'));assert e['returncode']==0 and e['id']==j['id']
        assert j['source']==(helper if j['phase'].startswith('fullstate') else src)
        exits.append(e)
    oldplan=read(OLD/'plan.json')
    priorpath=B/'reports/local-basis600-20261031/results.json';prior=read(priorpath)
    assert prior['completed_endpoint_count']==20
    rows,checks,pairs=[],[],[]
    for rec in proto['allocations']:
        name,task,seed=rec['id'],rec['task'],rec['training_seed'];run=ROOT/name
        ref=OLD/f'{task}-spatial-local-seed{seed}';parent=Path(proto['parent'])/name
        cfg,oldcfg=read(run/'config.json'),read(ref/'config.json')
        expected=copy.deepcopy(oldcfg);expected['output']=str(run)
        expected['model']['spatial_basis']='orthogonal'
        expected['audit_expectations']['initial_unchanged_prefixes'].append('spatial_orthogonal_basis')
        assert cfg==expected and semantic(cfg)==semantic(read(parent/'config.json'))
        pre=read(run/'pre-resume-fullstate.json')
        assert pre['passed'] and pre['copy_byte_identical'] and pre['step']==600 and all(pre['checks'].values())
        pc=next(x for x in prior['checks'] if x['task']==task and x['seed']==seed)
        assert digest(pc['initial']['path'])==pc['initial']['sha256']
        assert pc['original_initial_tensors_exact']==553 and pc['frozen600_exact']==211
        assert pre['portable_sha256']==pc['final_weights']['sha256']
        for step in [1200,2000]:
            fp=ROOT/f'fullstate{step}-{name}'/'fullstate.json';f=read(fp)
            assert f['passed'] and all(f['checks'].values()) and f['step']==step
            assert f['source_commit']==src['source_commit'] and f['helper_source_commit']==helper['source_commit']
            assert f['frozen_tensors']['equal'] and f['frozen_tensors']['compared_tensors']==211
            assert 'spatial_orthogonal_basis' in f['frozen_tensors']['names'] and f['basis_sha256']==QHASH
            assert f['native_stage_complete']['initial_step']==(600 if step==1200 else 1200)
            assert f['native_stage_complete']['stop_step']==step and f['campaign_samples']==step*128
            assert f['config_sha256']==digest(run/'config.json')
            assert digest(f['archived_checkpoint'])==f['checkpoint_sha256']
            assert digest(f['portable_path'])==f['portable_sha256']
            checks.append(dict(task=task,seed=seed,step=step,report=artifact(fp),
                archive=dict(path=f['archived_checkpoint'],sha256=f['checkpoint_sha256']),
                portable=dict(path=f['portable_path'],sha256=f['portable_sha256']),
                frozen_tensors=211,basis_sha256=f['basis_sha256'],passed=True))
        phases=['learn1200','fullstate1200','learn2000','fullstate2000','health','transition','planning','fresh-planning']
        previous=[]
        for phase in phases:
            j=jobs[phase+'-'+name];assert j['depends_on']==previous;previous=[j['id']]
        pairrows=[]
        for arm,root,rr in [('identity',OLD,ref),('orthogonal',ROOT,run)]:
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
                training=rr/'metrics.jsonl')
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
            assert finite(trace) and finite(h) and finite(s)
            c=s['conditions'];clean=c['clean_rollout']['mse']
            row=dict(task=task,seed=seed,arm=arm,values=outcomes,
                rollout=clean/c['initial_carry']['mse'],one_step=c['teacher_forced_prefix']['mse']/c['shifted_observed_carry']['mse'],
                wrongstate_ratio=c['other_initial']['mse']/clean,otheractions_ratio=c['other_episode_actions']['mse']/clean,
                wrongorder_ratio=c['wrong_action_order']['mse']/clean,
                renderer={k:h['summary']['fm_t0.95_'+k]/h['summary']['fm_t0.95_forecast'] for k in ['wrong_state','wrong_actions','carry']},
                rms=s['observed_state_rms'],coordinate_std_mean=s['observed_state_std_mean'],model_size=size,
                planning_seconds=dict(primary=p['seconds'],fresh=f['seconds']),state=s,health=h,training=trace,
                planning_signatures=signatures,windows=w,artifacts={k:artifact(v) for k,v in paths.items()})
            rows.append(row);pairrows.append(row)
        a,b=pairrows
        assert canonical(a['planning_signatures'])==canonical(b['planning_signatures'])
        assert a['windows']==b['windows'] and a['model_size']==b['model_size']
        assert [{k:v[k] for k in ['episode','start']} for v in a['health']['cases']]==[{k:v[k] for k in ['episode','start']} for v in b['health']['cases']]
        assert abs(a['rms']-b['rms'])<1e-6
        for k in ['initial_carry','shifted_observed_carry','zero_prediction']:
            assert abs(a['state']['conditions'][k]['mse']-b['state']['conditions'][k]['mse'])<1e-6
        for case in ['primary','fresh']:
            pairs.append(dict(task=task,seed=seed,case_set=case,
                **paired_outcomes(b['values'][case]['episode_successes'],a['values'][case]['episode_successes'],10000)))
        print(json.dumps(dict(checked=name,archives=2,primary=b['values']['primary']['success_rate'],fresh=b['values']['fresh']['success_rate'])),flush=True)
    summaries=[dict(task=t,case_set=c,**paired_summary([x for x in pairs if x['task']==t and x['case_set']==c],10000))
               for t in ['tworoom','cube'] for c in ['primary','fresh']]
    return dict(created_at=time.time(),completed_endpoint_count=32,exits=exits,sources=sources,checks=checks,rows=rows,
        comparisons=pairs,comparison_summaries=summaries,collector=artifact(__file__),basis600_report=artifact(priorpath),
        scope='Completed CPU file/hash checks of all8 archived native full-state audits, including Q211. No repeated model forward, probe, GPU, training, or new simulation.')


def report(s):
    lines=['2000步旋转组在TwoRoom的primary与fresh均变差；Cube方向也为负，但两seed配对区间包含0。'
        '32/32端点exit0，8份1200/2000完整状态归档及portable哈希核对通过，含Q的211个冻结张量审计通过；'
        '沿用600步553初始张量一致证据，同数据、恢复边界、归一化、64诊断窗口和两套各50病例/CEM预算。','',
        '| 任务/病例集 | 原local成功率，seed1/2 | 旋转成功率，seed1/2 | 旋转−原local均值pp [配对95%区间] |',
        '|---|---:|---:|---:|']
    for q in s['comparison_summaries']:
        task,case=q['task'],q['case_set'];vals=[]
        for arm in ['identity','orthogonal']:
            rr=sorted([r for r in s['rows'] if r['task']==task and r['arm']==arm],key=lambda r:r['seed'])
            vals.append('/'.join(f"{r['values'][case]['success_rate']:g}" for r in rr))
        lo,hi=q['shared_case_bootstrap_ci95_pp']
        lines.append(f"| {task}/{case} | {vals[0]} | {vals[1]} | {q['mean_difference_pp']:+g} [{lo:+g},{hi:+g}] |")
    lines += ['', '| 任务/seed | 基底 | rollout/初态复制 | 单步/前帧复制 | 错初态/正确 | 别段动作/正确 | DiT错初态/错动作 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in s['rows']:
        lines.append(f"| {r['task']}/{r['seed']} | {r['arm']} | {r['rollout']:.3f} | {r['one_step']:.3f} | "
            f"{r['wrongstate_ratio']:.3f} | {r['otheractions_ratio']:.3f} | {r['renderer']['wrong_state']:.3f}/{r['renderer']['wrong_actions']:.3f} |")
    lines += ['', '误差比值越低代表更好的对应预测；干预比值衡量输入使用，DiT在t=.95相对正确forecast。'
        '600步更快拟合不能推出2000步更好控制；Q保留观测信息和欧氏目标距离至舍入误差，故信息量不是唯一解释。'
        '但Q同时改变predictor与renderer的坐标因子化，噪声仅同分布且训练未做完整网络共轭，不能唯一归因某一部件。'
        '2000步已过1249步warmup，仍非收敛性能证明。区间按各自50配对病例共同重采样10000次，条件于两训练seed，'
        '不含seed总体不确定性或多重比较校正；两套病例集不是独立训练重复。没有续训。']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    result=collect();OUT.mkdir(exist_ok=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    keep=['task','seed','arm','values','rollout','one_step','wrongstate_ratio','otheractions_ratio','wrongorder_ratio','renderer',
          'rms','coordinate_std_mean','model_size','planning_seconds']
    summary=dict(full_results=artifact(OUT/'results.json'),completed_endpoint_count=32,
        rows=[{k:r[k] for k in keep} for r in result['rows']],checks=result['checks'],
        comparison_summaries=result['comparison_summaries'],sources=result['sources'],collector=result['collector'],
        basis600_report=result['basis600_report'],scope=result['scope'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (OUT/'RESULTS.md').write_text(report(summary))
    print(json.dumps(summary['comparison_summaries'],ensure_ascii=False),flush=True)
