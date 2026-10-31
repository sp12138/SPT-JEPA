"""Completed local/dense6000 CPU review, preserving the2000 audit contracts."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

REPO=next(p for p in Path(__file__).resolve().parents if (p/'experiments/ideas_validate_20261031/collect.py').is_file())
sys.path.insert(0,str(REPO))
from experiments.ideas_validate_20261031.collect import read,canonical,paired_outcomes
from experiments.carrier_progress_20261031.collect_learning import artifact,paired_summary,planning_signature,successes,verify_manifest
from experiments.carrier_progress_20261031.prepare_learning import semantic,digest

B=Path('/efs/prescottsun/SPT-JEPA');ROOT=B/'runs/local_isotropic6000_20261031'
OUT=B/'reports/local-isotropic6000-20261031'


def state_summary(s):
    c=s['conditions'];clean=c['clean_rollout']['mse']
    return dict(clean_mse=clean,clean_over_initial_carry=clean/c['initial_carry']['mse'],
        clean_over_shifted_carry=clean/c['shifted_observed_carry']['mse'],
        teacher_over_shifted_carry=c['teacher_forced_prefix']['mse']/c['shifted_observed_carry']['mse'],
        wrong_order_over_clean=c['wrong_action_order']['mse']/clean,
        other_actions_over_clean=c['other_episode_actions']['mse']/clean,
        other_initial_over_clean=c['other_initial']['mse']/clean,
        observed_state_std=s['observed_state_std_mean'])


def health_summary(h):
    s=h['summary'];f=s['fm_t0.95_forecast']
    return dict(feature_std=h['feature_std'],forecast_fm_t095=f,
        wrong_state_over_forecast=s['fm_t0.95_wrong_state']/f,
        wrong_actions_over_forecast=s['fm_t0.95_wrong_actions']/f,
        carry_over_forecast=s['fm_t0.95_carry']/f,
        zero_over_forecast=s['fm_t0.95_zero']/f)


def collect():
    src,helper,proto,plan,compat=[read(ROOT/n) for n in ['source-record.json','helper-source-record.json','protocol.json','plan.json','source-compatibility.json']]
    assert src['source_commit']==proto['training_source_commit']=='32fad0cc04ca46cb9e6f5516a9aa0525b65b4cea'
    assert helper['source_commit']==proto['helper_source_commit']=='da26748c866137ad1ea06cf7250cabd64431e9ac'
    assert len(plan['jobs'])==64 and proto['initial_step']==2000 and proto['intermediate_step']==4000 and proto['stop_step']==6000
    provenance={}
    for k,source in [('training',src),('helper',helper)]:
        provenance[k]=verify_manifest(source)
        assert provenance[k]['sha256']==compat[k]['manifest_sha256'] and compat[k]['source']==source
    runtime=read(src['runtime_manifest']);assert src['runtime_manifest']==helper['runtime_manifest']
    for name,sha in runtime['files'].items():assert digest(Path(src['runtime_source'])/name)==sha
    provenance['runtime']=artifact(src['runtime_manifest'])
    jobs={j['id']:j for j in plan['jobs']};exits=[]
    for j in plan['jobs']:
        e=read(ROOT/'exits'/(j['id']+'.json'));assert e['id']==j['id'] and e['returncode']==0
        assert j['source']==(helper if j['phase'].startswith('fullstate') else src)
        exits.append(dict(id=j['id'],phase=j['phase'],returncode=0))
    previous_path=B/'reports/local-isotropic2000-20261031/results.json';previous=read(previous_path)
    assert previous['completed_endpoint_count']==64 and previous['source']==src
    old={r['id']:r for r in previous['rows']};parent=Path(proto['parent'])
    signatures={};rows=[]
    def load_planning(path,task,case):
        x=read(path);values,rate=successes(x);signature=planning_signature(x);key=(task,case)
        if key in signatures:assert canonical(signatures[key])==canonical(signature)
        else:signatures[key]=signature
        return dict(success_rate=rate,outcomes=values,artifact=artifact(path)),x
    for rec in proto['allocations']:
        name=rec['id'];run=ROOT/name;before=old[name]
        paths=dict(config=run/'config.json',status=run/'status.json',stage=run/'stage-complete.json',size=run/'model_size.json',
            health=ROOT/('health-'+name)/'health.json',state=ROOT/('transition-'+name)/'state-audit.json',windows=ROOT/('transition-'+name)/'windows.json')
        cfg,status,stage,size,h,s,w=[read(paths[k]) for k in paths]
        assert semantic(cfg)==semantic(before['config'])==semantic(read(parent/name/'config.json'))
        assert cfg['batch_size']==128 and cfg['max_steps']==124910 and cfg['model_class']=='spatial_carrier'
        assert cfg['retain_steps']==[4000,6000] and cfg['checkpoint_every']==2000
        assert status['state']=='finished' and status['step']==stage['stop_step']==h['audit']['step']==s['step']==6000
        assert stage['stage_complete'] and stage['initial_step']==4000
        assert all(v for v in h['audit'].values() if isinstance(v,bool))
        for k,e in h['audit']['gradient_expectations'].items():
            if e=='active':assert h['audit']['gradient_max'][k]>0
            if e=='frozen':assert h['audit']['gradient_max'][k]==0
        cfgsha=digest(paths['config'])
        for phase in ['learn4000','learn6000']:assert jobs[phase+'-'+name]['config_sha256']==cfgsha
        assert s['config_sha256']==cfgsha and size==before['model_size'] and size['total']==h['parameters']
        assert s['manifest_sha256']==digest(paths['windows'])==before['state']['manifest_sha256']
        assert w['count']==s['cases']==64 and read(before['artifacts']['windows']['path'])==w
        assert [{k:r[k] for k in ['episode','start']} for r in h['cases']]==[
            {k:r[k] for k in ['episode','start']} for r in before['health']['cases']]
        pre=read(run/'pre-resume-fullstate.json')
        assert pre['passed'] and pre['copy_byte_identical'] and pre['step']==2000 and all(pre['checks'].values())
        assert pre['checkpoint_sha256']==digest(parent/name/'checkpoint-step0002000.ckpt')
        assert pre['checkpoint_sha256']==before['fullstate'][-1]['audit']['checkpoint_sha256']
        full=[]
        for step in [4000,6000]:
            q=ROOT/f'fullstate{step}-{name}/fullstate.json';a=read(q)
            assert a['passed'] and all(a['checks'].values()) and a['step']==step and a['campaign_samples']==step*128
            assert a['frozen_tensors']['equal'] and a['frozen_tensors']['compared_tensors']==210 and not a['frozen_tensors']['different']
            assert a['source_commit']==src['source_commit'] and a['helper_source_commit']==helper['source_commit']
            assert a['config_sha256']==cfgsha and a['native_stage_complete']['stop_step']==step
            assert a['native_stage_complete']['initial_step']==(2000 if step==4000 else 4000)
            archive=run/f'checkpoint-step{step:07d}.ckpt';assert Path(a['archived_checkpoint'])==archive
            assert digest(archive)==a['checkpoint_sha256']
            portable=run/f'model-step{step:07d}';assert digest(portable/'weights.pt')==a['portable_sha256']
            export=read(portable/'config.json');assert all(export[k]==v for k,v in cfg['model'].items())
            full.append(dict(audit=a,artifact=artifact(q),archive_sha_verified=True,portable_sha_verified=True))
        assert digest(run/'last.ckpt')==full[-1]['audit']['checkpoint_sha256']==read(ROOT/('fullstate6000-'+name)/'fullstate.json')['checkpoint_sha256']
        assert s['weights_sha256']==full[-1]['audit']['portable_sha256']
        cal=cfg['spatial_calibration'];assert digest(cal['path'])==cal['sha256']==before['config']['spatial_calibration']['sha256']
        assert read(cal['path'])['normalization']=='global_rms'
        planning={};parentplanning={}
        for case,phase in [('primary','planning'),('fresh','fresh-planning')]:
            p=ROOT/(phase+'-'+name)/'metrics.json';planning[case],x=load_planning(p,rec['task'],case)
            job=jobs[phase+'-'+name];manifest=Path(job['command'][job['command'].index('--manifest')+1])
            assert digest(manifest)==x['manifest_sha256']==job['manifest_sha256'] and read(manifest)['pairs']==x['pairs']
            assert x['checkpoint']==str(run/'model-step0006000') and x['checkpoint_audit']['parameters']==size['total']
            if case=='primary':desc=before['artifacts']['planning']
            else:desc=dict(path=proto['fresh2000_evidence'][name]['evaluation'],sha256=proto['fresh2000_evidence'][name]['sha256'])
            assert digest(desc['path'])==desc['sha256'];parentplanning[case],oldp=load_planning(desc['path'],rec['task'],case)
            assert oldp['checkpoint']==str(parent/name/'model-step0002000')
        trace=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines() if x.strip()]
        assert all(math.isfinite(v) for x in trace for v in x.values() if isinstance(v,float))
        rows.append(dict(**rec,seed=rec['training_seed'],step=6000,model_size=size,planning=planning,parent2000=parentplanning,
            fullstate=full,pre_resume=pre,health=h,state=s,old_health=health_summary(before['health']),old_state=state_summary(before['state']),
            health_summary=health_summary(h),state_summary=state_summary(s),artifacts={k:artifact(v) for k,v in paths.items()}))
        print(json.dumps(dict(audited=name)),flush=True)
    reference_path=B/'reports/carrier6000-20261031/results.json';reference=read(reference_path)
    native_path=B/'reports/native-lewm6000-20261031/results.json';native=read(native_path)
    contexts=[]
    for r in reference['rows']:
        arm='F-'+r['arm'] if r['family']=='carrier' else 'TF'
        planning={}
        for case in ['primary','fresh']:
            desc=r['planning'][case]['source'];assert digest(desc['path'])==desc['sha256']
            planning[case],_=load_planning(desc['path'],r['task'],case)
        contexts.append(dict(task=r['task'],seed=r['seed'],arm=arm,model_size=r['model_size'],planning=planning))
    for r in native['rows']:
        assert r['fullstate_passed'] and r['direct_buffer_only_check']
        for mode in ['raw','cal']:
            planning={}
            for case in ['primary','fresh']:
                root=B/'runs/native_lewm6000_20261031'
                if case=='primary':ident=('planning-' if mode=='raw' else 'planning-calibrated-')+r['id']
                elif mode=='cal':ident='fresh-planning-calibrated-'+r['id']
                else:root=B/'runs/native_rawfresh6000_20261031';ident='fresh-planning-raw-'+r['id']
                assert read(root/'exits'/(ident+'.json'))['returncode']==0
                planning[case],x=load_planning(root/ident/'metrics.json',r['task'],case)
                assert planning[case]['outcomes']==r['planning'][case][mode]['outcomes']
                assert abs(planning[case]['success_rate']-r['planning'][case][mode]['success'])<1e-8
            contexts.append(dict(task=r['task'],seed=r['seed'],arm='native-'+mode,model_size=r['model_size'],planning=planning))
    assert len(rows)==8 and len(contexts)==20
    lookup={(r['task'],r['seed'],r['arm']):r for r in rows+contexts}
    comparisons=[];summaries=[]
    for task in ['tworoom','cube']:
        refs=['dense','F-identity','TF','native-raw','native-cal']
        if task=='tworoom':refs+=['F-unconstrained','F-bounded']
        for case in ['primary','fresh']:
            for arm,ref in [('local',b) for b in refs]+[('local','same2000'),('dense','same2000')]:
                group=[]
                for seed in [1,2]:
                    a=lookup[task,seed,arm]
                    b=a['parent2000'][case] if ref=='same2000' else lookup[task,seed,ref]['planning'][case]
                    x=dict(task=task,case_set=case,seed=seed,candidate=arm,reference=ref,**paired_outcomes(a['planning'][case]['outcomes'],b['outcomes']))
                    comparisons.append(x);group.append(x)
                summaries.append(dict(task=task,case_set=case,candidate=arm,reference=ref,**paired_summary(group,10000)))
    return dict(root=str(ROOT),completed_endpoint_count=64,exits=exits,source=src,helper=helper,provenance=provenance,
        protocol=proto,rows=rows,contexts=contexts,comparison_summaries=summaries,comparisons=comparisons,
        parent_report=artifact(previous_path),reference_report=artifact(reference_path),native_report=artifact(native_path),collector=artifact(__file__),
        scope='Read-only CPU audit of completed runs. Archive/portable hashes verified; existing immutable CPU audit proves finite '
              'fullstate/native load/optimizer/scheduler/RNG/sampler and210 frozen tensors. No new model load or forward. '
              'All paired CIs jointly resample50cases over fixed2 seeds10000times seed1042; no training-seed-population or multiplicity adjustment.')


def compact(r,path):
    out={k:v for k,v in r.items() if k in ['root','completed_endpoint_count','source','helper','provenance','comparison_summaries','parent_report','reference_report','native_report','collector','scope']}
    out['full_results']=artifact(path)
    out['rows']=[dict(task=x['task'],seed=x['seed'],arm=x['arm'],model_size=x['model_size'],
        planning={k:v['success_rate'] for k,v in x['planning'].items()},parent2000={k:v['success_rate'] for k,v in x['parent2000'].items()},
        state=x['state_summary'],old_state=x['old_state'],health=x['health_summary'],old_health=x['old_health'],
        fullstate_checks=[dict(step=a['audit']['step'],passed=a['audit']['passed'],frozen210=a['audit']['frozen_tensors']['equal'],
            archive_sha_verified=a['archive_sha_verified'],portable_sha_verified=a['portable_sha_verified']) for a in x['fullstate']]) for x in r['rows']]
    out['references']=[dict(task=x['task'],seed=x['seed'],arm=x['arm'],model_size=x['model_size'],planning={k:v['success_rate'] for k,v in x['planning'].items()}) for x in r['contexts']]
    return out


def report(s):
    lines=['# Local/dense6000：Cube小幅候选收益，未超过强基线','',
        '全部64端点exit0；Cube local在fresh较dense有小幅改善，但primary差距较小，对native/TF没有稳定胜出证据。TwoRoom local明显好于dense，仍低于64维F与native。两任务、两seed和raw/calibrated native均完整保留，不按任务选择较好native变体。','',
        '| 任务 | 方法 | primary seed1/2 | fresh seed1/2 |','|---|---|---:|---:|']
    for task in ['tworoom','cube']:
        for arm in ['local','dense','F-identity','F-unconstrained','F-bounded','TF','native-raw','native-cal']:
            rows=sorted([x for x in s['rows']+s['references'] if x['task']==task and x['arm']==arm],key=lambda x:x['seed'])
            if not rows:continue
            vals=['/'.join(f"{x['planning'][c]:g}" for x in rows) for c in ['primary','fresh']]
            lines.append(f"| {task} | {arm} | {vals[0]}% | {vals[1]}% |")
    lines+=['','| 任务/病例 | local减对照 | seed1/2 Δpp | 平均 [配对case95%CI] |','|---|---|---:|---:|']
    for x in s['comparison_summaries']:
        if x['candidate']!='local' or x['reference'] not in ['dense','native-raw','native-cal','F-identity','TF','same2000']:continue
        a,b=x['differences_by_seed_pp'];lo,hi=x['shared_case_bootstrap_ci95_pp']
        lines.append(f"| {x['task']}/{x['case_set']} | {x['reference']} | {a:+g}/{b:+g} | {x['mean_difference_pp']:+g} [{lo:+g},{hi:+g}] |")
    lines+=['','区间按50相同病例联合重采样两个固定训练seed（10000次，seed1042），不是训练seed总体区间，未多重校正。primary与fresh分别比较；fresh仍来自同validation pool，不能视为独立test。Cube free/bounded未训练6000，不以2000替代。','',
        '| 任务/方法 | seed | clean/initial-copy 2000→6000 | wrong-order/clean 6000 | renderer t=.95 forecast MSE 2000→6000 |','|---|---:|---:|---:|---:|']
    for r in s['rows']:
        lines.append(f"| {r['task']}/{r['arm']} | {r['seed']} | {r['old_state']['clean_over_initial_carry']:.3f}→{r['state']['clean_over_initial_carry']:.3f} | {r['state']['wrong_order_over_clean']:.2f} | {r['old_health']['forecast_fm_t095']:.5f}→{r['health']['forecast_fm_t095']:.5f} |")
    lines+=['','潜变量误差、动作干预和renderer去噪衡量不同机制：latent readout是64固定窗口的clean rollout；health是16窗口带噪初态的条件renderer诊断。它们的改善不等于更准确的物理状态或更高任务成功率；不跨不同表示直接比较MSE。冻结观察编码未改变，未重复训练/执行observed probe。','',
        '4000/6000各8个完整状态审计通过，原32fad0c训练源、da26748 helper及runtime全文件哈希通过；本收集重新核验所有完整归档/portable SHA。每次审计确认210冻结参数/测量buffer不变、原global-RMS校准不变、optimizer/scheduler/RNG/sampler恢复完整。原有50病例顺序、manifest、CEM/budget、strict判据及train动作归一化与2000和全部6000对照一致。两阶段边界2000→4000→6000、总scheduler124910不变。','',
        'local/dense约55.9M参数、1024维状态；F/TF约51.1M、64维；native约18.04M、192维。更新/数据/规划预算匹配不代表容量和算力匹配，local/dense固定投影与计算也有区别。native-cal额外固定128训练clips做无梯度BN统计，是单独报告的统一变体。当前结果不支持大幅超过基线或自动续训；保留所有checkpoint/原报告。完整逐case数据在EFS，summary.json含其SHA。','']
    return '\n'.join(lines)


if __name__=='__main__':
    r=collect();OUT.mkdir(parents=True,exist_ok=True);p=OUT/'results.json'
    p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');s=compact(r,p)
    (OUT/'summary.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    (OUT/'RESULTS.md').write_text(report(s));print(report(s))
