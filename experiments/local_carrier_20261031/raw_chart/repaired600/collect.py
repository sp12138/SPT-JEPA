"""CPU-only complete collection: three prior successes plus17 new endpoints."""
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from as_jepa_lewm.carrier import ResidualCarrierChart
from experiments.local_carrier_20261031.results.basis600.collect import finite
from experiments.ideas_validate_20261031.collect import read, digest, canonical, paired_outcomes
from experiments.carrier_progress_20261031.collect_learning import artifact, paired_summary, planning_signature, successes, verify_manifest

B=Path('/efs/prescottsun/SPT-JEPA')
ROOT=B/'runs/local_raw_chart_repaired600_20261031'
PRIOR=B/'runs/raw_chart_public_termination_gate600_retry_20261031'
OLD=B/'runs/local_isotropic600_20261031'
OUT=B/'reports/local-raw-chart-repaired600-20261031'


def collect():
    torch.set_num_threads(2)
    plan,proto,src=map(read,[ROOT/'plan.json',ROOT/'protocol.json',ROOT/'source-record.json'])
    assert len(plan['jobs'])==17 and (ROOT/'HOLD_DISPATCH').exists()
    assert read(ROOT/'closed-dev13.json')['all17_exit0']
    sources=[dict(source=s,manifest=verify_manifest(s)) for s in [src,read(ROOT/'helper-source-record.json'),read(OLD/'source-record.json')]]
    exits=[]
    for j in plan['jobs']:
        path=ROOT/'exits'/(j['id']+'.json'); e=read(path)
        assert e['returncode']==0 and j['source']==src
        exits.append(dict(id=j['id'],artifact=artifact(path),returncode=0))
    prior=[]
    for x in proto['prior_successful_endpoints']:
        assert digest(x['path'])==x['sha256'] and read(x['path'])['returncode']==0
        prior.append(x)
    assert len(prior)==3 and read(PRIOR/'final-review.json')['direct_all_initial557_equal']
    for path,record in read(ROOT/'activation.json')['preserved_holds'].items():
        assert digest(path)==record['sha256'] and Path(path).stat().st_mtime==record['mtime']
    rows,checks,pairs=[],[],[]
    for rec in proto['runs']:
        name,task,seed=rec['id'],rec['task'],rec['training_seed']
        rr=(PRIOR if rec['training_reused'] else ROOT)/name
        healthroot=PRIOR if rec['training_reused'] else ROOT
        ref=OLD/f'{task}-spatial-local-seed{seed}'
        cfg,base=read(rr/'config.json'),read(ref/'config.json')
        expected=copy.deepcopy(base); expected['output']=str(rr)
        expected['model'].update(spatial_chart='bounded',spatial_state_interface='raw')
        # Checkpoint cadence is operational; retained optimizer schedule unchanged.
        expected['checkpoint_every']=200;expected['retain_steps']=[200,400,600]
        assert cfg==expected,(name,set(cfg)^set(expected))
        cal=cfg['spatial_calibration'];assert digest(cal['path'])==cal['sha256']
        fp=healthroot/('fullstate-'+name)/'preflight.json';f=read(fp)
        assert f['passed'] and all(f['checkpoint']['checks'].values()) and f['checkpoint']['step']==600
        assert f['checkpoint']['campaign_samples']==76800 and f['optimizer_parameter_count']['passed']
        assert f['frozen_tensors_unchanged']['equal'] and f['frozen_tensors_unchanged']['compared_tensors']==210
        assert all(x['passed'] for x in f['gradients'].values()) and f['all_parameter_gradients_finite']
        assert digest(rr/'last.ckpt')==f['checkpoint_sha256'] and digest(rr/'model-step0000600/weights.pt')==f['weights_sha256']
        if not rec['training_reused']:
            pre=read(rr/'post-copy-state400.json')
            assert pre['passed'] and pre['checkpoint_sha256']==rec['fullstate_sha256']
            oldpre=read(B/'reports/raw-chart-precision600-failure-20261031'/(name+'-state400.json'))
            assert pre['rng_component_sha256']==oldpre['rng_component_sha256'] and pre['sampler']==oldpre['sampler']
            assert f['paired_initial_equal']['compared_tensors']==557
        initial=torch.load(rr/'initial-model/weights.pt',map_location='cpu',weights_only=True)
        original=torch.load(Path(rec['original_raw'])/'initial-model/weights.pt',map_location='cpu',weights_only=True)
        identity=torch.load(ref/'initial-model/weights.pt',map_location='cpu',weights_only=True)
        final=torch.load(rr/'model-step0000600/weights.pt',map_location='cpu',weights_only=True)
        assert len(initial)==len(original)==557 and all(torch.equal(v,initial[k]) for k,v in original.items())
        assert len(identity)==553 and all(torch.equal(v,initial[k]) for k,v in identity.items())
        frozen=[k for k in initial if k.startswith(('encoder.','pool.','spatial_'))]
        assert len(frozen)==210 and all(torch.equal(initial[k],final[k]) for k in frozen)
        prefix='carrier_chart_module.';chartweights={k.removeprefix(prefix):v for k,v in final.items() if k.startswith(prefix)}
        assert len(chartweights)==4 and sum(v.numel() for v in chartweights.values())==2128
        changed=[k for k in initial if k.startswith(prefix) and not torch.equal(initial[k],final[k])];assert changed
        chart=ResidualCarrierChart(16,64,'bounded',.5);chart.load_state_dict(chartweights,strict=True)
        certificate=chart.norm_certificate();assert certificate['residual_lipschitz_upper']<=.500001
        checks.append(dict(task=task,seed=seed,initial557_exact=True,identity553_exact=True,frozen210=True,
            fullstate=artifact(fp),fullstate_sha256=f['checkpoint_sha256'],optimizer=f['optimizer_parameter_count'],
            chart_gradient=f['gradients']['carrier_chart_module'],certificate=certificate,changed_chart_tensors=changed))
        del initial,original,identity,final
        local=[]
        for arm,run,evaluations,healthbase in [('identity',ref,OLD,OLD),('raw-chart',rr,ROOT,healthroot)]:
            rn=run.name
            paths=dict(config=run/'config.json',planning=evaluations/('planning-'+rn)/'metrics.json',
                state=evaluations/('transition-'+rn)/'state-audit.json',windows=evaluations/('transition-'+rn)/'windows.json',
                cache=evaluations/('transition-'+rn)/'state-audit.npz',health=healthbase/('health-'+rn)/'health.json',
                status=run/'status.json',stage=run/'stage-complete.json',size=run/'model_size.json',trace=run/'metrics.jsonl')
            p,s,w,h,status,stage,size=[read(paths[k]) for k in ['planning','state','windows','health','status','stage','size']]
            assert stage['stage_complete'] and stage['stop_step']==status['step']==s['step']==h['audit']['step']==600
            assert status['state']=='finished' and all(v for v in h['audit'].values() if isinstance(v,bool))
            assert s['cases']==w['count']==64 and s['manifest_sha256']==digest(paths['windows'])
            assert s['config_sha256']==digest(paths['config']) and Path(p['checkpoint'])==run/'model-step0000600'
            oo,rate=successes(p);assert size['total']==p['checkpoint_audit']['parameters']==h['parameters']
            trace=[json.loads(x) for x in paths['trace'].read_text().splitlines() if x.strip()]
            assert finite(trace) and finite(s) and finite(h)
            c=s['conditions'];clean=c['clean_rollout']['mse']
            row=dict(task=task,seed=seed,arm=arm,success=rate,episode_successes=oo,
                rollout=clean/c['initial_carry']['mse'],one_step=c['teacher_forced_prefix']['mse']/c['shifted_observed_carry']['mse'],
                wrongstate_ratio=c['other_initial']['mse']/clean,otheractions_ratio=c['other_episode_actions']['mse']/clean,
                wrongorder_ratio=c['wrong_action_order']['mse']/clean,
                renderer={k:h['summary']['fm_t0.95_'+k]/h['summary']['fm_t0.95_forecast'] for k in ['wrong_state','wrong_actions','carry','zero'] if 'fm_t0.95_'+k in h['summary']},
                model_size=size,planning_seconds=p['seconds'],state=s,health=h,status=status,training_tail=trace[-4:],
                planning_signature=planning_signature(p),windows=w,artifacts={k:artifact(v) for k,v in paths.items()})
            local.append(row)
        a,b=local
        assert canonical(a['planning_signature'])==canonical(b['planning_signature']) and a['windows']==b['windows']
        assert [{k:x[k] for k in ['episode','start']} for x in a['health']['cases']]==[{k:x[k] for k in ['episode','start']} for x in b['health']['cases']]
        assert b['model_size']['total']-a['model_size']['total']==b['model_size']['trainable']-a['model_size']['trainable']==2128
        with np.load(a['artifacts']['cache']['path']) as ac,np.load(b['artifacts']['cache']['path']) as bc:
            assert np.array_equal(ac['actions'],bc['actions']) and np.array_equal(ac['observed'],bc['observed'])
            u=torch.from_numpy(bc['observed']).float();assert u.shape==(64,8,1024)
            with torch.no_grad():z=chart(u.reshape(64,8,64,16)).reshape_as(u)
            b['internal_chart_relative_rms']=float((z-u).double().square().mean().sqrt()/u.double().square().mean().sqrt())
            target=bc['observed'][:,1:].astype(np.float64)
            mse=float(np.mean((bc['clean_rollout'].astype(np.float64)-target)**2))
            assert abs(mse-b['state']['conditions']['clean_rollout']['mse'])<1e-6
        b['certificate']=certificate;b['public_observed_codes_exact_identity']=True
        rows.extend(local);pairs.append(dict(task=task,seed=seed,**paired_outcomes(b['episode_successes'],a['episode_successes'],10000)))
    summaries=[dict(task=t,**paired_summary([p for p in pairs if p['task']==t],10000)) for t in ['tworoom','cube']]
    return dict(created_at=time.time(),new_endpoints=17,prior_successful_endpoints=3,total_evidence_endpoints=20,
        new_exits=exits,prior_exits=prior,sources=sources,checks=checks,rows=rows,comparisons=pairs,comparison_summaries=summaries,
        collector=artifact(__file__),numerical_contracts=[artifact(Path(src['source'])/'experiments/local_carrier_20261031/raw_chart/public_termination_cpu'/n) for n in ['execution.json','native-contracts.json','termination-contracts.json']],
        coordinate_rule='Public/cached predictions already RAW. No second inverse or new model/encoder/GPU inference.',
        numerical_scope='1060->6e only continues the same fixed-point iterations when internal passes but actual public rounding fails. CPU passing-path output/input-and-parameter-gradient/RNG equality is proven for tested contracts, not every counterfactual training trajectory. Original400 states exact; all old failed/unknown/unstarted roots retained.')


def report(s):
    lines=['四条raw-chart600短门已完成：既有3个成功端点加新root17个真实exit0，全部封顶600并HOLD。4组557原初值、553 identity初值、210冻结张量、完整600 optimizer/RNG/cursor及原50病例/64窗口检查通过。新增2128参数；数据、sample-level目标和CEM预算未变。','',
        '|任务|identity seed1/2|raw-chart seed1/2|均值Δpp [配对95%区间]|','|---|---:|---:|---:|']
    for q in s['comparison_summaries']:
        vals=['/'.join(f"{r['success']:g}" for r in s['rows'] if r['task']==q['task'] and r['arm']==a) for a in ['identity','raw-chart']]
        lo,hi=q['shared_case_bootstrap_ci95_pp'];lines.append(f"|{q['task']}|{vals[0]}|{vals[1]}|{q['mean_difference_pp']:+g} [{lo:+g},{hi:+g}]|")
    lines+=['','|任务/seed|模型|rollout/初态复制|单步/移位复制|错初态/错动作|DiT错初态/错动作|rate600|','|---|---|---:|---:|---:|---:|---:|']
    for r in s['rows']:
        lines.append(f"|{r['task']}/{r['seed']}|{r['arm']}|{r['rollout']:.4f}|{r['one_step']:.4f}|{r['wrongstate_ratio']:.3f}/{r['otheractions_ratio']:.3f}|{r['renderer']['wrong_state']:.4f}/{r['renderer']['wrong_actions']:.4f}|{r['status']['rate']:.3f}|")
    lines+=['','缓存状态本身已是原始坐标，逐项验证观测码与identity完全相同，未再次inverse；chart只改变内部参数化，不能靠收缩降低raw rate。600仍在1249步warmup内；区间按同50病例共同重采样10000次，条件于两个训练seed，不代表seed总体不确定性。数值修复只改变内部证书通过但public舍入未通过时的继续迭代，原容差/64次上限未放宽；已通过路径的精确等价来自既有CPU合同，不能推广为所有轨迹反事实等价。原失败后缀和未启动schema记录保留，其额外计算是历史开销。没有>600续跑。']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    s=collect();OUT.mkdir(exist_ok=False);(OUT/'results.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    keep=['task','seed','arm','success','rollout','one_step','wrongstate_ratio','otheractions_ratio','wrongorder_ratio','renderer','status','model_size','planning_seconds','certificate','internal_chart_relative_rms','public_observed_codes_exact_identity']
    small={k:v for k,v in s.items() if k not in ['rows','new_exits','prior_exits','comparisons']}
    small['rows']=[{k:r[k] for k in keep if k in r} for r in s['rows']];small['full_results']=artifact(OUT/'results.json')
    (OUT/'summary.json').write_text(json.dumps(small,indent=2,allow_nan=False)+'\n');(OUT/'RESULTS.md').write_text(report(s))
    print(json.dumps(s['comparison_summaries']),flush=True)
