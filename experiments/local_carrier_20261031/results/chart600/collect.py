"""CPU-only completed chart600 collection, including cached common coordinates."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from as_jepa_lewm.carrier import ResidualCarrierChart
from experiments.local_carrier_20261031.results.basis600.collect import finite
from experiments.ideas_validate_20261031.collect import read,digest,canonical,paired_outcomes
from experiments.carrier_progress_20261031.collect_learning import artifact,paired_summary,planning_signature,successes,verify_manifest

B=Path('/efs/prescottsun/SPT-JEPA');ROOT=B/'runs/local_spatial_chart600_20261031'
OLD=B/'runs/local_isotropic600_20261031';OUT=B/'reports/local-spatial-chart600-20261031'


@torch.no_grad()
def cached_geometry(chart,oldpath,newpath):
    with np.load(oldpath) as a,np.load(newpath) as b:
        x=torch.from_numpy(a['observed']).float();y=torch.from_numpy(b['observed']).float()
        assert x.shape==y.shape==(64,8,1024) and np.array_equal(a['actions'],b['actions'])
        apply=lambda z:chart(z.reshape(*z.shape[:-1],64,16)).reshape_as(z)
        inverse=lambda z:chart.inverse(z.reshape(*z.shape[:-1],64,16),iterations=32).reshape_as(z)
        check=(apply(x)-y).abs().max().item();assert check<1e-4
        invcheck=(inverse(y)-x).abs().max().item();assert invcheck<1e-4
        d=(y-x).double();flat=d.flatten(0,1);xf=x.double().flatten(0,1)
        rms=lambda z:float(z.double().square().mean().sqrt())
        change=dict(rms=rms(d),relative_rms=rms(d)/rms(x),max_abs=float(d.abs().max()),
            centered_relative_rms=rms(flat-flat.mean(0))/rms(xf-xf.mean(0)),
            chart_reapplication_max_abs=check,inverse_observed_vs_original_max_abs=invcheck)
        target=x[:,1:].double();initial=float((target-x[:,:1]).square().mean());local=float((target-x[:,:-1]).square().mean())
        common={}
        for name in ['clean_rollout','teacher_forced_prefix']:
            z=torch.from_numpy(b[name]).float();decoded=inverse(z)
            residual=(apply(decoded)-z).abs().max().item();assert residual<1e-4
            mse=float((decoded.double()-target).square().mean())
            common[name]=dict(mse=mse,over_initial_copy=mse/initial,over_shifted_copy=mse/local,inverse_residual_max_abs=residual)
    return dict(identity_deviation=change,common_coordinates=common,inverse_iterations=32,
        scope='Only saved observed/prediction arrays and2128 chart parameters on CPU. No encoder/dynamics/renderer forward or new cases; contraction inverse checked by reapplication.')


def collect():
    torch.set_num_threads(2)
    plan,proto,src=read(ROOT/'plan.json'),read(ROOT/'protocol.json'),read(ROOT/'source-record.json')
    oldplan=read(OLD/'plan.json');assert len(plan['jobs'])==20 and proto['step']==600
    assert (ROOT/'HOLD_DISPATCH').exists() and read(ROOT/'closed-dev13.json')['all20_exit0']
    sources=[dict(source=s,manifest=verify_manifest(s)) for s in [src,read(OLD/'source-record.json')]]
    exits=[]
    for j in plan['jobs']:
        e=read(ROOT/'exits'/(j['id']+'.json'));assert e['returncode']==0 and j['source']==src;exits.append(e)
    rows,checks,pairs=[],[],[]
    for rec in proto['runs']:
        name,task,seed=rec['id'],rec['task'],rec['training_seed'];run=ROOT/name
        ref=OLD/f'{task}-spatial-local-seed{seed}';cfg=read(run/'config.json');expected=copy.deepcopy(read(ref/'config.json'))
        expected['output']=str(run);expected['model']['spatial_chart']='bounded';assert cfg==expected
        cal=cfg['spatial_calibration'];assert digest(cal['path'])==cal['sha256'] and read(cal['path'])['normalization']=='global_rms'
        fp=ROOT/('fullstate-'+name)/'preflight.json';full=read(fp)
        assert full['passed'] and all(full['checkpoint']['checks'].values()) and full['checkpoint']['step']==600
        assert full['frozen_tensors_unchanged']['equal'] and full['frozen_tensors_unchanged']['compared_tensors']==210
        assert full['gradients']['carrier_chart_module']['expectation']=='active' and full['gradients']['carrier_chart_module']['passed']
        assert all(v['passed'] for v in full['gradients'].values()) and full['all_parameter_gradients_finite']
        assert digest(run/'last.ckpt')==full['checkpoint_sha256']
        original=torch.load(ref/'initial-model/weights.pt',map_location='cpu',weights_only=True)
        initial=torch.load(run/'initial-model/weights.pt',map_location='cpu',weights_only=True)
        final=torch.load(run/'model-step0000600/weights.pt',map_location='cpu',weights_only=True)
        prefix='carrier_chart_module.';extra={prefix+k for k in ['input.weight','input.bias','output.weight','output.bias']}
        assert len(original)==553 and set(initial)==set(original)|extra
        assert all(torch.equal(v,initial[k]) for k,v in original.items())
        frozen=[k for k in original if k.startswith(('encoder.','pool.','spatial_'))]
        assert len(frozen)==210 and all(torch.equal(initial[k],final[k]) for k in frozen)
        assert digest(run/'model-step0000600/weights.pt')==full['weights_sha256']
        changed=[k for k in extra if not torch.equal(initial[k],final[k])];assert changed
        chart=ResidualCarrierChart(16,64,'bounded',.5)
        chart.load_state_dict({k.removeprefix(prefix):final[k] for k in extra},strict=True)
        certificate=chart.norm_certificate();assert certificate['residual_lipschitz_upper']<=.500001
        checks.append(dict(task=task,seed=seed,original_initial_tensors_exact=553,frozen_tensors600=210,
            changed_chart_tensors=changed,chart_gradient_norm=full['gradients']['carrier_chart_module']['gradient_norm'],
            fullstate=artifact(fp),initial=artifact(run/'initial-model/weights.pt'),
            original_initial=artifact(ref/'initial-model/weights.pt'),certificate=certificate,passed=True))
        del original,initial,final
        localrows=[]
        for arm,root,rr in [('identity',OLD,ref),('shared-chart',ROOT,run)]:
            rn=rr.name;pp=root/('planning-'+rn)/'metrics.json';sp=root/('transition-'+rn)/'state-audit.json'
            hp=root/('health-'+rn)/'health.json';wp=root/('transition-'+rn)/'windows.json'
            p,s,h,w=map(read,[pp,sp,hp,wp]);stage=read(rr/'stage-complete.json');status=read(rr/'status.json');size=read(rr/'model_size.json')
            relevant=[j for j in (oldplan if arm=='identity' else plan)['jobs'] if j['id'].endswith('-'+rn) or j['output']==str(rr)]
            assert len(relevant)==5 and all(read(root/'exits'/(j['id']+'.json'))['returncode']==0 for j in relevant)
            assert stage['initial_step']==0 and stage['stop_step']==status['step']==s['step']==h['audit']['step']==600
            assert stage['stage_complete'] and status['state']=='finished'
            assert all(v for v in h['audit'].values() if isinstance(v,bool))
            assert s['manifest_sha256']==digest(wp) and s['cases']==w['count']==64
            assert s['config_sha256']==digest(rr/'config.json')
            assert size['total']==p['checkpoint_audit']['parameters']==h['parameters'] and Path(p['checkpoint'])==rr/'model-step0000600'
            oo,rate=successes(p);trace=[json.loads(x) for x in (rr/'metrics.jsonl').read_text().splitlines() if x.strip()]
            assert finite(trace) and finite(s) and finite(h)
            c=s['conditions'];clean=c['clean_rollout']['mse']
            row=dict(task=task,seed=seed,arm=arm,success=rate,episode_successes=oo,
                rollout=clean/c['initial_carry']['mse'],one_step=c['teacher_forced_prefix']['mse']/c['shifted_observed_carry']['mse'],
                wrongstate_ratio=c['other_initial']['mse']/clean,otheractions_ratio=c['other_episode_actions']['mse']/clean,
                wrongorder_ratio=c['wrong_action_order']['mse']/clean,
                renderer={k:h['summary']['fm_t0.95_'+k]/h['summary']['fm_t0.95_forecast'] for k in ['wrong_state','wrong_actions','carry']},
                model_size=size,planning_seconds=p['seconds'],state=s,health=h,training=trace,planning_signature=planning_signature(p),windows=w,
                artifacts={k:artifact(v) for k,v in dict(config=rr/'config.json',planning=pp,state=sp,health=hp,windows=wp,
                    cached_states=root/('transition-'+rn)/'state-audit.npz',trace=rr/'metrics.jsonl').items()})
            localrows.append(row)
        a,b=localrows
        assert canonical(a['planning_signature'])==canonical(b['planning_signature']) and a['windows']==b['windows']
        assert [{k:r[k] for k in ['episode','start']} for r in a['health']['cases']]==[{k:r[k] for k in ['episode','start']} for r in b['health']['cases']]
        assert b['model_size']['total']-a['model_size']['total']==b['model_size']['trainable']-a['model_size']['trainable']==2128
        recorded=b['state']['carrier_chart_certificate']
        for key in ['input_operator_norm','output_operator_norm','residual_lipschitz_upper']:
            assert abs(recorded[key]-certificate[key])<1e-6
        b['chart_analysis']=cached_geometry(chart,a['artifacts']['cached_states']['path'],b['artifacts']['cached_states']['path'])
        b['certificate']=certificate
        a['common_rollout'],a['common_one_step']=a['rollout'],a['one_step']
        common=b['chart_analysis']['common_coordinates']
        b['common_rollout']=common['clean_rollout']['over_initial_copy'];b['common_one_step']=common['teacher_forced_prefix']['over_shifted_copy']
        rows.extend(localrows);pairs.append(dict(task=task,seed=seed,**paired_outcomes(b['episode_successes'],a['episode_successes'],10000)))
        print(json.dumps(dict(checked=name,success=b['success'],chart_rms_fraction=b['chart_analysis']['identity_deviation']['relative_rms'],common_rollout=b['common_rollout'],native_rollout=b['rollout'])),flush=True)
    summaries=[dict(task=t,**paired_summary([p for p in pairs if p['task']==t],10000)) for t in ['tworoom','cube']]
    return dict(created_at=time.time(),completed_endpoint_count=20,exits=exits,sources=sources,checks=checks,rows=rows,comparisons=pairs,
        comparison_summaries=summaries,collector=artifact(__file__),scope='CPU completed files/tensors/cached-code analysis only. Native stage held; no new encoder/dynamics/probe/GPU or training.')


def report(s):
    lines=['共享local chart的600步短门未建立稳定性能提升。20/20端点exit0，4对553初始张量一致、210冻结张量不变，chart梯度有效且4个张量已学习；新增2128个可训练参数。原校准、病例和CEM预算匹配，调度949185及监控949186均已关闭并HOLD。','',
        '| 任务 | 原local成功率 seed1/2 | chart成功率 seed1/2 | 均值Δpp [配对95%区间] |','|---|---:|---:|---:|']
    for q in s['comparison_summaries']:
        values=[]
        for arm in ['identity','shared-chart']:
            rr=sorted([r for r in s['rows'] if r['task']==q['task'] and r['arm']==arm],key=lambda r:r['seed'])
            values.append('/'.join(f"{r['success']:g}" for r in rr))
        lo,hi=q['shared_case_bootstrap_ci95_pp'];lines.append(f"| {q['task']} | {values[0]} | {values[1]} | {q['mean_difference_pp']:+g} [{lo:+g},{hi:+g}] |")
    lines+=['','| 任务/seed | 基底 | 同坐标rollout/复制 | 同坐标单步/复制 | 错初态/别段动作 | DiT错初态/错动作 | code偏移RMS% / 残差L上界 |','|---|---|---:|---:|---:|---:|---:|']
    for r in s['rows']:
        extra='0 / 0' if r['arm']=='identity' else f"{100*r['chart_analysis']['identity_deviation']['relative_rms']:.2f} / {r['certificate']['residual_lipschitz_upper']:.3f}"
        lines.append(f"| {r['task']}/{r['seed']} | {r['arm']} | {r['common_rollout']:.3f} | {r['common_one_step']:.3f} | "
            f"{r['wrongstate_ratio']:.3f}/{r['otheractions_ratio']:.3f} | {r['renderer']['wrong_state']:.4f}/{r['renderer']['wrong_actions']:.4f} | {extra} |")
    lines+=['','同坐标误差把已缓存chart预测用其收缩逆映射回原local载体后计算，避免坐标收缩冒充预测改善；没有新encoder或世界模型推理。干预比值及t=.95 DiT比值使用各自训练坐标。'
        '四组W1谱范数归一化均活跃，W2均未触及1；实际残差L约.067–.082，远低于配置.5上限，不能据此声称上限约束带来性能收益。'
        'chart已偏离identity，但证书仅证明数学上的可逆/距离界，不证明任务相关几何。600步仍在1249步warmup内；区间按50病例重采样10000次，条件于两个训练seed。未续训。']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    result=collect();OUT.mkdir(exist_ok=False);(OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    keys=['task','seed','arm','success','rollout','one_step','common_rollout','common_one_step','wrongstate_ratio','otheractions_ratio','wrongorder_ratio','renderer','model_size','planning_seconds','chart_analysis','certificate']
    summary=dict(full_results=artifact(OUT/'results.json'),completed_endpoint_count=20,rows=[{k:r[k] for k in keys if k in r} for r in result['rows']],
        checks=result['checks'],comparison_summaries=result['comparison_summaries'],sources=result['sources'],collector=result['collector'],scope=result['scope'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');(OUT/'RESULTS.md').write_text(report(summary))
    print(json.dumps(summary['comparison_summaries']),flush=True)
