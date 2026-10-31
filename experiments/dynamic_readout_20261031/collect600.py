"""Complete-only CPU collection with explicit three-root recovery provenance."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

B = Path('/efs/prescottsun/SPT-JEPA')
TRAIN = B/'runs/dynamic_readout600_20261031'
FULL = B/'runs/dynamic_readout600_recovery_20261031'
EVAL = B/'runs/dynamic_readout600_health_recovery_20261031'
OLD = B/'runs/local_isotropic600_20261031'
PREFIX = 'dynamic_readout_module.'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8<<20), b''):
            h.update(block)
    return h.hexdigest()


def artifact(path):
    return dict(path=str(path), sha256=digest(path), bytes=Path(path).stat().st_size)


def signature(p):
    return {key:p[key] for key in ['manifest_sha256','pairs','solver','plan','protocol','split','normalization']}


def paired(a,b):
    a,b=np.asarray(a,dtype=int),np.asarray(b,dtype=int);assert a.shape==b.shape==(50,)
    delta=a-b;draws=delta[np.random.default_rng(1042).integers(50,size=(10000,50))].mean(1)*100
    return dict(difference_pp=float(delta.mean()*100),gained=int(((a==1)&(b==0)).sum()),
                lost=int(((a==0)&(b==1)).sum()),ci95_pp=np.quantile(draws,[.025,.975]).tolist(),
                per_case_difference=delta.tolist())


def collect(output):
    assert (TRAIN/'HOLD_DISPATCH').exists() and (FULL/'HOLD_DISPATCH').exists() and (EVAL/'HOLD_DISPATCH').exists()
    assert read(EVAL/'closure.json')['all24actual_exit0']
    # This head collector enforces all three roots, actual exits, fullstate and byte hashes.
    head_script=Path(__file__).with_name('head_mechanism.py')
    subprocess.run([sys.executable,'-B',str(head_script),'--root',str(TRAIN),'--recovery-root',str(FULL),
                    '--evaluation-root',str(EVAL),'--output',str(output/'head')],check=True)
    head_result=read(output/'head/head-mechanism.json');heads={x['id']:x for x in head_result['rows']}
    sources=[]
    for p in [OLD/'source-record.json',TRAIN/'source-record.json',
              B/'runs/dynamic_readout_recovery_source_20261031/source-record.json',
              B/'runs/dynamic_readout_health_source_20261031/source-record.json']:
        s=read(p);manifest=read(s['manifest']);assert manifest['source_commit']==s['source_commit']
        for name,h in manifest['files'].items():assert digest(Path(s['source'])/name)==h
        sources.append(dict(record=s,manifest=artifact(s['manifest'])))
    oldplan=read(OLD/'plan.json');rows=[];checks=[]
    for task in ['tworoom','cube']:
        for seed in [1,2]:
            reference=OLD/f'{task}-spatial-local-seed{seed}';refcfg=read(reference/'config.json')
            refstate=OLD/('transition-'+reference.name)
            reference_planning=read(OLD/('planning-'+reference.name)/'metrics.json')
            reference_health=read(OLD/('health-'+reference.name)/'health.json')
            for arm in ['identity','static','state']:
                run=reference if arm=='identity' else TRAIN/f'{task}-readout-{arm}-seed{seed}'
                resultroot=OLD if arm=='identity' else EVAL
                name=run.name;cfg=read(run/'config.json');stage=read(run/'stage-complete.json')
                assert stage['stage_complete'] and stage['stop_step']==600 and stage['initial_step']==0
                assert stage['config_sha256']==digest(run/'config.json')
                if arm!='identity':
                    expected=copy.deepcopy(refcfg)
                    expected.update(output=str(run),checkpoint_every=200,retain_steps=[200,400,600])
                    expected['model']['spatial_readout_mode']=arm
                    for suffix in ['input.weight','input.bias','output.weight','output.bias']:
                        expected['audit_expectations']['gradients'][PREFIX+suffix]='active'
                    assert cfg==expected
                else:
                    jobs=[j for j in oldplan['jobs'] if j['id'].endswith('-'+name) or j['output']==str(run)]
                    assert len(jobs)==5
                    assert all(read(OLD/'exits'/(j['id']+'.json'))['returncode']==0 for j in jobs)
                calibration=cfg['spatial_calibration'];assert digest(calibration['path'])==calibration['sha256']
                assert read(calibration['path'])['normalization']=='global_rms'
                ppath=resultroot/('planning-'+name)/'metrics.json'
                spath=resultroot/('transition-'+name)/'state-audit.json'
                hpath=resultroot/('health-'+name)/'health.json'
                wpath=resultroot/('transition-'+name)/'windows.json'
                p,s,h,w=map(read,[ppath,spath,hpath,wpath]);size=read(run/'model_size.json')
                assert signature(p)==signature(reference_planning)
                assert s['step']==h['audit']['step']==600 and s['config_sha256']==digest(run/'config.json')
                assert s['manifest_sha256']==digest(wpath) and w==read(refstate/'windows.json') and s['cases']==64
                assert all(v for v in h['audit'].values() if isinstance(v,bool))
                assert [(x['episode'],x['start'])for x in h['cases']]==[(x['episode'],x['start'])for x in reference_health['cases']]
                assert len(h['cases'])==16
                assert p['checkpoint_audit']['strict_equal'] and Path(p['checkpoint'])==run/'model-step0000600'
                assert size['total']==p['checkpoint_audit']['parameters']==h['parameters']
                outcomes=p['metrics']['episode_successes'];assert len(outcomes)==len(p['pairs'])==50
                assert all(v in [0,1,False,True]for v in outcomes)
                success=sum(outcomes)*2;assert abs(success-p['metrics']['success_rate'])<1e-8
                c=s['conditions'];clean=c['clean_rollout']['mse']
                row=dict(task=task,seed=seed,arm=arm,id=name,success=success,outcomes=outcomes,
                    rollout_over_copy=clean/c['initial_carry']['mse'],
                    one_step_over_copy=c['teacher_forced_prefix']['mse']/c['shifted_observed_carry']['mse'],
                    other_initial_ratio=c['other_initial']['mse']/clean,
                    other_actions_ratio=c['other_episode_actions']['mse']/clean,
                    wrong_order_ratio=c['wrong_action_order']['mse']/clean,
                    renderer={key:h['summary']['fm_t0.95_'+key]/h['summary']['fm_t0.95_forecast']for key in ['wrong_state','wrong_actions','carry']},
                    state=s,health=h,size=size,planning_signature=signature(p),
                    trace=[json.loads(line)for line in(run/'metrics.jsonl').read_text().splitlines()],
                    artifacts={key:artifact(path)for key,path in dict(config=run/'config.json',planning=ppath,
                        state=spath,health=hpath,windows=wpath).items()})
                if arm!='identity':
                    native=read(FULL/('fullstate-'+name)/'preflight.json')
                    extra=read(FULL/('fullstate-'+name)/'readout-audit.json')
                    side=read(resultroot/('health-'+name)/'exact-parameter-evidence.json')
                    assert native['passed'] and extra['passed'] and side['passed']
                    assert extra['original_initial_tensors_exact']==553 and extra['frozen_tensors_unchanged']==210
                    assert extra['reference_initial_sha256']==digest(reference/'initial-model/weights.pt')
                    assert extra['weights_sha256']==s['weights_sha256']==heads[name]['weights_sha256']
                    refsize=read(reference/'model_size.json')
                    assert size['total']-refsize['total']==size['trainable']-refsize['trainable']==5376
                    with np.load(refstate/'state-audit.npz')as a,np.load(resultroot/('transition-'+name)/'state-audit.npz')as b:
                        assert np.array_equal(a['observed'],b['observed']) and np.array_equal(a['actions'],b['actions'])
                    row['head']=heads[name]
                    checks.append(dict(id=name,original_initial553=True,frozen210=True,observed_and_actions_byte_identical=True,
                        same_cases_solver_normalization=True,extra_parameters=5376,fullstate=artifact(FULL/('fullstate-'+name)/'preflight.json'),
                        direct_gradient=extra['head_gradient'],strict_health_sidecar=artifact(resultroot/('health-'+name)/'exact-parameter-evidence.json')))
                rows.append(row)
    comparisons=[];summaries=[]
    for task in ['tworoom','cube']:
        for candidate,reference in [('static','identity'),('state','identity'),('state','static')]:
            pairrows=[]
            for seed in [1,2]:
                a=next(x for x in rows if(x['task'],x['seed'],x['arm'])==(task,seed,candidate))
                b=next(x for x in rows if(x['task'],x['seed'],x['arm'])==(task,seed,reference))
                pairrows.append(dict(task=task,seed=seed,candidate=candidate,reference=reference,**paired(a['outcomes'],b['outcomes'])))
            comparisons.extend(pairrows)
            matrix=np.asarray([x['per_case_difference']for x in pairrows],dtype=float)
            draws=matrix.mean(0)[np.random.default_rng(1042).integers(50,size=(10000,50))].mean(1)*100
            summaries.append(dict(task=task,candidate=candidate,reference=reference,
                mean_difference_pp=float(matrix.mean()*100),differences_by_seed_pp=[x['difference_pp']for x in pairrows],
                paired_shared_case_ci95_pp=np.quantile(draws,[.025,.975]).tolist()))
    return dict(rows=rows,checks=checks,comparisons=comparisons,comparison_summaries=summaries,sources=sources,
        completion=head_result['completion'],head_artifact=artifact(output/'head/head-mechanism.json'),collector=artifact(__file__),
        scope='Complete-only CPU artifact/head checks. Paired intervals resample the same50 cases jointly across the two fixed seeds,10000 draws seed1042; conditional on these trained models, not training-seed uncertainty or multiplicity-adjusted.600<1249warmup. No newcohort/model forward/GPU/training or continuation.')


def report(s):
    lines=['# Dynamic readout600 complete review','',
        '训练8组均有600步stage-complete和通过的完整状态实物审计；原OS退出码未知。首恢复保留8个fullstate退出0、8个旧health退出1和16个阻断端点；最终恢复24个实际退出0。没有补造40/40exit0。所有8组原553初始张量、210冻结张量、原病例/规划器/归一化匹配；新增5376参数，观测状态与原identity逐元素一致。','',
        '|任务|identity seed1/2|STATIC seed1/2|STATE seed1/2|STATE−STATIC pp [条件95%CI]|','|---|---:|---:|---:|---:|']
    for task in ['tworoom','cube']:
        values=['/'.join(f"{x['success']:g}"for x in sorted([r for r in s['rows']if r['task']==task and r['arm']==arm],key=lambda x:x['seed']))for arm in ['identity','static','state']]
        q=next(x for x in s['comparison_summaries']if(x['task'],x['candidate'],x['reference'])==(task,'state','static'))
        lines.append('|{}|{}|{}|{}|{:+.1f} [{:+.1f},{:+.1f}]|'.format(task,*values,q['mean_difference_pp'],*q['paired_shared_case_ci95_pp']))
    lines+=['','|任务/seed|臂|clean rollout/初态复制|单步/相邻复制|错初态/别段动作 MSE比|M entry RMS|同位置状态方差/总能量|','|---|---|---:|---:|---:|---:|---:|']
    for r in s['rows']:
        h=r.get('head',{}).get('clean_prefix',{})
        lines.append('|{}/{}|{}|{:.3f}|{:.3f}|{:.3f}/{:.3f}|{}|{}|'.format(r['task'],r['seed'],r['arm'],r['rollout_over_copy'],r['one_step_over_copy'],r['other_initial_ratio'],r['other_actions_ratio'],
            f"{h['matrix_entry_rms']:.5f}"if h else '—',f"{h['same_position_variance_over_total']:.3f}"if h else '—'))
    lines+=['','所有CI仅按共同50病例、固定两个训练seed条件计算，未校正多重比较。Cube已有原协议重复波动，不把少量成功翻转当强证据。600仍处于1249步warmup内；非零梯度/M或更小预测误差都不是成熟控制效果。M(0,pos)参照不是保证在分布内的状态。缓存缺少修正前p和δ，故不报告Mδ/δ，也不反推缺失量。原逐参数训练日志伪0被保留；600完整审计直接p.grad及严格health sidecar是相应检查依据。尚未续训。']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    result=collect(args.output)
    (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    keys=['task','seed','arm','id','success','rollout_over_copy','one_step_over_copy','other_initial_ratio','other_actions_ratio','wrong_order_ratio','renderer','size','head']
    summary=dict(full_results=artifact(args.output/'results.json'),rows=[{k:r[k]for k in keys if k in r}for r in result['rows']],
        completion=result['completion'],checks=result['checks'],comparisons=[{k:v for k,v in p.items()if k!='per_case_difference'}for p in result['comparisons']],
        comparison_summaries=result['comparison_summaries'],sources=result['sources'],collector=result['collector'],scope=result['scope'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (args.output/'RESULTS.md').write_text(report(summary))
    print(json.dumps(dict(output=str(args.output),comparison_summaries=result['comparison_summaries'])),flush=True)
