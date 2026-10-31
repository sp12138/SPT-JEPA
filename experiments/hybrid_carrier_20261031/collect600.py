"""Completed-only CPU review of hybrid600. No forward, fitting, or GPU call."""
import argparse
import gc
import hashlib
import json
import random
from functools import cache
from pathlib import Path

import numpy as np
import torch

from as_jepa_lewm.train import TrainModule,WindowSampler
from as_jepa_tools.benchmark_data import BenchmarkWindows
from experiments.ideas_validate_20261031.collect import paired_outcomes
from experiments.ideas_validate_20261031.preflight import finite
from vforge.ops import OpsConfig,initialize_ops
from collect_preflight import same

B=Path('/efs/prescottsun/SPT-JEPA')
def read(p):return json.loads(Path(p).read_text())
@cache
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def native_restore(run,cfg,datasets):
    """Same native CPU restoration used in the reviewed2-step collection, at600."""
    checkpoint=torch.load(run/'last.ckpt',map_location='cpu',weights_only=False)
    assert checkpoint['global_step']==600 and checkpoint['campaign_samples']==600*cfg['batch_size']
    torch.manual_seed(cfg['seed']);module=TrainModule(cfg)
    module.load_state_dict(checkpoint['state_dict'],strict=True)
    assert same(module.state_dict(),checkpoint['state_dict'])
    configured=module.configure_optimizers();optimizer=configured['optimizer'];scheduler=configured['lr_scheduler']['scheduler']
    optimizer.load_state_dict(checkpoint['optimizer_states'][0]);scheduler.load_state_dict(checkpoint['lr_schedulers'][0])
    assert same(optimizer.state_dict(),checkpoint['optimizer_states'][0])
    assert same(scheduler.state_dict(),checkpoint['lr_schedulers'][0])
    qstates=sum(p in optimizer.state for p in module.model.pool.parameters())
    assert qstates==(0 if cfg['model']['hybrid_query']=='fixed' else 9)
    rng=checkpoint['campaign_rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
    assert same(random.getstate(),rng['python']) and same(np.random.get_state(),rng['numpy']) and torch.equal(torch.get_rng_state(),rng['torch'])
    assert rng['cuda'] and all(torch.is_tensor(s) and s.dtype==torch.uint8 and s.numel()>0 for s in rng['cuda'])
    key=(cfg['data'],cfg['seed'])
    if key not in datasets:datasets[key]=BenchmarkWindows(cfg['data'],window=cfg['model']['window'],split='train',seed=cfg['seed'])
    def upcoming():
        it=iter(WindowSampler(datasets[key],start=checkpoint['campaign_samples'],seed=cfg['seed'],batch_size=cfg['batch_size']))
        return [next(it) for _ in range(16)]
    indices=upcoming();assert upcoming()==indices
    result=dict(passed=True,strict_native_module_optimizer_scheduler_roundtrip=True,model_tensors=len(module.model.state_dict()),
        restored_Q_optimizer_states=qstates,cursor=checkpoint['campaign_samples'],native_next16_window_indices=indices,
        CPU_python_numpy_torch_rng_roundtrip=True,cuda_rng_records_verified=len(rng['cuda']),cuda_rng_applied=False,
        model_forward_calls=0,optimizer_updates=0,checkpoint_sha256=sha(run/'last.ckpt'))
    del checkpoint,module,configured,optimizer,scheduler;gc.collect()
    return result


def shared_case_summary(rows):
    # Exact existing collect_learning.paired_summary logic: jointly resample the
    # same50 case indices, conditional on these two already trained seeds.
    rows=sorted(rows,key=lambda r:r['seed']);assert [r['seed'] for r in rows]==[1,2]
    matrix=np.asarray([r['per_case_difference'] for r in rows],dtype=float);assert matrix.shape==(2,50)
    draws=matrix.mean(0)[np.random.default_rng(1042).integers(50,size=(10000,50))].mean(1)*100
    return dict(seeds=[1,2],differences_by_seed_pp=[r['difference_pp'] for r in rows],mean_difference_pp=float(matrix.mean()*100),
        shared_case_bootstrap_ci95_pp=np.quantile(draws,[.025,.975]).tolist(),
        scope='Conditional on these two trained seeds; shared-case bootstrap, not seed-population inference or multiplicity correction.')


def planning_signature(p):
    return {k:p[k] for k in ['pairs','manifest_sha256','solver','plan','normalization','split','protocol']}


def branch_stats(data,sl):
    x=data['observed'][...,sl].astype(np.float64);flat=x.reshape(-1,64);mean=flat.mean(0);target=x[:,1:]
    errors={k:float(np.mean((data[k][...,sl].astype(np.float64)-target)**2)) for k in data.files if k not in ['observed','actions']}
    assert np.isfinite(flat).all() and all(np.isfinite(v) for v in errors.values())
    return dict(mean=mean.tolist(),mean_rms=float(np.sqrt(np.mean(mean**2))),centered_rms=float(np.sqrt(np.mean((flat-mean)**2))),
        total_rms=float(np.sqrt(np.mean(flat**2))),mse=errors,clean_over_initial_copy=errors['clean_rollout']/errors['initial_carry'],
        teacher_over_shifted_copy=errors['teacher_forced_prefix']/errors['shifted_observed_carry'],
        wrong_state_over_clean=errors['other_initial']/errors['clean_rollout'],wrong_actions_over_clean=errors['other_episode_actions']/errors['clean_rollout'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(B/'runs/hybrid_carrier600_20261031'))
    p.add_argument('--output',default=str(B/'reports/hybrid-carrier600-20261031'));args=p.parse_args()
    root=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    plan=read(root/'plan.json');closed=read(root/'closed.json');assert (root/'HOLD').exists() and (root/'HOLD_DISPATCH').exists()
    exits=[read(root/'exits'/(j['id']+'.json')) for j in plan['jobs']]
    assert closed['completed']==len(exits)==48 and closed['failed']==0 and all(e['returncode']==0 for e in exits)
    sources={k:read(root/(k+'-source-record.json')) for k in ['model','helper','controller']}
    for source in sources.values():
        for key in ['manifest','runtime_manifest']:
            m=read(source[key])
            for name,digest in m['files'].items():assert sha(Path(m['source'])/name)==digest,name
    assert sources['model']['source_commit'].startswith('f2ae93d') and sources['helper']['source_commit'].startswith('109e69a')
    assert sources['controller']['source_commit'].startswith('f4fd69a')
    halves=read(out/'core-half-stats.json');half_rows={r['id']:r for r in halves['rows']}
    rows=[];signatures={};observed_F={};configs={}
    for job in plan['jobs']:
        if job['phase']!='train600':continue
        run=Path(job['output']);name=run.name;cfg=read(run/'config.json');configs[name]=cfg
        assert sha(run/'config.json')==job['config_sha256'] and cfg['batch_size']==128 and cfg['max_steps']==124910
        status=read(run/'status.json');assert status['state']=='finished' and status['step']==600
        fullpath=root/('fullstate-'+name)/'preflight.json';full=read(fullpath)
        qpath=fullpath.with_name('hybrid-query-audit.json');q=read(qpath)
        assert full['passed'] and q['passed'] and full['paired_initial_equal']['compared_tensors']==q['reviewed_initial_tensors_exact']==555
        assert full['frozen_tensors_unchanged']['equal'] and q['native_report_sha256']==sha(fullpath)
        for path,digest in q['binding']['files'].items():assert sha(path)==digest,path
        healthpath=root/('health-'+name)/'health.json';health=read(healthpath)
        hq=health['audit']['hybrid_Q_compatibility'];assert hq['passed'] and hq['proof_sha256']==sha(qpath)
        assert hq['strict_Q9']==q['query_stop_gradient'] and hq['binding']==q['binding']
        statepath=root/('transition-'+name)/'state-audit.json';state=read(statepath)
        assert state['step']==600 and state['cases']==64 and state['state_dim']==128
        statejob=next(j for j in plan['jobs'] if j['id']=='transition-'+name)
        assert state['manifest_sha256']==statejob['manifest_sha256']
        assert state['config_sha256']==sha(run/'config.json') and state['weights_sha256']==sha(run/'model-step0000600/weights.pt')
        half=half_rows[name];assert half['cache_sha256']==sha(half['cache'])
        with np.load(half['cache']) as a:
            assert half['branches']=={k:branch_stats(a,sl) for k,sl in [('F',slice(0,64)),('Q',slice(64,128))]}
            observed_F[(job['task'],job['training_seed'],job['arm'])]=a['observed'][...,:64].copy()
        evaluations={}
        for controller,phase in [('original','planning'),('correlated','correlated-planning')]:
            path=root/(phase+'-'+name)/'metrics.json';value=read(path)
            assert len(value['pairs'])==len(value['metrics']['episode_successes'])==50 and value['checkpoint_audit']['strict_equal']
            assert value['checkpoint']==str(run/'model-step0000600') and value['normalization']=='train episodes'
            assert sha(value['manifest'])==value['manifest_sha256'] and read(value['manifest'])['pairs']==value['pairs']
            sig=planning_signature(value)
            if job['task'] not in signatures:signatures[job['task']]=sig
            assert sig==signatures[job['task']]
            if controller=='correlated':
                d=value['controller_diagnostic'];assert d['mode']=='correlated' and d['checkpoint_files_unchanged'] and d['matched_cases_checkpoint_solver_plan_normalization']
                assert d['reference_sha256']==sha(root/('planning-'+name)/'metrics.json')
                spec=read(root/(phase+'-'+name)/'executed-spec.json')
                assert spec['source']==sources['model'] and d['spec_sha256']==sha(root/(phase+'-'+name)/'executed-spec.json')
                for ip,digest in spec['immutable_inputs'].items():assert sha(ip)==digest
            evaluations[controller]=dict(success_rate=value['metrics']['success_rate'],successes=value['metrics']['episode_successes'],seconds=value['seconds'],
                path=str(path),sha256=sha(path),manifest_sha256=value['manifest_sha256'],checkpoint_audit=value['checkpoint_audit'])
        trace=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()];assert all(finite(x) for x in trace)
        bystep={int(x['step']):x for x in trace};assert {100,200,300,400,500,600}<=set(bystep)
        gradients=[dict(step=s,**{k:v for k,v in row.items() if k.startswith('grad_norm_')}) for s,row in sorted(bystep.items()) if 'grad_norm_pool' in row]
        qs=max(gradients,key=lambda r:r['grad_norm_pool'])
        hs=health['summary'];renderer={}
        for t in [.2,.5,.8,.95]:
            base=hs[f'fm_t{t}_forecast'];renderer[str(t)]={c:hs[f'fm_t{t}_{c}']/base for c in ['wrong_state','wrong_actions','carry','zero']}
            renderer[str(t)]['shuffled_posterior_over_posterior']=hs[f'fm_t{t}_shuffled_posterior']/hs[f'fm_t{t}_posterior']
        rows.append(dict(id=name,task=job['task'],seed=job['training_seed'],arm=job['arm'],host=job['host'],gpu=job['gpu'],
            status=status,planning=evaluations,branches=half['branches'],native_fullstate=full,strict_Q=q,
            health_summary=hs,health_audit=health['audit'],renderer_relative_errors=renderer,
            transition={k:v for k,v in state.items() if k!='conditions'},
            transition_conditions={k:{n:v for n,v in value.items() if not n.startswith('per_')} for k,value in state['conditions'].items()},
            native_trace=list(bystep.values()),native_gradient_trace=gradients,largest_logged_Q_gradient=qs,
            size=read(run/'model_size.json'),effective_Q_parameters=152464 if job['arm']=='adaptive' else 0,
            artifacts={str(path):sha(path) for path in [run/'config.json',fullpath,qpath,healthpath,statepath,Path(half['cache']),run/'metrics.jsonl']}))
    assert len(rows)==8
    protected=[]
    for task in ['tworoom','cube']:
        for seed in [1,2]:
            a=observed_F[(task,seed,'fixed')];b=observed_F[(task,seed,'adaptive')]
            protected.append(dict(task=task,seed=seed,observed_F_exact=np.array_equal(a,b),max_abs_difference=float(np.max(np.abs(a.astype(float)-b)))))
    paired=[];aggregate=[]
    for task in ['tworoom','cube']:
        for controller in ['original','correlated']:
            group=[]
            for seed in [1,2]:
                fixed=next(r for r in rows if (r['task'],r['seed'],r['arm'])==(task,seed,'fixed'))
                adaptive=next(r for r in rows if (r['task'],r['seed'],r['arm'])==(task,seed,'adaptive'))
                result=dict(task=task,controller=controller,seed=seed,**paired_outcomes(adaptive['planning'][controller]['successes'],fixed['planning'][controller]['successes']))
                paired.append(result);group.append(result)
            aggregate.append(dict(task=task,controller=controller,**shared_case_summary(group)))
    controller_pairs=[dict(task=r['task'],seed=r['seed'],arm=r['arm'],**paired_outcomes(r['planning']['correlated']['successes'],r['planning']['original']['successes'])) for r in rows]
    contextpath=B/'code/experiments/local_carrier_20261031/results/gate600/summary.json'
    context=dict(F64_identity600=read(contextpath)['context'],source_path=str(contextpath),source_sha256=sha(contextpath),
        historical_TF600=dict(tworoom=[6,8],cube=[14,10],scope='Previously recorded scalar-only primary context; original case-level artifacts not located, no pairedCI.'),
        limits='128 vs64, native downstream size and effective learning capacity differ. F64 historical diagonal normalization differs from current globalRMS; both protected carriers use frozen randomViT, TF encoder learns. No claim of a capacity/metric-matched historical contrast.')
    core=dict(runs=rows,paired=paired,aggregate=aggregate,controller_pairs=controller_pairs,protected_F=protected,context=context)
    write(out/'core.json',core)
    print(json.dumps(dict(core_ready=True,aggregate=aggregate,protected_F=protected)),flush=True)
    torch.set_num_threads(2);initialize_ops(OpsConfig(backend='sdpa_math',allow_fallback=False),device=torch.device('cpu'),parallel_state=None)
    datasets={}
    for row in rows:
        row['native_CPU_restore']=native_restore(root/row['id'],configs[row['id']],datasets)
        print(json.dumps(dict(restored=row['id'],passed=True)),flush=True)
    result=dict(passed=True,root=str(root),all48_exit0=True,closed=closed,sources=sources,exits=exits,**core,
        collector_sha256=sha(__file__),half_stats_sha256=sha(out/'core-half-stats.json'),CPU_only=True,model_forward_calls=0,
        new_cohorts=0,new_fits=0,extra_training_updates=0,
        scope='Completed600 short gate inside1249 warmup. Functional performance, clean prediction and renderer use reported separately; no automatic continuation, novelty or cross-task-win claim.')
    write(out/'summary.json',result)
    print(json.dumps(dict(passed=True,runs=8,endpoints=48,CPU_native_restores=8,output=str(out))),flush=True)


if __name__=='__main__':main()
