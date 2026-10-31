"""Prepare eight reviewed fresh600 comparisons,48 endpoints, without launch."""
import argparse
import copy
import json
from pathlib import Path

from experiments.hybrid_carrier_20261031 import prepare_preflight as base
from experiments.hybrid_carrier_20261031.gate_audit import read,sha

B,PY,PREFIX=base.B,base.PY,base.PREFIX
PRE=B/'runs/hybrid_carrier_preflight_strict_20261031'


def verify_source(source):
    for key in ['manifest','runtime_manifest']:
        manifest=read(source[key])
        for name,digest in manifest['files'].items():assert sha(Path(manifest['source'])/name)==digest,name


def config(alloc,root):
    c=copy.deepcopy(read(PRE/alloc['id']/'config.json'))
    assert c['model_class']=='hybrid_carrier' and c['batch_size']==128 and c['max_steps']==124910
    assert c['model']['hybrid_query']==alloc['arm'] and c['seed']==alloc['training_seed']
    assert c['model']['beta']==.25 and c['model']['rho']==.8 and c['model']['forecast_fraction']==.5
    assert c['model'].get('sigma0',.1)==.1 and c['model']['state_slots']==8 and c['model']['state_width']==16
    assert sha(c['hybrid_calibration']['path'])==c['hybrid_calibration']['sha256']
    c.update(output=str(root/alloc['id']),checkpoint_every=200,retain_steps=[200,400,600])
    return c


def jobs_for(root,model,helper,controller,configs):
    jobs=[];templates={};directory=Path(helper['source'])/PREFIX
    for a in base.allocations():
        name=a['id'];run=root/name;cfg=configs[name]
        common={k:v for k,v in a.items() if k!='id'};common['source']=model
        previous=[]
        def add(phase,out,command,**extra):
            nonlocal previous
            ident=phase+'-'+name
            jobs.append(dict(**common,id=ident,phase=phase,output=str(out),depends_on=previous,command=command,**extra));previous=[ident]
        add('train600',run,[PY,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(run/'config.json'),'--stop-step','600'],
            config=str(run/'config.json'),config_sha256=sha(run/'config.json'))
        peer=f'{a["task"]}-hybrid-{"adaptive" if a["arm"]=="fixed" else "fixed"}-seed{a["training_seed"]}'
        proof=root/('fullstate-'+name);bindings=['--model-source-record',str(root/'model-source-record.json'),
            '--helper-source-record',str(root/'helper-source-record.json')]
        cmd=[PY,'-B','-u',str(directory/'gate_audit.py'),'fullstate',*bindings,'--reviewed-run',str(PRE/name),
            '--run',str(run),'--output',str(proof),'--expected-step','600','--gradient-cases','2',
            '--reference-initial',str(root/peer/'initial-model/weights.pt')]
        for prefix in ['encoder','pool','predictor','action_encoder','renderer','sigreg',*base.STAT_KEYS]:
            cmd+=['--reference-prefix',prefix]
        add('fullstate',proof,cmd)
        jobs[-1]['depends_on'].append('train600-'+peer)
        out=root/('health-'+name)
        add('health',out,[PY,'-B','-u',str(directory/'gate_audit.py'),'health',*bindings,
            '--query-proof',str(proof/'hybrid-query-audit.json'),'--run',str(run),'--output',str(out),'--cases','16'])
        out=root/('transition-'+name)
        audit_manifest=Path(model['source'])/f'experiments/ideas_validate_20261031/audit-windows-{a["task"]}.json'
        add('transition',out,[PY,'-B','-u',str(Path(model['source'])/'experiments/ideas_validate_20261031/audit.py'),
            '--run',str(run),'--output',str(out),'--cases','64','--manifest',str(audit_manifest)],
            manifest_sha256=sha(audit_manifest))
        manifest=Path(cfg['data'])/'manifests/validation-strict.json';out=root/('planning-'+name)
        assert len(read(manifest)['pairs'])==50
        add('planning',out,[PY,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',str(manifest),
            '--checkpoint',str(run/'model-step0000600'),'--data',cfg['data'],'--output',str(out)],
            manifest_sha256=sha(manifest),environment={'MUJOCO_EGL_DEVICE_ID':str(a['gpu'])})
        cal=B/'reports/action-covariance-20261031/calibration'/a['task']/'calibration.json'
        matrix=cal.with_name('matrix.npz');card=Path(cfg['data'])/'dataset_card.json'
        caldata=read(cal);assert caldata['matrix_sha256']==sha(matrix) and caldata['dataset_card_sha256']==sha(card)
        assert caldata['zero_variance_coordinates']==0 and caldata['task']==a['task']
        corr=root/('correlated-planning-'+name);spec=root/'specs'/f'{name}.json'
        templates[name]=dict(id='correlated-planning-'+name,task=a['task'],family='hybrid-'+a['arm'],step=600,
            training_seed=a['training_seed'],case_set='primary',mode='correlated',source=model,
            checkpoint=str(run/'model-step0000600'),manifest=str(manifest),data=cfg['data'],output=str(corr),
            reference_metrics=str(out/'metrics.json'),native_evaluator=str(Path(model['source'])/'as_jepa_tools/benchmark_evaluate.py'),
            calibration=str(cal),controller_source=controller,
            prepared_inputs={str(p):sha(p) for p in [cal,matrix,card,manifest,run/'config.json',root/'controller-source-record.json']})
        add('correlated-planning',corr,[PY,'-B','-u',str(directory/'correlated_endpoint.py'),'--template',str(spec)],
            environment={'MUJOCO_EGL_DEVICE_ID':str(a['gpu'])})
    assert len(jobs)==48 and len({j['id'] for j in jobs})==48
    ids={j['id'] for j in jobs};assert all(set(j['depends_on'])<=ids for j in jobs)
    return jobs,templates


def main():
    p=argparse.ArgumentParser();p.add_argument('--helper-source-record',required=True)
    p.add_argument('--controller-source-record',default=str(B/'runs/action_covariance_source_20261031/source-record.json'))
    p.add_argument('--root',default=str(B/'runs/hybrid_carrier600_20261031'));a=p.parse_args()
    root=Path(a.root);assert not root.exists()
    assert (PRE/'HOLD').exists() and (PRE/'HOLD_DISPATCH').exists()
    review=read(B/'reports/hybrid-carrier-preflight-20261031/summary.json')
    assert review['passed']
    oldplan=read(PRE/'plan.json');assert len(oldplan['jobs'])==16
    assert all(read(PRE/'exits'/(j['id']+'.json'))['returncode']==0 for j in oldplan['jobs'])
    model=read(PRE/'model-source-record.json');assert model['source_commit']==base.MODEL_COMMIT
    helper=read(a.helper_source_record);controller=read(a.controller_source_record)
    assert controller['source_commit']=='f4fd69a5e80abc68d9ecf156175ea1fd54f1af37'
    for source in [model,helper,controller]:verify_source(source)
    hm=read(helper['manifest']);mm=read(model['manifest'])
    assert sha(__file__)==hm['files'][PREFIX+'prepare600.py']
    proof=read(Path(helper['source'])/PREFIX/'cpu/gate-contracts.json');assert proof['passed'] and proof['tests']==2
    for name,digest in proof['source_sha256'].items():assert hm['files'][PREFIX+name]==digest
    for name in ['as_jepa_lewm/hybrid_carrier.py','as_jepa_lewm/factory.py','as_jepa_lewm/train.py',
        'as_jepa_lewm/robust.py','as_jepa_tools/train_stage.py','as_jepa_tools/benchmark_health.py']:
        assert hm['files'][name]==mm['files'][name],name
    root.mkdir();configs={a['id']:config(a,root) for a in base.allocations()}
    for alloc in base.allocations():
        own=copy.deepcopy(configs[alloc['id']]);own.pop('output');own['model'].pop('hybrid_query')
        peer=f'{alloc["task"]}-hybrid-{"adaptive" if alloc["arm"]=="fixed" else "fixed"}-seed{alloc["training_seed"]}'
        other=copy.deepcopy(configs[peer]);other.pop('output');other['model'].pop('hybrid_query')
        assert own==other
    for name,cfg in configs.items():base.write(root/name/'config.json',cfg)
    for name,value in [('model',model),('helper',helper),('controller',controller)]:base.write(root/(name+'-source-record.json'),value)
    jobs,templates=jobs_for(root,model,helper,controller,configs)
    for name,spec in templates.items():base.write(root/'specs'/f'{name}.json',spec)
    for j in jobs:
        if j['phase']=='correlated-planning':j['template_sha256']=sha(j['command'][-1])
    base.write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment=oldplan['environment'],jobs=jobs))
    base.write(root/'protocol.json',dict(runs=8,endpoints=48,stop_step=600,allocations=base.allocations(),
        model_source=model,helper_source=helper,controller_source=controller,review_sha256=sha(B/'reports/hybrid-carrier-preflight-20261031/summary.json'),
        unchanged='Native train_stage, batch128, max_steps124910 scheduler, original TF beta/sigma/rho/forecast, exact initial calibration and all555 initial tensors. No two-step observer wrapper.',
        checkpoints='Portable200/400/600; final native fullstate600. Do not call intermediate portable exports resumable fullstates.',
        audits='Original fullstate/objective backward +3986 Q9 semantics, all555 paired/reviewed initial equality, original health16 with hash-bound Q compatibility; original transition64.',
        controllers='Original CEM primary50 followed by existing frozen f4 correlated CEM on exactly same50 and checkpoint. Report both, allseeds; no controller selection.',
        CPU_collection='F/Q branch mean, centered and totalRMS plus clean/copy/zero errors only from existing transition cache. No extra representation calls.',
        stopping='Root authorized this one600 short gate after completed16 preflights. No freshcases or automatic continuation above600. All oldHOLDs unchanged.',
        interpretation='Fixed64+fixedQ64 versus fixed64+adaptiveQ64; different effective trainable capacity and joint clipping. Neither originality nor mature-baseline superiority follows from short results.'))
    (root/'HOLD').write_text('Prepared only; activation requires actual target-slot availability and immutable-source review.\n')
    (root/'HOLD_DISPATCH').write_text('Prepared48 endpoints; cap600, no automatic continuation.\n')
    print(json.dumps(dict(root=str(root),runs=8,endpoints=48,active=False)))


if __name__=='__main__':main()
