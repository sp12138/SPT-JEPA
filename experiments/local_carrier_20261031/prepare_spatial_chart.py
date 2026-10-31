"""Candidate-only shared local chart gate; no calibration, trainer, or launch."""
import argparse, copy, itertools, json
from pathlib import Path
from experiments.local_carrier_20261031 import prepare_basis as base

B,PY,OLD=base.B,base.PY,base.OLD
read,sha,write=base.read,base.sha,base.write
PRE=B/'runs/local_spatial_chart_preflight_20261031'
ROOT=B/'runs/local_spatial_chart600_20261031'
CONTRACT=B/'reports/local-spatial-chart-contracts-20261031/contracts.json'
SELF='experiments/local_carrier_20261031/prepare_spatial_chart.py'
MODEL='as_jepa_lewm/spatial_carrier.py'
TEST='experiments/local_carrier_20261031/test_spatial_chart.py'
PREFIX='carrier_chart_module.'


def source(path,contract_path):
    s=read(path);m=read(s['manifest']);assert s['source_commit']==m['source_commit']
    for name,h in m['files'].items():assert sha(Path(s['source'])/name)==h,name
    assert sha(__file__)==m['files'][SELF]
    proof=read(contract_path)
    assert proof['passed'] and proof['tests']==4 and proof['device']=='CPU'
    assert proof['model_source_sha256']==m['files'][MODEL] and proof['test_source_sha256']==m['files'][TEST]
    assert proof['geometry']['parameters']==2128 and proof['geometry']['shared_slots']==64
    return s


def allocations():
    for gpu,(task,seed) in enumerate(itertools.product(('tworoom','cube'),(1,2)),2):
        yield dict(id=f'{task}-spatial-chart-bounded-seed{seed}',task=task,training_seed=seed,
            host='dev13',gpu=gpu,family='spatial_carrier',arm='local-shared-bounded')


def config(a,out,step):
    old=OLD/f'{a["task"]}-spatial-local-seed{a["training_seed"]}'
    c=copy.deepcopy(read(old/'config.json'));cal=c['spatial_calibration']
    assert sha(cal['path'])==cal['sha256'] and read(cal['path'])['normalization']=='global_rms'
    assert c['model']['spatial_mode']=='local' and c['model'].get('spatial_basis','identity')=='identity'
    assert 'spatial_chart' not in c['model'] and len(set(c['model']['spatial_scale']))==1
    assert c['batch_size']==128 and c['max_steps']==124910 and c['record_initial_weights']
    assert not c.get('initial_weights') and not c.get('initial_encoder_checkpoint')
    c.update(output=str(out),checkpoint_every=2 if step==2 else 200,retain_steps=[2] if step==2 else [200,400,600])
    c['model']['spatial_chart']='bounded'
    return c,old


def prepare(a):
    s=source(a.source_record,a.cpu_contract);stage=a.phase
    root=PRE if stage=='preflight' else ROOT;step=2 if stage=='preflight' else 600
    assert not root.exists()
    if stage=='training':
        review=read(PRE/'review.json');assert review['passed'] and read(PRE/'source-record.json')==s
        assert review['cpu_contract_sha256']==sha(a.cpu_contract)
    oldplan=read(OLD/'plan.json');root.mkdir();write(root/'source-record.json',s);jobs=[];records=[]
    for alloc in allocations():
        name=alloc['id'];out=root/name;c,old=config(alloc,out,step);write(out/'config.json',c)
        common={k:v for k,v in alloc.items() if k!='id'};common['source']=s;previous=[]
        def add(phase,dest,command,**extra):
            nonlocal previous
            ident=phase+'-'+name
            jobs.append(dict(**common,id=ident,phase=phase,output=str(dest),depends_on=previous,command=command,**extra));previous=[ident]
        add('train'+str(step),out,[PY,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(out/'config.json'),'--stop-step',str(step)],
            config=str(out/'config.json'),config_sha256=sha(out/'config.json'))
        dest=root/('fullstate-'+name)
        add('fullstate',dest,[PY,'-B','-u',str(Path(s['source'])/'experiments/ideas_validate_20261031/preflight.py'),
            '--run',str(out),'--output',str(dest),'--expected-step',str(step),'--reference-initial',str(old/'initial-model/weights.pt')])
        if stage=='training':
            dest=root/('health-'+name);add('health',dest,[PY,'-B','-u','-m','as_jepa_tools.benchmark_health','--run',str(out),'--output',str(dest)])
            dest=root/('transition-'+name);add('transition',dest,[PY,'-B','-u',str(Path(s['source'])/'experiments/ideas_validate_20261031/audit.py'),
                '--run',str(out),'--output',str(dest),'--cases','64','--manifest',
                str(Path(s['source'])/f'experiments/ideas_validate_20261031/audit-windows-{alloc["task"]}.json')])
            manifest=Path(c['data'])/'manifests/validation-strict.json';dest=root/('planning-'+name)
            add('planning',dest,[PY,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',str(manifest),
                '--checkpoint',str(out/'model-step0000600'),'--data',c['data'],'--output',str(dest)],
                manifest_sha256=sha(manifest),environment=dict(MUJOCO_EGL_DEVICE_ID=str(alloc['gpu'])))
        records.append(dict(**alloc,reference=str(old),calibration=c['spatial_calibration']))
    assert len(jobs)==(8 if stage=='preflight' else 20)
    write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment=oldplan['environment'],jobs=jobs))
    write(root/'protocol.json',dict(source=s,step=step,runs=records,candidate_only=True,
        cpu_contract=dict(path=str(a.cpu_contract),sha256=sha(a.cpu_contract)),
        intervention='One shared16→64→16 ResidualCarrierChart, bound.5, applied to64 original local slots after fixed isotropic calibration.2128 additional trainable parameters; no spatial mixing.',
        comparator='Completed local600 uses no chart. Frozen identity chart was proved function/loss/all-original-gradient/RNG exact to that default on full native CPU configs. No new identity/free grid.',
        parameter_departure=dict(extra_total=2128,extra_trainable=2128,extra_parameter_tensors=4),
        unchanged='Existing sample-level TF loss, fixed measurement/calibration, observation/action data, original native Trainer/optimizer/scheduler/checkpoints and CEM/strict50 cases.',
        boundary='Known invertible residual chart composed with a local carrier; conditional research hypothesis, not an academic novelty or performance claim. Bound is exact-real; no quantized-arithmetic guarantee.',
        policy='Root alone activates. All8 real two-step/preflight endpoints and all553 original initial-tensor equality checks required before4 fresh600 gates. Stop600; no automatic continuation.'))
    assert not (root/'plan.json').exists()
    print(json.dumps(dict(root=str(root),jobs=len(jobs),launched=False)))


def review(a):
    import torch
    from as_jepa_lewm.carrier import ResidualCarrierChart
    torch.set_num_threads(2);s=source(a.source_record,a.cpu_contract)
    assert read(PRE/'source-record.json')==s
    p=read(PRE/'candidate-plan.json');assert len(p['jobs'])==8
    for j in p['jobs']:assert read(PRE/'exits'/(j['id']+'.json'))['returncode']==0
    rows=[];common_chart=None
    for alloc in allocations():
        name=alloc['id'];run=PRE/name;c,old=config(alloc,run,2);assert c==read(run/'config.json')
        fp=PRE/('fullstate-'+name)/'preflight.json';r=read(fp)
        assert r['passed'] and r['frozen_tensors_unchanged']['equal'] and r['frozen_tensors_unchanged']['compared_tensors']==210
        assert r['gradients']['carrier_chart_module']['expectation']=='active' and r['gradients']['carrier_chart_module']['passed']
        original=torch.load(old/'initial-model/weights.pt',map_location='cpu',weights_only=True)
        initial=torch.load(run/'initial-model/weights.pt',map_location='cpu',weights_only=True)
        final=torch.load(run/'model-step0000002/weights.pt',map_location='cpu',weights_only=True)
        extra={PREFIX+k for k in ['input.weight','input.bias','output.weight','output.bias']}
        assert len(original)==553 and set(initial)==set(original)|extra
        assert all(torch.equal(v,initial[k]) for k,v in original.items())
        initial_chart={k.removeprefix(PREFIX):initial[k] for k in extra}
        assert torch.count_nonzero(initial_chart['output.weight'])==torch.count_nonzero(initial_chart['output.bias'])==0
        if common_chart is None:common_chart=initial_chart
        else:assert all(torch.equal(v,common_chart[k]) for k,v in initial_chart.items())
        assert any(not torch.equal(initial[k],final[k]) for k in extra)
        chart=ResidualCarrierChart(16,64,'bounded',.5)
        chart.load_state_dict({k.removeprefix(PREFIX):final[k] for k in extra},strict=True)
        certificate=chart.norm_certificate();assert certificate['residual_lipschitz_upper']<=.500001
        size=read(run/'model_size.json');oldsize=read(old/'model_size.json')
        assert size['total']-oldsize['total']==size['trainable']-oldsize['trainable']==2128
        rows.append(dict(id=name,original_initial_tensors_exact=553,added_parameters=2128,chart_initial_shared_exact=True,
            initial_output_identity=True,chart_updated_after_two_native_steps=True,certificate=certificate,
            fullstate_report_sha256=sha(fp),initial_sha256=sha(run/'initial-model/weights.pt'),
            reference_initial_sha256=sha(old/'initial-model/weights.pt'),passed=True))
        del original,initial,final,chart
    write(PRE/'review.json',dict(passed=True,source=s,cpu_contract_sha256=sha(a.cpu_contract),rows=rows,
        scope='All real two-step/native fullstate and active-chart gradient checks. This does not establish learning or planning quality at600.'))
    print(json.dumps(dict(passed=True,runs=4)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','review','training'])
    p.add_argument('--source-record',required=True);p.add_argument('--cpu-contract',default=str(CONTRACT));a=p.parse_args()
    review(a) if a.phase=='review' else prepare(a)
