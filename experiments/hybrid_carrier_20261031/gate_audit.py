"""600-step native audits with the reviewed stop-gradient Q interpretation.

No training, additional representation assay, or replacement health objective.
The native objective backward establishes Q health once; original health16 uses
that exact checkpoint-bound proof instead of equating frozen with requires_grad.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

# Executed by absolute frozen-helper path while PYTHONPATH pins model f2.
# Sibling import keeps the newer audit out of model-source resolution.
from preflight import query_contract
from experiments.ideas_validate_20261031 import preflight as native


def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def binding(run, model_source, helper_source):
    """Bind the strict proof to actual inputs, not a checkpoint name or step."""
    run=Path(run)
    files=[run/'config.json',run/'initial-model/weights.pt',run/'last.ckpt',
           run/'model-step0000600/weights.pt']
    sources={}
    for label,path in [('model',model_source),('helper',helper_source)]:
        s=read(path);m=read(s['manifest'])
        assert m['source']==s['source'] and m['source_commit']==s['source_commit']
        for name,digest in m['files'].items():assert sha(Path(s['source'])/name)==digest,name
        files += [Path(path),Path(s['manifest'])]
        sources[label]=dict(commit=s['source_commit'],source=s['source'])
    return dict(files={str(p):sha(p) for p in files},sources=sources)


def validate_proof(proof, current, arm, trace):
    assert proof['passed'] and proof['binding']==current
    q=proof['query_stop_gradient']
    assert q['arm']==arm and q['tensors']==9 and q['passed'] and len(q['rows'])==9
    assert q['active_group_nonzero']
    for row in q['rows']:
        assert row['passed']
        if arm=='fixed':
            assert row['initial_equal'] and row['gradient_is_none'] and not row['optimizer_state_present']
        else:
            assert row['gradient_finite'] and row['optimizer_state_present']
    assert trace and all(native.finite(row) for row in trace)
    # Keep the native logged-gradient requirement as well as actual-backward Q9.
    logged=max(row.get('grad_norm_pool',0.) for row in trace)
    if arm=='fixed':assert q['gradient_norm']==0. and logged==0.
    else:assert q['gradient_norm']>0. and logged>0.
    return dict(expectation='frozen' if arm=='fixed' else 'active',
        logged_gradient_max=logged,strict_Q9=q,passed=True,
        interpretation='Original Q flag check delegated to checkpoint-bound actual backward, initial tensor and native AdamW-ID proof.')


def fullstate(args, native_args):
    argv=sys.argv;sys.argv=[argv[0],*native_args]
    run=Path(native_args[native_args.index('--run')+1]);out=Path(native_args[native_args.index('--output')+1])
    cfg=read(run/'config.json');assert cfg['model_class']=='hybrid_carrier'
    initial=torch.load(run/'initial-model/weights.pt',map_location='cpu',weights_only=True)
    reference=torch.load(Path(args.reviewed_run)/'initial-model/weights.pt',map_location='cpu',weights_only=True)
    assert len(initial)==len(reference)==555 and initial.keys()==reference.keys()
    assert all(torch.equal(v,reference[n]) for n,v in initial.items())
    del reference
    required={'hybrid_projection','hybrid_fixed_mean','hybrid_fixed_scale','hybrid_query_mean','hybrid_query_scale'}
    assert required<=set(cfg['audit_expectations']['initial_unchanged_prefixes'])
    checkpoint=torch.load(run/'last.ckpt',map_location='cpu',weights_only=False)
    assert checkpoint['global_step']==600
    before=binding(run,args.model_source_record,args.helper_source_record)
    old=native.gradient_report;extra={}
    def gradients(model,expectations):
        assert expectations['pool']==('frozen' if model.hybrid_query=='fixed' else 'active')
        rows=old(model,{k:v for k,v in expectations.items() if k!='pool'})
        extra['query_stop_gradient']=query_contract(model,cfg,checkpoint,initial)
        rows['pool']=dict(expectation=expectations['pool'],**extra['query_stop_gradient'])
        return rows
    try:
        native.gradient_report=gradients;native.main()
        report=read(out/'preflight.json')
        assert report['passed'] and report['paired_initial_equal']['compared_tensors']==555
        assert binding(run,args.model_source_record,args.helper_source_record)==before
        extra.update(passed=True,binding=before,reviewed_initial_tensors_exact=555,
            reviewed_initial_path=str(Path(args.reviewed_run)/'initial-model/weights.pt'),
            reviewed_initial_sha256=sha(Path(args.reviewed_run)/'initial-model/weights.pt'),
            native_report_sha256=sha(out/'preflight.json'),additional_encoder_or_Q_assays=0)
    finally:
        native.gradient_report=old;sys.argv=argv
        extra.setdefault('passed',False);out.mkdir(parents=True,exist_ok=True)
        (out/'hybrid-query-audit.json').write_text(json.dumps(extra,indent=2,allow_nan=False)+'\n')


def health(args,native_args):
    from as_jepa_tools import benchmark_health as original
    run=Path(native_args[native_args.index('--run')+1]);out=Path(native_args[native_args.index('--output')+1])
    cfg=read(run/'config.json');proof=read(args.query_proof)
    assert cfg['model_class']=='hybrid_carrier'
    assert sha(Path(args.query_proof).with_name('preflight.json'))==proof['native_report_sha256']
    bound=binding(run,args.model_source_record,args.helper_source_record)
    trace=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
    delegated=validate_proof(proof,bound,cfg['model']['hybrid_query'],trace)
    old=original.gradient_expectations;argv=sys.argv
    def expectations(model,config):
        values=old(model,config)
        assert values['pool']==delegated['expectation']
        return {k:v for k,v in values.items() if k!='pool'}
    try:
        original.gradient_expectations=expectations;sys.argv=[argv[0],*native_args]
        original.main()
        assert binding(run,args.model_source_record,args.helper_source_record)==bound
        report=read(out/'health.json')
        report['audit']['gradient_expectations']['pool']=delegated['expectation']
        report['audit']['gradient_max']['pool']=delegated['logged_gradient_max']
        report['audit']['hybrid_Q_compatibility']=dict(**delegated,
            proof_path=str(args.query_proof),proof_sha256=sha(args.query_proof),binding=bound)
        (out/'health.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    finally:original.gradient_expectations=old;sys.argv=argv


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['fullstate','health'])
    p.add_argument('--model-source-record',required=True);p.add_argument('--helper-source-record',required=True)
    p.add_argument('--reviewed-run');p.add_argument('--query-proof')
    args,rest=p.parse_known_args()
    if args.phase=='fullstate':assert args.reviewed_run;fullstate(args,rest)
    else:assert args.query_proof;health(args,rest)
