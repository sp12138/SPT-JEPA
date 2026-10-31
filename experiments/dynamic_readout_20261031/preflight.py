"""Delegate the native audit; add readout-specific checks, never an update."""
import argparse,json,sys
from pathlib import Path
import torch
from experiments.ideas_validate_20261031 import preflight as native

PREFIX='dynamic_readout_module.'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--expected-step',type=int,required=True);parser.add_argument('--reference-initial',required=True)
    args=parser.parse_args();run,out=Path(args.run),Path(args.output);cfg=json.loads((run/'config.json').read_text())
    mode=cfg['model']['spatial_readout_mode'];assert mode in {'static','state'}
    original=native.gradient_report;details={}
    def audit_gradients(model,expectations):
        result=original(model,expectations);head=model.dynamic_readout_module
        values={name:dict(max_abs=float(p.grad.detach().abs().max()),finite=bool(torch.isfinite(p.grad).all()))
            for name,p in head.named_parameters()if p.grad is not None}
        state_columns=float(head.input.weight.grad[:,:16].detach().abs().max())
        position_columns=float(head.input.weight.grad[:,16:].detach().abs().max())
        passed=len(values)==4 and all(x['finite']and x['max_abs']>0 for x in values.values())
        passed=passed and position_columns>0 and (state_columns==0 if mode=='static'else state_columns>0)
        details.update(mode=mode,per_tensor=values,state_input_gradient_max=state_columns,position_input_gradient_max=position_columns,
            static_inactive_parameter_elements=1024 if mode=='static'else 0,passed=passed,
            scope='Actual native checkpoint eval FP32 backward, no optimizer step. Zero STATIC state-input columns are intentional. Positive current gradient does not require first-layer weights already changed under tiny warmup LR.')
        result['dynamic_readout_module']['readout_detail']=details.copy()
        result['dynamic_readout_module']['passed'] &= passed
        return result
    native.gradient_report=audit_gradients
    try:native.main()
    finally:native.gradient_report=original
    report=json.loads((out/'preflight.json').read_text());assert report['passed']
    initial=torch.load(run/'initial-model/weights.pt',map_location='cpu',weights_only=True,mmap=True)
    reference=torch.load(args.reference_initial,map_location='cpu',weights_only=True,mmap=True)
    final=torch.load(run/f'model-step{args.expected_step:07d}/weights.pt',map_location='cpu',weights_only=True,mmap=True)
    extra={PREFIX+k for k in ['input.weight','input.bias','output.weight','output.bias']}
    initial_exact=len(reference)==553 and set(initial)==set(reference)|extra and set(final)==set(initial)
    initial_exact=initial_exact and all(torch.equal(v,initial[k])for k,v in reference.items())
    counts={k:int(initial[k].numel())for k in extra}
    output_zero=all(torch.count_nonzero(initial[PREFIX+k])==0 for k in ['output.weight','output.bias'])
    assert initial_exact and sum(counts.values())==5376 and output_zero
    status=json.loads((run/'status.json').read_text());assert status['state']=='finished'and status['step']==args.expected_step
    assert 0<status['peak_gib']<140 and status['seconds_per_step']>0
    assert report['frozen_tensors_unchanged']['equal']and report['frozen_tensors_unchanged']['compared_tensors']==210
    row=dict(passed=True,step=args.expected_step,mode=mode,original_initial_tensors_exact=553,new_tensors=4,
        new_parameters=5376,new_parameter_shapes={k:list(initial[k].shape)for k in sorted(extra)},
        initial_output_zero=True,frozen_tensors_unchanged=210,optimizer=report['optimizer_parameter_count'],head_gradient=details,
        head_tensor_changed={k:not torch.equal(initial[k],final[k])for k in sorted(extra)},
        change_requirement='Informational only: no requirement for input weights to change after3 tiny-LR warmup updates.',
        peak_gib=status['peak_gib'],seconds_per_step=status['seconds_per_step'],learning_rates=status['learning_rates'],
        checkpoint_sha256=report['checkpoint_sha256'],weights_sha256=report['weights_sha256'],
        initial_sha256=report['initial_weights']['sha256'],reference_initial_sha256=native.file_hash(args.reference_initial))
    with(out/'readout-audit.json').open('x')as f:json.dump(row,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(row),flush=True)
if __name__=='__main__':main()
