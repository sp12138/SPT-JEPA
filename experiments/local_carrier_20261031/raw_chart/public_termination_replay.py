"""One full captured-shape GPU verification; never instantiate/train a model."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch
from torch.nn import functional as F
import as_jepa_lewm.carrier as carrier


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    source = json.loads(Path(args.source_record).read_text()); manifest = json.loads(Path(source['manifest']).read_text())
    assert Path(carrier.__file__).resolve() == (Path(source['source'])/'as_jepa_lewm/carrier.py').resolve()
    assert sha(carrier.__file__) == manifest['files']['as_jepa_lewm/carrier.py']
    assert sha(__file__) == manifest['files']['experiments/local_carrier_20261031/raw_chart/public_termination_replay.py']
    root = Path(args.capture_root); path = root/'capture/capture.pt'
    metadata = json.loads((root/'capture/capture.json').read_text()); assert sha(path)==metadata['capture_sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False); data = saved['tensors']
    tensors = {}
    for key in ['target','closure_first','closure_second','closure_derivative']:
        record = data[key]; value = record['value'].to('cuda')
        assert list(value.shape)==record['shape'] and list(value.stride())==record['stride']
        tensors[key]=value
    target, first, second, derivative = (tensors[key] for key in ['target','closure_first','closure_second','closure_derivative'])
    assert target.shape==(128,4,64,16) and target.dtype==torch.float32
    bound, cap, rtol = metadata['bound'], metadata['iterations'], metadata['rtol']
    assert bound==.5 and cap==64 and rtol==1e-6 and metadata['atol']==0
    torch.set_float32_matmul_precision(metadata['float32_matmul_precision']); torch.backends.cuda.matmul.allow_tf32=metadata['allow_tf32']
    flags = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32
    torch.cuda.synchronize(); before=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    with patch.object(carrier,'_rounded_solution_certified',wraps=carrier._rounded_solution_certified) as checker:
        public, solved = carrier._precise_adjoint(target,first,second,derivative,bound,cap,rtol)
    torch.cuda.synchronize(); seconds=time.perf_counter()-start; peak=torch.cuda.max_memory_allocated()
    assert checker.call_count==3  # candidate4, candidate8, defensive final check
    assert flags==(torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32)
    target64=target.double(); residual=lambda x:bound*F.linear(F.linear(x,second.T)*derivative,first.T)
    expected=target64.clone()
    for _ in range(8):expected=target64-residual(expected)
    assert torch.equal(solved,expected) and torch.equal(public,expected.float())
    error=(public.double()+residual(public.double())-target64).norm(dim=-1)/(1-bound)
    tolerance=rtol*target64.norm(dim=-1); assert bool((error<=tolerance).all())
    bad=data['public_error_bound']['value']>data['tolerance']['value'];bad=bad.to('cuda')
    g=target64[bad]; d=derivative[bad]
    matrix=torch.eye(16,dtype=torch.float64,device='cuda')[None]+bound*(second[None]*d[:,None,:])@first
    oracle=torch.linalg.solve(matrix.transpose(-1,-2),g.unsqueeze(-1)).squeeze(-1)
    oracle_error=(public[bad].double()-oracle).norm(dim=-1)/g.norm(dim=-1)
    report=dict(passed=True,source=source,helper_sha256=sha(__file__),capture_sha256=sha(path),
        actual_full_shape=list(target.shape),actual_full_stride=list(target.stride()),public_dtype=str(public.dtype),
        original_atol=0.,original_rtol=rtol,original_cap=cap,actual_updates=8,
        internal_and_public_certificates_pass=True,all_outputs_exact_eight_update_reference=True,
        max_public_error_over_tolerance=float((error/tolerance.clamp_min(torch.finfo(torch.float64).tiny)).max()),
        original_violating_slot_solution_error_over_rhs=float(oracle_error.max()),
        global_flags_before=list(flags),global_flags_unchanged=True,wall_seconds_single_call=seconds,
        peak_allocated_bytes=peak,incremental_peak_bytes=peak-before,
        scope='One frozen full-shape production-solver replay only. No model/optimizer/training, tolerance change or general numerical reliability claim. Single-call cost is not a stable model-throughput benchmark.')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:handle.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-record',required=True);p.add_argument('--capture-root',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
