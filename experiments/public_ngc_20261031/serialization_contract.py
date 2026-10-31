"""CPU serialization of the actual report expression for all twelve fixtures."""
import argparse
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from common import *


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--failed-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    helper=Path(__file__).with_name('stable_graph.py');tree=ast.parse(helper.read_text());body=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main').body
    assignment=lambda name:next(n for n in body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets))
    expression=compile(ast.Expression(assignment('report').value),str(helper),'eval')
    bound=compile(ast.Module(body=[assignment('relative_bound')],type_ignores=[]),str(helper),'exec')
    rows=[];root=Path(a.failed_root)
    for j in read(root/'plan.json')['jobs']:
        assert read(root/'exits'/(j['id']+'.json'))['returncode']==1
        s=read(j['spec']);out=Path(j['output']);log=(out/'dispatch.log').read_text()
        assert 'Object of type float32 is not JSON serializable' in log
        origin=read(s['origin_report']);rebuilt=s['origin_kind']=='rebuild-fp64-from-saved-nodes'
        with np.load(out/'radius-audit.npz')as f:radii=f['direct32'].tolist();refs=f['direct64'].tolist();oldradii=f['original_cuda_mm'].tolist()
        with np.load(out/'graph.npz')as f:bridges=f['bridges'].tolist()
        components=len(bridges)+1 if rebuilt else origin['original_components']
        fixture=SimpleNamespace(spec=j['spec'])
        env=dict(globals(),s=s,out=out,a=fixture,origin=origin,components=components,bridges=bridges,
            bridge_lengths=[]if rebuilt else origin['bridge_lengths'],radii=radii,refs=refs,oldradii=oldradii,
            old=True if rebuilt else None,maxabs=0.,maxrel=0.,sumabs=0.,pairs=25000000,mm_false_zero=0,old_fail_chunks=0,
            before=origin['rng_before'],graph_seconds=0.,start=time.perf_counter(),
            torch=SimpleNamespace(__version__=torch.__version__,cuda=SimpleNamespace(max_memory_allocated=lambda:0)))
        exec(bound,env);report=eval(expression,env)
        json.dumps(report,allow_nan=False)
        for value in [float('nan'),float('inf'),-float('inf')]:
            invalid=copy.deepcopy(report);invalid['radius_quantiles'][0]=value
            try:json.dumps(invalid,allow_nan=False)
            except ValueError:pass
            else:raise AssertionError('Nonfinite report accepted')
        invalid=copy.deepcopy(report);invalid['distance_reference']['dimension_relative_bound']=np.float32(env['relative_bound'])
        try:json.dumps(invalid,allow_nan=False)
        except TypeError:pass
        else:raise AssertionError('Original NumPy scalar error not reproduced')
        rows.append(dict(id=j['id'],passed=True,original_failure_reproduced=True,nonfinite_rejected=True,
            dimension_relative_bound_type=type(env['relative_bound']).__name__,fields=len(report)))
    assert len(rows)==12
    write(a.output,dict(passed=True,rows=rows,stable_graph_sha256=sha(helper),contract_sha256=sha(__file__),
        model_calls=0,gpu_calls=0,
        scope='Actual production report AST, all12 real failed specs/origin/radius/bridge fixtures. Unsaved timing/error/allocator counters use their production Python scalar types with neutral values; no diagnostic numbers reconstructed or reported as actual GPU results. All nested fields serialized; NaN/infinities and original NumPy scalar rejection checked.'))
    print(json.dumps(dict(passed=True,reports=12,gpu_calls=0)))
