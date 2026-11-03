"""Compare every native loader output before approving an IO-only continuation."""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def main(old,new,data,output):
    old,new=Path(old),Path(new);torch.set_num_threads(2)
    left=load(old/'as_jepa_tools/benchmark_data.py','old_benchmark').BenchmarkWindows(data,window=8)
    right=load(new/'as_jepa_tools/benchmark_data.py','cached_benchmark').BenchmarkWindows(data,window=8)
    assert left.natural_length==right.natural_length
    nn={str(p.relative_to(old)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (old/'as_jepa_lewm').glob('*.py')}
    assert all((new/name).exists() and hashlib.sha256((new/name).read_bytes()).hexdigest()==sha for name,sha in nn.items())
    unchanged_tools=['train_stage.py','data.py']
    assert all((old/'as_jepa_tools'/p).read_bytes()==(new/'as_jepa_tools'/p).read_bytes() for p in unchanged_tools)
    indices=np.random.default_rng(20261103).choice(left.natural_length,32,replace=False).tolist()
    rows=[];timings=[];handle=None
    for repeat in range(2):
        started=time.monotonic()
        for index in indices:
            a,b=left[index],right[index]
            assert a.keys()==b.keys()
            checks={key:bool(torch.equal(a[key],b[key])) if torch.is_tensor(a[key]) else a[key]==b[key] for key in a}
            assert all(checks.values()),(index,checks)
            if handle is None:handle=id(right.pixels)
            assert id(right.pixels)==handle
            if repeat==0:rows.append(dict(index=index,episode=a['episode'],start=a['start'],rows=a['rows'].tolist(),checks=checks,
                                          pixels_sha256=hashlib.sha256(a['pixels'].numpy().tobytes()).hexdigest()))
        timings.append(time.monotonic()-started)
    result=dict(passed=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(new),original_source=str(old),
                dataset=str(data),nn_source_sha256=nn,windows=rows,complete_comparison_seconds=timings,
                old_data_sha256=hashlib.sha256((old/'as_jepa_tools/benchmark_data.py').read_bytes()).hexdigest(),
                new_data_sha256=hashlib.sha256((new/'as_jepa_tools/benchmark_data.py').read_bytes()).hexdigest(),
                scope='64 paired real train-window reads;all pixels,render,anchor,actions,IDs,rows andflags byte/value equal. '
                      'All NN module bytes and native stage/sampler source unchanged. Only HDF5 Dataset handle lifetime changes. '
                      'No SGD,new model,loss,schedule,data/scaler/split,task-specific configuration or planning change.')
    Path(output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,windows=len(rows),nn_files=len(nn))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['old','new','data','output']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args();main(args.old,args.new,args.data,args.output)
