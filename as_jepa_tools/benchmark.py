"""Measured full forward/backward/AdamW throughput; no quality claim."""
import argparse,json,time,statistics
from pathlib import Path
import torch
from as_jepa_lewm.model import ASJepaLeWM
from vforge.ops import initialize_ops,OpsConfig

p=argparse.ArgumentParser();p.add_argument("--batch",type=int,default=128);p.add_argument("--output",required=True);p.add_argument("--steps",type=int,default=30);a=p.parse_args()
torch.set_num_threads(2);torch.set_float32_matmul_precision("high");initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=torch.device("cuda",0),parallel_state=None)
model=ASJepaLeWM().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,fused=True)
batch={"pixels":torch.randint(0,256,(a.batch,8,3,224,224),dtype=torch.uint8,device="cuda"),"render":torch.randint(0,256,(a.batch,8,3,64,64),dtype=torch.uint8,device="cuda"),"anchor":torch.randint(0,256,(a.batch,3,64,64),dtype=torch.uint8,device="cuda"),"anchor224":torch.randint(0,256,(a.batch,3,224,224),dtype=torch.uint8,device="cuda"),"action":torch.randn(a.batch,8,10,device="cuda")}
times=[]
for i in range(a.steps):
    torch.cuda.synchronize();started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.bfloat16):loss,metrics=model.loss(batch)
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();torch.cuda.synchronize();times.append(time.perf_counter()-started)
    print(json.dumps(dict(step=i+1,seconds=times[-1],loss=float(loss.detach()))),flush=True)
record=dict(batch=a.batch,median_step_seconds=statistics.median(times[5:]),mean_step_seconds=statistics.mean(times[5:]),peak_gib=torch.cuda.max_memory_allocated()/2**30,parameters=sum(p.numel() for p in model.parameters()),steps=a.steps,gpu=torch.cuda.get_device_name(),scope="synthetic full-model optimizer steps; excludes disk loading, evaluation and checkpoints")
Path(a.output).write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
