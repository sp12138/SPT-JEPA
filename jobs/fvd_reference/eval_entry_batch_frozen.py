"""Use Native EvalRunner; replace only fixed input loader and artifact writer with I3D features."""
from pathlib import Path
from dataclasses import replace
import os,sys,json,hashlib,time
import numpy as np
J=Path(__file__).resolve().parent;S=json.loads((J/'state.json').read_text());R=Path(S['report'])
S=dict(S,**S.get('evaluation_profile',{}))
sys.path[:0]=[S['runtime_snapshot'],str(J/'detector')]
import torch
from omegaconf import OmegaConf
from vforge.evaluation.runner import EvalOutputWriter,sample_artifact_name
from vforge.data.loader import build_eval_loader
from vforge.distributed.parallel_state import get_parallel_state
from vforge.tasks.rep_world.eval import collate_sample, batch_eval_loader
from fvd import I3DExtractor
import vforge.tasks.eval as native_eval
name,ckpt,output=sys.argv[1:4];smoke='--smoke' in sys.argv
BATCH_SIZE=2 if smoke else int(S['sampling_batch_sizes'][name])
assert 2048%(8*BATCH_SIZE)==0
rows=[json.loads(x) for x in (R/'data/eval_2048.jsonl').read_text().splitlines()]
B=R/'fvd/bank';extractor=None
FVD_PROTOCOL=json.loads((R/'fvd/protocol.json').read_text())
FVD_PROTOCOL_HASH=hashlib.sha256(json.dumps(dict(protocol=FVD_PROTOCOL,
 evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 detector_sha256=hashlib.sha256((J/'detector/fvd.py').read_bytes()).hexdigest(),
 native_source_digest=S['source_digest']),sort_keys=True).encode()).hexdigest()
class FVDOutputWriter(EvalOutputWriter):
 def is_complete(self,output_dir,*,sample_id,seed,provenance):
  if not super().is_complete(output_dir,sample_id=sample_id,seed=seed,provenance=provenance):return False
  try:sidecar=json.loads(Path(self.expected_path(output_dir,sample_id)+'.json').read_text())
  except (OSError,ValueError):return False
  return sidecar.get('metadata',{}).get('fvd_protocol_sha256')==FVD_PROTOCOL_HASH
class Bank(torch.utils.data.Dataset):
 def __len__(self):return len(rows)
 def __getitem__(self,index):
  r=rows[index];pixels=np.load(B/(r['sample_id']+'.npy'),allow_pickle=False)
  meta=json.loads((B/(r['sample_id']+'.json')).read_text())
  assert meta['pixel_sha256']==hashlib.sha256(pixels.tobytes()).hexdigest()
  rgb=torch.from_numpy(pixels[:16]).float().div_(255).mul_(2).sub_(1).permute(3,0,1,2).contiguous()
  return dict(sample_id=r['sample_id'],rgb=rgb)
def write_feature(result,output_dir):
 global extractor
 if extractor is None:
  extractor=I3DExtractor(S['detector_weights'],device=f"cuda:{torch.cuda.current_device()}",cache_dir=None,batch_size=1)
 pred=result['prediction']
 assert pred.shape==(1,3,144,256,256),pred.shape
 pixels=pred[0].detach().float().clamp(-1,1).add(1).mul(127.5).round().to(torch.uint8).permute(1,2,3,0).contiguous().cpu().numpy()
 with torch.autocast('cuda',enabled=False):feature=extractor._infer([pixels[16:]])[0]
 meta=json.loads((B/(str(result['sample_id'])+'.json')).read_text())
 assert extractor.identity==meta['detector'],'GT/pred detector mismatch'
 path=Path(output_dir)/sample_artifact_name(result['sample_id'],suffix='.npy')
 temp=path.with_suffix('.npy.tmp')
 with temp.open('wb') as f:np.save(f,feature,allow_pickle=False)
 os.replace(temp,path)
 result['metadata'].update(fvd_protocol_sha256=FVD_PROTOCOL_HASH,detector=extractor.identity,metric_frames=[16,144],generated_rgb_sha256=hashlib.sha256(pixels[16:].tobytes()).hexdigest(),reference_rgb_sha256=meta['pixel_sha256'])
 print(json.dumps(dict(event='fvd_feature',sample_id=result['sample_id'],time=time.time(),rank=os.environ.get('RANK'))),flush=True)
 return str(path)
original=native_eval.resolve_builder
def custom_builder(model_name,verb):
 base=original(model_name,verb)
 def build(request):
  global extractor
  # Runtime has initialized the rank device. Apply detector/generation flags
  # BEFORE the first prediction, so first cases and resumed cases agree.
  extractor=I3DExtractor(S['detector_weights'],
                       device=f"cuda:{torch.cuda.current_device()}",cache_dir=None,batch_size=1)
  components=base(request)
  loader=build_eval_loader(Bank(),parallel_state=get_parallel_state(),collate_fn=collate_sample,
                            num_samples=request.num_samples,sample_selection='first')
  loader=batch_eval_loader(loader,BATCH_SIZE)
  return replace(components,dataloader=loader,output_writer=FVDOutputWriter(
   artifact_name=lambda sid:sample_artifact_name(sid,suffix='.npy'),write_output=write_feature))
 return build
native_eval.resolve_builder=custom_builder
identity=next(r for r in json.loads((J/'eval_runs.json').read_text()) if r['name']==name)
cfg=OmegaConf.load(identity['config'])
cfg.model.extras.rollout.batch_size=BATCH_SIZE
cfg.runtime.output_dir=output
cfg.runtime.weights_strict=True
cfg.parallel.dp_mode='fsdp2'
cfg.parallel.gradient_checkpointing=False
cfg.checkpoint.enabled=False
cfg.evaluation.data_source='validation'
cfg.evaluation.manifest=str(R/'data/eval_2048.jsonl')
cfg.evaluation.checkpoint_path=ckpt
cfg.evaluation.checkpoint_format='dcp'
cfg.evaluation.output_dir=output
cfg.evaluation.num_samples=16 if smoke else 2048
cfg.evaluation.seed=20260920
cfg.evaluation.guidance_scale=1
cfg.evaluation.num_inference_steps=50
cfg.evaluation.skip_existing=True
cfg.model.extras.rollout.total_states=9
cfg.model.extras.rollout.history_window_states=8
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
native_eval.run_eval(cfg)
if int(os.environ.get('RANK','0'))==0:
 path=Path(output)/'eval_manifest.json'
 manifest=json.loads(path.read_text())
 manifest['sampling_batch_size']=BATCH_SIZE
 manifest['evaluation_weights']=cfg.model.extras.ema.evaluation_weights
 manifest['ema_decay']=cfg.model.extras.ema.decay
 manifest['evaluation_profile']=S['profile_name']
 manifest['fvd_protocol']=FVD_PROTOCOL
 manifest['fvd_protocol_sha256']=FVD_PROTOCOL_HASH
 tmp=path.with_suffix('.json.protocol.tmp');tmp.write_text(json.dumps(manifest,indent=2));os.replace(tmp,path)
