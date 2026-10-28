"""Read a committed Native checkpoint through torchDCP and export model weights."""
import argparse,json,pickle,hashlib
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--model-config",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 root=Path(a.checkpoint);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
 manifest=json.loads((root/"manifest.json").read_text());assert manifest["state"]=="full"
 meta=pickle.loads((root/"weights/.metadata").read_bytes());weights={key:torch.empty(item.size,dtype=item.properties.dtype) for key,item in meta.state_dict_metadata.items()}
 dcp.load(weights,checkpoint_id=str(root/"weights"),no_dist=True)
 if all(key.startswith("model.") for key in weights):weights={key.removeprefix("model."):value for key,value in weights.items()}
 torch.save(weights,out/"weights.pt");(out/"config.json").write_text(Path(a.model_config).read_text())
 (out/"manifest.json").write_text(json.dumps(dict(checkpoint=str(root),completed_steps=manifest["completed_steps"],weights_sha256=hashlib.sha256((out/"weights.pt").read_bytes()).hexdigest(),reader="torch.distributed.checkpoint.load(no_dist=True)",tensors=len(weights)),indent=2))
 print(len(weights))
if __name__=="__main__":main()
