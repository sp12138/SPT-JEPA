"""Acquire official task data and released reference weights into campaign-owned EFS."""
import argparse, hashlib, json, tarfile, time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import zstandard

def main():
    a=argparse.ArgumentParser();a.add_argument("task",choices=["cube","tworoom"]);args=a.parse_args()
    b=Path("/efs/prescottsun/SPT-JEPA");root=b/"data/hardtasks-v1";root.mkdir(parents=True,exist_ok=True)
    repo="quentinll/lewm-"+dict(cube="cube",tworoom="tworooms")[args.task]
    name=dict(cube="cube_single_expert.tar.zst",tworoom="tworoom.tar.zst")[args.task]
    api=HfApi();revision=api.dataset_info(repo).sha
    path=Path(hf_hub_download(repo_id=repo,filename=name,repo_type="dataset",revision=revision,cache_dir=b/"cache/huggingface",local_dir=root/"downloads"))
    out=root/args.task;out.mkdir(exist_ok=True)
    marker=out/"acquisition.json"
    if marker.exists():
        print(marker.read_text());return
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):digest.update(block)
    with path.open("rb") as f,zstandard.ZstdDecompressor().stream_reader(f) as z,tarfile.open(fileobj=z,mode="r|") as tar:
        tar.extractall(out,filter="data")
    h5=list(out.rglob("*.h5"));assert h5,h5
    model_revision=api.model_info(repo).sha
    weights=b/"weights/hardtasks"/args.task;weights.mkdir(parents=True,exist_ok=True)
    for file in ["config.json","weights.pt"]:
        hf_hub_download(repo_id=repo,filename=file,revision=model_revision,cache_dir=b/"cache/huggingface",local_dir=weights)
    record=dict(task=args.task,dataset_repo=repo,dataset_revision=revision,archive_sha256=digest.hexdigest(),files=[str(x) for x in h5],model_revision=model_revision,weights=str(weights),finished_at=time.time())
    marker.write_text(json.dumps(record,indent=2)+"\n");print(json.dumps(record),flush=True)
if __name__=="__main__":main()
