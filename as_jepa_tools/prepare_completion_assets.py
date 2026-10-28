"""Project-owned immutable asset preparation for the full protocol cohort."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--install-lpips",action="store_true");a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    os.environ["HF_HOME"]=str(root/"huggingface-cache")
    from huggingface_hub import snapshot_download,HfApi
    info=HfApi().dataset_info("galilai-group/lewm-pusht",revision="ea321e3")
    pinned=info.sha
    folder=snapshot_download("galilai-group/lewm-pusht",repo_type="dataset",revision=pinned,local_dir=root/"lewm-pusht-ea321e3",max_workers=3)
    hashes={}
    for name in HfApi().list_repo_files("galilai-group/lewm-pusht",repo_type="dataset",revision=pinned):
        path=Path(folder)/name
        if not path.is_file():continue
        with path.open("rb") as handle:hashes[name]=dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(handle,"sha256").hexdigest())
    if a.install_lpips:
        target=root.parent.parent/"environments/completion-overlay"
        subprocess.run([sys.executable,"-m","pip","install","--no-deps","--target",str(target),"lpips==0.1.4"],check=True)
    (root/"assets.json").write_text(json.dumps(dict(dataset="galilai-group/lewm-pusht",revision=pinned,path=folder,files=hashes,completed_at=time.time()),indent=2))
    print(json.dumps(dict(revision=pinned,files=len(hashes),folder=folder)))

if __name__=="__main__":main()
