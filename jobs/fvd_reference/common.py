from pathlib import Path
import hashlib, json, os, tempfile, uuid
J=Path(__file__).resolve().parent
S=json.loads((J/'state.json').read_text())
R=Path(S['report'])
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,ensure_ascii=False)
    for attempt in range(3):
        tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with tmp.open('x') as f:
                f.write(text);f.flush();os.fsync(f.fileno())
            try: os.replace(tmp,path);return
            except FileNotFoundError:
                if path.exists() and path.read_text()==text:return
                if attempt==2:raise
        finally:tmp.unlink(missing_ok=True)
def environment():
    return dict(os.environ,PYTHONPATH=S['runtime_snapshot'],PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
        HYDRA_FULL_ERROR='1',FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED='1',
        FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=str(Path(S['cache_root'])/'scaling_fa4_h200/fa4'),
        CUTE_DSL_CACHE_DIR=str(Path(S['cache_root'])/'scaling_fa4_h200/cute'))
def verify_runtime():
    for f,h in json.loads((J/'source_hashes.json').read_text()).items():
        assert hashlib.sha256((Path(S['runtime_snapshot'])/f).read_bytes()).hexdigest()==h,f
