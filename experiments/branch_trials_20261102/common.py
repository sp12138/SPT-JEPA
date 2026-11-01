"""Finite three-host trials; helpers have no background research scheduler."""
import hashlib
import json
from pathlib import Path

B=Path('/efs/prescottsun/SPT-JEPA')
PY='/efs/prescottsun/project_jepa/envs/lewm/bin/python'


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8<<20),b''):digest.update(part)
    return digest.hexdigest()


def source_check(source):
    for key in ['manifest','runtime_manifest']:
        if key not in source:continue
        manifest=read(source[key])
        for name,digest in manifest['files'].items():assert sha(Path(manifest['source'])/name)==digest,name


def dataset(cfg,split):
    if cfg.get('dataset_kind')=='benchmark_hdf5':
        from as_jepa_tools.benchmark_data import BenchmarkWindows
        from as_jepa_tools.benchmark_gap import GapBenchmarkWindows
        cls=GapBenchmarkWindows if cfg.get('benchmark_anchor_rule')=='past_gap' else BenchmarkWindows
        extra=dict(gap=cfg['gap']) if cls is GapBenchmarkWindows else {}
        ds=cls(cfg['data'],window=8,seed=cfg['seed'],split=split,**extra)
    else:
        from as_jepa_tools.data import FullPushTWindows
        ds=FullPushTWindows(cfg['data'],window=8,gap=cfg.get('gap',8),seed=cfg['seed'],split='train' if split=='train' else 'test')
    if cfg.get('goal_query_reference'):
        from as_jepa_tools.goal_query_data import QueryReferenceWindows
        ds=QueryReferenceWindows(ds,seed=cfg['seed'],mode=cfg.get('goal_query_reference_mode','global'))
    return ds
