"""Bind a newly completed checkpoint to the existing frozen f4 controller.

Only builds the immutable-input spec. Evaluation and action mapping are owned
by the already reviewed action_covariance evaluator, without reimplementation.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

from gate_audit import read,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--template',required=True);a=p.parse_args()
    template=read(a.template)
    for path,digest in template.pop('prepared_inputs').items():assert sha(path)==digest,path
    helper=template.pop('controller_source');manifest=read(helper['manifest'])
    for name,digest in manifest['files'].items():assert sha(Path(helper['source'])/name)==digest,name
    directory=Path(helper['source'])/'experiments/action_covariance_20261031'
    checkpoint=Path(template['checkpoint']);reference=read(template['reference_metrics'])
    assert len(reference['pairs'])==50 and reference['manifest_sha256']==sha(template['manifest'])
    assert reference['checkpoint']==str(checkpoint) and reference['checkpoint_audit']['strict_equal']
    assert reference['pairs']==read(template['manifest'])['pairs']
    assert reference['normalization']=='train episodes'
    template['helper_files']={n:sha(directory/n) for n in ['adapter.py','evaluate.py']}
    paths=[Path(template['reference_metrics']),checkpoint/'weights.pt',checkpoint/'config.json',
        Path(template['manifest']),Path(template['data'])/'dataset_card.json',Path(template['native_evaluator']),
        Path(template['calibration']),Path(template['calibration']).with_name('matrix.npz')]
    template['immutable_inputs']={str(path):sha(path) for path in paths}
    out=Path(template['output']);out.mkdir(parents=True,exist_ok=True)
    spec=out/'executed-spec.json'
    with spec.open('x') as f:f.write(json.dumps(template,indent=2,allow_nan=False)+'\n')
    sys.path.insert(0,str(directory));sys.argv=[str(directory/'evaluate.py'),'--spec',str(spec)]
    runpy.run_path(str(directory/'evaluate.py'),run_name='__main__')


if __name__=='__main__':main()
