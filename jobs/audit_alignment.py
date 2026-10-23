"""Audit a frozen SPT pretraining run against its supplied upstream launch."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument('reference', type=Path)
parser.add_argument('run', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--comparison', type=Path)
args = parser.parse_args()
original = json.loads((args.reference/'plan.json').read_text())
plan = json.loads((args.run/'plan.json').read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(set(a) | set(b)):
            result.extend(differences(a.get(key), b.get(key), prefix+'.'+key))
        return result
    return [] if a == b else [dict(field=prefix, original=a, spt=b)]

a = OmegaConf.to_container(OmegaConf.load(args.reference/'workers/pretrain/pretrain_effective.yaml'), resolve=True)
b = OmegaConf.to_container(OmegaConf.load(args.run/'workers/pretrain/pretrain_effective.yaml'), resolve=True)
diffs = differences(a,b)
allowed = {'.checkpoint.dirpath', '.data.train_manifest', '.data.val_manifest', '.runtime.output_dir'}
if not a['model']['extras']['repa']['enabled'] and not b['model']['extras']['repa']['enabled']:
    allowed.add('.model.extras.repa.teacher_checkpoint')
assert all(item['field'] in allowed for item in diffs), diffs
manifest_hashes = {}
for field in ['train_manifest','val_manifest']:
    hashes = dict(original=digest(Path(a['data'][field])), spt=digest(Path(b['data'][field])))
    assert hashes['original'] == hashes['spt']
    manifest_hashes[field] = hashes
expected = json.loads((Path(original['jobs'])/'runtime_hashes.json').read_text())
for name, expected_hash in expected.items():
    assert digest(Path(original['runtime'])/name) == expected_hash, name
    assert digest(Path(plan['runtime'])/name) == expected_hash, name
assert (Path(original['jobs'])/'train_entry.py').read_bytes() == (Path(plan['jobs'])/'train_entry.py').read_bytes()
functions = {}
for name in ['train','checkpoint','run','check_hardware','wait_idle']:
    def node(path):
        return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name==name)
    functions[name] = ast.dump(node(Path(original['jobs'])/'worker.py'), include_attributes=False) == ast.dump(node(Path(plan['jobs'])/'worker.py'), include_attributes=False)
assert all(functions.values())

def steps(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    return {r['step']:r for r in records if r.get('event')=='step'}

old = steps(Path(original['pretrain_output'])/'train_steps.jsonl')
new = steps(Path(plan['pretrain_output'])/'train_steps.jsonl')
keys = sorted(set(old) & set(new))
sample_mismatches = sum(old[k]['step_metadata']['sample_ids'] != new[k]['step_metadata']['sample_ids'] for k in keys)
lr_mismatches = sum(old[k]['learning_rate'] != new[k]['learning_rate'] for k in keys)
assert sample_mismatches == lr_mismatches == 0
loss_deltas = [abs(old[k]['loss']-new[k]['loss']) for k in keys]
loss_windows = []
for start in [0,2400,2500,4900,9900]:
    row = dict(first_completed_step=start+1,last_completed_step=start+100)
    for label, series in [('original',old),('spt',new)]:
        values = [series[k]['loss'] for k in range(start,start+100) if k in series]
        row[label] = dict(count=len(values),mean_loss=statistics.mean(values) if values else None)
    loss_windows.append(row)
report = dict(time=datetime.now(timezone.utc).isoformat(), reference=str(args.reference), run=str(args.run),
    source_commit=plan['source_commit'], status='PASS', config_differences=diffs,
    manifest_hashes=manifest_hashes, runtime_files_verified=len(expected),
    train_entry_byte_identical=True, worker_functions_ast_identical=functions,
    paired_training_steps=len(keys), sample_order_mismatches=sample_mismatches, learning_rate_mismatches=lr_mismatches,
    mean_absolute_loss_difference=statistics.mean(loss_deltas), max_absolute_loss_difference=max(loss_deltas),
    loss_windows=loss_windows, original_evaluation_protocol=original['evaluation'],
    limits=['FP16/BF16 kernels are not deterministic; training and decoded pixels are not bitwise identical.',
            'Eight raw-video cases were sampled; the complete FVD bank, detector and 2048-case metric were not reproduced.',
            'Later upstream checkpoints validate the sampler, not the future quality of the current SPT run.'])
if args.comparison:
    report['checkpoint_comparison'] = json.loads(args.comparison.read_text())
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['status','runtime_files_verified','paired_training_steps','sample_order_mismatches','learning_rate_mismatches','mean_absolute_loss_difference']},indent=2))
