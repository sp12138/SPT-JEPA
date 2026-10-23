"""Use the standard DCP reader to compare saved tensors on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import torch
import torch.distributed.checkpoint as dcp

torch.set_num_threads(4)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('original', type=Path)
parser.add_argument('candidate', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
roots = [args.original.resolve(), args.candidate.resolve()]
metadata = [pickle.loads((root/'weights/.metadata').read_bytes()) for root in roots]
assert metadata[0].state_dict_metadata.keys() == metadata[1].state_dict_metadata.keys()
states = []
for root, meta in zip(roots, metadata):
    state = {key: torch.empty(item.size, dtype=item.properties.dtype) for key, item in meta.state_dict_metadata.items()}
    dcp.load(state, checkpoint_id=str(root/'weights'), no_dist=True)
    states.append(state)
    print(json.dumps(dict(checkpoint=str(root), tensors_loaded=len(state))), flush=True)
groups = {}
rows = []
for key in states[0]:
    x, y = states[0][key], states[1][key]
    assert x.shape == y.shape and x.dtype == y.dtype
    group = 'ema_shadows' if key.startswith('model.ema.shadow_') else 'online_parameters' if key.startswith(('model.net.', 'model.causal_semantic_encoder.')) else 'ema_metadata'
    identical = torch.equal(x, y)
    if x.is_floating_point():
        xf, yf = x.double(), y.double()
        delta = yf-xf
        reference_squared = float(xf.square().sum())
        delta_squared = float(delta.square().sum())
        maximum = float(delta.abs().max()) if delta.numel() else 0.0
    else:
        reference_squared = delta_squared = maximum = 0.0
    row = dict(name=key, group=group, shape=list(x.shape), dtype=str(x.dtype), identical=identical, relative_l2=(delta_squared/reference_squared)**0.5 if reference_squared else None, max_absolute_difference=maximum)
    rows.append(row)
    aggregate = groups.setdefault(group, dict(tensors=0, identical_tensors=0, elements=0, reference_squared=0.0, delta_squared=0.0, max_absolute_difference=0.0))
    aggregate['tensors'] += 1
    aggregate['identical_tensors'] += identical
    aggregate['elements'] += x.numel()
    aggregate['reference_squared'] += reference_squared
    aggregate['delta_squared'] += delta_squared
    aggregate['max_absolute_difference'] = max(aggregate['max_absolute_difference'], maximum)
for key in ['model.ema.initialized', 'model.ema.parameter_names_utf8', 'model.ema.num_updates', 'model.ema.decay_value']:
    assert torch.equal(states[0][key], states[1][key]), key
ema_updates = int(states[0]['model.ema.num_updates'])
for group in groups.values():
    group['relative_l2'] = (group['delta_squared']/group['reference_squared'])**0.5 if group['reference_squared'] else None
result = dict(status='PASS', checkpoint_paths=[str(p) for p in roots], device='cpu', reader='torch.distributed.checkpoint.load(no_dist=True)', tensors=len(rows), groups=groups, ema_update_count=ema_updates, ema_metadata_identical=True, tensor_differences=rows, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
output = args.output
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'tensor_differences'}, indent=2))
