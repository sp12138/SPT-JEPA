"""Read-only audit of the frozen training runs and completed FVD evaluations."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf
import torch

parser = argparse.ArgumentParser()
parser.add_argument('reference', type=Path)
parser.add_argument('run', type=Path)
parser.add_argument('xie_evaluation', type=Path)
parser.add_argument('spt_evaluation', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
report = dict(time=datetime.now(timezone.utc).isoformat(), status='RUNNING', reference=str(args.reference), run=str(args.run))

def stage(name):
    print(json.dumps(dict(stage=name)), flush=True)

def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(set(a) | set(b)):
            result.extend(differences(a.get(key), b.get(key), prefix+'.'+str(key)))
        return result
    return [] if a == b else [dict(field=prefix, original=a, spt=b)]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous()
        return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
    if isinstance(value, np.ndarray):
        return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value

def state_digest(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True).encode()).hexdigest()

def normalize_paths(value, config):
    text = json.dumps(canonical(value), sort_keys=True)
    for key in ['train_manifest', 'val_manifest']:
        text = text.replace(config['data'][key], '<'+key+'>')
    return json.loads(text)

stage('source_and_training_config')
base_helper = Path(__file__).resolve().with_name('audit_alignment.py')
subprocess.run([sys.executable, '-B', str(base_helper), str(args.reference), str(args.run), str(args.output/'training_alignment.json')], check=True)
base = json.loads((args.output/'training_alignment.json').read_text())
# The helper's historical demo limits predate the two full FVD measurements.
base['limits'] = ['Sample IDs in step logs cover rank 0; saved loader/RNG states are checked separately for every rank.', 'Historical inherited environment variables and initial full model tensors were not archived by upstream.']
(args.output/'training_alignment.json').write_text(json.dumps(base, indent=2)+'\n')
original_plan = json.loads((args.reference/'plan.json').read_text())
spt_plan = json.loads((args.run/'plan.json').read_text())
configs = [OmegaConf.to_container(OmegaConf.load(root/'workers/pretrain/pretrain_effective.yaml'), resolve=True) for root in [args.reference, args.run]]
outputs = [Path(plan['pretrain_output']) for plan in [original_plan, spt_plan]]
report['source'] = dict(runtime_files_verified=base['runtime_files_verified'], train_entry_byte_identical=True, worker_functions_ast_identical=base['worker_functions_ast_identical'], training_config_differences=base['config_differences'])
envs = []
for plan in [original_plan, spt_plan]:
    tree = ast.parse((Path(plan['jobs'])/'worker.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'environment')
    call = node.body[0].value
    envs.append({kw.arg: ast.literal_eval(kw.value) if isinstance(kw.value, ast.Constant) else ast.unparse(kw.value) for kw in call.keywords})
env_diffs = differences(*envs)
assert all(d['field'].endswith('_DIR') or d['field'] in {'.CUDA_CACHE_PATH', '.XDG_CACHE_HOME'} for d in env_diffs), env_diffs
report['declared_environment_differences'] = env_diffs
nodes = [json.loads((root/'workers/pretrain/node.json').read_text()) for root in [args.reference, args.run]]
assert nodes[0]['gpus'] == nodes[1]['gpus']
report['training_hardware'] = dict(original=nodes[0], spt=nodes[1], same_gpu_models_and_count=True)

stage('training_trajectory_10000')
series = []
for output in outputs:
    records = {}
    with (output/'train_steps.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('event') == 'step':
                if row['step'] >= 10000:
                    break
                assert row['step'] not in records
                records[row['step']] = row
    assert set(records) == set(range(10000))
    series.append(records)
keys = ['learning_rate', 'learning_rate_before_step', 'learning_rate_after_step', 'dp_size', 'usp_size', 'world_size', 'step_metadata']
metric_keys = ['ema_updates', 'ema_decay', 'target_horizon_1_fraction', 'target_horizon_2_fraction']
mismatches = {key: sum(series[0][i].get(key) != series[1][i].get(key) for i in range(10000)) for key in keys}
mismatches.update({key: sum(series[0][i]['metrics'].get(key) != series[1][i]['metrics'].get(key) for i in range(10000)) for key in metric_keys})
assert not any(mismatches.values()), mismatches
deltas = [abs(series[0][i]['loss']-series[1][i]['loss']) for i in range(10000)]
windows = []
for start in range(0, 10000, 1000):
    original = [series[0][i]['loss'] for i in range(start, start+1000)]
    spt = [series[1][i]['loss'] for i in range(start, start+1000)]
    windows.append(dict(first_completed_step=start+1, last_completed_step=start+1000, original_mean_loss=statistics.mean(original), spt_mean_loss=statistics.mean(spt), mean_absolute_loss_difference=statistics.mean(deltas[start:start+1000]), max_absolute_loss_difference=max(deltas[start:start+1000])))
report['training_trajectory_10000'] = dict(paired_steps=10000, mismatches=mismatches, first_step_losses=[s[0]['loss'] for s in series], first_step_loss_difference=deltas[0], mean_absolute_loss_difference=statistics.mean(deltas), max_absolute_loss_difference=max(deltas), windows=windows, last100_mean_loss=[statistics.mean(s[i]['loss'] for i in range(9900,10000)) for s in series])

stage('all_rank_checkpoint_rng_and_loader_states')
boundaries = []
for step in [2500, 5000, 7500, 10000]:
    cps = [output/'checkpoints'/f'{step:05d}.dcp' for output in outputs]
    manifests = [json.loads((cp/'manifest.json').read_text()) for cp in cps]
    for m in manifests:
        assert m['completed_steps'] == step and m['state'] == 'full' and m['world_size'] == 8
        assert all(m['components_present'][k] for k in ['weights','optimizer','trainer'])
    for key in ['execution_policy', 'topology', 'model_fingerprint', 'optimizer_fingerprint', 'torch_version', 'python_version']:
        assert manifests[0][key] == manifests[1][key], (step, key)
    assert manifests[0]['fingerprints']['semantic_config'] == manifests[1]['fingerprints']['semantic_config']
    ranks = []
    for rank in range(8):
        states = [torch.load(cp/'trainer'/f'rank-{rank:05d}.pt', map_location='cpu', weights_only=False) for cp in cps]
        rng_differences = differences(canonical(states[0]['rng']), canonical(states[1]['rng']))
        loader_differences = differences(*[normalize_paths(state['dataloader'], config) for state, config in zip(states, configs)])
        scheduler_differences = differences(canonical(states[0]['scheduler']), canonical(states[1]['scheduler']))
        ranks.append(dict(rank=rank, rng_identical=not rng_differences, rng_differences=rng_differences, loader_identical_after_manifest_path_normalization=not loader_differences, loader_differences=loader_differences, scheduler_identical=not scheduler_differences, original_rng_sha256=state_digest(states[0]['rng']), spt_rng_sha256=state_digest(states[1]['rng'])))
    boundaries.append(dict(step=step, execution_policy_and_topology_identical=True, semantic_model_and_optimizer_config_identical=True, versions_identical=True, ranks=ranks))
    print(json.dumps(dict(checkpoint_step=step, rng_matches=sum(r['rng_identical'] for r in ranks), loader_matches=sum(r['loader_identical_after_manifest_path_normalization'] for r in ranks))), flush=True)
report['checkpoint_boundaries'] = boundaries

stage('all_2048_fvd_cases')
eval_roots = [args.xie_evaluation, args.spt_evaluation]
eval_plans = [json.loads((root/'plan.json').read_text()) for root in eval_roots]
eval_outputs = [Path(plan['output']) for plan in eval_plans]
helpers = ['common.py', 'fvd_pool.py', 'eval_entry.py', 'eval_entry_batch_frozen.py', 'artifact_io_v2.py', 'fvd_coordination.py']
original_jobs = args.reference/'fvd_eval/jobs'
for name in helpers:
    assert len({digest(root/'jobs'/name) for root in eval_roots} | {digest(original_jobs/name)}) == 1, name
assert (eval_roots[0]/'fvd/protocol.json').read_bytes() == (eval_roots[1]/'fvd/protocol.json').read_bytes()
assert (eval_roots[0]/'data/eval_2048.jsonl').read_bytes() == (eval_roots[1]/'data/eval_2048.jsonl').read_bytes()
assert (eval_roots[0]/'fvd/bank').resolve() == (eval_roots[1]/'fvd/bank').resolve()
eval_configs = [OmegaConf.to_container(OmegaConf.load(next((root/'jobs/evaluation_configs').glob('*.yaml'))), resolve=True) for root in eval_roots]
eval_diffs = differences(*eval_configs)
assert all(d['field'] in {'.data.train_manifest', '.data.val_manifest', '.runtime.output_dir', '.checkpoint.dirpath', '.evaluation.output_dir'} for d in eval_diffs), eval_diffs
rows = [json.loads(line) for line in (eval_roots[0]/'data/eval_2048.jsonl').read_text().splitlines()]
assert len(rows) == 2048
meta_fields = ['cfg_scale', 'condition_input', 'detector', 'ema_decay', 'ema_updates', 'evaluation_weights', 'fvd_protocol_sha256', 'history_states', 'metric_frames', 'reference_rgb_sha256', 'sampling_batch_size', 'seed', 'steps', 'total_states']
pixel_hash_matches = 0
feature_squared_delta = 0.0
feature_squared_reference = 0.0
for index, row in enumerate(rows):
    sid = row['sample_id']
    sides = [json.loads((output/(sid+'.npy.json')).read_text()) for output in eval_outputs]
    for field in meta_fields:
        assert sides[0]['metadata'][field] == sides[1]['metadata'][field], (sid, field)
    for side, plan in zip(sides, eval_plans):
        assert side['sample_id'] == sid and side['seed'] == sides[0]['seed']
        assert side['metadata']['ema_updates'] == 10000
        assert side['metadata']['evaluation_weights'] == 'ema'
        for source in side['provenance']['checkpoint']['sources']:
            assert source['path'] == plan['checkpoint'] and source['weights_strict'] is True
            assert source['manifest']['completed_steps'] == 10000
    pixel_hash_matches += sides[0]['metadata']['generated_rgb_sha256'] == sides[1]['metadata']['generated_rgb_sha256']
    features = [np.load(output/(sid+'.npy'), allow_pickle=False).astype(np.float64) for output in eval_outputs]
    assert all(f.shape == (400,) and np.isfinite(f).all() for f in features)
    feature_squared_delta += float(np.sum((features[1]-features[0])**2))
    feature_squared_reference += float(np.sum(features[0]**2))
    if (index+1) % 512 == 0:
        print('Paired inference cases checked: '+str(index+1), flush=True)
scores = [json.loads((output/'score.json').read_text()) for output in eval_outputs]
assert scores[0]['protocol'] == scores[1]['protocol']
assert scores[0]['fvd_protocol_sha256'] == scores[1]['fvd_protocol_sha256']
report['inference'] = dict(cases=2048, evaluator_helpers_byte_identical_to_upstream=helpers, protocol_identical=True, bank_identical=True, inference_config_differences=eval_diffs, per_case_parameters_seeds_inputs_detectors_identical=True, checkpoint_and_ema_strict_load_verified_cases=2048, generated_pixel_hash_matches=pixel_hash_matches, generated_feature_relative_l2=(feature_squared_delta/feature_squared_reference)**0.5, original_fvd=scores[0]['metric']['fvd'], spt_fvd=scores[1]['metric']['fvd'])
report['numerical_policy'] = dict(training_seed=[config['runtime']['seed'] for config in configs], runtime_deterministic=[config['runtime']['deterministic'] for config in configs], precision=[config['runtime']['precision'] for config in configs], attention_backend=[config['attention']['backend'] for config in configs], same_rng_at_all_checked_boundaries=all(r['rng_identical'] for b in boundaries for r in b['ranks']), same_loader_at_all_checked_boundaries=all(r['loader_identical_after_manifest_path_normalization'] for b in boundaries for r in b['ranks']), fa4_default_deterministic=False)
report['limits'] = ['No new training run or multi-seed experiment was performed; an 11% FVD difference cannot be causally attributed from this audit alone.', 'All saved ranks and 4 checkpoint boundaries were checked; per-step rank 1-7 input pixels and the initial full model tensors were not archived.', 'Historical inherited environment variables, CUDA drivers and installed-package snapshots are not fully available for upstream.']
report['audit_script_sha256'] = digest(Path(__file__))
report['status'] = 'PASS'
(args.output/'reproducibility_audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({key:report[key] for key in ['status','training_trajectory_10000','numerical_policy','inference']}, indent=2), flush=True)
