"""Bounded, latent-only factual history diagnostic using the ORIGINAL native rollout."""
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

B = Path('/efs/prescottsun/SPT-JEPA')
PY = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'
PATHS = {'cold5': (1, 5), 'warm5': (3, 5), 'cold7': (1, 7)}
ROOTS = {'identity': 'local_isotropic2000_20261031',
         'static': 'dynamic_readout2000_20261031', 'state': 'dynamic_readout2000_20261031'}


def read(p): return json.loads(Path(p).read_text())
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda: f.read(8 << 20), b''): h.update(x)
    return h.hexdigest()
def source_check(s):
    for key in ['manifest', 'runtime_manifest']:
        m = read(s[key])
        for f, v in m['files'].items(): assert sha(Path(m['source'])/f) == v, f
def state_hash(model):
    import torch
    h = hashlib.sha256()
    for kind, values in [('parameter', model.named_parameters()), ('buffer', model.named_buffers())]:
        for name, a in values:
            a = a.detach().cpu().contiguous()
            h.update(str((kind, name, tuple(a.shape), a.dtype)).encode())
            h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def rng_hash():
    import random
    import numpy as np
    import torch
    h = hashlib.sha256(repr(random.getstate()).encode())
    h.update(repr(np.random.get_state()).encode()); h.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_initialized():
        for x in torch.cuda.get_rng_state_all(): h.update(x.cpu().numpy().tobytes())
    return h.hexdigest()


def check_cache(s):
    import numpy as np
    for key in ['config', 'weights', 'cache', 'state', 'windows']:
        assert sha(s[key]) == s[key+'_sha256'], key
    cfg, meta, windows = read(s['config']), read(s['state']), read(s['windows'])
    assert meta['step'] == 2000 and meta['cases'] == 64 and meta['window'] == 8 and meta['state_dim'] == 1024
    assert meta['weights_sha256'] == s['weights_sha256'] and meta['config_sha256'] == s['config_sha256']
    assert meta['manifest_sha256'] == s['windows_sha256']
    assert windows['split'] == 'validation' and windows['count'] == 64 and windows['frameskip'] == 5
    data = Path(cfg['data']); assert sha(data/'dataset_card.json') == windows['dataset_card_sha256']
    for f, digest in windows['metadata_sha256'].items(): assert sha(data/f) == digest
    eps = np.load(data/'episode_ids.npy'); offsets = np.load(data/'offsets.npy')
    train = set(np.load(data/'train_episodes.npy').tolist()); val = set(np.load(data/'validation_episodes.npy').tolist())
    assert not train & val
    for w in windows['windows']:
        i = w['episode_array_index']; assert int(eps[i]) == w['episode'] and w['episode'] in val
        assert w['rows'] == (int(offsets[i])+w['start']+np.arange(8)*5).tolist()
    assert len({w['episode'] for w in windows['windows']}) == 64
    with np.load(s['cache']) as f: a = {k: f[k].copy() for k in ['observed', 'actions', 'clean_rollout']}
    assert a['observed'].shape == (64, 8, 1024) and a['clean_rollout'].shape == (64, 7, 1024)
    assert a['actions'].shape == (64, 8, 10 if s['task'] == 'tworoom' else 25)
    assert all(np.isfinite(v).all() for v in a.values()) and np.count_nonzero(a['actions'][:, -1]) == 0
    assert cfg['model']['window'] == 8 and cfg['seed'] == s['seed']
    assert cfg['model'].get('spatial_readout_mode', 'identity') == s['arm']
    assert cfg['model']['spatial_mode'] == 'local' and cfg['model'].get('spatial_chart', 'none') == 'none'
    assert cfg['model'].get('spatial_basis', 'identity') == 'identity'
    calibration = cfg['spatial_calibration']; assert sha(calibration['path']) == calibration['sha256']
    assert read(calibration['path'])['normalization'] == 'global_rms'
    return a, cfg


def prepare(args):
    import numpy as np
    root = Path(args.root); assert not root.exists(), 'Preserve prior roots'
    helper = read(args.source_record) if args.source_record else None
    script = Path(helper['source'])/'experiments/planner_context_20261031/assay.py' if helper else Path(__file__).resolve()
    if helper: source_check(helper); assert sha(script) == sha(__file__)
    jobs, specs, refs, checked = [], [], {}, set()
    for task in ['tworoom', 'cube']:
        for seed in [1, 2]:
            for arm in ['identity', 'static', 'state']:
                parent = B/'runs'/ROOTS[arm]
                name = f'{task}-spatial-local-seed{seed}' if arm == 'identity' else f'{task}-readout-{arm}-seed{seed}'
                run = parent/name; folder = parent/('transition-'+name)
                source = read(parent/'source-record.json')
                if source['source_commit'] not in checked: source_check(source); checked.add(source['source_commit'])
                s = dict(id=name, task=task, seed=seed, arm=arm, model_source=source, helper_source=helper, helper_sha256=sha(__file__),
                    output=str(root/name), budget=dict(predict=136, action_encoder=24, encoder=0, renderer=0))
                for key, path in dict(config=run/'config.json', weights=run/'model-step0002000/weights.pt', cache=folder/'state-audit.npz',
                                      state=folder/'state-audit.json', windows=folder/'windows.json').items():
                    s[key] = str(path); s[key+'_sha256'] = sha(path)
                a, cfg = check_cache(s); key = (task, seed)
                sig = (cfg['model']['spatial_mean'], cfg['model']['spatial_scale'], s['windows_sha256'])
                if key in refs:
                    old, oldsig = refs[key]; assert sig == oldsig
                    assert np.array_equal(a['observed'], old['observed']) and np.array_equal(a['actions'], old['actions'])
                else: refs[key] = (a, sig)
                sp = root/'specs'/(name+'.json'); write(sp, s); specs.append(s)
                i = len(jobs); host, gpu = ('dev2', i) if i < 8 else ('dev13', i-4)
                jobs.append(dict(id=name, host=host, gpu=gpu, hold=True, output=s['output'], source=helper,
                    command=[PY, '-B', '-u', str(script), 'run', '--spec', str(sp)]))
    write(root/'candidate-plan.json', dict(root=str(root), deadline=None, environment={'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2'}, jobs=jobs))
    write(root/'protocol.json', dict(specs=specs, paths=PATHS, batch_size=8, windows=64, total_predict_calls=1632, total_action_encoder_calls=288,
        encoder_renderer_simulator_calls=0, target_frames={'cold5':[1,2,3,4,5], 'warm5':[3,4,5,6,7], 'cold7':[1,2,3,4,5,6,7]},
        semantics='Original LeWM.rollout, cached emb and shape-only pixels, original action_encoder/predict. No noise, no labels, no simulator. Warm versus cold7 changes real information AND prediction depth; factual recorded actions only, not ranking/planning causality. Zero is fixed training-center coordinates.',
        precision='FP32 highest; original SDPA; eval no_grad', helper=helper))
    (root/'HOLD_DISPATCH').write_text('Candidate; source freeze and actual availability gate required.\n')
    write(root/'cache-contracts.json', dict(passed=True, models=12, matching_windows_actions_observed_within_task_seed=True,
        original_sources=list(checked), new_RGB_reads=0, model_calls=0))
    print(json.dumps(dict(root=str(root), models=12, passed=True)), flush=True)


def native_path(model, observed, actions, history, horizon):
    # The native rollout inspects pixels.size(2) only when emb is already supplied.
    # A zero-element tensor makes this shape-only; forbidden encode hooks fail closed.
    info = dict(pixels=observed.new_empty(len(observed), 1, history, 0), emb=observed[:, None, :history],
                action_history=actions[:, None, :history-1])
    result = model.rollout(info, actions[:, None, history-1:history-1+horizon])['predicted_emb']
    import torch
    assert result.shape == (len(observed), 1, history+horizon, observed.shape[-1])
    assert torch.equal(result[:, 0, :history], observed[:, :history]) and torch.isfinite(result).all()
    return result[:, 0, history:]


def enter(args):
    s = read(args.spec); env = {**os.environ, 'PYTHONPATH':s['model_source']['pythonpath'], 'PYTHONDONTWRITEBYTECODE':'1'}
    os.chdir(s['model_source']['source'])
    os.execve(PY, [PY, '-B', '-u', str(Path(__file__).resolve()), 'inner', '--spec', args.spec]+(['--cpu-contract'] if args.cpu_contract else []), env)


def inner(args):
    import numpy as np
    import torch
    from as_jepa_lewm.factory import build_model
    from stable_worldmodel.wm.lewm.lewm import LeWM
    from vforge.ops import OpsConfig, initialize_ops
    s = read(args.spec); assert sha(__file__) == s['helper_sha256']; source_check(s['model_source'])
    a, cfg = check_cache(s); torch.set_num_threads(2); torch.set_float32_matmul_precision('highest'); torch.manual_seed(1042)
    device = torch.device('cpu' if args.cpu_contract else 'cuda')
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=device, parallel_state=None)
    model = build_model(cfg).to(device).eval(); model.requires_grad_(False)
    model.load_state_dict(torch.load(s['weights'], map_location='cpu', weights_only=True), strict=True)
    assert model.rollout.__func__ is LeWM.rollout and model.predictor.num_frames == 7
    imports = {}
    for name in dict.fromkeys(['as_jepa_lewm.factory', 'vforge.ops']+[c.__module__ for c in type(model).__mro__ if c.__module__.startswith(('as_jepa', 'stable_worldmodel'))]):
        path = Path(inspect.getsourcefile(importlib.import_module(name))).resolve()
        if name.startswith('as_jepa'): assert path.is_relative_to(Path(s['model_source']['source']))
        imports[name] = dict(path=str(path), sha256=sha(path))
    counts = dict(predict=0, action_encoder=0, encoder=0, renderer=0); lengths = []
    budget = dict(predict=1, action_encoder=1, encoder=0, renderer=0) if args.cpu_contract else s['budget']
    def forbidden(*a, **kw): raise AssertionError('Encoding/rendering forbidden in this latent-only assay')
    model.encode = forbidden
    handles = [model.encoder.register_forward_pre_hook(forbidden), model.renderer.register_forward_pre_hook(forbidden)]
    def action_hook(module, inputs):
        counts['action_encoder'] += 1; assert counts['action_encoder'] <= budget['action_encoder']
    handles.append(model.action_encoder.register_forward_pre_hook(action_hook)); original_predict = model.predict
    def predict(z, act):
        counts['predict'] += 1; assert counts['predict'] <= budget['predict']
        assert z.shape[:2] == act.shape[:2]; lengths.append(z.shape[1]); return original_predict(z, act)
    model.predict = predict
    obs = torch.from_numpy(a['observed']).to(device); act = torch.from_numpy(a['actions']).to(device)
    if device.type == 'cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    before, rngbefore, start = state_hash(model), rng_hash(), time.time(); values = {}
    with torch.no_grad():
        if args.cpu_contract:
            # Actual first model + pinned native path, one short CPU forward only.
            test = native_path(model, obs[:1], act[:1], 3, 1)
            assert test.shape == (1, 1, 1024) and lengths == [3]
        else:
            for key, (history, horizon) in PATHS.items():
                offset = len(lengths)
                values[key] = torch.cat([native_path(model, obs[i:i+8], act[i:i+8], history, horizon) for i in range(0, 64, 8)]).cpu().numpy()
                assert lengths[offset:] == list(range(history, history+horizon))*8
    if device.type == 'cuda': torch.cuda.synchronize()
    seconds = time.time()-start; after, rngafter = state_hash(model), rng_hash()
    assert before == after and rngbefore == rngafter and counts == budget
    assert all(not x.training for x in model.modules()) and sha(s['weights']) == s['weights_sha256']
    assert all(sha(v['path']) == v['sha256'] for v in imports.values())
    report = dict(passed=True, spec=s, actual_calls=counts, state_hash_before=before, state_hash_after=after,
                  rng_hash_before=rngbefore, rng_hash_after=rngafter, seconds=seconds, actual_imports=imports,
                  predict_input_lengths=lengths, torch_version=torch.__version__, device=str(device),
                  peak_gib=0 if args.cpu_contract else torch.cuda.max_memory_allocated()/2**30)
    if args.cpu_contract:
        write(Path(args.spec).parent.parent/'actual-model-cpu-contract.json', report)
    else:
        out = Path(s['output']); out.mkdir(exist_ok=True)
        d = values['cold7'].astype(np.float64)-a['clean_rollout'].astype(np.float64)
        report['old_cache_numerical_replay'] = dict(max_abs=float(np.abs(d).max()), mse=float(np.square(d).mean()), bit_exact=bool(np.array_equal(values['cold7'],a['clean_rollout'])))
        # Check gross path disagreement, not bitwise identity across differently shaped GEMMs.
        report['old_cache_numerical_replay']['guard'] = 'relative RMS error <= 1e-4 of max(1, old forecast RMS)'
        assert np.sqrt(np.square(d).mean()) <= 1e-4*max(1.,np.sqrt(np.square(a['clean_rollout'].astype(np.float64)).mean()))
        report['cold5_vs_cold7_prefix'] = dict(max_abs=float(np.abs(values['cold5']-values['cold7'][:,:5]).max()))
        np.savez_compressed(out/'predictions.npz', **values)
        report['predictions_sha256'] = sha(out/'predictions.npz'); write(out/'result.json', report)
    print(json.dumps({k:report[k] for k in ['passed','actual_calls','seconds','device']}), flush=True)


def collect(args):
    import numpy as np
    root = Path(args.root); plan = read(root/'plan.json'); assert len(plan['jobs']) == 12
    rows = []; totals = dict(predict=0, action_encoder=0, encoder=0, renderer=0)
    for job in plan['jobs']:
        assert read(root/'exits'/(job['id']+'.json'))['returncode'] == 0
        out = Path(job['output']); r = read(out/'result.json'); s = r['spec']; a, cfg = check_cache(s)
        assert r['passed'] and r['state_hash_before'] == r['state_hash_after'] and r['rng_hash_before'] == r['rng_hash_after']
        assert sha(out/'predictions.npz') == r['predictions_sha256']
        with np.load(out/'predictions.npz') as f: predictions = {k:f[k].astype(np.float64) for k in PATHS}
        conditions = {}
        for key, (h, t) in PATHS.items():
            y = a['observed'][:,h:h+t].astype(np.float64); p = predictions[key]; anchor = a['observed'][:,h-1:h].astype(np.float64)
            metrics = dict(mse=np.square(p-y).mean(2), zero_mse=np.square(y).mean(2), copy_mse=np.square(anchor-y).mean(2),
                           prediction_energy=np.square(p).mean(2), twice_alignment=2*(p*y).mean(2))
            assert np.max(np.abs(metrics['mse']-(metrics['zero_mse']+metrics['prediction_energy']-metrics['twice_alignment']))) < 1e-12
            aggregate = {k:float(v.mean()) for k,v in metrics.items()}; horizons = {k:v.mean(0).tolist() for k,v in metrics.items()}
            aggregate.update(over_zero=aggregate['mse']/aggregate['zero_mse'], over_copy=aggregate['mse']/aggregate['copy_mse'])
            horizons['over_zero'] = (metrics['mse'].mean(0)/metrics['zero_mse'].mean(0)).tolist()
            horizons['over_copy'] = (metrics['mse'].mean(0)/metrics['copy_mse'].mean(0)).tolist()
            conditions[key] = dict(mean=aggregate, horizons=horizons, target_frames=list(range(h,h+t)), per_window={k:v.tolist() for k,v in metrics.items()})
        delta = np.asarray(conditions['warm5']['per_window']['mse'])[:,-1]-np.asarray(conditions['cold7']['per_window']['mse'])[:,-1]
        boot = delta[np.random.default_rng(1042).integers(64,size=(10000,64))].mean(1)
        rows.append(dict(id=s['id'],task=s['task'],seed=s['seed'],arm=s['arm'],conditions=conditions,
            same_endpoint_warm5_minus_cold7=dict(mean=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),per_window=delta.tolist()),
            replay=r['old_cache_numerical_replay'],result=str(out/'result.json'),result_sha256=sha(out/'result.json'),windows_sha256=s['windows_sha256']))
        for k in totals: totals[k] += r['actual_calls'][k]
    assert totals == dict(predict=1632,action_encoder=288,encoder=0,renderer=0)
    comparisons=[]
    for task in ['tworoom','cube']:
        for seed in [1,2]:
            ref=next(r for r in rows if (r['task'],r['seed'],r['arm'])==(task,seed,'identity'))
            for arm in ['static','state']:
                row=next(r for r in rows if (r['task'],r['seed'],r['arm'])==(task,seed,arm))
                assert row['windows_sha256']==ref['windows_sha256']
                for mode in PATHS:
                    delta=np.asarray(row['conditions'][mode]['per_window']['mse']).mean(1)-np.asarray(ref['conditions'][mode]['per_window']['mse']).mean(1)
                    boot=delta[np.random.default_rng(1042).integers(64,size=(10000,64))].mean(1)
                    comparisons.append(dict(task=task,seed=seed,candidate=arm,reference='identity',mode=mode,mean=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),per_window=delta.tolist()))
    write(Path(args.output)/'results.json',dict(rows=rows,paired_model_comparisons=comparisons,actual_calls=totals,all12_pass=True,
        limitation='Factual recorded actions only. Cold5 ends at obs5; warm5 at obs7, so their raw errors are not a context effect. Warm5 versus cold7 uses the same endpoint but adds observations and reduces forecast depth. Bootstrap conditions on fixed models/windows, not independent seeds or counterfactual action rankings. No planning result.'))
    brief=[dict(task=r['task'],seed=r['seed'],arm=r['arm'],**{k:dict(mean_over_zero=r['conditions'][k]['mean']['over_zero'],endpoint_over_zero=r['conditions'][k]['horizons']['over_zero'][-1],endpoint_over_copy=r['conditions'][k]['horizons']['over_copy'][-1])for k in PATHS},same_endpoint_delta=r['same_endpoint_warm5_minus_cold7']['mean'])for r in rows]
    write(Path(args.output)/'summary.json',dict(rows=brief,actual_calls=totals,all12_pass=True));print(json.dumps(brief,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['prepare','run','inner','collect']);p.add_argument('--root');p.add_argument('--source-record');p.add_argument('--spec');p.add_argument('--cpu-contract',action='store_true');p.add_argument('--output');a=p.parse_args()
    {'prepare':prepare,'run':enter,'inner':inner,'collect':collect}[a.mode](a)
