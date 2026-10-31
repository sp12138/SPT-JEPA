"""One fixed t=.95 goal-denoising energy on the existing actual Cube bank.

This is a read-only candidate-ranking diagnostic, never a likelihood claim or
an equal-FLOP planning result. No fitting, noise/time sweep, or new simulation.
"""
import argparse
import os
from pathlib import Path
import sys
from common import B, PY, read, sha, source_check, state_hash, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', required=True, type=int, choices=[1, 2])
    p.add_argument('--output', required=True)
    p.add_argument('--inner', action='store_true')
    a = p.parse_args()
    root = B / 'runs/cube_pool_maturity23342_20261101'
    report_path = root / f'tf23342-seed{a.seed}/result.json'
    report = read(report_path); s = report['spec']
    if not a.inner:
        env = dict(os.environ, PYTHONPATH=s['model_source']['pythonpath'], PYTHONDONTWRITEBYTECODE='1')
        os.chdir(s['model_source']['source'])
        os.execve(PY, [PY, '-B', '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--inner'], env)
    import inspect
    import numpy as np
    import torch
    from torch.nn import functional as F
    from as_jepa_lewm.factory import build_model
    from vforge.models.as_jepa.pixels import pack
    from vforge.ops import OpsConfig, initialize_ops
    assert report['passed'] and report['state_before'] == report['state_after']
    source_check(s['model_source'])
    assert sha(s['weights']) == s['weights_sha256'] and sha(s['config']) == s['config_sha256']
    out = Path(a.output); assert not out.exists(); out.mkdir(parents=True)
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=torch.device('cuda'), parallel_state=None)
    model = build_model(read(s['config'])).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(s['weights'], map_location='cpu', weights_only=True), strict=True)
    assert Path(inspect.getfile(type(model))).is_relative_to(s['model_source']['source'])
    before = state_hash(model)
    inventory = read(s['inventory']); pools = {x['case']: x for x in inventory['cases_provenance']}
    counts = dict(encoder=0, predictor=0, renderer=0, rendered_images=0)
    def forbidden(*args):
        raise AssertionError('Use existing frozen state predictions;no encoding/prediction allowed')
    handles = [model.encoder.register_forward_pre_hook(forbidden), model.action_encoder.register_forward_pre_hook(forbidden),
               model.predictor.register_forward_pre_hook(forbidden)]
    def rendered(module, args):
        counts['renderer'] += 1; counts['rendered_images'] += len(args[0])
    handles.append(model.renderer.register_forward_pre_hook(rendered))
    rows = []
    def rgb64(array):
        pixels = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().cuda()
        return F.interpolate(pixels, size=(64, 64), mode='area').round() / 127.5 - 1
    with torch.inference_mode():
        for rec in report['cases']:
            assert sha(rec['file']) == rec['sha256']
            pool = pools[rec['case']]; assert sha(pool['path']) == pool['sha256'] == rec['pool_sha256']
            with np.load(rec['file']) as f, np.load(pool['path']) as raw:
                goal = rgb64(raw['goal_pixels'][None])
                initial = rgb64(raw['start_pixels'][None])
                true_rgb = rgb64(raw['pixels'][:, -1])
                y, anchor = pack(goal), pack(model.render_anchor(initial))
                generator = torch.Generator(device='cuda').manual_seed(1042 + rec['case'])
                noise = torch.randn(y.shape, dtype=y.dtype, device='cuda', generator=generator)
                x = .05*y + .95*noise
                target = y-noise
                costs = dict(latent_pred=f['predicted_cost'].copy(), latent_true=f['encoded_cost'].copy(),
                             pixel_true=(true_rgb-goal).square().flatten(1).mean(1).cpu().numpy())
                for name, states in [('denoise_pred', f['predicted_endpoint']), ('denoise_true', f['true_endpoint'])]:
                    pieces = []
                    for lo in range(0, 39, 13):
                        z = torch.from_numpy(states[lo:lo+13].copy()).cuda().reshape(-1, model.state_slots, model.state_width)
                        n = len(z); dims = (n,) + (-1,)*(x.ndim-1)
                        t = torch.full((n, 1), .95, device='cuda')
                        with model.renderer.anchored(anchor.expand((n,)+(-1,)*(anchor.ndim-1))):
                            velocity = model.renderer(x.expand(dims), t, condition=z)
                        pieces.append((target-velocity).float().square().flatten(1).mean(1).cpu().numpy())
                    costs[name] = np.concatenate(pieces)
                assert all(v.shape == (39,) and np.isfinite(v).all() for v in costs.values())
                family=f['family'];groups=dict(nonoracle=np.flatnonzero(np.isin(family,['zero','gaussian'])|np.char.startswith(family,'cem_')),
                                             all=np.arange(39));assert len(groups['nonoracle'])==22
                for group, ids in groups.items():
                    r=dict(case=rec['case'], group=group, seed=a.seed)
                    for name, value in costs.items():
                        i=int(ids[np.argmin(value[ids])]);pos=float(f['position_error_m'][i,-1]);angle=float(f['orientation_error_deg'][i,-1])
                        r[name]=dict(index=i,hold=float(f['success'][i].any()),joint=float(pos<=.03 and angle<=15),
                                     pose=float(f['physical_cost'][i,-1]),position_cm=100*pos,angle_deg=angle)
                    rows.append(r)
                path=out/f'case{rec["case"]:04d}.npz';np.savez_compressed(path,**costs)
    for h in handles:h.remove()
    assert before==state_hash(model) and counts==dict(encoder=0,predictor=0,renderer=300,rendered_images=3900)
    summary={g:{name:{metric:float(np.mean([r[name][metric] for r in rows if r['group']==g]))
                    for metric in ['hold','joint','pose','position_cm','angle_deg']} for name in costs}
             for g in ['nonoracle','all']}
    write(out/'results.json',dict(passed=True,seed=a.seed,t=.95,noise_seed_rule='1042+case;one shared draw across all candidates/endpoints',
          counts=counts,state_before=before,state_after=before,source_report_sha256=sha(report_path),rows=rows,summary=summary,
          limitation='Fixed old first-segment bank. High-noise flow residual is an energy,not log likelihood. Renderer work is extra inference compute;no planner/equal-FLOP/task/novelty claim.'))
    print(str(summary),flush=True)


if __name__=='__main__':main()
