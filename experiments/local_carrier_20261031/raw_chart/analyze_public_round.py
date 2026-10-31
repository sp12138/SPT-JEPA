"""CPU-only frozen-operand diagnosis; no model/trainer or precision repair."""
import argparse
import hashlib
import json
from pathlib import Path


def main(args):
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2)
    root = Path(args.root); out = root/'capture'
    reproduction = json.loads((out/'capture-exit.json').read_text())
    assert reproduction['captured'] and reproduction['original_failure_exact'], reproduction
    captured = torch.load(out/'capture.pt', map_location='cpu', weights_only=False)
    meta, records = captured['metadata'], captured['tensors']
    assert meta['label'] == 'Precise chart inverse adjoint'
    value = lambda name: records[name]['value']
    target, solved, public = value('target').double(), value('solved64'), value('public')
    first, second, derivative = (value('closure_'+name).double() for name in ['first', 'second', 'derivative'])
    bound = meta['bound']; tolerance = value('tolerance'); width = target.shape[-1]

    def residual(x): return bound*F.linear(F.linear(x, second.T)*derivative, first.T)
    def error(x): return (x.double()+residual(x.double())-target).norm(dim=-1)/(1-bound)
    def summary(x):
        e = error(x); den = tolerance.clamp_min(torch.finfo(torch.float64).tiny)
        return dict(violations=int((e>tolerance).sum()), max_error_over_tolerance=float((e/den).max()))

    bad = value('public_error_bound')>tolerance
    assert bad.any() and not (value('solved_error_bound')>tolerance).any()
    target_norm = target.norm(dim=-1); normalizer = target_norm.clamp_min(torch.finfo(torch.float64).tiny)
    agreement = float(((error(public)-value('public_error_bound')).abs()/normalizer).max())
    assert agreement < 1e-12, agreement
    x = target.clone(); internal_stop = None
    for i in range(meta['iterations']):
        x = target-residual(x)
        if (i+1)%4 == 0 or i+1 == meta['iterations']:
            if not (error(x)>tolerance).any():
                internal_stop = i+1
                break
    assert internal_stop is not None
    replay_difference = float(((x-solved).norm(dim=-1)/normalizer).max())
    assert replay_difference < 1e-12, replay_difference
    extras = []; x = solved.clone()
    for i in range(4):
        x = target-residual(x)
        extras.append(dict(additional_updates=i+1, internal=summary(x), public=summary(x.to(public.dtype))))

    # Independent16x16 direct solve for EVERY actually violating vector.
    g = target[bad]; d = derivative[bad]
    jacobian = torch.eye(width, dtype=torch.float64)[None]+bound*(second[None]*d[:, None, :])@first
    oracle = torch.linalg.solve(jacobian.transpose(-1, -2), g.unsqueeze(-1)).squeeze(-1)
    oracle_public = oracle.to(public.dtype).double()
    oracle_residual = oracle_public+bound*F.linear(F.linear(oracle_public, second.T)*d, first.T)-g
    oracle_error = oracle_residual.norm(dim=-1)/(1-bound)
    den = tolerance[bad]
    oracle_public_pass = bool((oracle_error<=den).all())
    nonzero = target[bad].abs(); nonzero = nonzero[nonzero>0]
    tiny = torch.finfo(public.dtype).tiny
    ratios = value('public_error_bound')[bad]/den
    report = dict(cpu_only=True, helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        capture_sha256=hashlib.sha256((out/'capture.pt').read_bytes()).hexdigest(), source=meta['source'],
        original_failure_exact=True, label=meta['label'], shape=list(target.shape), public_dtype=str(public.dtype),
        bound=bound, atol=meta['atol'], rtol=meta['rtol'], original_iteration_cap=meta['iterations'],
        recorded_violations=int(bad.sum()), cpu_recorded_residual_agreement_relative_to_rhs=agreement,
        original_internal_stop_updates=internal_stop, internal_replay_difference_relative_to_rhs=replay_difference,
        recorded_internal=summary(solved), recorded_public=summary(public),
        violating_slots=dict(indices=bad.nonzero().tolist(),
            internal_error_over_tolerance=(value('solved_error_bound')[bad]/den).tolist(),
            public_error_over_tolerance=ratios.tolist(), target_norm=target_norm[bad].tolist(),
            tolerance=den.tolist(), minimum_nonzero_rhs_component=float(nonzero.min()),
            subnormal_rhs_components=int(((target[bad]!=0)&(target[bad].abs()<tiny)).sum()),
            subnormal_public_components=int(((public[bad]!=0)&(public[bad].abs()<tiny)).sum())),
        extra_updates_same_full_cpu_shape=extras,
        independent_oracle=dict(every_violating_slot=True, direct_matrix_dimension=width,
            public_all_pass=oracle_public_pass, public_error_over_tolerance=(oracle_error/den).tolist(),
            solved_distance_from_oracle_over_rhs=((solved[bad]-oracle).norm(dim=-1)/target_norm[bad]).tolist()),
        conclusion=('For this exactly reproduced failure, representable FP32 solutions satisfy the unchanged external certificate, and additional unchanged fixed-point updates pass after rounding. The original FP64 internal stopping criterion leaves insufficient margin for final public rounding. This is a captured-case diagnosis, not a universal representability guarantee.'
            if oracle_public_pass and extras[-1]['public']['violations']==0 else
            'The tested FP32 witnesses do not jointly resolve the cause. Direct-solve rounding or further updates failing would not prove that every FP32 solution is impossible; inspect the recorded per-vector scales and residuals before any repair.'),
        scope='Frozen operands only. No change to source, model, tolerance, iteration cap, training, or planner.')
    (root/'public-round-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    main(parser.parse_args())
