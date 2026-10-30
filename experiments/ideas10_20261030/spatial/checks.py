"""CPU-only constructive checks, not training or benchmark performance evidence."""
import argparse
import json
from pathlib import Path

import numpy as np


def annihilator(cumulative_actions):
    x = np.column_stack([np.ones(len(cumulative_actions)), cumulative_actions])
    return np.eye(len(x)) - x @ np.linalg.pinv(x), x


def action_annihilator():
    # This path is not closed; no synthetic counterfactual labels are needed.
    u = np.array([-1., -.4, .1, .7, 1.2, .3, -.6, .2])[:, None]
    p, x = annihilator(u)
    scales = np.array([.2, .1, .05, .025, .0125])
    smooth, contact = [], []
    for eps in scales:
        q = .7 + eps * u
        visible = np.column_stack([np.sin(q[:, 0]), np.cos(q[:, 0])])
        smooth.append(float(np.linalg.norm(p @ visible)))
        # At a contact boundary, the map is Lipschitz but not differentiable.
        contact.append(float(np.linalg.norm(p @ np.maximum(eps * u, 0))))
    smooth_order = np.log(np.array(smooth[:-1]) / smooth[1:]) / np.log(2)
    contact_order = np.log(np.array(contact[:-1]) / contact[1:]) / np.log(2)
    affine = 17. + u @ np.array([[2., -3., .7]])
    affine_error = float(np.linalg.norm(p @ affine))
    shifted_p, _ = annihilator(73 * u + 5)
    # Global action-only conditional-mean removal is a different operator:
    # state-dependent initial offset and slope remain, local P kills both.
    state = np.array([-2., -1., 1., 2.])
    y = state[:, None, None] + (state[:, None, None] + 3) * u[None]
    global_residual = y - y.mean(axis=0, keepdims=True)
    local_residual = np.einsum('ij,njk->nik', p, y)
    assert affine_error < 1e-11 and np.max(np.abs(p @ x)) < 1e-12
    assert np.all(smooth_order > 1.99) and np.allclose(contact_order, 1)
    assert np.max(np.abs(p - shifted_p)) < 1e-12
    return dict(scales=scales.tolist(), smooth_residual=smooth, contact_residual=contact,
                smooth_observed_order=smooth_order.tolist(), contact_observed_order=contact_order.tolist(),
                affine_annihilation_error=affine_error, action_coordinate_invariance_error=float(np.max(abs(p-shifted_p))),
                global_action_mean_residual_norm=float(np.linalg.norm(global_residual)),
                local_annihilator_residual_norm=float(np.linalg.norm(local_residual)),
                rank=int(np.linalg.matrix_rank(x)), residual_dimension=int(len(x)-np.linalg.matrix_rank(x)))


def contact_projection():
    # Closed convex halfspace n.q >= b; only this fixed set gets the guarantee.
    n, boundary = np.array([-1., 1.]), .1
    def project(q):
        violation = np.minimum(q @ n - boundary, 0)
        return q - violation[..., None] * n / (n @ n)
    rng = np.random.default_rng(20261030)
    x, y = rng.normal(size=(2, 10000, 2))
    px, py = project(x), project(y)
    difference = px - py
    firm_gap = (difference ** 2).sum(-1) - (difference * (x-y)).sum(-1)
    proposal = np.array([.4, .2])
    corrected = project(proposal)
    # A physical saturating action is not invertible, unlike a group action.
    start, action = -.2, .5
    after = min(start + action, 0.)
    back = min(after - action, 0.)
    assert np.max(firm_gap) < 1e-12
    assert np.max(abs(project(px)-px)) < 1e-12
    return dict(firm_nonexpansion_max_gap=float(firm_gap.max()),
                feasibility_min_slack=float(np.min(px @ n-boundary)),
                idempotence_max_error=float(np.max(abs(project(px)-px))),
                free_proposal=proposal.tolist(), projected_contact=corrected.tolist(),
                blocked_round_trip=dict(start=start, contact=after, returned=back, defect=back-start),
                scope="Known convex halfspace, no learned image coordinates or nonconvex global guarantee; existing contact-model baseline.")


def first_passage():
    # S,G,D. Action A toggles S<->G; B stays. Endpoints after AA and BB match.
    a = np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]])
    b = np.eye(3)
    start = np.array([1.,0.,0.]); survive = np.diag([1.,0.,1.])
    result = {}
    for name, sequence in [('AA',[a,a]),('BB',[b,b])]:
        endpoint, alive = start.copy(), start.copy()
        for transition in sequence:
            endpoint = endpoint @ transition
            alive = alive @ transition @ survive
        result[name] = dict(endpoint=endpoint.tolist(), hit_probability=float(1-alive.sum()))
    # Right-censored geometric event data: ignoring non-events overestimates rate.
    true_hazard, cap = .2, 3
    probs = true_hazard * (1-true_hazard)**np.arange(cap)
    expected_event_rate = probs.sum()
    expected_observed_time = sum((1-true_hazard)**k for k in range(cap))
    correctly_censored = expected_event_rate / expected_observed_time
    event_only = expected_event_rate / np.dot(np.arange(1,cap+1), probs)
    assert result['AA']['endpoint'] == result['BB']['endpoint']
    assert result['AA']['hit_probability'] == 1 and result['BB']['hit_probability'] == 0
    assert abs(correctly_censored-true_hazard) < 1e-14
    return dict(path_query_counterexample=result, true_hazard=true_hazard,
                correct_censored_mle=correctly_censored, event_only_mle=event_only,
                observed_event_fraction=float(expected_event_rate),
                scope="Classical killed-kernel and survival identities; not a new theorem or proof that image-defined physical events are identifiable.")


def aliasing_glue():
    # Equal observation/embedding but opposite velocity; endpoint-only joins fail.
    x, v_left, v_right, horizon = 0., 1., -1., 5
    return dict(endpoint_distance=0., continuation_distance=abs((x+horizon*v_left)-(x+horizon*v_right)),
                scope="A counterexample to endpoint-only stitching, not an innovation claim; history-aware models can already fix it.")


def real_data(base, episodes=128):
    import hdf5plugin  # registers the existing dataset's read-only compression filter
    import h5py
    rows = {}
    for task in ['tworoom','cube']:
        meta = base / (task+'-metadata')
        card = json.loads((meta/'dataset_card.json').read_text())
        train = np.load(meta/'train_episodes.npy')
        rng = np.random.default_rng(20261030)
        selected = sorted(rng.choice(train, min(episodes,len(train)),replace=False).tolist())
        residuals, absolute_residuals, pixel_ratios, ranks, proxy, contacts = [], [], [], [], [], []
        label_scope = 'diagnostic crossing of recorded continuous gripper-contact .5; also report 0/.9 sensitivity' if task=='cube' else 'diagnostic departure from nominal 5*action displacement, not certified contact labels'
        with h5py.File(card['h5_path'],'r',swmr=True) as f:
            offsets, lengths = f['ep_offset'][:], f['ep_len'][:]
            # Smooth fixed image measurement. No learned or privileged target.
            t = np.arange(224)+.5
            basis = np.cos(np.pi*np.arange(4)[:,None]*t[None]/224).astype(np.float32)/224
            for ep in selected:
                length = int(lengths[ep]); offset=int(offsets[ep])
                if length < 36: continue
                start = int(rng.integers(0,length-35))
                ids = offset+start+np.arange(8)*5
                actions = np.asarray(f['action'][int(ids[0]):int(ids[-1])],dtype=float)
                integrated = np.vstack([np.zeros(actions.shape[-1]), actions.reshape(7,5,-1).sum(1).cumsum(0)])
                p,x = annihilator(integrated); ranks.append(int(np.linalg.matrix_rank(x)))
                images = np.asarray(f['pixels'][ids],dtype=np.float32)/255.
                visual = np.einsum('ih,thwc,jw->tijc',basis,images,basis,optimize=True).reshape(8,-1)
                centered = visual-visual.mean(0)
                pixel_ratios.append(float(np.sum((p@visual)**2)/max(1e-16,np.sum(centered**2))))
                if task=='cube':
                    positions = np.asarray(f['privileged_block_0_pos'][ids],dtype=float)
                    event = np.asarray(f['proprio_gripper_contact'][int(ids[0]):int(ids[-1])+1],dtype=float).ravel()
                    contacts.append(event.tolist())
                    proxy.append(bool(np.any(np.diff(event>.5)!=0)))
                else:
                    positions = np.asarray(f['pos_agent'][ids],dtype=float)
                    raw = np.asarray(f['pos_agent'][int(ids[0]):int(ids[-1])+1],dtype=float)
                    error = np.linalg.norm(np.diff(raw,axis=0)-5*actions,axis=1)
                    proxy.append(bool(np.any(error>1e-3)))
                residuals.append(float(np.linalg.norm(p@positions)/max(1e-16,np.linalg.norm(positions-positions.mean(0)))))
                absolute_residuals.append(float(np.linalg.norm(p@positions)))
        def summarize_groups(flags):
            groups={}
            for event in [False,True]:
                indices=np.array(flags)==event
                groups[str(event)]=dict(n=int(indices.sum()),
                    median_visual_residual_energy_ratio=float(np.median(np.asarray(pixel_ratios)[indices])) if indices.any() else None,
                    median_position_residual_amplitude_ratio=float(np.median(np.asarray(residuals)[indices])) if indices.any() else None,
                    median_absolute_position_residual=float(np.median(np.asarray(absolute_residuals)[indices])) if indices.any() else None)
            return groups
        groups=summarize_groups(proxy)
        rows[task]=dict(episodes=selected,windows=len(ranks),feature='fixed 4x4 spatial cosine coefficients per RGB channel',
                       window=8,frameskip=5,action_design_ranks=ranks,groups=groups,label_scope=label_scope,
                       visual_residual_energy_ratios=pixel_ratios,position_residual_amplitude_ratios=residuals,
                       absolute_position_residuals=absolute_residuals,event_proxy=proxy,
                       scope='CPU audit of existing train data, not training or task evaluation; physical fields used only for diagnosis.')
        if contacts:
            rows[task]['contact_threshold_sensitivity']={str(threshold):summarize_groups([bool(np.any(np.diff(np.asarray(c)>threshold))) for c in contacts]) for threshold in [0,.5,.9]}
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--real-data')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result=dict(action_annihilator=action_annihilator(),contact_projection=contact_projection(),first_passage=first_passage(),aliasing_glue=aliasing_glue())
    if args.real_data:result['real_data']=real_data(Path(args.real_data))
    (out/'checks.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='real_data'},indent=2))
    if 'real_data' in result:print(json.dumps({k:v['groups'] for k,v in result['real_data'].items()},indent=2))


if __name__=='__main__':main()
