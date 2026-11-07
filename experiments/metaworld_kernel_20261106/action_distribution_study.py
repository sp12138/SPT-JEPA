"""Bounded actor tests: clipping likelihood, goal correspondence, seed robustness.

All actors are fresh scratch and supervise only the current native action.
No per-task choice, expert future action, extra data or external teacher.
"""
import argparse
import copy
import json
from pathlib import Path
import time

from experiments.metaworld_kernel_20261106.campaign import save, digest

BASE = Path('/efs/prescottsun/SPT-JEPA')


def prepare(a):
    import numpy as np
    a.root.mkdir(); (a.root/'results').mkdir()
    card = json.loads((BASE/'data/metaworld-lerobot-b46a1c18-packed-v1/dataset_card.json').read_text())
    mean = np.asarray(card['action_mean'][:4], np.float32)
    std = np.asarray(card['action_std'][:4], np.float32)
    recipes = []
    variants = [('censored', 0), ('point', 1), ('correspondence', 2), ('baseline-replica', 3),
                ('censored-single', 4), ('point-single', 5)]
    for host, paired_seed in [('dev13', 3), ('dev15', 4)]:
        for arm, gpu in variants:
            seed = paired_seed+2 if arm == 'baseline-replica' else paired_seed
            cfg = json.loads((BASE/f'runs/metaworld_native_maturity40_20261107/training/directed-seed{paired_seed}/config.json').read_text())
            run = a.root/'training'/f'{arm}-seed{seed}'; run.mkdir(parents=True)
            cfg.update(output=str(run), seed=seed, max_steps=40000, retain_steps=[2, 600, 40000])
            if arm.startswith(('point', 'censored')):
                cfg['model_class'] = 'censored_action_bridge'
                cfg['model'].update(objective='mse', components=1 if arm.endswith('-single') else 4, observation_model=arm.split('-')[0],
                    action_lower=((-np.ones(4, np.float32)-mean)/std).tolist(),
                    action_upper=((np.ones(4, np.float32)-mean)/std).tolist())
            elif arm == 'correspondence':
                cfg['model_class'] = 'correspondence_bridge'
            save(run/'config.json', cfg)
            world = BASE/f'runs/metaworld_native_transport_20261106/training/stencil-seed{paired_seed}'
            assert json.loads((world/'audit-20000.json').read_text())['passed']
            save(run/'ORIGIN.json', dict(all_actor_weights_random=True, actor_steps=40000, actor_lr_horizon=40000,
                target='ONLY current native executed a_t4D; NO a_t+1 or later command labels.',
                goal='Original future RGB; same default goal grid5/anchors/episode balancing for all arms.',
                own_world=str(world), world_steps=20000, world_ancestry='Original-data scratch future-RGB ONLY; no inverse teacher.',
                scope='Fixed-cap tests, not automatically promoted. Original public LeRobot corpus/common cases/RGB sensor5/500 any-success. Not public SOTA/equalFLOPs. World3/4 each shared by6 actors, not12 full-pipeline independent pretrainings.'))
            recipes.append(dict(host=host, gpu=gpu, seed=seed, arm=arm, run=str(run), world=str(world),
                wait_output=str(BASE/f'runs/metaworld_native_solver_control_20261107/euler24-old-seed{paired_seed-2}'),
                config_sha256=digest(run/'config.json')))
    # Every selected GPU is already free; a completed marker is the explicit queue dependency.
    dependency = BASE/'runs/metaworld_native_goalclock_20261107/training/native-goal-seed3'
    assert (dependency/'complete.json').exists()
    for r in recipes: r['wait_output'] = str(dependency)
    screen = json.loads((BASE/'runs/metaworld_current_control50_20261107/plan.json').read_text())
    save(a.root/'training-plan.json', dict(source=str(a.source), evaluator=str(a.evaluator), evaluator_sha256=digest(a.evaluator),
        recipes=recipes, stages=[2, 600, 40000], promotion_kind='bounded_mechanism_test',
        manifest=str(BASE/'runs/metaworld_eval_20261103/cases-v2/manifest.json'), deadline_unix=time.time()+18*3600,
        confirmation=dict(manifest=screen['manifest'], tasks=screen['tasks'], seeds=[2042, 2043, 3042, 3043], episodes=200, cohort='screen50'),
        scope='12 fixed-cap actors: censor-v-point xunimodal/4-component identical model pairs3/4; learned-correspondence Flow3/4; mature baseline replicas5/6. Scratch/currenta_t-only/sample-level/original-data. Default Euler12 or one-pass mixture. OwnF20k3/4 shared within actor groups. No automatic120k.'))


def contract(a):
    import numpy as np
    import torch
    from as_jepa_tools.native_inverse import NativeInversePairs
    from as_jepa_lewm.factory import build_model
    from experiments.metaworld_kernel_20261106.audit import equal
    torch.set_num_threads(2)
    plan = json.loads((a.root/'training-plan.json').read_text())
    pairs = {}; reports = []
    for r in plan['recipes']:
        cfg = json.loads((Path(r['run'])/'config.json').read_text())
        assert not any(cfg.get(k) for k in ['initial_checkpoint', 'initial_model_checkpoint', 'initial_encoder_checkpoint'])
        data = NativeInversePairs(cfg, 'test'); assert data.steps == data.reserve == 1
        for i in np.linspace(0, len(data)-1, 16, dtype=int):
            value = data[int(i)]; start = value['current_interval_start']
            record = next(v for v in data.records if v['episode']*5 == value['episode'])
            shard = data.shards[record['shard']]; original = shard['controls']
            class Guard:
                def __getitem__(self, key):
                    assert isinstance(key, slice) and key.start == start and key.stop == start+1
                    return original[key]
            shard['controls'] = Guard()
            try: assert equal(value, data[int(i)])
            finally: shard['controls'] = original
        torch.manual_seed(cfg['seed']); model = build_model(cfg)
        base_cfg = copy.deepcopy(cfg)
        base_cfg['model_class'] = 'temporal_bridge'
        for key in ['components', 'observation_model', 'action_lower', 'action_upper']:
            base_cfg['model'].pop(key, None)
        base_cfg['model'].pop('retain_goal_chart', None)
        base_cfg['model']['objective'] = 'flow'
        torch.manual_seed(cfg['seed']); baseline = build_model(base_cfg)
        common = ['encoder.', 'predictor.', 'action_input.', 'condition.']
        if r['arm'] != 'correspondence': common.append('goal_fusion.')
        shared = {k:v for k,v in baseline.state_dict().items() if any(k.startswith(p) for p in common)}
        assert equal(shared, {k:model.state_dict()[k] for k in shared})
        if r['arm'].startswith(('point', 'censored')):
            pair = (r['seed'], cfg['model']['components'])
            if pair in pairs: assert equal(pairs[pair], model.state_dict())
            else: pairs[pair] = {k:v.clone() for k,v in model.state_dict().items()}
        batch = {k:torch.stack([data[0][k], data[len(data)//2][k]]) if torch.is_tensor(data[0][k])
                 else torch.tensor([data[0][k], data[len(data)//2][k]]) for k in data[0]}
        model.eval(); loss, _ = model.loss(batch); assert torch.isfinite(loss); loss.backward()
        gradients = {}
        for prefix, expected in model.audit_expected_gradients().items():
            grads = [p.grad.float() for n,p in model.named_parameters() if n.startswith(prefix+'.') and p.grad is not None]
            norm = sum(float(g.square().sum()) for g in grads)**.5
            assert grads and all(bool(torch.isfinite(g).all()) for g in grads) and norm > 0
            gradients[prefix] = norm
        generator = torch.Generator().manual_seed(993)
        proposal = model.propose(batch, generator=generator)
        assert proposal.shape == (2, 1, 4) and bool(torch.isfinite(proposal).all())
        reports.append(dict(arm=r['arm'], seed=r['seed'], shared_initial_tensors=len(shared), loss=float(loss), gradients=gradients,
            current_action_guards=16, proposal_shape=list(proposal.shape)))
    save(a.root/'initial-contract.json', dict(passed=True, new_updates=0, shared_CNN_decoder_initial_exact=True,
        point_censored_entire_initial_state_exact=True if pairs else None, current_native_action_only=True, reports=reports,
        scope='Actual real-window CPU gradients/proposals/current-action guards. Actual native2 full optimizer restore/GPU backward/20physics gates still mandatory.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['prepare', 'contract'], required=True)
    for key in ['root', 'source', 'evaluator']: p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args(); globals()[a.mode](a)
