"""Frozen-world-model action-proposal pilot;common native RGB-goal protocol."""
import argparse
import datetime
import hashlib
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np

from experiments.metaworld_eval_20261103.environment import (make_environment, pixels, physics_state,
                                                           rgb_replay_check, close_environment)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):digest.update(block)
    return digest.hexdigest()


def resize_tensor(frames):
    import torch
    from torch.nn import functional as F
    value = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    return F.interpolate(value, size=(224, 224), mode='bilinear', align_corners=False,
                         antialias=True).round().clamp(0, 255).byte()


def model_sha(model):
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def main(checkpoint, data, manifest, output, task=None, bank=None, limit=None, budget=500,
         success_mode='any', proposal='held', library=None):
    import gymnasium as gym
    import torch
    import stable_worldmodel as swm
    from stable_worldmodel.planning import ShootingCostEvaluator, GoalMSE
    from stable_worldmodel.planning.solver import CEMSolver
    from vforge.ops import OpsConfig, initialize_ops
    from experiments.metaworld_proposal_20261104.planners import (held_projection, build_library,
                                                                  LibrarySolver, audit_plan)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=torch.device('cuda'), parallel_state=None)
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    card = json.loads((Path(data) / 'dataset_card.json').read_text())
    assert card['physical_frameskip'] == 5 and len(card['action_mean']) == 20
    mean, std = np.asarray(card['action_mean'], np.float32), np.asarray(card['action_std'], np.float32)
    model = swm.wm.utils.load_pretrained(checkpoint).cuda().eval().requires_grad_(False)
    before = model_sha(model)
    lower = torch.tensor((-1 - mean) / std, device='cuda')
    upper = torch.tensor((1 - mean) / std, device='cuda')

    class BoundedCost(ShootingCostEvaluator):
        def __init__(self):
            super().__init__(model, GoalMSE()); self.first_bank_sha256 = None
        def get_cost(self, info, candidates):
            if proposal == 'held':
                held_projection(candidates, torch.as_tensor(mean,device='cuda'), torch.as_tensor(std,device='cuda'))
            # In-place projection ensures native elite updates use the actual
            # bounded candidates,not an unprojected shadow. Common to all arms.
            candidates.clamp_(min=lower, max=upper)
            if self.first_bank_sha256 is None:
                self.first_bank_sha256 = hashlib.sha256(candidates.detach().cpu().numpy().tobytes()).hexdigest()
            return super().get_cost(info, candidates)

    features = actions = None
    if proposal == 'library':
        assert library is not None
        library = Path(library)
        if not (library/'manifest.json').exists():
            features, actions = build_library(model, data, library)
        else:
            proof = json.loads((library/'manifest.json').read_text())
            assert proof['passed'] and proof['model_state_before'] == before
            assert proof['split_sha256'] == file_sha(Path(data)/'split-manifest.json')
            assert proof['library_sha256'] == file_sha(library/'library.pt')
            values = torch.load(library/'library.pt', map_location='cuda', weights_only=True)
            features, actions = values['features'], values['actions']

    cases = json.loads(Path(manifest).read_text())['rows']
    if task:cases = [c for c in cases if c['task'] == task]
    if bank:cases = [c for c in cases if (c['seed'] >= 1000) == (bank == 'fresh')]
    if limit:cases = cases[:limit]
    assert cases
    results = []; started = time.monotonic(); audit_inputs = None
    for case in cases:
        assert file_sha(case['path']) == case['sha256']
        cached = np.load(case['path'])
        env, initial, vector = make_environment(case['task'], case['seed'], cached['rand_vec'])
        assert np.array_equal(initial, cached['initial_state'])
        assert all(np.array_equal(value, cached['initial_physics_'+key])
                   for key,value in physics_state(env).items())
        actual_frame = pixels(env); rgb_check = rgb_replay_check(cached['initial'], actual_frame)
        assert rgb_check['passed'], rgb_check
        frame = cached['initial'].copy()  # identical first RGB input for all methods
        frames, past_actions, successes, executed = [frame], [], [], []
        goal = model.normalize(resize_tensor([cached['goal']])[None].cuda())
        cost = BoundedCost()
        solver = CEMSolver(cost, batch_size=1, num_samples=300, n_steps=15, topk=10,
                           device='cuda', seed=20261103 + case['seed'] * 1009)
        solver.configure(action_space=gym.spaces.Box(-1., 1., shape=(1,4), dtype=np.float32), n_envs=1,
                         config=SimpleNamespace(horizon=6, action_block=5))
        if proposal == 'library':
            solver = LibrarySolver(model, cost, features, actions)
        warm = None; plan_calls = 0; terminated = truncated = False
        try:
            while len(successes) < budget:
                context = min(2, len(frames))
                history = np.asarray(past_actions[-(context-1):], np.float32) if context > 1 else np.empty((0,20), np.float32)
                info = dict(pixels=model.normalize(resize_tensor(frames[-context:])[None].cuda()), goal=goal,
                            action=torch.zeros(1, context, 20, device='cuda'),
                            action_history=torch.from_numpy(history)[None].cuda())
                assert set(info) == {'pixels', 'goal', 'action', 'action_history'}
                with torch.inference_mode():solution = solver.solve(info, init_action=warm)
                plan_calls += 1
                plan = solution['actions'].numpy()[0]
                if plan_calls == 1 and case == cases[0]:
                    # Defer simulation until EVERY online environment is closed.
                    # Live MuJoCo viewers can interfere even with equal physics.
                    audit_inputs = ({k:v.clone() for k,v in info.items()}, plan.copy(), dict(case),
                                    {k:cached[k].copy() for k in ['initial','rand_vec']})
                warm = solution['actions'][:,3:].clone()
                for normalized in plan[:3]:
                    raw = np.clip(normalized * std + mean, -1, 1).reshape(5,4)
                    for action in raw:
                        _, reward, terminated, truncated, outcome = env.step(action.astype(np.float32))
                        successes.append(bool(outcome['success'])); executed.append(action.copy())
                        if len(successes) >= budget or terminated or truncated or (success_mode == 'any' and successes[-1]):break
                    if len(successes) >= budget or terminated or truncated or (success_mode == 'any' and successes[-1]):break
                    frames.append(pixels(env)); past_actions.append(((raw.reshape(-1)-mean)/std).astype(np.float32))
                if terminated or truncated or (success_mode == 'any' and successes[-1]):break
        finally:close_environment(env)
        success = any(successes) if success_mode == 'any' else successes[-1]
        trace = output / f"{case['task']}-seed{case['seed']}.npz"
        np.savez_compressed(trace, actions=np.asarray(executed), successes=np.asarray(successes))
        results.append(dict(task=case['task'], seed=case['seed'], success=success, any_success=any(successes),
                            final_success=successes[-1], steps=len(successes), plan_calls=plan_calls,
                            expert_goal_success=case['expert_any_success'], case_sha256=case['sha256'],
                            first_bank_sha256=cost.first_bank_sha256, trace=str(trace), trace_sha256=file_sha(trace)))
        results[-1]['initial_rgb_replay'] = rgb_check
        print(json.dumps(results[-1]), flush=True)
    assert audit_inputs is not None
    audit_info,audit_actions,audit_case,audit_cache = audit_inputs
    audit = audit_plan(model,audit_info,audit_actions,audit_case,audit_cache,mean,std)
    audit['execution_order'] = 'After all online environments closed;never interrupts online controller.'
    (output/'physical-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    after = model_sha(model); assert before == after
    groups = {}
    for name in sorted({r['task'] for r in results}):
        selected = [r for r in results if r['task'] == name]
        groups[name] = dict(episodes=len(selected), success_rate=100 * np.mean([r['success'] for r in selected]),
                            goal_construction_failures=sum(not r['expert_goal_success'] for r in selected),
                            goal_valid_episodes=sum(r['expert_goal_success'] for r in selected),
                            goal_valid_success_rate=(100 * np.mean([r['success'] for r in selected if r['expert_goal_success']])
                                                     if any(r['expert_goal_success'] for r in selected) else None))
    report = dict(passed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), checkpoint=checkpoint,
                  weights_sha256=file_sha(Path(checkpoint) / 'weights.pt'), model_state_before=before, model_state_after=after,
                  model_updates=0, parameters=sum(p.numel() for p in model.parameters()),
                  renderer=dict(uniform_first_frame_warmup=True, explicit_mjr_context_cleanup=True,
                                helper_environment_sha256=file_sha(inspect.getfile(make_environment))),
                  data_card_sha256=file_sha(Path(data) / 'dataset_card.json'), manifest_sha256=file_sha(manifest),
                  groups=groups, episodes=results, budget=budget, success_mode=success_mode,
                  controller=dict(samples=300, iterations=15, elites=10, horizon_macro=6, execute_macro=3,
                                  max_context=2, frameskip=5, native_box_projection=True, warm_start=True,
                                  proposal=proposal, forecasts_per_plan=4500, library=str(library) if library else None),
                  seconds=time.monotonic()-started,
                  scope='Independent LeRobot RGB-only image-goal extension. All cases including failed expert goals retained; '
                        '500-step any-success and100-step final-success are separate runs if requested. '
                        'Not a published JEPA-WMs or canonical task-ID-conditionedMT50 score. '
                        'No environment state,reward,success or optimal action enters learner or planner. '
                        'New proposal mechanism;compare paired cases with archived standard CEM. '
                        'Library is extra train-memory access,not identical plannerinformation/compute or novelty evidence.')
    (output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['checkpoint','data','manifest','output']:parser.add_argument('--'+name, required=True)
    parser.add_argument('--task'); parser.add_argument('--bank', choices=['primary','fresh'])
    parser.add_argument('--limit', type=int); parser.add_argument('--budget', type=int, default=500)
    parser.add_argument('--proposal', choices=['held','library','standard'], required=True)
    parser.add_argument('--library')
    parser.add_argument('--success-mode', choices=['any','final'], default='any')
    args = parser.parse_args()
    main(args.checkpoint,args.data,args.manifest,args.output,args.task,args.bank,args.limit,args.budget,
         args.success_mode,args.proposal,args.library)
