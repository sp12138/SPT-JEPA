"""Actual RGB-only control: bridge proposals or a changed MPC target/cost.

Native reset/rendering/physics contracts are inherited. Private state/success
are checked and logged only after decisions,never supplied to the controller.
"""
import argparse
import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.metaworld_contact_20261106.diagnose import save, digest
from experiments.metaworld_eval_20261103.environment import (
    make_environment, close_environment, pixels, physics_state, rgb_replay_check,
)
from experiments.metaworld_eval_20261103.evaluate import resize_tensor, model_sha


class ObservedTargets:
    """Observed route successors: a known AP mechanism control,not novelty."""
    def __init__(self, root, local=False):
        self.local = local
        self.features, self.visual, self.records = [], [], []
        offset = 0
        for shard in range(4):
            directory = Path(root)/str(shard)
            m = json.loads((directory/'manifest.json').read_text())
            f = torch.load(directory/'features.pt', map_location='cpu', weights_only=True)
            for r in m['records']:
                if r['split'] == 'train':
                    selected = slice(r['offset'], r['offset']+r['length'])
                    self.features.append(f['embedding'][selected])
                    self.visual.append(f['patches'][selected].flatten(1).float() if local else f['embedding'][selected])
                    self.records.append(dict(start=offset, length=r['length'], episode=r['episode']))
                    offset += r['length']
        self.features = torch.cat(self.features).cuda()
        self.visual = torch.cat(self.visual).cuda()
        maximum = max(r['length'] for r in self.records)
        rows = torch.arange(maximum, device='cuda')[None]
        starts = torch.tensor([r['start'] for r in self.records], device='cuda')[:, None]
        self.lengths = torch.tensor([r['length'] for r in self.records], device='cuda')[:, None]
        self.valid = rows < self.lengths
        self.indices = (starts+rows).clamp_max(len(self.features)-1)

    def retrieve(self, current, goal):
        start_distance = (self.visual-current.flatten()).square().mean(-1)
        end_distance = (self.visual-goal.flatten()).square().mean(-1)
        values = end_distance[self.indices].masked_fill(~self.valid, float('inf'))
        suffix, back_index = values.flip(1).cummin(1)
        suffix, end = suffix.flip(1), values.shape[1]-1-back_index.flip(1)
        # j>i: every chosen successor was actually observed after its start.
        score = start_distance[self.indices[:, :-1]]+suffix[:, 1:]
        score = score.masked_fill(~self.valid[:, 1:], float('inf'))
        flat = int(score.argmin()); episode, i = divmod(flat, score.shape[1])
        j = int(end[episode, i+1]); target = min(i+6, j)
        index = int(self.indices[episode, target])
        return self.features[index], dict(episode=self.records[episode]['episode'], start=i, end=j,
                                          target=target, retrieval_score=float(score[episode, i]))


def patch_key(model, normalized):
    tokens = model.encoder(normalized, interpolate_pos_encoding=True).last_hidden_state[:, 1:]
    side = int(tokens.shape[1]**.5)
    return torch.nn.functional.adaptive_avg_pool2d(tokens.transpose(1, 2).reshape(len(tokens), -1, side, side), (4, 4)).flatten(2).transpose(1, 2).flatten(1)


def run(a):
    import gymnasium as gym
    import stable_worldmodel as swm
    from stable_worldmodel.planning.solver import CEMSolver
    from vforge.ops import OpsConfig, initialize_ops
    from as_jepa_lewm.factory import build_model
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=torch.device('cuda'), parallel_state=None)
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    output = Path(a.output); output.mkdir(parents=True, exist_ok=False)
    card = json.loads((Path(a.data)/'dataset_card.json').read_text())
    mean, std = np.asarray(card['action_mean'], np.float32), np.asarray(card['action_std'], np.float32)
    learner = None
    if a.run:
        cfg = json.loads((Path(a.run)/'config.json').read_text())
        learner = build_model(cfg).cuda().eval().requires_grad_(False)
        learner.load_state_dict(torch.load(Path(a.run)/f'model-step{a.step:07d}/weights.pt', map_location='cuda', weights_only=True))
        learner_before = model_sha(learner)
    needs_parent = a.mode != 'bridge' or learner.visual_source not in ['local', 'vit']
    parent = swm.wm.utils.load_pretrained(a.parent).cuda().eval().requires_grad_(False) if needs_parent else None
    before = model_sha(parent) if parent is not None else None
    bank = ObservedTargets(a.feature_cache, local=a.mode == 'ap-local') if a.mode.startswith('ap-') else None
    lower = torch.tensor((-1-mean)/std, device='cuda'); upper = torch.tensor((1-mean)/std, device='cuda')

    class Cost(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.model = parent; self.target = None; self.goal = None
        def get_cost(self, info, candidates):
            candidates.clamp_(min=lower, max=upper)
            pred = self.model.rollout(info, candidates)['predicted_emb'][:, :, -1]
            if a.mode == 'metric':
                pair = torch.stack([pred.flatten(0, 1), self.goal.expand(pred.shape[1], -1)], 1)
                return learner.duration(learner.memory(dict(features=pair))).reshape(pred.shape[:2])
            return (pred-self.target[None, None]).square().sum(-1)

    cases = json.loads(Path(a.manifest).read_text())['rows']
    cases = [c for c in cases if c['task'] in a.tasks and c['seed'] in a.seeds]
    cases.sort(key=lambda c:(a.tasks.index(c['task']), a.seeds.index(c['seed'])))
    assert len(cases) == len(a.tasks)*len(a.seeds)
    rows = []
    for case in cases:
        assert digest(case['path']) == case['sha256']
        cached = np.load(case['path'])
        env, state, _ = make_environment(case['task'], case['seed'], cached['rand_vec'])
        assert np.array_equal(state, cached['initial_state'])
        assert all(np.array_equal(v, cached['initial_physics_'+k]) for k, v in physics_state(env).items())
        gate = rgb_replay_check(cached['initial'], pixels(env)); assert gate['passed'], gate
        frames, history, actions, successes, targets = [cached['initial'].copy()], [], [], [], []
        goal_rgb = resize_tensor([cached['goal']]).cuda()
        generator = torch.Generator(device='cuda').manual_seed(20261106+case['seed']*1009)
        cost = Cost(); warm = None; calls = 0
        if a.mode != 'bridge':
            solver = CEMSolver(cost, batch_size=1, num_samples=300, n_steps=15, topk=10, device='cuda', seed=20261103+case['seed']*1009)
            solver.configure(action_space=gym.spaces.Box(-1., 1., shape=(1, 4), dtype=np.float32), n_envs=1,
                             config=SimpleNamespace(horizon=a.horizon, action_block=5))
        with torch.inference_mode():
            goal = parent.encode(dict(pixels=parent.normalize(goal_rgb[None])))['emb'][0, 0] if needs_parent else None
            goal_key = patch_key(parent, parent.normalize(goal_rgb)) if a.mode == 'ap-local' else goal
            if a.mode == 'bridge' and learner.visual_source == 'patch':
                goal_patches = parent.encoder(parent.normalize(goal_rgb), interpolate_pos_encoding=True).last_hidden_state[:, 1:].half()
            try:
                while len(actions) < a.budget:
                    current_rgb = resize_tensor(frames[-1:]).cuda()
                    current = parent.encode(dict(pixels=parent.normalize(current_rgb[None])))['emb'][0, 0] if needs_parent else None
                    if a.mode == 'bridge':
                        batch = dict(features=torch.stack([current, goal])[None]) if needs_parent else {}
                        if learner.visual_source in ['local', 'hybrid', 'vit']:
                            if learner.history_context:
                                previous_rgb = resize_tensor(frames[-2:-1] if len(frames)>1 else frames[-1:]).cuda()[0]
                                batch['pixels'] = torch.stack([previous_rgb, current_rgb[0], goal_rgb[0]])[None]
                                batch['previous_action'] = torch.from_numpy(history[-1] if history else np.zeros(20,np.float32))[None].cuda()
                                batch['history_valid'] = torch.tensor([bool(history)], device='cuda')
                            else:
                                batch['pixels'] = torch.stack([current_rgb[0], goal_rgb[0]])[None]
                        if learner.visual_source == 'patch':
                            current_patches = parent.encoder(parent.normalize(current_rgb), interpolate_pos_encoding=True).last_hidden_state[:, 1:].half()
                            batch['patches'] = torch.stack([current_patches[0], goal_patches[0]])[None]
                        plan = learner.propose(batch, generator=generator)[0].float().cpu().numpy()
                    else:
                        cost.goal = goal
                        if bank:
                            key = patch_key(parent, parent.normalize(current_rgb)) if a.mode == 'ap-local' else current
                            target, record = bank.retrieve(key, goal_key)
                            cost.target = target; targets.append(dict(native_step=len(actions), **record))
                        else:
                            cost.target = goal
                        context = min(2, len(frames))
                        past = np.asarray(history[-(context-1):], np.float32) if context > 1 else np.empty((0, 20), np.float32)
                        info = dict(pixels=parent.normalize(resize_tensor(frames[-context:])[None].cuda()),
                                    action=torch.zeros(1, context, 20, device='cuda'), action_history=torch.from_numpy(past)[None].cuda())
                        solution = solver.solve(info, init_action=warm)
                        plan = solution['actions'].numpy()[0]
                        warm = solution['actions'][:, a.execute:].clone()
                    calls += 1
                    for block in plan[:a.execute]:
                        raw = np.clip(block*std+mean, -1, 1).reshape(5, 4)
                        for action in raw:
                            _, _, terminated, truncated, outcome = env.step(action.astype(np.float32))
                            actions.append(action.copy()); successes.append(bool(outcome['success']))
                            if successes[-1] or terminated or truncated or len(actions) >= a.budget:
                                break
                        if successes[-1] or terminated or truncated or len(actions) >= a.budget:
                            break
                        frames.append(pixels(env)); history.append(((raw.reshape(-1)-mean)/std).astype(np.float32))
                    if successes[-1] or terminated or truncated:
                        break
            finally:
                close_environment(env)
        trace = output/f"{case['task']}-seed{case['seed']}.npz"
        np.savez_compressed(trace, actions=np.asarray(actions), successes=np.asarray(successes))
        result = dict(task=case['task'], seed=case['seed'], success=bool(any(successes)), steps=len(actions),
                      case_sha256=case['sha256'], expert_goal_success=case['expert_any_success'], initial_rgb=gate,
                      trace=str(trace), trace_sha256=digest(trace), calls=calls, targets=targets)
        rows.append(result); save(output/'progress.json', dict(rows=rows))
        print(json.dumps({k:v for k, v in result.items() if k not in ['targets', 'initial_rgb']}), flush=True)
    if parent is not None:
        assert model_sha(parent) == before
    if learner:
        assert model_sha(learner) == learner_before
    save(output/'summary.json', dict(passed=True, mode=a.mode, rows=rows, model_unchanged=True,
         parent_sha256=before, learner_sha256=learner_before if learner else None, step=a.step, execute=a.execute,
         frozen_parent_used=needs_parent,
         finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         scope='Independent LeRobot data,RGB-only factual-action method,500native any-success.Known cases,not officialJEPA-WMs or canonicalMT50.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['data', 'manifest', 'parent', 'output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--mode', choices=['bridge', 'metric', 'ap-global', 'ap-local', 'cem'], required=True)
    p.add_argument('--run'); p.add_argument('--feature-cache')
    p.add_argument('--step', type=int, default=0); p.add_argument('--horizon', type=int, default=6)
    p.add_argument('--execute', type=int, default=3); p.add_argument('--budget', type=int, default=500)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 1042]); p.add_argument('--tasks', nargs='+', required=True)
    run(p.parse_args())
