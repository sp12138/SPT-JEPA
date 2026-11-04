"""Action-proposal controls using frozen world models and factual train actions."""
import datetime
import json
from pathlib import Path

import numpy as np
import torch


def held_projection(candidates, mean, std):
    """Preserve the first primitive's exploration scale;hold it for five steps."""
    raw = candidates * std + mean
    # The actual layout is five consecutive XYZG blocks.
    held = raw[..., :4].clamp(-1, 1).unsqueeze(-2).expand(*raw.shape[:-1], 5, 4).reshape_as(raw)
    candidates.copy_((held - mean) / std)
    return candidates


def build_library(model, data, output):
    """All valid phase0 train starts;each action path stays in one trajectory."""
    import h5py
    import hdf5plugin
    import errno
    import time
    from experiments.metaworld_eval_20261103.evaluate import model_sha, file_sha
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before = model_sha(model)
    card = json.loads((data/'dataset_card.json').read_text())
    mean, std = np.asarray(card['action_mean'], np.float32), np.asarray(card['action_std'], np.float32)
    ids, offsets, lengths = [np.load(data/(key+'.npy')) for key in ['episode_ids', 'offsets', 'lengths']]
    train, test = np.load(data/'train_episodes.npy'), np.load(data/'test_episodes.npy')
    selected = np.flatnonzero(np.isin(ids, train) & (ids % 5 == 0) & (lengths >= 7))
    assert not set(ids[selected]) & set(test)
    actions, embeddings, records = [], [], []
    file = h5py.File(card['h5_path'], 'r')
    with torch.inference_mode():
        for index, ep in enumerate(selected):
            offset, n = int(offsets[ep]), int(lengths[ep])
            # Last action row is padding;no path uses it.
            for attempt in range(3):
                try:
                    raw = np.asarray(file['action'][offset:offset+n-1], np.float32)
                    frames = np.asarray(file['pixels'][offset:offset+n-6])
                    break
                except OSError as error:
                    if error.errno != errno.EIO or attempt == 2:
                        raise
                    file.close()
                    time.sleep(attempt+1)
                    file = h5py.File(card['h5_path'], 'r')
            assert raw.shape == (n-1, 20) and np.isfinite(raw).all() and np.max(np.abs(raw)) <= 1
            paths = np.stack([raw[start:start+6] for start in range(n-6)])
            assert paths.shape == (n-6, 6, 20)
            actions.append((paths-mean)/std)
            encoded = []
            for start in range(0, len(frames), 64):
                rgb = torch.from_numpy(frames[start:start+64]).permute(0,3,1,2)[:,None].cuda()
                encoded.append(model.encode(dict(pixels=model.normalize(rgb)))['emb'][:,0].cpu())
            embeddings.append(torch.cat(encoded))
            records.extend((int(ids[ep]), start, offset+start) for start in range(n-6))
            if index % 100 == 0:
                print(json.dumps(dict(library_episodes=index, total_episodes=len(selected))), flush=True)
    file.close()
    features, paths = torch.cat(embeddings), torch.from_numpy(np.concatenate(actions))
    assert len(features) == len(paths) == len(records) and len(paths) >= 4499
    assert model_sha(model) == before
    np.save(output/'records.npy', np.asarray(records, np.int64))
    torch.save(dict(features=features, actions=paths), output/'library.pt')
    proof = dict(passed=True, windows=len(paths), episodes=len(selected), phase=0, horizon=6,
                 train_only=True, no_terminal_padding=True, learner_fields=['pixels','action'],
                 privileged_fields=[], model_updates=0, model_state_before=before, model_state_after=before,
                 split_sha256=file_sha(data/'split-manifest.json'), records_sha256=file_sha(output/'records.npy'),
                 library_sha256=file_sha(output/'library.pt'),
                 created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 scope='Factual action paths,not presumed optimal. No task-ID/reward/success/state filtering. '
                       'Image embeddings rank start proximity;no next-optimal-action supervised learning.')
    (output/'manifest.json').write_text(json.dumps(proof, indent=2)+'\n')
    return features.cuda(), paths.cuda()


class LibrarySolver:
    """4499 nearest observed starts +one native warm-tail candidate;4500 forecasts."""
    def __init__(self, model, cost, features, actions):
        self.model, self.cost, self.features, self.actions = model, cost, features, actions

    @torch.inference_mode()
    def solve(self, info, init_action=None):
        current = self.model.encode(dict(pixels=info['pixels']))['emb'][:,-1]
        distance = (self.features-current).square().mean(-1)
        selected = distance.topk(4499, largest=False).indices
        warm = torch.zeros((1,6,20), device=self.actions.device)
        if init_action is not None:
            tail = init_action.to(warm.device)
            warm[:,:tail.shape[1]] = tail
        bank = torch.cat([warm, self.actions[selected]], 0)
        costs = []
        for start in range(0, 4500, 300):
            expanded = {k:v[:,None].expand(-1,300,*v.shape[1:]) for k,v in info.items()}
            costs.append(self.cost.get_cost(expanded, bank[start:start+300][None]).flatten())
        scores = torch.cat(costs)
        assert scores.shape == (4500,) and torch.isfinite(scores).all()
        best = scores.argmin()
        return dict(actions=bank[best][None].cpu(), costs=scores.min().cpu(),
                    selected_record=(-1 if int(best)==0 else int(selected[best-1])))


@torch.inference_mode()
def audit_plan(model, info, plan, case, cached, mean, std):
    """Actually realize the proposed path and physicalzero from the SAME reset.

    State/success are diagnostic outputs,never planner inputs or candidate labels.
    This is a two-path local check,not a complete counterfactual ranking audit.
    """
    from experiments.metaworld_eval_20261103.environment import (
        make_environment, pixels, close_environment, rgb_replay_check)
    from experiments.metaworld_eval_20261103.evaluate import resize_tensor
    paths = [np.asarray(plan, np.float32), np.broadcast_to(-mean/std, (6,20)).copy()]
    goal = model.encode(dict(pixels=info['goal']))['emb'][:,-1]
    rows = []
    for name, path in zip(['proposed', 'physical_zero'], paths):
        candidate = torch.from_numpy(path)[None,None].cuda()
        expanded = {k:v[:,None] for k,v in info.items()}
        prediction = model.rollout(expanded, candidate)['predicted_emb'][0,0,-6:]
        env, _, _ = make_environment(case['task'], case['seed'], cached['rand_vec'])
        frames, outcomes = [], []
        try:
            assert rgb_replay_check(cached['initial'], pixels(env))['passed']
            for block in path:
                for action in np.clip(block*std+mean,-1,1).reshape(5,4):
                    _,_,terminated,truncated,result = env.step(action.astype(np.float32))
                    outcomes.append(bool(result['success']))
                    assert not terminated and not truncated
                frames.append(pixels(env))
        finally:
            close_environment(env)
        observed = model.encode(dict(pixels=model.normalize(resize_tensor(frames)[None].cuda())))['emb'][0]
        rows.append(dict(candidate=name, predicted_goal_mse=float((prediction[-1]-goal[0]).square().mean()),
                         realized_goal_mse=float((observed[-1]-goal[0]).square().mean()),
                         forecast_mse_by_horizon=(prediction-observed).square().mean(-1).tolist(),
                         any_success=any(outcomes), actions=path.tolist()))
    return dict(rows=rows, model_updates=0, physical_paths=2, native_steps=60,
                predicted_prefers_proposed=rows[0]['predicted_goal_mse']<rows[1]['predicted_goal_mse'],
                realized_prefers_proposed=rows[0]['realized_goal_mse']<rows[1]['realized_goal_mse'],
                scope='Two actually realized alternatives;latent-error units are model-specific. '
                      'Success is diagnostic only. No oracle action/state/label enters online planning.')
