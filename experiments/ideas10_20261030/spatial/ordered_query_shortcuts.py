"""Episode-disjoint CPU shortcut audit. Fixed rules, no hyperparameter selection."""
import argparse
import json
from pathlib import Path

import numpy as np

from ordered_query_checks import transfer


REPRESENTATIVES = [(), (0,), (1,), (0, 1), (1, 0)]
LOOKUP = {tuple(transfer(word).ravel()): i for i, word in enumerate(REPRESENTATIVES)}
PARTS = ['whole', 'first_half', 'second_half']


def collect(base, task, split, episodes=128):
    import hdf5plugin
    import h5py
    meta = base/(task+'-metadata')
    card = json.loads((meta/'dataset_card.json').read_text())
    episode_ids = np.load(meta/(split+'_episodes.npy'))
    rng = np.random.default_rng(20261030 if split=='train' else 20261032)
    selected = sorted(rng.choice(episode_ids, min(episodes, len(episode_ids)), replace=False).tolist())
    query_rng = np.random.default_rng(20261031 if split=='train' else 20261033)
    rows = {'actions':[], 'queries':[], 'initial':[], 'labels':[], 'episodes':[]}
    with h5py.File(card['h5_path'], 'r', swmr=True) as data:
        offsets, lengths = data['ep_offset'][:], data['ep_len'][:]
        for episode in selected:
            if lengths[episode] < 36:
                continue
            start = int(rng.integers(0, int(lengths[episode])-35))
            indices = int(offsets[episode])+start+5*np.arange(8)
            pixels = np.asarray(data['pixels'][indices], dtype=np.float32)/255
            features = pixels.reshape(8, 8, 28, 8, 28, 3).mean(axis=(2, 4)).reshape(8, -1)
            actions = np.asarray(data['action'][int(indices[0]):int(indices[-1])], dtype=np.float32).ravel()
            for _ in range(8):
                prototype_indices = query_rng.choice(8, 3, replace=False)
                centers = features[prototype_indices]
                word = ((features[:, None]-centers[None])**2).sum(-1).argmin(-1)
                label = [LOOKUP[tuple(transfer(segment).ravel())] for segment in [word, word[:4], word[4:]]]
                rows['actions'].append(actions)
                rows['queries'].append(centers.ravel())
                rows['initial'].append(features[0])
                rows['labels'].append(label)
                rows['episodes'].append(episode)
    return {key:np.asarray(value) for key,value in rows.items()}


def distances(x, y):
    return np.maximum(0., (x*x).sum(-1)[:, None]+(y*y).sum(-1)[None]-2*x@y.T)


def fit_predict(x, y, targets, kind):
    mean, std = x.mean(0), x.std(0)
    keep = std > 1e-6
    x = (x[:, keep]-mean[keep])/std[keep]
    y = (y[:, keep]-mean[keep])/std[keep]
    centered = targets-targets.mean(0)
    if kind == 'linear':
        if x.shape[1] <= len(x):
            gram = x.T@x
            gram.flat[::len(gram)+1] += 1.
            coefficients = np.linalg.solve(gram, x.T@centered)
            train, validation = x@coefficients, y@coefficients
        else:
            gram = x@x.T
            gram.flat[::len(gram)+1] += 1.
            coefficients = np.linalg.solve(gram, centered)
            train, validation = (x@x.T)@coefficients, (y@x.T)@coefficients
        detail = dict(features=int(keep.sum()), ridge=1.)
    else:
        squared = distances(x, x)
        nonzero = squared[np.triu_indices(len(x), 1)]
        nonzero = nonzero[nonzero>1e-10]
        bandwidth = float(np.median(nonzero)) if len(nonzero) else 1.
        kernel = np.exp(-squared/bandwidth)
        coefficients = np.linalg.solve(kernel+np.eye(len(x)), centered)
        train, validation = kernel@coefficients, np.exp(-distances(y, x)/bandwidth)@coefficients
        detail = dict(features=int(keep.sum()), ridge=1., kernel='exp(-distance_squared/median_train_distance_squared)', bandwidth_squared=bandwidth)
    return train+targets.mean(0), validation+targets.mean(0), detail


def metric(scores, labels):
    prediction = scores.argmax(-1)
    recall = [float(np.mean(prediction[labels==c]==c)) if np.any(labels==c) else None for c in range(5)]
    return dict(accuracy=float(np.mean(prediction==labels)), macro_recall=float(np.mean([r for r in recall if r is not None])), class_recall=recall)


def run_task(train, validation):
    assert not set(train['episodes']).intersection(validation['episodes'])
    inputs = {'actions':['actions'], 'queries':['queries'], 'queries_actions':['queries','actions'],
              'queries_actions_initial':['queries','actions','initial']}
    result, predictions = {}, {}
    # One solve learns the three segment readouts simultaneously; no segment timestamp is an input.
    targets = np.eye(5)[train['labels']].reshape(len(train['labels']), 15)
    for kind in ['linear', 'rbf']:
        for name, keys in inputs.items():
            if kind=='rbf' and name not in ['queries_actions', 'queries_actions_initial']:
                continue
            x = np.concatenate([train[key] for key in keys], -1).astype(np.float64)
            y = np.concatenate([validation[key] for key in keys], -1).astype(np.float64)
            train_scores, scores, details = fit_predict(x, y, targets, kind)
            train_scores, scores = train_scores.reshape(-1,3,5), scores.reshape(-1,3,5)
            key = kind+'/'+name
            result[key] = dict(details=details, train={}, validation={})
            predictions[key] = scores.argmax(-1)
            for i, part in enumerate(PARTS):
                result[key]['train'][part] = metric(train_scores[:,i], train['labels'][:,i])
                result[key]['validation'][part] = metric(scores[:,i], validation['labels'][:,i])
            result[key]['validation']['combined_accuracy'] = float(np.mean(predictions[key]==validation['labels']))
    unique = np.unique(validation['episodes'])
    paired = {}
    for kind in ['linear', 'rbf']:
        blind = predictions[kind+'/queries_actions']==validation['labels']
        state = predictions[kind+'/queries_actions_initial']==validation['labels']
        episode_delta = np.array([(state[validation['episodes']==episode].astype(float)-blind[validation['episodes']==episode]).mean(0) for episode in unique])
        bootstrap_rng = np.random.default_rng(20261034)
        draws = bootstrap_rng.integers(0,len(unique),size=(5000,len(unique)))
        bootstrap = episode_delta[draws].mean(1)
        paired[kind] = {part:dict(delta_accuracy=float(episode_delta[:,i].mean()),
                                  episode_bootstrap_95ci=np.quantile(bootstrap[:,i],[.025,.975]).tolist()) for i,part in enumerate(PARTS)}
        paired[kind]['combined'] = dict(delta_accuracy=float(episode_delta.mean()),
                                       episode_bootstrap_95ci=np.quantile(bootstrap.mean(1),[.025,.975]).tolist())
    return dict(train_episodes=sorted(set(map(int,train['episodes']))), validation_episodes=unique.tolist(),
                train_queries=len(train['labels']), validation_queries=len(validation['labels']),
                train_class_counts={part:np.bincount(train['labels'][:,i],minlength=5).tolist() for i,part in enumerate(PARTS)},
                validation_class_counts={part:np.bincount(validation['labels'][:,i],minlength=5).tolist() for i,part in enumerate(PARTS)},
                readouts=result, paired_initial_state_effect=paired)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    target = Path(args.output)
    target.mkdir(parents=True,exist_ok=True)
    results = {}
    for task in ['tworoom','cube']:
        splits = {}
        for split in ['train','validation']:
            cache = target/(task+'-query-shortcuts-'+split+'.npz')
            if cache.exists():
                with np.load(cache) as loaded:
                    splits[split] = {key:loaded[key] for key in loaded.files}
            else:
                splits[split] = collect(Path(args.data),task,split)
                np.savez_compressed(cache,**splits[split])
        results[task] = run_task(splits['train'],splits['validation'])
        (target/'ordered-query-shortcuts.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
        print(json.dumps({task:results[task]['paired_initial_state_effect']},indent=2),flush=True)


if __name__=='__main__':
    main()
