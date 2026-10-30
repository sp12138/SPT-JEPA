"""CPU oracle for a proposed observation-query transfer objective; no training."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np


# Rows are prior monitor states; columns are new monitor states.
# 0: A not seen; 1: A seen, waiting for B; 2: completed A then B.
OPERATORS = np.array([
    [[0., 1., 0.], [0., 1., 0.], [0., 0., 1.]],  # A
    [[1., 0., 0.], [0., 0., 1.], [0., 0., 1.]],  # B
    np.eye(3),  # unrelated observation O
])


def transfer(word):
    matrix = np.eye(3)
    for symbol in word:
        matrix = matrix @ OPERATORS[symbol]
    return matrix


def oracle(word, initial):
    progress = initial
    for symbol in word:
        if progress == 0 and symbol == 0:
            progress = 1
        elif progress == 1 and symbol == 1:
            progress = 2
    return progress


def constructions():
    words = [word for n in range(8) for word in itertools.product(range(3), repeat=n)]
    max_composition_error = max_oracle_error = max_stutter_error = 0.
    for word in words:
        matrix = transfer(word)
        expected = np.eye(3)[[oracle(word, initial) for initial in range(3)]]
        max_oracle_error = max(max_oracle_error, float(abs(matrix-expected).max()))
        for cut in range(len(word)+1):
            max_composition_error = max(max_composition_error, float(abs(matrix-transfer(word[:cut])@transfer(word[cut:])).max()))
        doubled = tuple(symbol for symbol in word for _ in range(2))
        max_stutter_error = max(max_stutter_error, float(abs(matrix-transfer(doubled)).max()))
    # Same endpoint and occupation counts, but different order.
    left, right = (2, 0, 1, 2), (2, 1, 0, 2)
    assert left[-1] == right[-1] and sorted(left) == sorted(right)
    # B and O are both 'no success yet' from progress 0; only B completes A-prefix.
    scalar_failure = {
        'B_scalar': transfer((1,))[0, 2], 'O_scalar': transfer((2,))[0, 2],
        'AB_scalar': transfer((0, 1))[0, 2], 'AO_scalar': transfer((0, 2))[0, 2],
        'B_transfer': transfer((1,)).tolist(), 'O_transfer': transfer((2,)).tolist(),
    }
    # Correlated symbols: with probability half AA, half BB. There is never AB.
    correct_joint = .5*transfer((0, 0)) + .5*transfer((1, 1))
    marginal = .5*OPERATORS[0] + .5*OPERATORS[1]
    independent_marginals = marginal @ marginal
    # Brier loss has a unique conditional-mean minimizer in an unconstrained matrix class.
    target_mean = .3*transfer(left) + .7*transfer(right)
    risks = []
    for prediction in [target_mean, transfer(left), transfer(right)]:
        risks.append(float(.3*np.mean((prediction-transfer(left))**2)+.7*np.mean((prediction-transfer(right))**2)))
    assert max_composition_error == max_oracle_error == max_stutter_error == 0
    assert transfer(left)[0, 2] == 1 and transfer(right)[0, 2] == 0
    assert correct_joint[0, 2] == 0 and independent_marginals[0, 2] == .25
    assert risks[0] < min(risks[1:])
    return dict(exhaustive_words=len(words), max_length=7,
                oracle_error=max_oracle_error, composition_error=max_composition_error,
                hard_symbol_stutter_error=max_stutter_error,
                same_endpoint_histogram=dict(left_word='OABO', right_word='OBAO',
                                            left_hit=float(transfer(left)[0, 2]), right_hit=float(transfer(right)[0, 2])),
                scalar_nonclosure=scalar_failure,
                correlated_future=dict(true_joint_hit=float(correct_joint[0, 2]), false_independent_hit=float(independent_marginals[0, 2])),
                brier_risk=dict(conditional_mean=risks[0], always_left=risks[1], always_right=risks[2]))


def gradients():
    import torch
    logits = torch.tensor([[.1, -.1, 0.], [-.3, .2, .1], [.4, -.2, .1], [.1, .2, -.2]], dtype=torch.float64, requires_grad=True)
    ops = torch.from_numpy(OPERATORS)
    matrix = torch.eye(3, dtype=torch.float64)
    for probability in logits.softmax(-1):
        matrix = matrix @ torch.einsum('s,sij->ij', probability, ops)
    target = torch.from_numpy(transfer((2, 0, 1, 2)))
    loss = ((matrix[:2]-target[:2])**2).mean()  # absorbing row is constant, carries no learning signal
    loss.backward()
    assert torch.isfinite(logits.grad).all() and (logits.grad.norm(dim=-1)>0).all()
    return dict(loss=float(loss.detach()), per_time_gradient_norm=logits.grad.norm(dim=-1).tolist(),
                scope='Gradient through soft monitor, not a trained model or calibrated correlated-path probability.')


def real_data(base, episodes=128):
    import hdf5plugin  # Existing read-only dataset compression.
    import h5py
    out = {}
    for task in ['tworoom', 'cube']:
        metadata = base/(task+'-metadata')
        card = json.loads((metadata/'dataset_card.json').read_text())
        train = np.load(metadata/'train_episodes.npy')
        rng = np.random.default_rng(20261030)
        chosen = sorted(rng.choice(train, min(episodes, len(train)), replace=False).tolist())
        query_rng = np.random.default_rng(20261031)
        records = []
        with h5py.File(card['h5_path'], 'r', swmr=True) as data:
            offsets, lengths = data['ep_offset'][:], data['ep_len'][:]
            for episode in chosen:
                if lengths[episode] < 36:
                    continue
                start = int(rng.integers(0, int(lengths[episode])-35))
                indices = int(offsets[episode])+start+5*np.arange(8)
                pixels = np.asarray(data['pixels'][indices], dtype=np.float32)/255
                # Exact 224->8 area pool: no model, simulator labels, or task thresholds.
                features = pixels.reshape(8, 8, 28, 8, 28, 3).mean(axis=(2, 4)).reshape(8, -1)
                for query in range(8):
                    prototype_indices = query_rng.choice(8, 3, replace=False)
                    centers = features[prototype_indices]
                    squared_distance = ((features[:, None]-centers[None])**2).sum(-1)
                    word = squared_distance.argmin(-1)
                    matrix = transfer(word)
                    shuffled = word.copy()
                    shuffled[1:-1] = query_rng.permutation(shuffled[1:-1])
                    # Keep endpoint and initial observation fixed; intermediate visitation order can change.
                    reverse = np.r_[word[0], word[-2:0:-1], word[-1]]
                    records.append(dict(episode=int(episode), start=start, query=query,
                                        prototype_indices=prototype_indices.tolist(), word=word.tolist(),
                                        unique_prototypes=int(np.unique(centers, axis=0).shape[0]),
                                        hit=float(matrix[0, 2]),
                                        shuffle_changes_transfer=bool(np.any(matrix!=transfer(shuffled))),
                                        shuffle_changes_hit=bool(matrix[0, 2]!=transfer(shuffled)[0, 2]),
                                        reverse_changes_transfer=bool(np.any(matrix!=transfer(reverse)))))
        def average(key):
            return float(np.mean([record[key] for record in records]))
        out[task] = dict(windows=len(records)//8, queries=len(records),
                         degenerate_prototype_fraction=float(np.mean([r['unique_prototypes']<3 for r in records])),
                         ordered_hit_fraction=average('hit'),
                         endpoint_preserving_shuffle_transfer_change_fraction=average('shuffle_changes_transfer'),
                         endpoint_preserving_shuffle_hit_change_fraction=average('shuffle_changes_hit'),
                         endpoint_preserving_reverse_transfer_change_fraction=average('reverse_changes_transfer'),
                         records=records,
                         scope='Observation-only target support. Shuffled words are diagnostic queries, never factual training outcomes or new actions.')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--real-data')
    args = parser.parse_args()
    result = dict(constructions=constructions(), gradients=gradients())
    if args.real_data:
        result['real_data'] = real_data(Path(args.real_data))
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    small = {**result}
    if 'real_data' in small:
        small['real_data'] = {task:{key:value for key,value in row.items() if key!='records'} for task,row in small['real_data'].items()}
    print(json.dumps(small, indent=2))


if __name__ == '__main__':
    main()
