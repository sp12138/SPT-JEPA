"""Read-only train-input statistics; no learner, target change or policy."""
import argparse
import hashlib
import json
from pathlib import Path


def main(data, cache, output):
    import h5py
    import numpy as np
    proof = json.loads((cache/'COMPLETE.json').read_text())
    assert proof['passed'] and proof['original'] == str(data)
    assert hashlib.sha256((data/'dataset.h5').read_bytes()).hexdigest() == proof['original_vds_sha256']
    card = json.loads((data/'dataset_card.json').read_text())
    offsets, lengths, episode_ids, selected = [np.load(data/(key+'.npy')) for key in
        ['offsets', 'lengths', 'episode_ids', 'train_episodes']]
    valid = np.zeros(sum(lengths), dtype=bool)
    for index in np.flatnonzero(np.isin(episode_ids, selected) & (lengths >= 8)):
        valid[offsets[index]:offsets[index]+lengths[index]-1] = True
    with h5py.File(cache/'dataset.h5', 'r') as f:
        actions = np.asarray(f['action'])[valid].astype(np.float64)
    assert actions.ndim == 2 and actions.shape[1] == 20 and np.isfinite(actions).all()
    assert np.abs(actions).max() <= 1
    mean, std = np.asarray(card['action_mean']), np.asarray(card['action_std'])
    z = (actions-mean)/std
    center = z-z.mean(0)
    eigenvalues = np.linalg.eigvalsh(center.T@center/len(center))[::-1].clip(0)
    shares = np.cumsum(eigenvalues)/eigenvalues.sum()
    words = actions.reshape(-1, 5, 4)
    repeated_mean = np.broadcast_to(words.mean(1, keepdims=True), words.shape).reshape(-1, 20)
    residual = (actions-repeated_mean)/std
    correlations = []
    for component in range(4):
        x, y = words[:, :-1, component].reshape(-1), words[:, 1:, component].reshape(-1)
        correlations.append(float(np.corrcoef(x, y)[0, 1]))
    report = dict(passed=True, training_only=True, valid_train_action_words=len(actions),
        source_data=str(data), cache_vds_sha256=proof['cache_vds_sha256'],
        standardized_target_mean_square=float(np.mean(z*z)),
        within_word_nonconstant_standardized_mean_square=float(np.mean(residual*residual)),
        covariance_eigenvalues=eigenvalues.tolist(),
        covariance_participation_rank=float(eigenvalues.sum()**2/np.square(eigenvalues).sum()),
        covariance_dimensions95=int(np.searchsorted(shares, .95)+1),
        covariance_dimensions99=int(np.searchsorted(shares, .99)+1),
        adjacent_native_command_correlations=correlations,
        within_word_gripper_switch_fraction=float(np.any(np.diff(words[:, :, 3], axis=1) != 0, axis=1).mean()),
        native_input_component_saturation=float(np.mean(np.abs(actions) == 1)),
        model_updates=0, no_new_training_fields=True,
        scope='Original TRAIN inputs only,excluding padding/short unusable episodes. Repeated word mean uses the observed '
              'inputs themselves and is a descriptive projection,NOT a state-derived predictor or proposed target. '
              'Covariance rank/correlation does NOT establish invertibility from RGB,physical control,or task success.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key not in ['covariance_eigenvalues', 'scope']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); main(a.data, a.cache, a.output)
