"""Audit public RGB/action trajectories and pack all five temporal phases on CPU.

Phase packing changes storage order only: it retains every valid raw-frame start,
while allowing the existing native loader to read eight consecutive macro frames.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import datetime
from functools import lru_cache
import hashlib
import io
import json
import multiprocessing
from pathlib import Path

import numpy as np

REPO = 'lerobot/metaworld_mt50'
REVISION = 'b46a1c185a2368660650a82ac5a5083a56722f48'
STRIDE = 5
WINDOW = 8


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2); handle.write('\n')


def initialize_worker(cache, output, ranges, validation):
    global CACHE, OUTPUT, RANGES, VALIDATION
    CACHE, OUTPUT, RANGES = Path(cache), Path(output), ranges
    VALIDATION = set(validation)
    import torch
    torch.set_num_threads(2)


@lru_cache(maxsize=2)
def load_rows(index):
    import pyarrow.parquet as pq
    return pq.read_table(CACHE / RANGES[index]['path'], columns=['index', 'observation.image', 'action']).to_pylist()


def episode_rows(start, end):
    ends = np.array([r['end'] for r in RANGES])
    file_index = int(np.searchsorted(ends, start, side='right'))
    result = []
    while start < end:
        descriptor = RANGES[file_index]
        stop = min(end, descriptor['end'])
        rows = load_rows(file_index)
        result.extend(rows[start - descriptor['start']:stop - descriptor['start']])
        start = stop; file_index += 1
    return result


def prepare_bundle(argument):
    bundle_index, episodes = argument
    import h5py
    from PIL import Image
    import torch
    from torch.nn import functional as F
    size = sum(e['length'] for e in episodes)
    path = OUTPUT / 'shards' / f'bundle{bundle_index:04d}.h5'
    phases, cursor = [], 0
    total, squares, count = np.zeros(20), np.zeros(20), 0
    clipped, raw_count = 0, 0
    with h5py.File(path, 'x') as file:
        pixels = file.create_dataset('pixels', (size, 224, 224, 3), dtype='u1', chunks=(1, 224, 224, 3), compression='lzf')
        controls = file.create_dataset('action', (size, 20), dtype='f4')
        raw_rows = file.create_dataset('raw_row', (size,), dtype='i8')
        for episode in episodes:
            ep = episode['episode_index']; start, end = episode['dataset_from_index'], episode['dataset_to_index']
            rows = episode_rows(start, end)
            assert [r['index'] for r in rows] == list(range(start, end))
            actions = np.asarray([r['action'] for r in rows], dtype=np.float32)
            assert actions.shape == (len(rows), 4) and np.isfinite(actions).all()
            clipped += int((np.abs(actions) > 1).sum()); raw_count += actions.size
            # Meta-World clips XYZ inside set_xyz_action; clipping the command to
            # its declared Box(-1,1) records the common effective control domain.
            actions = np.clip(actions, -1, 1)
            images = []
            for row in rows:
                data = row['observation.image']
                assert isinstance(data['bytes'], bytes)
                image = np.asarray(Image.open(io.BytesIO(data['bytes'])).convert('RGB'))
                assert image.shape == (480, 480, 3)
                images.append(image)
            resized = []
            for index in range(0, len(images), 32):
                tensor = torch.from_numpy(np.stack(images[index:index + 32])).permute(0, 3, 1, 2).float()
                small = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False, antialias=True).round().clamp(0, 255).byte()
                resized.append(small.permute(0, 2, 3, 1).numpy())
            images = np.concatenate(resized)
            for phase in range(STRIDE):
                indices = np.arange(phase, len(rows), STRIDE)
                length = len(indices)
                packed = np.zeros((length, 20), dtype=np.float32)
                for i, raw_index in enumerate(indices[:-1]):
                    block = actions[raw_index:raw_index + STRIDE]
                    assert block.shape == (STRIDE, 4)
                    packed[i] = block.reshape(-1)
                pixels[cursor:cursor + length] = images[indices]
                controls[cursor:cursor + length] = packed
                raw_rows[cursor:cursor + length] = start + indices
                phases.append(dict(episode_id=ep * STRIDE + phase, original_episode=ep, task_id=episode['task_id'],
                                   phase=phase, length=length, offset=cursor))
                if ep not in VALIDATION and length > 1:
                    values = packed[:-1].astype(np.float64)
                    total += values.sum(0); squares += np.square(values).sum(0); count += len(values)
                cursor += length
        assert cursor == size
    return dict(bundle=bundle_index, path=str(path), rows=size, bytes=path.stat().st_size, phases=phases,
                train_sum=total.tolist(), train_squares=squares.tolist(), train_controls=count,
                clipped_components=clipped, raw_components=raw_count)


def main(cache, output, download_proof, task_config, workers):
    import h5py
    import pyarrow.parquet as pq
    cache, output = Path(cache), Path(output)
    proof = read(download_proof)
    assert proof['passed'] and proof['repository'] == REPO and proof['revision'] == REVISION
    info = read(cache / 'meta/info.json')
    assert info['total_episodes'] == 2500 and info['total_frames'] == 204806 and info['total_tasks'] == 50
    output.mkdir(parents=True, exist_ok=False); (output / 'shards').mkdir()
    columns = ['episode_index', 'dataset_from_index', 'dataset_to_index', 'length', 'stats/task_id/min', 'stats/task_id/max']
    episodes = pq.read_table(cache / 'meta/episodes/chunk-000/file-000.parquet', columns=columns).to_pylist()
    episodes.sort(key=lambda e: e['episode_index'])
    assert [e['episode_index'] for e in episodes] == list(range(2500))
    for e in episodes:
        assert e['stats/task_id/min'] == e['stats/task_id/max']
        e['task_id'] = int(e['stats/task_id/min'][0])
        assert e['dataset_to_index'] - e['dataset_from_index'] == e['length']
    assert sum(e['length'] for e in episodes) == 204806
    assert Counter(e['task_id'] for e in episodes) == {task: 50 for task in range(50)}
    config = read(task_config)
    task_names = {value: key for key, value in config['TASK_NAME_TO_ID'].items()}
    assert set(task_names) == set(range(50))
    expected_episode = np.concatenate([np.full(e['length'], e['episode_index']) for e in episodes])
    expected_frame = np.concatenate([np.arange(e['length']) for e in episodes])
    expected_task = np.concatenate([np.full(e['length'], e['task_id']) for e in episodes])
    ranges, cursor = [], 0
    for path in sorted((cache / 'data').rglob('*.parquet')):
        table = pq.read_table(path, columns=['index', 'episode_index', 'frame_index', 'task_id', 'task_index'])
        values = {name: table[name].to_numpy() for name in table.column_names}
        n = len(table)
        assert np.array_equal(values['index'], np.arange(cursor, cursor + n))
        assert np.array_equal(values['episode_index'], expected_episode[cursor:cursor + n])
        assert np.array_equal(values['frame_index'], expected_frame[cursor:cursor + n])
        assert np.array_equal(values['task_id'], expected_task[cursor:cursor + n])
        ranges.append(dict(path=str(path.relative_to(cache)), start=cursor, end=cursor + n, bytes=path.stat().st_size))
        cursor += n
    assert cursor == 204806
    validation = []
    for task in range(50):
        ids = [e['episode_index'] for e in episodes if e['task_id'] == task]
        validation.extend(np.random.default_rng(20261103 + task).permutation(ids)[:5].tolist())
    validation = sorted(validation); training = sorted(set(range(2500)) - set(validation))
    assert len(validation) == 250 and len(training) == 2250 and not set(validation) & set(training)
    write(output / 'split-manifest.json', dict(seed=20261103, train_original_episodes=training,
          validation_original_episodes=validation, task_names=task_names, train_per_task=45, validation_per_task=5,
          scope='All phases of an original trajectory remain in one split; IDs only group/split data,never enter learner.'))
    bundles = [(i // 10, episodes[i:i + 10]) for i in range(0, len(episodes), 10)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker, initargs=(str(cache), str(output), ranges, validation)) as pool:
        completed = []
        for item in pool.map(prepare_bundle, bundles):
            completed.append(item)
            if len(completed) % 25 == 0:print('Packed bundles', len(completed), '/', len(bundles), flush=True)
    phases = []; total = np.zeros(20); squares = np.zeros(20); count = 0; cursor = 0
    pixels = h5py.VirtualLayout((204806, 224, 224, 3), dtype='u1')
    actions = h5py.VirtualLayout((204806, 20), dtype='f4')
    raw_rows = h5py.VirtualLayout((204806,), dtype='i8')
    for bundle in completed:
        n = bundle['rows']; path = bundle['path']
        pixels[cursor:cursor + n] = h5py.VirtualSource(path, 'pixels', shape=(n, 224, 224, 3))
        actions[cursor:cursor + n] = h5py.VirtualSource(path, 'action', shape=(n, 20))
        raw_rows[cursor:cursor + n] = h5py.VirtualSource(path, 'raw_row', shape=(n,))
        for phase in bundle['phases']:phases.append(dict(phase, offset=phase['offset'] + cursor))
        total += np.array(bundle['train_sum']); squares += np.array(bundle['train_squares']); count += bundle['train_controls']
        cursor += n
    assert cursor == 204806 and len(phases) == 12500
    with h5py.File(output / 'dataset.h5', 'x', libver='latest') as file:
        file.create_virtual_dataset('pixels', pixels); file.create_virtual_dataset('action', actions)
        file.create_virtual_dataset('raw_row', raw_rows)
        assert set(file) == {'pixels', 'action', 'raw_row'}
    mean = total / count; variance = np.maximum(squares / count - mean ** 2, 0); std = np.sqrt(variance)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 1e-6).all()
    for name, values in [('offsets', [p['offset'] for p in phases]), ('lengths', [p['length'] for p in phases]),
                         ('episode_ids', [p['episode_id'] for p in phases]),
                         ('train_episodes', [p['episode_id'] for p in phases if p['original_episode'] in set(training)]),
                         ('test_episodes', [p['episode_id'] for p in phases if p['original_episode'] in set(validation)])]:
        np.save(output / f'{name}.npy', np.asarray(values, dtype=np.int64))
    card = dict(repository=REPO, revision=REVISION, h5_path=str(output / 'dataset.h5'), frameskip=1,
                physical_frameskip=5, primitive_action_dim=4, macro_action_dim=20,
                action_mean=mean.tolist(), action_std=std.tolist(),
                action_semantics='Clip native commands to environment Box[-1,1],then concatenate five consecutive4Dcontrols. '
                                 'Last phase-row control is zero padding,never a target/valid transition.',
                storage='All5phases; contiguous macro frames,all valid original starts retained; no repeated stride5.',
                raw_frames=204806, raw_episodes=2500, tasks=50, phase_episodes=12500,
                learner_fields=['pixels', 'action'], label_fields_in_hdf5=[],
                image_transform='Torch bilinear224 antialias,round uint8; identical resizing for simulation inputs.',
                comparison='Independent expert data; not gated JEPA-WMs mixture or canonicalMT50/ML45 score.')
    write(output / 'dataset_card.json', card)
    write(output / 'pack-index.json', dict(bundles=completed, phases=phases, raw_files=ranges))
    natural = {name: sum(max(0, p['length'] - WINDOW + 1) for p in phases
                        if (p['original_episode'] in set(validation)) == (name == 'test')) for name in ['train', 'test']}
    report = dict(passed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  repository=REPO, revision=REVISION, raw_rows_checked=204806, episodes_checked=2500,
                  actual_task_ids=50, registry_rows=len(pq.read_table(cache / 'meta/tasks.parquet')),
                  natural_windows=natural, frameskip=5, phase_storage_stride=1, window=8,
                  clipped_native_command_fraction=sum(b['clipped_components'] for b in completed) / sum(b['raw_components'] for b in completed),
                  train_scaler_controls=count, split_manifest_sha256=hashlib.sha256((output / 'split-manifest.json').read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Full scalar/index and PNG decode audit; no model/SGD/GPU/simulator. '
                        'Alignment follows LeRobot obs_t/action_t rows; exact physical replay remains a separate check.')
    write(output / 'complete.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['cache', 'output', 'download-proof', 'task-config']:parser.add_argument('--' + name, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    main(args.cache, args.output, args.download_proof, args.task_config, args.workers)
