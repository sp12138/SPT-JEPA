"""Frozen train/test-separated RGB/action and feature caches, no privileged data."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.metaworld_contact_20261106.diagnose import save, digest


def run(a):
    import h5py
    import hdf5plugin
    import stable_worldmodel as swm
    from experiments.metaworld_eval_20261103.evaluate import model_sha
    from vforge.ops import OpsConfig, initialize_ops
    initialize_ops(OpsConfig(backend='sdpa', allow_fallback=False), device=torch.device('cuda'), parallel_state=None)
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    output = Path(a.output); output.mkdir(parents=True, exist_ok=False)
    data = Path(a.data); card = json.loads((data/'dataset_card.json').read_text())
    ids, offsets, lengths = [np.load(data/(k+'.npy')) for k in ['episode_ids', 'offsets', 'lengths']]
    train, test = [np.load(data/(s+'_episodes.npy')) for s in ['train', 'test']]
    eligible = np.flatnonzero((ids % 5 == 0) & (lengths >= 2))
    eligible = eligible[a.shard::a.shards]
    assert not set(train) & set(test)
    model = swm.wm.utils.load_pretrained(a.checkpoint).cuda().eval().requires_grad_(False)
    before = model_sha(model)
    total = int(lengths[eligible].sum())
    images = np.lib.format.open_memmap(output/'pixels.npy', mode='w+', dtype=np.uint8, shape=(total, 224, 224, 3)) if a.rgb else None
    dense = np.lib.format.open_memmap(output/'patches.npy', mode='w+', dtype=np.float16, shape=(total, 256, 192)) if a.dense else None
    embeddings, patches, actions, records = [], [], [], []
    cursor = 0; start_time = time.monotonic()
    with h5py.File(card['h5_path'], 'r', rdcc_nbytes=64*1024*1024) as file, torch.inference_mode():
        for index, ep in enumerate(eligible):
            offset, n = int(offsets[ep]), int(lengths[ep])
            raw = np.asarray(file['pixels'][offset:offset+n])
            control = np.asarray(file['action'][offset:offset+n], np.float32)
            assert raw.shape == (n, 224, 224, 3) and control.shape == (n, 20)
            if images is not None:
                images[cursor:cursor+n] = raw
            for j in range(0, n, 48):
                x = torch.from_numpy(raw[j:j+48]).cuda().permute(0, 3, 1, 2)[:, None]
                inp = model.normalize(x)
                z = model.encode(dict(pixels=inp))['emb'][:, 0]
                embeddings.append(z.float().cpu())
                # Preserve local visual information before global64D pooling.
                tokens = model.encoder(inp[:, 0], interpolate_pos_encoding=True).last_hidden_state[:, 1:]
                side = int(tokens.shape[1]**.5)
                assert side*side == tokens.shape[1]
                if dense is not None:
                    assert tokens.shape[1:] == (256, 192)
                    dense[cursor+j:cursor+j+len(tokens)] = tokens.half().cpu().numpy()
                local = torch.nn.functional.adaptive_avg_pool2d(tokens.transpose(1, 2).reshape(len(tokens), -1, side, side), (4, 4))
                patches.append(local.flatten(2).transpose(1, 2).half().cpu())
            actions.append(control)
            split = 'train' if ids[ep] in train else 'test'
            assert ids[ep] in (train if split == 'train' else test)
            records.append(dict(episode=int(ids[ep]), offset=cursor, length=n, split=split,
                                source_offset=offset, source_phase=0))
            cursor += n
            if index % 25 == 0:
                print(json.dumps(dict(episodes=index, frames=cursor, total_frames=total,
                                      elapsed=time.monotonic()-start_time)), flush=True)
    assert cursor == total
    if images is not None:
        images.flush(); del images
    if dense is not None:
        dense.flush(); del dense
    torch.save(dict(embedding=torch.cat(embeddings), patches=torch.cat(patches)), output/'features.pt')
    np.save(output/'actions.npy', np.concatenate(actions))
    after = model_sha(model); assert after == before
    save(output/'manifest.json', dict(passed=True, checkpoint=a.checkpoint, model_sha256=before,
         model_unchanged=True, model_updates=0, records=records, frames=total,
         shard=a.shard, shards=a.shards, rgb=a.rgb, dense=a.dense,
         dense_sha256=digest(output/'patches.npy') if a.dense else None, input_fields=['pixels', 'action'],
         features_sha256=digest(output/'features.pt'), actions_sha256=digest(output/'actions.npy'),
         split_sha256=digest(data/'split-manifest.json'),
         scope='All50tasks,phase0 only,trajectory-disjoint train/test.Factual controls,not optimal labels.No reward/state/success/taskID in features.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ['data', 'checkpoint', 'output']:
        p.add_argument('--'+k, required=True)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--shards', type=int, default=4)
    p.add_argument('--rgb', action='store_true')
    p.add_argument('--dense', action='store_true')
    run(p.parse_args())
