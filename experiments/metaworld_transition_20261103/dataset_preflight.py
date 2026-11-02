"""Inspect one authorized official shard before any model training; no credential arguments."""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path

REPO = 'facebook/jepa-wms'
REVISION = '6116f042ae7ae4c8e3f1fd2f194f432615664182'
FILE = 'metaworld/data/train-00000-of-00126.parquet'


def main(cache, output):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
    import numpy as np
    import pyarrow.parquet as pq
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    try:
        path = Path(hf_hub_download(REPO, FILE, repo_type='dataset', revision=REVISION, local_dir=cache))
    except GatedRepoError:
        record = dict(passed=False, blocked_on='Existing Hugging Face account must be authorized and logged in.',
            repo=REPO, revision=REVISION, file=FILE, training_updates=0,
            access_url='https://huggingface.co/datasets/facebook/jepa-wms')
        (output/'ACCESS_REQUIRED.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record)); return 2
    table = pq.read_table(path)
    required = {'task', 'states', 'actions', 'video'}
    assert required.issubset(table.column_names), table.schema
    rows = table.to_pylist(); assert rows
    lengths, tasks = [], {}
    for row in rows:
        states = np.asarray(row['states'], dtype=np.float32)
        actions = np.asarray(row['actions'], dtype=np.float32)
        assert states.ndim == 2 and states.shape[1] == 39
        assert actions.ndim == 2 and actions.shape[1] == 4
        assert len(states) == len(actions)+1
        assert np.isfinite(states).all() and np.isfinite(actions).all()
        tasks[row['task']] = tasks.get(row['task'], 0)+1
        lengths.append(dict(states=len(states), actions=len(actions)))
    row = rows[0]
    import imageio.v2 as imageio
    from PIL import Image
    video = row['video']; assert isinstance(video, dict) and isinstance(video['bytes'], bytes)
    reader = imageio.get_reader(io.BytesIO(video['bytes']), format='mp4')
    try:frames = np.stack([frame for frame in reader])
    finally:reader.close()
    assert frames.shape == (lengths[0]['states'], 224, 224, 3) and frames.dtype == np.uint8
    sampled = []
    for step in sorted({0, min(25,len(frames)-1), min(50,len(frames)-1), len(frames)-1}):
        Image.fromarray(frames[step]).save(output/f'sample-{step:03d}.png')
        sampled.append(dict(step=step, rgb_sha256=hashlib.sha256(frames[step].tobytes()).hexdigest()))
    value = dict(passed=True, checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        repo=REPO, revision=REVISION, file=FILE, file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        file_bytes=path.stat().st_size, shard_rows=len(rows), tasks=tasks, lengths=lengths,
        sample_video_shape=list(frames.shape), sample_frames=sampled,
        source_contract='Public loader pairs video/states then discards final state/frame, retaining99; macro-step f5 concatenates5 executed4Dactions to20D.',
        scope='One real shard schema/video integrity check only; not entire126shard coverage, exact action-timing proof, simulator replay or method-performance evidence.',
        learner_fields=['observations', 'executed_actions'], excluded_from_learner=['states','rewards','success','optimal_actions'],
        training_updates=0, model_forwards=0)
    (output/'proof.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ['lengths','sample_frames']})); return 0


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache', required=True);p.add_argument('--output', required=True)
    a=p.parse_args();raise SystemExit(main(a.cache,a.output))
