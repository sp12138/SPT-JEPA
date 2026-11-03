"""Download one independently released, ungated dataset at an immutable revision."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

REPOSITORY = 'lerobot/metaworld_mt50'
REVISION = 'b46a1c185a2368660650a82ac5a5083a56722f48'


def main(cache, output):
    from huggingface_hub import HfApi, snapshot_download
    cache, output = Path(cache), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    info = HfApi(token=False).dataset_info(REPOSITORY, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION and info.gated is False and not info.private
    files = [dict(path=f.rfilename, bytes=f.size) for f in info.siblings
             if f.rfilename.startswith(('data/', 'meta/')) or f.rfilename == 'README.md']
    request = dict(repository=REPOSITORY, revision=REVISION, token_used=False, cache=str(cache), files=files,
                   expected_bytes=sum(f['bytes'] for f in files),
                   scope='Independent Apache-2.0 LeRobot expert dataset,not a mirror of gated JEPA-WMs data. '
                         'No training, simulator, GPU, private endpoint or authorization bypass.')
    (output / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    snapshot_download(REPOSITORY, repo_type='dataset', revision=REVISION, token=False,
                      local_dir=str(cache), allow_patterns=['data/**', 'meta/**', 'README.md'], max_workers=8)
    assert all((cache / f['path']).is_file() and (cache / f['path']).stat().st_size == f['bytes'] for f in files)
    record = dict(**request, passed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope_validation='All listed paths/byte sizes present. Content/timing/split audit is a separate gate.')
    (output / 'complete.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(passed=True, files=len(files), bytes=request['expected_bytes'], cache=str(cache))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.cache, args.output)
