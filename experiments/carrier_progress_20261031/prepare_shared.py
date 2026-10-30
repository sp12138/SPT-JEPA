"""Freeze committed helpers and unchanged native model code for this campaign."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

B = Path('/efs/prescottsun/SPT-JEPA')
ROOT = B/'runs/carrier_progress_20261031'
CODE = B/'code'


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD'], cwd=CODE, check=True,
                   stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip()
    paths = ['AGENTS.md', 'as_jepa_lewm', 'as_jepa_tools', 'experiments/carrier_progress_20261031',
             'experiments/ideas_validate_20261031/audit.py',
             'experiments/ideas_validate_20261031/preflight.py',
             'experiments/ideas_validate_20261031/audit-windows-cube.json',
             'experiments/ideas_validate_20261031/audit-windows-tworoom.json']
    archive = subprocess.check_output(['git', 'archive', commit, *paths], cwd=CODE)
    ROOT.mkdir()
    source = ROOT/'source-v1'
    source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter='data')
    write(ROOT/'source-v1.json', dict(source=str(source), source_commit=commit,
        files={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in source.rglob('*') if p.is_file()}))
    old_root = B/'runs/ideas_validate600_20261031'
    old = json.loads((old_root/'source-record.json').read_text())
    record = dict(old, source=str(source), source_commit=commit,
        manifest=str(ROOT/'source-v1.json'), pythonpath=old['pythonpath'].replace(old['source'], str(source)))
    write(ROOT/'source-record.json', record)
    environment = json.loads((old_root/'plan.json').read_text())['environment']
    temp = B/'tmp/carrier_progress_20261031'
    temp.mkdir(exist_ok=True)
    environment['TMPDIR'] = str(temp)
    write(ROOT/'environment.json', environment)
    write(ROOT/'scope.json', dict(created_at=time.time(), source_commit=commit,
        authorization='2026-10-31: persistent sample-level research; approximately30minute review; fix and improve, then continue.',
        training='12 full-state carrier continuations600->1200->2000, then review. No S3 continuation.',
        diagnostics='7 freshTwoRoom comparisons,12 primaryCube metric comparisons,12 predeclared freshCube metric comparisons and3 freshCube baselines; existing600weights only.',
        monitoring=dict(interval_seconds=1800, read_only=True, no_api_calls=True,
            distinction='Server snapshots are not an autonomous algorithm-editing agent; research review remains in the active goal session.'),
        limits='Preserve old holds/source/data; no optimal-action or physical training labels. No automatic continuation beyond each reviewed cap.',
        success='Require substantial task gains on both tasks over strong disclosed baselines, independent training seeds and new cases. Short/noisy improvements and novelty claims do not satisfy the goal.'))
    print(json.dumps(dict(root=str(ROOT), source_commit=commit, source_record=str(ROOT/'source-record.json'))))


if __name__ == '__main__':
    main()
