"""Export committed Python sources without weights,data or result arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main(output):
    output=Path(output);assert not output.exists()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],text=True).splitlines()
    selected=[p for p in names if (p.endswith('.py') and p.startswith(('as_jepa_lewm/','as_jepa_tools/','experiments/','tests/'))) or p=='experiments/branch_trials_20261102/upstream/provenance.json']
    subprocess.run(['git','diff','--exit-code',commit,'--',*selected],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','archive','--format=tar.gz','--output',str(output),commit,*selected],check=True)
    hashes={}
    with tarfile.open(output) as archive:
        for member in archive:
            if member.isfile():hashes[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    meta=dict(commit=commit,files=hashes,archive_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    with output.with_suffix('.json').open('x') as f:json.dump(meta,f,indent=2);f.write('\n')
    print(json.dumps(dict(commit=commit,files=len(hashes),archive_sha256=meta['archive_sha256'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);main(parser.parse_args().output)
