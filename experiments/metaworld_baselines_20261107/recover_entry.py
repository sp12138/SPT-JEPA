"""Retain all pre-SGD v2 failures and create fresh v3 outputs; never erase errors."""
import argparse
import os
from pathlib import Path
import signal

from experiments.metaworld_baselines_20261107.campaign import ROOT, read, save, sha, now


def main(helper):
    plan=read(ROOT/'plan.json');errors=[]
    assert not (ROOT/'PANEL_APPROVED.json').exists()
    for job in plan['jobs']:
        out=Path(job['output'])
        assert not list(out.glob('checkpoint-*.pt'))
        assert not list(out.glob('health-rank*.json'))
        if (out/'error.json').exists():
            log=(out/'train-0000003.log').read_text()
            causes=[s for s in ['ModuleNotFoundError: No module named', "ValueError: resolver 'eval' is already registered"] if s in log]
            assert causes,(job['name'],'not an audited pre-SGD failure')
            errors.append(dict(name=job['name'],error=str(out/'error.json'),sha256=sha(out/'error.json'),causes=causes))
    assert len(errors)==16
    owner=read(ROOT/'collector-launch.json')
    proc=Path(f'/proc/{owner["pid"]}/cmdline')
    if proc.exists():
        command=proc.read_bytes().replace(b'\0',b' ').decode()
        assert 'helper-panel-v2' in command and '--mode collect' in command
        os.kill(owner['pid'],signal.SIGTERM)
    save(ROOT/'collector-retired-v2.json',dict(**owner,retired_utc=now(),reason='Fresh pre-SGD entry recovery; original logs/results retained'))
    save(ROOT/'plan-v2.json',plan)
    catalog=read(ROOT/'sources.json')
    catalog.update(helper=str(helper),helper_hashes={str(p.relative_to(helper)):sha(p) for p in helper.rglob('*.py')},
        operational_recovery='Explicit audited cache helper import; stable_pretraining owns resolver before Hydra compose. '
            'Public neural source/models/loss coefficients/data/batch/budget unchanged. All16 failed before SGD.')
    save(ROOT/'sources-v3.json',catalog)
    source_sha=sha(ROOT/'sources-v3.json')
    for job in plan['jobs']:
        old=Path(job['output']);new=ROOT/'training-v3'/job['name'];new.mkdir(parents=True,exist_ok=False)
        cfg=read(job['config']);cfg['output']=str(new);cfg['source_catalog_sha256']=source_sha
        save(new/'config.json',cfg)
        job.update(output=str(new),config=str(new/'config.json'),original_pre_sgd_attempt=str(old))
    plan.update(helper=str(helper),claims_dir='claims-v3',launch_tag='v3-',source_catalog=str(ROOT/'sources-v3.json'),
        source_catalog_sha256=source_sha,entry_recovery_utc=now())
    save(ROOT/'plan.json',plan)
    save(ROOT/'ENTRY_RECOVERY_v3.json',dict(passed=True,original_attempt_errors=errors,effective_model_negatives=0,
        completed_sgd_repeated=0,unsaved_sgd_repeated=0,original_updates=0,new_outputs='training-v3',
        new_helper=str(helper),deadline_unchanged=True,source_catalog_sha256=source_sha,utc=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--helper',type=Path,required=True);a=p.parse_args();main(a.helper)
