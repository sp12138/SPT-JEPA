"""CPU-derived isotropic statistics, then unchanged real native preflight/600 gates.

Never manufacture calibration jobs/exits or modify original snapshots. The model
and original prepare implementation remain unchanged; only separate calibration
records and run roots are selected here.
"""
import argparse
import hashlib
import json
import copy
import importlib.util
import math
import os
import platform
import time
from pathlib import Path

_spec=importlib.util.spec_from_file_location('local_carrier_prepare',Path(__file__).with_name('prepare.py'))
base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)
SELF='experiments/local_carrier_20261031/prepare_isotropic.py'
ORIGINAL_CAL=base.B/'runs/local_carrier_calibration_20261031'
ORIGINAL_SOURCE=base.B/'runs/local_carrier_preflight_20261031/source-record.json'
base.CAL=base.B/'runs/local_isotropic_calibration_20261031'
base.PRE=base.B/'runs/local_isotropic_preflight_20261031'
base.ROOT=base.B/'runs/local_isotropic600_20261031'
_original_source_check=base.source


def source(record):
    src=_original_source_check(record);root=Path(src['source']);manifest=base.read(src['manifest'])
    assert base.digest(__file__)==base.digest(root/SELF)==manifest['files'][SELF]
    old=base.read(ORIGINAL_SOURCE);oldmanifest=base.read(old['manifest'])
    assert old['source_commit']=='96aca13f99b25dcf880346a1e699318150139ae5'
    for name in ('as_jepa_lewm/spatial_carrier.py','as_jepa_lewm/robust.py','as_jepa_lewm/model.py',
                 'as_jepa_lewm/train.py','as_jepa_lewm/factory.py'):
        assert manifest['files'][name]==oldmanifest['files'][name]
        assert base.digest(root/name)==base.digest(Path(old['source'])/name)==oldmanifest['files'][name]
    assert src['runtime_source']==old['runtime_source'] and src['runtime_manifest']==old['runtime_manifest']
    return src


def isotropic_record(original,original_path,src):
    """Pure derivation, with no image access or task-dependent choice."""
    assert original['measured_frames']==128 and original['source_commit']=='96aca13f99b25dcf880346a1e699318150139ae5'
    result=copy.deepcopy(original)
    for arm in ('local','dense'):
        old=original['arms'][arm];new=result['arms'][arm]
        assert old['scale_floor_coordinates']==0,'Cannot reconstruct raw variance from clipped statistics'
        values=old['model_kwargs']['spatial_scale'];means=old['model_kwargs']['spatial_mean']
        assert len(values)==len(means)==1024
        assert all(math.isfinite(v) and v>0 for v in values)
        assert all(math.isfinite(v) for v in means)
        scalar=math.sqrt(math.fsum(v*v for v in values)/1024)
        assert math.isfinite(scalar) and scalar>0
        new['model_kwargs']['spatial_scale']=[scalar]*1024
        new.update(normalization='global_rms',isotropic_scale=scalar,
            original_coordinate_scale_sha256=hashlib.sha256(
                json.dumps(values,separators=(',',':')).encode()).hexdigest(),
            original_raw_std_min=old['raw_std_min'],original_raw_std_mean=old['raw_std_mean'],original_raw_std_max=old['raw_std_max'])
        assert new['model_kwargs']['spatial_mean']==means
        assert new['projection_sha256']==old['projection_sha256']
    result.update(source_commit=src['source_commit'],source_sha256=base.digest(__file__),
        model_source_sha256=base.digest(Path(src['source'])/'as_jepa_lewm/spatial_carrier.py'),
        original_calibration=str(original_path),original_calibration_sha256=base.digest(original_path),
        original_source_commit=original['source_commit'],normalization='global_rms',
        original_scale_floor=original['scale_floor'],scale_floor=None,
        transformation='Keep original per-coordinate means; replace all1024 scales by sqrt(mean(original raw coordinate population variance)). '
                       'Original numerical floors are untriggered, so saved scales equal raw standard deviations.',
        execution=dict(kind='CPU JSON statistics derivation only',pid=os.getpid(),timestamp=time.time(),python=platform.python_version()),
        scope='Same original128 training-frame measurements; no new image access, GPU calibration, validation, fit, labels or optimizer. '
              'Original local/dense frozen projection and model are unchanged. This is a normalization repair, not a new algorithm.')
    assert result['encoder_initial_sha256']==original['encoder_initial_sha256']
    return result


def derive(args):
    src=source(args.source_record)
    assert not base.CAL.exists() and not base.PRE.exists() and not base.ROOT.exists()
    original_plan=base.read(ORIGINAL_CAL/'plan.json');assert len(original_plan['jobs'])==4
    results=[]
    for job in original_plan['jobs']:
        assert base.read(ORIGINAL_CAL/'exits'/(job['id']+'.json'))['returncode']==0
        original_path=Path(job['output'])/'calibration.json';original=base.read(original_path)
        record=isotropic_record(original,original_path,src)
        results.append((job['id'],record))
    base.CAL.mkdir();base.write(base.CAL/'source-record.json',src)
    records=[]
    for name,result in results:
        out=base.CAL/name;out.mkdir();path=out/'calibration.json';base.write(path,result)
        records.append(dict(id=name,path=str(path),sha256=base.digest(path),
            original_calibration=result['original_calibration'],original_sha256=result['original_calibration_sha256']))
    base.write(base.CAL/'derivation.json',dict(passed=True,kind='executed CPU derivation; not GPU jobs',
        source_commit=src['source_commit'],helper_sha256=base.digest(__file__),environment=original_plan['environment'],
        original_source_record_sha256=base.digest(ORIGINAL_SOURCE),records=records,
        no_new_images=True,no_jobs_or_exits_fabricated=True))
    print(json.dumps(dict(derived=str(base.CAL),calibrations=len(records),gpu_jobs=0)))


def calibration_ready(src):
    assert base.read(base.CAL/'source-record.json')==src
    record=base.read(base.CAL/'derivation.json');assert record['passed'] and len(record['records'])==4
    assert record['source_commit']==src['source_commit'] and record['helper_sha256']==base.digest(__file__)
    assert record['no_jobs_or_exits_fabricated'] and record['no_new_images']
    assert not (base.CAL/'exits').exists() and not (base.CAL/'plan.json').exists()
    for item in record['records']:
        assert base.digest(item['path'])==item['sha256']
        assert base.digest(item['original_calibration'])==item['original_sha256']
        derived=base.read(item['path']);original=base.read(item['original_calibration'])
        expected=isotropic_record(original,item['original_calibration'],src)
        expected.pop('execution');actual=copy.deepcopy(derived);actual.pop('execution')
        assert actual==expected,'Derived statistics or provenance changed'
    return record['environment']


base.source=source
base.calibration_ready=calibration_ready

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['derive','preflight','training'])
    p.add_argument('--source-record',required=True);args=p.parse_args()
    dict(derive=derive,preflight=base.preflight,training=base.training)[args.phase](args)
