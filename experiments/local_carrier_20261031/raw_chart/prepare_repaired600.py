"""Complete17 remaining real endpoints; preserve the three prior successes."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil

from experiments.local_carrier_20261031 import prepare2000 as core
from experiments.local_carrier_20261031.raw_chart.prepare import allocations

B, PY = core.B, core.PYTHON
read, write, sha = core.read, core.write, core.digest
OLD = B/'runs/local_raw_chart_precision600_20261031'
SUCCESS = B/'runs/raw_chart_public_termination_gate600_retry_20261031'
ROOT = B/'runs/local_raw_chart_repaired600_20261031'
SELF = 'experiments/local_carrier_20261031/raw_chart/prepare_repaired600.py'
MODEL_COMMIT = '6e1693345786f2c190d0191685e4b485d24b4ab8'
LEGACY_HELPER = B/'runs/raw_chart_precision600_helpers_20261031/source-record.json'


def verify_source(record):
    source = read(record); manifest = read(source['manifest'])
    for key in ['source', 'source_commit', 'manifest', 'pythonpath']:
        assert isinstance(source[key], str) and source[key]
    assert manifest['source'] == source['source'] and manifest['source_commit'] == source['source_commit']
    assert Path(source['source']).is_absolute()
    for name, digest in manifest['files'].items(): assert sha(Path(manifest['source'])/name) == digest, name
    assert all(Path(x).exists() for x in source['pythonpath'].split(':'))
    return source


def remaining_jobs(builder, alloc, run, root, source, reference, already600):
    jobs = builder(alloc, run, source, reference, resume=not already600)
    if already600:
        jobs = jobs[-2:]
        for job in jobs:
            job['output'] = str(root/(job['phase']+'-'+alloc['id']))
            job['command'][job['command'].index('--output')+1] = job['output']
        jobs[0]['depends_on'] = []
    else:
        jobs[0]['command'].insert(3, '-v')  # actual native model import provenance
        full = next(job for job in jobs if job['phase']=='fullstate')
        full['command'] += ['--reference-prefix', 'sigreg']
    assert len(jobs) == (2 if already600 else 5)
    return jobs


def prepare(args):
    import torch
    torch.set_num_threads(2)
    source, helper = verify_source(args.model_source_record), verify_source(args.helper_source_record)
    assert source['source_commit'] == MODEL_COMMIT
    assert sha(__file__) == read(helper['manifest'])['files'][SELF]
    assert Path(core.__file__).is_relative_to(Path(source['source']))
    success = read(SUCCESS/'final-review.json')
    assert success['passed'] and success['direct_all_initial557_equal'] and success['additional_three_buffers_unchanged']
    assert success['source'] == source and (SUCCESS/'HOLD_DISPATCH').exists() and (OLD/'HOLD_DISPATCH').exists()
    prior_plan = read(SUCCESS/'plan.json'); assert len(prior_plan['jobs']) == 3
    previous = []
    for job in prior_plan['jobs']:
        path = SUCCESS/'exits'/(job['id']+'.json'); assert read(path)['returncode'] == 0
        previous.append(dict(id=job['id'], path=str(path), sha256=sha(path), returncode=0))
    # Reuse the previously reviewed pure five-endpoint builder. Its own legacy
    # helper manifest is provenance only, never a workflow/model source.
    legacy = read(LEGACY_HELPER); legacy_path = Path(legacy['source'])/'experiments/local_carrier_20261031/raw_chart/prepare_precision600.py'
    assert sha(legacy_path) == read(legacy['manifest'])['files']['experiments/local_carrier_20261031/raw_chart/prepare_precision600.py']
    spec = importlib.util.spec_from_file_location('raw_precision600_builder814', legacy_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert not ROOT.exists(); ROOT.mkdir(); jobs=[]; records=[]
    for alloc in allocations():
        name = alloc['id']; already600 = alloc['task']=='tworoom' and alloc['training_seed']==1
        parent = (SUCCESS if already600 else OLD)/name
        original = B/'runs/local_raw_chart600_20261031'/name
        identity = B/'runs/local_isotropic600_20261031'/f'{alloc["task"]}-spatial-local-seed{alloc["training_seed"]}'
        cfg = read(parent/'config.json'); cal = cfg['spatial_calibration']
        assert sha(cal['path']) == cal['sha256'] and read(cal['path'])['normalization']=='global_rms'
        initial = torch.load(parent/'initial-model/weights.pt', map_location='cpu', weights_only=True)
        raw = torch.load(original/'initial-model/weights.pt', map_location='cpu', weights_only=True)
        base = torch.load(identity/'initial-model/weights.pt', map_location='cpu', weights_only=True)
        assert len(initial)==len(raw)==557 and initial.keys()==raw.keys()
        assert all(torch.equal(v, initial[k]) for k,v in raw.items())
        assert len(base)==553 and all(k in initial and torch.equal(v,initial[k]) for k,v in base.items())
        del initial,raw,base
        primary=Path(cfg['data'])/'manifests/validation-strict.json'
        reference=read(identity.parent/('planning-'+identity.name)/'metrics.json')
        assert reference['manifest_sha256']==sha(primary) and len(reference['pairs'])==50 and reference['checkpoint_audit']['strict_equal']
        rec=dict(**alloc,parent=str(parent),original_raw=str(original),identity_reference=str(identity),
            original557_equal=True,identity553_equal=True,primary_manifest_sha256=sha(primary),calibration=cal)
        if already600:
            run=parent; rec.update(initial_step=600,training_reused=True,prior_three_actual_exits=previous)
        else:
            pre=read(B/'reports/raw-chart-precision600-failure-20261031'/(name+'-state400.json'))
            assert pre['passed'] and pre['step']==400 and sha(parent/'last.ckpt')==pre['checkpoint_sha256']
            run=ROOT/name;run.mkdir();nextcfg=copy.deepcopy(cfg);nextcfg['output']=str(run)
            assert core.semantic(nextcfg)==core.semantic(cfg)
            assert nextcfg['checkpoint_every']==200 and nextcfg['retain_steps']==[200,400,600]
            write(run/'config.json',nextcfg);shutil.copy2(parent/'last.ckpt',run/'last.ckpt')
            shutil.copytree(parent/'initial-model',run/'initial-model')
            for step in [200,400]:shutil.copytree(parent/f'model-step{step:07d}',run/f'model-step{step:07d}')
            assert sha(run/'last.ckpt')==pre['checkpoint_sha256']
            post=core.audit_state(run,400,parent,run/'initial-model/weights.pt',source)
            assert post['passed'] and post['checkpoint_sha256']==pre['checkpoint_sha256']
            assert post['rng_component_sha256']==pre['rng_component_sha256'] and post['sampler']==pre['sampler']
            assert post['frozen_tensors']['compared_tensors']==210
            write(run/'post-copy-state400.json',post)
            rec.update(initial_step=400,training_reused=False,complete400_byte_equal=True,post_copy_native_restore_passed=True,
                       fullstate_sha256=pre['checkpoint_sha256'])
        jobs.extend(remaining_jobs(module.jobs_for,alloc,run,ROOT,source,original,already600));records.append(rec)
        print(json.dumps(dict(prepared=name,initial_step=rec['initial_step'],initial557_and553_exact=True)),flush=True)
    assert len(jobs)==17 and len({j['id'] for j in jobs})==17 and sum('--resume' in j['command'] for j in jobs)==3
    assert all(j['source']==source for j in jobs)
    write(ROOT/'source-record.json',source);write(ROOT/'helper-source-record.json',helper)
    write(ROOT/'candidate-plan.json',dict(root=str(ROOT),deadline=None,environment=read(OLD/'plan.json')['environment'],jobs=jobs))
    write(ROOT/'protocol.json',dict(model_source=source,helper_source=helper,legacy_pure_job_builder=legacy,
        runs=records,new_endpoints=17,prior_successful_endpoints=previous,total_evidence_endpoints=20,
        scope='Exactly3 prior real successes plus17 new real endpoints across roots, not20 successes in one root.',
        authorization='After repairedTRseed1 full600/fullstate/health review: resume only other3 genuine400 states to600 and their original evaluations; TRseed1 adds only transition64 and primary50. No freshcases or budget>600.',
        schedule='dev13gpu0 existingTRseed1 evaluations;gpu1 TRseed2,gpu2 Cubeseed1,gpu3 Cubeseed2. All capped600.',
        unchanged='Model6e,validatedmetadata-v2,original observeddata/calibration/seed/batch128/nativeoptimizer/scheduler/RNG/sampler/sample-levelobjective/planner/cases.2128 chart parameters unchanged. CPU postcopy strictly restores the new source; original400states and holds remain immutable.',
        initial_scope='Direct557 raw and553 identity equality for every run; native paired-prefix audit now additionally includes sigreg for557, without activating SIGReg in the loss.',
        coordinates='Public/cachedstates areRAW; never apply another inverse in collection.',
        operations='Python-v records actual native imports only. Use existing22e control-plane wrapper, actual child/update startup checks and readonly1800smonitor. Do not equate dispatcher creation with training startup.',
        failures='All original numerical failures, unknown/unstarted metadata-attempt outcomes, captures and holds remain unchanged. Failed discarded suffix computations are historical overhead, not extra retained optimizer updates.'))
    assert not (ROOT/'plan.json').exists()
    print(json.dumps(dict(candidate=str(ROOT/'candidate-plan.json'),new_endpoints=17,prior_successes=3,launched=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model-source-record',required=True);parser.add_argument('--helper-source-record',required=True)
    prepare(parser.parse_args())
